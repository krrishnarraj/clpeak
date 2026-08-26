# CompileShaders.cmake
#
# Compiles GLSL compute shaders to SPIR-V at configure time using glslc,
# then embeds the SPIR-V as uint32_t arrays in a generated C++ source file.
# All work is done at configure time to avoid build-time CMake list/shell
# escaping problems when passing file lists across custom_command boundaries.
#
# Usage:
#   compile_shaders(TARGET <target> SHADERS <shader1.comp> ...)
#   Generates ${CMAKE_CURRENT_BINARY_DIR}/vk_shaders_generated.cpp

function(compile_shaders)
  cmake_parse_arguments(CS "" "TARGET" "SHADERS" ${ARGN})

  # Always prefer the host Vulkan SDK's glslc, including for Android builds.
  # glslc runs at configure time and emits portable, architecture-neutral
  # SPIR-V that is embedded as C++ arrays and reused unchanged across every
  # ABI, so it is decoupled from the NDK cross-compilation toolchain. The host
  # SDK ships a modern glslang that supports the newer GLSL extensions some
  # shaders here depend on (integer dot product, bfloat16, cooperative matrix,
  # fp8); the glslc bundled with the NDK is frozen at an old shaderc/glslang
  # (v2022.x as of NDK r30) that rejects them.
  #
  # An explicit -DCLPEAK_GLSLC=/path/to/glslc override wins over everything.
  set(_shader_tool_hints "$ENV{VULKAN_SDK}/bin" /usr/local/bin /opt/homebrew/bin)
  if(DEFINED CLPEAK_GLSLC)
    set(GLSLC "${CLPEAK_GLSLC}")
  else()
    find_program(GLSLC glslc HINTS ${_shader_tool_hints})
  endif()

  # Last-resort fallback: the glslc bundled with the Android NDK
  # (shader-tools/). It is older than the host SDK and will skip shaders that
  # need newer GLSL extensions (those benchmarks are then excluded). We only
  # use it when no host glslc is available.
  #
  # TODO: revisit and drop this fallback once the NDK ships a glslc new enough
  # to compile every shader in src/vulkan/shaders/.
  if(NOT GLSLC AND DEFINED ANDROID_NDK)
    # ANDROID_HOST_TAG is set by android.toolchain.cmake (e.g. "darwin-x86_64").
    if(NOT ANDROID_HOST_TAG)
      if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        set(ANDROID_HOST_TAG "darwin-x86_64")
      elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        set(ANDROID_HOST_TAG "linux-x86_64")
      elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        set(ANDROID_HOST_TAG "windows-x86_64")
      endif()
    endif()
    find_program(GLSLC glslc HINTS "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}")
    if(GLSLC)
      message(WARNING "Using the NDK's bundled glslc (${GLSLC}); it is older "
                      "than the host Vulkan SDK and will skip shaders that need "
                      "newer GLSL extensions. Install the Vulkan SDK / set "
                      "VULKAN_SDK to build all shaders.")
    endif()
  endif()

  if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK or set VULKAN_SDK env var.")
  endif()

  # Re-run cmake configure whenever a shader source changes.  Shared GLSL
  # headers (mad_chain.glsl) are picked up too -- they are not in CS_SHADERS
  # but every shader that includes one has to be rebuilt when they change.
  foreach(SHADER ${CS_SHADERS})
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SHADER}")
  endforeach()
  file(GLOB _CS_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")
  foreach(HDR ${_CS_HEADERS})
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${HDR}")
  endforeach()

  set(GEN_CPP "${CMAKE_CURRENT_BINARY_DIR}/vk_shaders_generated.cpp")
  set(CPP_CONTENT "#include <cstdint>\n#include <cstddef>\n\nnamespace vk_shaders {\n\n")

  # Compile one .comp into vk_shaders::<SYMBOL> and append it to CPP_CONTENT.
  # A macro, not a function, so the append lands in the caller's scope.
  # Sets _CS_OK to TRUE on success, FALSE when glslc rejected the shader.
  macro(_cs_embed SHADER SYMBOL EXTRA_ARG)
    set(_CS_SPV "${CMAKE_CURRENT_BINARY_DIR}/${SYMBOL}.spv")
    get_filename_component(_CS_DIR "${SHADER}" DIRECTORY)

    execute_process(
      # vulkan1.2 (SPIR-V 1.5), not vulkan1.3 (SPIR-V 1.6), and the difference
      # is not cosmetic.  At 1.6 a specialized work-group size compiles to
      # OpExecutionModeId LocalSizeId, which is conditional on the maintenance4
      # feature and is the one structural difference between our coopmat
      # modules and the ones a driver that faults on ours is known to compile.
      # At 1.5 the same GLSL compiles to the classic OpExecutionMode LocalSize
      # plus a gl_WorkGroupSize spec-constant composite, which every driver has
      # handled since Vulkan 1.0.  Nothing here needs 1.6, and 1.5 modules load
      # on Vulkan 1.2 devices that 1.6 modules cannot.
      COMMAND "${GLSLC}" --target-env=vulkan1.2 -O -I "${_CS_DIR}"
              ${EXTRA_ARG} "${SHADER}" -o "${_CS_SPV}"
      RESULT_VARIABLE _CS_RESULT
      ERROR_VARIABLE  _CS_ERROR
    )
    if(NOT _CS_RESULT EQUAL 0)
      # Some shaders need newer GLSL/SPIR-V features (integer dot product,
      # bfloat16, cooperative matrix) that older glslc versions -- notably the
      # one bundled with the Android NDK -- do not support.  Skip with a
      # warning so the build still produces a functional binary; host code is
      # gated on the VK_HAS_<SYMBOL> defines set below.
      message(WARNING "glslc could not build ${SYMBOL}; skipping it. Error:\n${_CS_ERROR}")
      set(_CS_OK FALSE)
    else()
      # Read SPIR-V binary as a hex string (two chars per byte).
      # SPIR-V files on all glslc targets use little-endian word order, matching
      # every platform Vulkan runs on, so we reconstruct each 4-byte word as
      # 0x<B3><B2><B1><B0> which equals the correct host uint32_t value.
      file(READ "${_CS_SPV}" _CS_HEX HEX)
      string(LENGTH "${_CS_HEX}" _CS_HEX_LEN)

      set(_CS_INIT "")
      set(_CS_COL 0)
      set(_CS_I 0)
      while(${_CS_I} LESS ${_CS_HEX_LEN})
        math(EXPR _CS_I0 "${_CS_I} + 0")
        math(EXPR _CS_I1 "${_CS_I} + 2")
        math(EXPR _CS_I2 "${_CS_I} + 4")
        math(EXPR _CS_I3 "${_CS_I} + 6")
        string(SUBSTRING "${_CS_HEX}" ${_CS_I0} 2 _CS_B0)
        string(SUBSTRING "${_CS_HEX}" ${_CS_I1} 2 _CS_B1)
        string(SUBSTRING "${_CS_HEX}" ${_CS_I2} 2 _CS_B2)
        string(SUBSTRING "${_CS_HEX}" ${_CS_I3} 2 _CS_B3)
        string(APPEND _CS_INIT "0x${_CS_B3}${_CS_B2}${_CS_B1}${_CS_B0},")
        math(EXPR _CS_COL "${_CS_COL} + 1")
        if(_CS_COL EQUAL 8)
          string(APPEND _CS_INIT "\n    ")
          set(_CS_COL 0)
        endif()
        math(EXPR _CS_I "${_CS_I} + 8")
      endwhile()

      # Use 'extern' so the symbols have external linkage.
      # (In C++, `const` at namespace scope defaults to internal linkage; since
      #  vk_shaders_generated.cpp doesn't include vk_peak.h the compiler can't
      #  see that the header already marked these `extern`.)
      string(APPEND CPP_CONTENT "// Auto-generated from ${SHADER}\n")
      string(APPEND CPP_CONTENT "extern const uint32_t ${SYMBOL}[] = {\n    ${_CS_INIT}\n};\n")
      string(APPEND CPP_CONTENT "extern const size_t ${SYMBOL}_size = sizeof(${SYMBOL});\n\n")
      set(_CS_OK TRUE)
    endif()
  endmacro()

  foreach(SHADER ${CS_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    string(TOUPPER "${SHADER_NAME}" SHADER_NAME_UPPER)

    message(STATUS "Compiling shader: ${SHADER_NAME}.comp -> SPIR-V")
    _cs_embed("${SHADER}" "${SHADER_NAME}" "")
    if(NOT _CS_OK)
      continue()
    endif()
    target_compile_definitions(${CS_TARGET} PUBLIC VK_HAS_${SHADER_NAME_UPPER})

    # A shader that pulls in mad_chain.glsl gets a second build with the
    # affine chain shape, embedded as <name>_alt.  runComputeKernel times both
    # and reports the faster -- neither shape reaches peak on every vendor.
    # Adopting the shared chain in a shader is the only step needed; this is
    # detected from the source rather than listed anywhere.
    file(READ "${SHADER}" _CS_SRC)
    if(_CS_SRC MATCHES "mad_chain\\.glsl")
      _cs_embed("${SHADER}" "${SHADER_NAME}_alt" "-DMAD_CHAIN_AFFINE")
      if(_CS_OK)
        target_compile_definitions(${CS_TARGET} PUBLIC VK_HAS_${SHADER_NAME_UPPER}_ALT)
      endif()
    endif()
  endforeach()

  string(APPEND CPP_CONTENT "} // namespace vk_shaders\n")
  file(WRITE "${GEN_CPP}" "${CPP_CONTENT}")

  target_sources(${CS_TARGET} PRIVATE "${GEN_CPP}")
endfunction()
