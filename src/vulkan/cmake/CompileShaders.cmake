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

  # Re-run cmake configure whenever a shader source changes
  foreach(SHADER ${CS_SHADERS})
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SHADER}")
  endforeach()

  set(GEN_CPP "${CMAKE_CURRENT_BINARY_DIR}/vk_shaders_generated.cpp")
  set(CPP_CONTENT "#include <cstdint>\n#include <cstddef>\n\nnamespace vk_shaders {\n\n")

  foreach(SHADER ${CS_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv")

    message(STATUS "Compiling shader: ${SHADER_NAME}.comp -> SPIR-V")
    execute_process(
      COMMAND "${GLSLC}" --target-env=vulkan1.3 -O "${SHADER}" -o "${SPV_FILE}"
      RESULT_VARIABLE GLSLC_RESULT
      ERROR_VARIABLE  GLSLC_ERROR
    )
    if(NOT GLSLC_RESULT EQUAL 0)
      # Some shaders require newer GLSL/SPIR-V features (e.g. integer dot
      # product) that older glslc versions (notably the one bundled with
      # the Android NDK) don't support. Skip with a warning so the build
      # still produces a functional binary; host code is gated with
      # VK_HAS_<SHADER_NAME_UPPER> defines (see below).
      message(WARNING "glslc could not build ${SHADER_NAME}.comp; skipping it. Error:\n${GLSLC_ERROR}")
      continue()
    endif()
    string(TOUPPER "${SHADER_NAME}" SHADER_NAME_UPPER)
    target_compile_definitions(${CS_TARGET} PUBLIC VK_HAS_${SHADER_NAME_UPPER})

    # Read SPIR-V binary as a hex string (two chars per byte)
    file(READ "${SPV_FILE}" SPV_HEX HEX)
    string(LENGTH "${SPV_HEX}" HEX_LEN)

    # Build the uint32_t array initializer.
    # SPIR-V files on all glslc targets use little-endian word order, matching
    # every platform Vulkan runs on, so we reconstruct each 4-byte word as
    # 0x<B3><B2><B1><B0> which equals the correct host uint32_t value.
    set(ARRAY_INIT "")
    set(COL 0)
    set(I 0)
    while(${I} LESS ${HEX_LEN})
      math(EXPR I0 "${I} + 0")
      math(EXPR I1 "${I} + 2")
      math(EXPR I2 "${I} + 4")
      math(EXPR I3 "${I} + 6")
      string(SUBSTRING "${SPV_HEX}" ${I0} 2 B0)
      string(SUBSTRING "${SPV_HEX}" ${I1} 2 B1)
      string(SUBSTRING "${SPV_HEX}" ${I2} 2 B2)
      string(SUBSTRING "${SPV_HEX}" ${I3} 2 B3)
      string(APPEND ARRAY_INIT "0x${B3}${B2}${B1}${B0},")
      math(EXPR COL "${COL} + 1")
      if(COL EQUAL 8)
        string(APPEND ARRAY_INIT "\n    ")
        set(COL 0)
      endif()
      math(EXPR I "${I} + 8")
    endwhile()

    # Use 'extern' so the symbols have external linkage.
    # (In C++, `const` at namespace scope defaults to internal linkage; since
    #  vk_shaders_generated.cpp doesn't include vk_peak.h the compiler can't
    #  see that the header already marked these `extern`.)
    string(APPEND CPP_CONTENT "// Auto-generated from ${SHADER_NAME}.comp\n")
    string(APPEND CPP_CONTENT "extern const uint32_t ${SHADER_NAME}[] = {\n    ${ARRAY_INIT}\n};\n")
    string(APPEND CPP_CONTENT "extern const size_t ${SHADER_NAME}_size = sizeof(${SHADER_NAME});\n\n")
  endforeach()

  string(APPEND CPP_CONTENT "} // namespace vk_shaders\n")
  file(WRITE "${GEN_CPP}" "${CPP_CONTENT}")

  target_sources(${CS_TARGET} PRIVATE "${GEN_CPP}")
endfunction()
