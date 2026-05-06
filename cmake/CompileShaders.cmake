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
  # Each shader in FP_SHADERS is compiled twice -- once with default flags
  # and once with `-DRELAXED_MATH=1`.  The second invocation embeds a
  # `<name>_relaxed` array next to `<name>` so vk_peak.cpp can pick the
  # variant matching the current PrecisionMode at runtime.  Shaders not
  # in FP_SHADERS are compiled once.
  cmake_parse_arguments(CS "" "TARGET" "SHADERS;FP_SHADERS" ${ARGN})

  # Search Vulkan SDK first, then the NDK's bundled shader-tools when building
  # for Android (Gradle passes ANDROID_NDK to CMake; ANDROID_HOST_TAG is set by
  # android.toolchain.cmake, e.g. "darwin-x86_64" / "linux-x86_64").
  set(_shader_tool_hints "$ENV{VULKAN_SDK}/bin")
  if(DEFINED ANDROID_NDK)
    if(NOT ANDROID_HOST_TAG)
      if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        set(ANDROID_HOST_TAG "darwin-x86_64")
      elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        set(ANDROID_HOST_TAG "linux-x86_64")
      elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        set(ANDROID_HOST_TAG "windows-x86_64")
      endif()
    endif()
    list(APPEND _shader_tool_hints "${ANDROID_NDK}/shader-tools/${ANDROID_HOST_TAG}")
  endif()
  find_program(GLSLC glslc HINTS ${_shader_tool_hints})
  if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK or set VULKAN_SDK env var.")
  endif()
  # spirv-opt is shipped alongside glslc in both the Vulkan SDK and the NDK
  # shader-tools tarball.  Used to produce relaxed-math FP shader variants
  # via `--relax-float-ops` (decorates every fp op with RelaxedPrecision).
  find_program(SPIRV_OPT spirv-opt HINTS ${_shader_tool_hints})

  # Re-run cmake configure whenever a shader source changes
  foreach(SHADER ${CS_SHADERS})
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${SHADER}")
  endforeach()

  set(GEN_CPP "${CMAKE_CURRENT_BINARY_DIR}/vk_shaders_generated.cpp")
  set(CPP_CONTENT "#include <cstdint>\n#include <cstddef>\n\nnamespace vk_shaders {\n\n")

  # Read a SPIR-V binary and emit the comma-separated uint32_t array
  # initializer text into `_ARRAY_INIT` in the parent scope.
  function(_spv_to_array SPV_FILE)
    file(READ "${SPV_FILE}" SPV_HEX HEX)
    string(LENGTH "${SPV_HEX}" HEX_LEN)
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
    set(_ARRAY_INIT "${ARRAY_INIT}" PARENT_SCOPE)
  endfunction()

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
      # CLPEAK_VK_HAS_<SHADER_NAME_UPPER> defines (see below).
      message(WARNING "glslc could not build ${SHADER_NAME}.comp; skipping it. Error:\n${GLSLC_ERROR}")
      continue()
    endif()
    string(TOUPPER "${SHADER_NAME}" SHADER_NAME_UPPER)
    target_compile_definitions(${CS_TARGET} PRIVATE CLPEAK_VK_HAS_${SHADER_NAME_UPPER})

    _spv_to_array("${SPV_FILE}")
    string(APPEND CPP_CONTENT "// Auto-generated from ${SHADER_NAME}.comp\n")
    string(APPEND CPP_CONTENT "extern const uint32_t ${SHADER_NAME}[] = {\n    ${_ARRAY_INIT}\n};\n")
    string(APPEND CPP_CONTENT "extern const size_t ${SHADER_NAME}_size = sizeof(${SHADER_NAME});\n\n")

    # Floating-point shaders also get a relaxed-math variant produced by
    # `spirv-opt --relax-float-ops -O <default.spv> -o <relaxed.spv>`, which
    # decorates every fp op with RelaxedPrecision.  Drivers that honour the
    # hint (notably mobile GPUs) may downcast to fp16; on desktop drivers it
    # is typically a no-op.
    list(FIND CS_FP_SHADERS "${SHADER}" _is_fp)
    if(NOT _is_fp EQUAL -1 AND SPIRV_OPT)
      set(SPV_FAST "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}_relaxed.spv")
      message(STATUS "Relaxing shader:  ${SHADER_NAME}_relaxed.spv (spirv-opt --relax-float-ops)")
      execute_process(
        COMMAND "${SPIRV_OPT}" "${SPV_FILE}" --relax-float-ops -O -o "${SPV_FAST}"
        RESULT_VARIABLE SPIRV_OPT_RESULT
        ERROR_VARIABLE  SPIRV_OPT_ERROR
      )
      if(SPIRV_OPT_RESULT EQUAL 0)
        target_compile_definitions(${CS_TARGET} PRIVATE CLPEAK_VK_HAS_${SHADER_NAME_UPPER}_RELAXED)
        _spv_to_array("${SPV_FAST}")
        string(APPEND CPP_CONTENT "// Auto-generated relaxed-math variant of ${SHADER_NAME}.comp\n")
        string(APPEND CPP_CONTENT "extern const uint32_t ${SHADER_NAME}_relaxed[] = {\n    ${_ARRAY_INIT}\n};\n")
        string(APPEND CPP_CONTENT "extern const size_t ${SHADER_NAME}_relaxed_size = sizeof(${SHADER_NAME}_relaxed);\n\n")
      else()
        message(WARNING "spirv-opt could not relax ${SHADER_NAME}.spv; relaxed variant skipped. Error:\n${SPIRV_OPT_ERROR}")
      endif()
    endif()
  endforeach()

  string(APPEND CPP_CONTENT "} // namespace vk_shaders\n")
  file(WRITE "${GEN_CPP}" "${CPP_CONTENT}")

  target_sources(${CS_TARGET} PRIVATE "${GEN_CPP}")
endfunction()
