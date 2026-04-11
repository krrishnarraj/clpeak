# CompileShaders.cmake
#
# Compiles GLSL compute shaders to SPIR-V using glslc (from Vulkan SDK),
# then generates a C++ source file with the SPIR-V embedded as uint32_t arrays.
#
# Usage:
#   compile_shaders(TARGET target SHADERS shader1.comp shader2.comp ...)
#   Generates ${CMAKE_CURRENT_BINARY_DIR}/vk_shaders_generated.cpp
#

function(compile_shaders)
  cmake_parse_arguments(CS "" "TARGET" "SHADERS" ${ARGN})

  find_program(GLSLC glslc HINTS "$ENV{VULKAN_SDK}/bin")
  if(NOT GLSLC)
    message(FATAL_ERROR "glslc not found. Install the Vulkan SDK or set VULKAN_SDK env var.")
  endif()

  set(SPV_FILES "")
  set(CPP_CONTENT "#include <cstdint>\n#include <cstddef>\n\nnamespace vk_shaders {\n\n")

  foreach(SHADER ${CS_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SPV_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv")
    set(SPV_HEX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv.hex")

    add_custom_command(
      OUTPUT ${SPV_FILE}
      COMMAND ${GLSLC} -O ${SHADER} -o ${SPV_FILE}
      DEPENDS ${SHADER}
      COMMENT "Compiling GLSL shader ${SHADER_NAME}.comp -> SPIR-V"
    )

    # Generate hex include at build time via a script
    add_custom_command(
      OUTPUT ${SPV_HEX_FILE}
      COMMAND ${CMAKE_COMMAND} -DSPV_FILE=${SPV_FILE} -DHEX_FILE=${SPV_HEX_FILE} -DNAME=${SHADER_NAME}
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SpvToHex.cmake
      DEPENDS ${SPV_FILE}
      COMMENT "Generating C++ hex array for ${SHADER_NAME}"
    )

    list(APPEND SPV_FILES ${SPV_HEX_FILE})
  endforeach()

  # Generate the combined C++ file
  set(GEN_CPP "${CMAKE_CURRENT_BINARY_DIR}/vk_shaders_generated.cpp")
  add_custom_command(
    OUTPUT ${GEN_CPP}
    COMMAND ${CMAKE_COMMAND} -DHEX_FILES="${SPV_FILES}" -DOUT_FILE=${GEN_CPP}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CombineHex.cmake
    DEPENDS ${SPV_FILES}
    COMMENT "Combining shader hex arrays"
  )

  target_sources(${CS_TARGET} PRIVATE ${GEN_CPP})
endfunction()
