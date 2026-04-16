# CombineHex.cmake
# Combines multiple .spv.hex files into a single C++ source file.
# Variables: MANIFEST_FILE (path to file containing ;-separated hex file list), OUT_FILE

file(READ ${MANIFEST_FILE} HEX_FILES_RAW)
# file(GENERATE) writes the list as a semicolon-separated string
string(REPLACE ";" ";" HEX_FILES "${HEX_FILES_RAW}")  # keep as-is; CMake will split on ;

set(CONTENT "#include <cstdint>\n#include <cstddef>\n\nnamespace vk_shaders {\n\n")

foreach(HEX_FILE ${HEX_FILES})
  if(NOT HEX_FILE STREQUAL "")
    file(READ "${HEX_FILE}" HEX_CONTENT)
    set(CONTENT "${CONTENT}${HEX_CONTENT}")
  endif()
endforeach()

set(CONTENT "${CONTENT}} // namespace vk_shaders\n")

file(WRITE ${OUT_FILE} "${CONTENT}")
