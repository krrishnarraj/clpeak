# CombineHex.cmake
# Combines multiple .spv.hex files into a single C++ source file.
# Variables: HEX_FILES (semicolon-separated list), OUT_FILE

set(CONTENT "#include <cstdint>\n#include <cstddef>\n\nnamespace vk_shaders {\n\n")

foreach(HEX_FILE ${HEX_FILES})
  file(READ ${HEX_FILE} HEX_CONTENT)
  set(CONTENT "${CONTENT}${HEX_CONTENT}")
endforeach()

set(CONTENT "${CONTENT}} // namespace vk_shaders\n")

file(WRITE ${OUT_FILE} "${CONTENT}")
