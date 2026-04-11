# SpvToHex.cmake
# Reads a SPIR-V binary file and writes a C++ hex array to HEX_FILE.
# Variables: SPV_FILE, HEX_FILE, NAME

file(READ ${SPV_FILE} SPV_DATA HEX)
string(LENGTH "${SPV_DATA}" SPV_LEN)

# Convert hex pairs to 0xNN format
set(HEX_ARRAY "")
set(BYTE_COUNT 0)
math(EXPR LAST "${SPV_LEN} - 1")

# Build uint32_t array (4 bytes at a time, little-endian)
set(UINT32_ARRAY "")
set(UINT32_COUNT 0)
set(BYTES "")

set(I 0)
while(I LESS SPV_LEN)
  string(SUBSTRING "${SPV_DATA}" ${I} 2 BYTE)
  list(APPEND BYTES "${BYTE}")
  math(EXPR BYTE_COUNT "${BYTE_COUNT} + 1")
  math(EXPR I "${I} + 2")

  list(LENGTH BYTES BLEN)
  if(BLEN EQUAL 4)
    # Little-endian uint32: bytes[0] is lowest
    list(GET BYTES 0 B0)
    list(GET BYTES 1 B1)
    list(GET BYTES 2 B2)
    list(GET BYTES 3 B3)
    set(UINT32_ARRAY "${UINT32_ARRAY}0x${B3}${B2}${B1}${B0},")
    math(EXPR UINT32_COUNT "${UINT32_COUNT} + 1")
    # Newline every 8 values
    math(EXPR MOD "${UINT32_COUNT} % 8")
    if(MOD EQUAL 0)
      set(UINT32_ARRAY "${UINT32_ARRAY}\n    ")
    endif()
    set(BYTES "")
  endif()
endwhile()

set(CONTENT "// Auto-generated from ${NAME}.comp\n")
set(CONTENT "${CONTENT}const uint32_t ${NAME}[] = {\n    ${UINT32_ARRAY}\n};\n")
set(CONTENT "${CONTENT}const size_t ${NAME}_size = sizeof(${NAME});\n\n")

file(WRITE ${HEX_FILE} "${CONTENT}")
