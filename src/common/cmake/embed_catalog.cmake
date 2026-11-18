# Embed the headline catalog (app/assets/measures.json) in the build, as a
# byte array the measure engine reads (HeadlineCatalog::bundled(),
# include/common/measures.h).  The app bundles the same file as an asset, so
# --summary and the app's Summary show the same headlines.
#
#   cmake -DIN=<measures.json> -DOUT=<measures_catalog.cpp> -P embed_catalog.cmake
#
# Run by src/common/CMakeLists.txt as a build step that depends on the file,
# so editing the catalog rebuilds the binaries without a reconfigure.  Bytes,
# not a string literal: MSVC caps one literal's length, and a catalog is free
# to hold any character.

if(NOT DEFINED IN OR NOT DEFINED OUT)
    message(FATAL_ERROR "embed_catalog.cmake: pass -DIN=<json> -DOUT=<cpp>")
endif()

file(READ "${IN}" hex HEX)
string(LENGTH "${hex}" hexLength)
math(EXPR size "${hexLength} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
# Sixteen to a line, so a diff of the generated file stays readable.
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){16})" "\\1\n    " bytes "${bytes}")

file(WRITE "${OUT}"
"// Generated from app/assets/measures.json by src/common/cmake/embed_catalog.cmake.
// Do not edit: edit the catalog, and the build regenerates this.

#include <cstddef>

namespace xpupeak {

extern const unsigned char kBundledCatalogJson[];
extern const std::size_t   kBundledCatalogJsonSize;

const unsigned char kBundledCatalogJson[] = {
    ${bytes}
    0x00};
const std::size_t kBundledCatalogJsonSize = ${size};

} // namespace xpupeak
")
