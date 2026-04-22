cmake_minimum_required(VERSION 3.20)

# Build OpenCL stub
add_subdirectory(libopencl-stub)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
include(${CLPEAK_ROOT}/cmake/version.cmake)
include(${CLPEAK_ROOT}/cmake/sources.cmake)

add_library(clpeak SHARED
    ${CLPEAK_CORE_SOURCES}
    ${CLPEAK_ROOT}/src/logger_android.cpp
    ${CLPEAK_ROOT}/src/entry_android.cpp
)

clpeak_setup_version(clpeak)

target_include_directories(clpeak PRIVATE
    libopencl-stub/include
    ${CLPEAK_CORE_INCLUDE_DIRS}
)

# Target OpenCL 3.0 headers so 3.0-only tests (e.g. INT8 dot-product) are
# reachable. The stub dlopens the device's libOpenCL at runtime, and each
# test checks its extension string before running — devices with a 1.2
# driver skip the 3.0 tests cleanly.
target_compile_definitions(clpeak PRIVATE ANDROID_LOGGER USE_STUB_OPENCL CL_TARGET_OPENCL_VERSION=300 CLPEAK_HAS_OPENCL_30)
target_compile_options(clpeak PRIVATE -fPIC -O2 -fexceptions)
set_target_properties(clpeak PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(clpeak log atomic dl OpenCL)
