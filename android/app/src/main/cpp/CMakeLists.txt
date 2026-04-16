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

target_compile_definitions(clpeak PRIVATE ANDROID_LOGGER USE_STUB_OPENCL CL_TARGET_OPENCL_VERSION=120)
target_compile_options(clpeak PRIVATE -fPIC -O2 -fexceptions)
set_target_properties(clpeak PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(clpeak log atomic dl OpenCL)
