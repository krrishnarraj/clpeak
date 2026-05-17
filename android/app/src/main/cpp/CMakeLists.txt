cmake_minimum_required(VERSION 3.20)

# Build OpenCL stub (provides headers + dlopen-based libOpenCL)
add_subdirectory(libopencl-stub)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
get_filename_component(CLPEAK_GIT_ROOT "${CLPEAK_ROOT}" ABSOLUTE)
include(${CLPEAK_ROOT}/src/common/cmake/version.cmake)

# -----------------------------------------------------------------------
# Shared common library — always built, no logger (Android provides its own)
# -----------------------------------------------------------------------
add_subdirectory(${CLPEAK_ROOT}/src/common ${CMAKE_BINARY_DIR}/common)

# -----------------------------------------------------------------------
# OpenCL backend — predefine OpenCL vars so src/opencl/CMakeLists.txt
# skips find_package and uses the stub
# -----------------------------------------------------------------------
set(CLPEAK_ENABLE_OPENCL ON)
set(CLPEAK_OPENCL_PREDEFINED ON)
set(OpenCL_LIBRARIES OpenCL)
set(OpenCL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/libopencl-stub/include
    ${CLPEAK_ROOT}/include
)
# HPP_FOUND is needed by the OpenCL CMakeLists for the opencl.hpp path
set(HPP_FOUND "${CMAKE_CURRENT_SOURCE_DIR}/libopencl-stub/include")
add_subdirectory(${CLPEAK_ROOT}/src/opencl ${CMAKE_BINARY_DIR}/opencl)

# -----------------------------------------------------------------------
# Vulkan backend — Vulkan is part of the NDK since API 24
# -----------------------------------------------------------------------
find_library(ANDROID_VULKAN_LIB vulkan)
if(ANDROID_VULKAN_LIB)
    set(CLPEAK_ENABLE_VULKAN ON)
    set(Vulkan_FOUND TRUE)
    set(Vulkan_LIBRARIES ${ANDROID_VULKAN_LIB})
    # NDK provides vulkan headers on the default include path;
    # Vulkan_INCLUDE_DIRS is not strictly needed here.
    add_subdirectory(${CLPEAK_ROOT}/src/vulkan ${CMAKE_BINARY_DIR}/vulkan)
endif()

# -----------------------------------------------------------------------
# JNI shared library — links common + backends + Android-specific sources
# -----------------------------------------------------------------------
add_library(clpeak SHARED
    logger_android.cpp
    entry_android.cpp
    enumerate_backends.cpp
)

target_link_libraries(clpeak
    peak_common
    peak_opencl
    log atomic dl
)

if(TARGET peak_vulkan)
    target_link_libraries(clpeak peak_vulkan)
    target_compile_definitions(clpeak PRIVATE ENABLE_VULKAN)
endif()

target_include_directories(clpeak PRIVATE
    ${CLPEAK_ROOT}/include
    ${CLPEAK_ROOT}/include/common
    ${CLPEAK_ROOT}/src/opencl
)

target_compile_definitions(clpeak PRIVATE ANDROID_LOGGER CL_TARGET_OPENCL_VERSION=300)
target_compile_options(clpeak PRIVATE -fPIC -O2 -fexceptions)
set_target_properties(clpeak PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Version
clpeak_setup_version(clpeak GIT_ROOT ${CLPEAK_GIT_ROOT})
