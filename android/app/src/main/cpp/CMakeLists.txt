cmake_minimum_required(VERSION 3.20)
project(clpeak_android)

# Support 16 KB page sizes on Android 15+
set(ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
# Native CPU backend — plain C++ / std::thread, no external deps.
# (Threads come from Bionic libc; no find_package(Threads) needed.)
# -----------------------------------------------------------------------
set(CLPEAK_ENABLE_CPU ON)
add_subdirectory(${CLPEAK_ROOT}/src/cpu ${CMAKE_BINARY_DIR}/cpu)

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

    # The NDK ships the runtime stub (libvulkan.so) plus headers, but those
    # headers lag the Vulkan spec (NDK r28 = VK_HEADER_VERSION 275), so they
    # lack newer symbols the backend uses (bfloat16 / float8 cooperative
    # matrix, etc.).  The headers are pure spec-defined declarations — enum
    # values, struct layouts, extension-name strings — and every non-core
    # entry point is loaded at runtime via vkGetInstanceProcAddr, so compiling
    # against newer headers while linking the NDK stub is safe.  Use the
    # vendored Khronos Vulkan-Headers submodule and put it ahead of the
    # sysroot copy.
    set(VULKAN_HEADERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include)
    if(NOT EXISTS ${VULKAN_HEADERS_DIR}/vulkan/vulkan_core.h)
        message(FATAL_ERROR
            "Vulkan-Headers submodule missing. Run: git submodule update --init")
    endif()
    # src/vulkan/CMakeLists.txt adds ${Vulkan_INCLUDE_DIRS} to peak_vulkan, and
    # the JNI lib gets it transitively via PUBLIC.
    set(Vulkan_INCLUDE_DIRS ${VULKAN_HEADERS_DIR})

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
    peak_cpu
    log atomic dl
)
target_compile_definitions(clpeak PRIVATE ENABLE_CPU)

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

# Version
clpeak_setup_version(clpeak GIT_ROOT ${CLPEAK_GIT_ROOT})
