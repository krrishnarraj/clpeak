cmake_minimum_required(VERSION 3.20)
project(xpupeak_android)

# Standalone superproject consumed by the Flutter app's Gradle build
# (app/android/app/build.gradle.kts → externalNativeBuild).  Builds the
# xpupeak_ffi shared library with the Android backend set:
# OpenCL (dlopen stub) + Vulkan (NDK) + CPU + ONNX Runtime + LiteRT.

# Support 16 KB page sizes on Android 15+
set(ANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES ON CACHE BOOL "" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(XPUPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
get_filename_component(XPUPEAK_ROOT "${XPUPEAK_ROOT}" ABSOLUTE)
set(XPUPEAK_GIT_ROOT "${XPUPEAK_ROOT}")
include(${XPUPEAK_ROOT}/src/common/cmake/version.cmake)

# Build OpenCL stub (provides headers + dlopen-based libOpenCL)
add_subdirectory(${XPUPEAK_ROOT}/third_party/libopencl-stub
                 ${CMAKE_BINARY_DIR}/libopencl-stub)

# -----------------------------------------------------------------------
# Shared common library
# -----------------------------------------------------------------------
add_subdirectory(${XPUPEAK_ROOT}/src/common ${CMAKE_BINARY_DIR}/common)

# -----------------------------------------------------------------------
# Native CPU backend — plain C++ / std::thread, no external deps.
# (Threads come from Bionic libc; no find_package(Threads) needed.)
# -----------------------------------------------------------------------
set(XPUPEAK_ENABLE_CPU ON)
add_subdirectory(${XPUPEAK_ROOT}/src/cpu ${CMAKE_BINARY_DIR}/cpu)

# -----------------------------------------------------------------------
# OpenCL backend — predefine OpenCL vars so src/opencl/CMakeLists.txt
# skips find_package and uses the stub
# -----------------------------------------------------------------------
set(XPUPEAK_ENABLE_OPENCL ON)
set(XPUPEAK_OPENCL_PREDEFINED ON)
set(OpenCL_LIBRARIES OpenCL)
set(OpenCL_INCLUDE_DIRS
    ${XPUPEAK_ROOT}/third_party/libopencl-stub/include
    ${XPUPEAK_ROOT}/include
)
# HPP_FOUND is needed by the OpenCL CMakeLists for the opencl.hpp path
set(HPP_FOUND "${XPUPEAK_ROOT}/third_party/libopencl-stub/include")
add_subdirectory(${XPUPEAK_ROOT}/src/opencl ${CMAKE_BINARY_DIR}/opencl)

# -----------------------------------------------------------------------
# Vulkan backend — Vulkan is part of the NDK since API 24
# -----------------------------------------------------------------------
find_library(ANDROID_VULKAN_LIB vulkan)
if(ANDROID_VULKAN_LIB)
    set(XPUPEAK_ENABLE_VULKAN ON)
    set(Vulkan_FOUND TRUE)
    set(Vulkan_LIBRARIES ${ANDROID_VULKAN_LIB})

    # The NDK ships the runtime stub (libvulkan.so) plus headers, but those
    # headers lag the Vulkan spec (NDK r30 = VK_HEADER_VERSION 335), so they
    # lack newer symbols the backend uses (VK_EXT_cooperative_matrix_maintenance1,
    # etc.).  The headers are pure spec-defined declarations — enum
    # values, struct layouts, extension-name strings — and every non-core
    # entry point is loaded at runtime via vkGetInstanceProcAddr, so compiling
    # against newer headers while linking the NDK stub is safe.  Use the
    # vendored Khronos Vulkan-Headers submodule and put it ahead of the
    # sysroot copy.
    set(VULKAN_HEADERS_DIR ${XPUPEAK_ROOT}/third_party/Vulkan-Headers/include)
    if(NOT EXISTS ${VULKAN_HEADERS_DIR}/vulkan/vulkan_core.h)
        message(FATAL_ERROR
            "Vulkan-Headers submodule missing. Run: git submodule update --init")
    endif()
    # src/vulkan/CMakeLists.txt adds ${Vulkan_INCLUDE_DIRS} to peak_vulkan, and
    # xpupeak_ffi gets it transitively via PUBLIC.
    set(Vulkan_INCLUDE_DIRS ${VULKAN_HEADERS_DIR})

    add_subdirectory(${XPUPEAK_ROOT}/src/vulkan ${CMAKE_BINARY_DIR}/vulkan)
endif()

# -----------------------------------------------------------------------
# ONNX Runtime backend — NNAPI / XNNPACK / CPU execution providers.
# Nothing to find at build time: only the vendored C-API header is needed,
# and libonnxruntime.so is dlopen'd at run time.  The app packages one
# (com.microsoft.onnxruntime:onnxruntime-android, see
# app/android/app/build.gradle.kts) so the bare soname resolves out of the
# APK's lib dir; a build without it still works, reporting the backend
# unavailable until the settings screen names a library.
# -----------------------------------------------------------------------
set(XPUPEAK_ENABLE_ONNX ON)
add_subdirectory(${XPUPEAK_ROOT}/src/onnx ${CMAKE_BINARY_DIR}/onnx)

# -----------------------------------------------------------------------
# LiteRT backend -- CPU (XNNPACK) / GPU (OpenCL) / NPU (vendor dispatch).
# Headers only at build time; libLiteRt.so comes from the
# com.google.ai.edge.litert AAR the app packages (see
# app/android/app/build.gradle.kts) and is dlopen'd by its bare soname.
# -----------------------------------------------------------------------
set(XPUPEAK_ENABLE_LITERT ON)
add_subdirectory(${XPUPEAK_ROOT}/src/litert ${CMAKE_BINARY_DIR}/litert)

# -----------------------------------------------------------------------
# The FFI bridge — the same library every other platform loads
# -----------------------------------------------------------------------
set(XPUPEAK_FFI_STANDALONE ON)
add_subdirectory(${XPUPEAK_ROOT}/src/ffi ${CMAKE_BINARY_DIR}/ffi)

target_link_libraries(xpupeak_ffi PRIVATE atomic dl)
target_compile_definitions(xpupeak_ffi PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_compile_options(xpupeak_ffi PRIVATE -O2 -fexceptions)
