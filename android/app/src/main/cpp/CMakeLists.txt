cmake_minimum_required(VERSION 3.20)

# Build OpenCL stub
add_subdirectory(libopencl-stub)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
include(version_android.cmake)
include(${CLPEAK_ROOT}/cmake/sources.cmake)

set(CLPEAK_ANDROID_SOURCES
    ${CLPEAK_CORE_SOURCES}
    logger_android.cpp
    entry_android.cpp
)

# Vulkan is part of the NDK since API 24; libvulkan.so and headers ship with
# every modern NDK. glslc for shader compilation lives at
# ${ANDROID_NDK}/shader-tools/<host>. We enable Vulkan unconditionally here;
# at runtime vkPeak::initInstance() fails gracefully on devices without a
# Vulkan driver.
find_library(ANDROID_VULKAN_LIB vulkan)
if(ANDROID_VULKAN_LIB)
  list(APPEND CLPEAK_ANDROID_SOURCES ${CLPEAK_VK_SOURCES})
endif()

add_library(clpeak SHARED ${CLPEAK_ANDROID_SOURCES})

clpeak_setup_version_android(clpeak)

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

if(ANDROID_VULKAN_LIB)
  target_compile_definitions(clpeak PRIVATE ENABLE_VULKAN)
  target_link_libraries(clpeak ${ANDROID_VULKAN_LIB})
  include(${CLPEAK_ROOT}/cmake/CompileShaders.cmake)
  compile_shaders(
    TARGET clpeak
    SHADERS ${CLPEAK_VK_SHADERS}
  )
endif()
