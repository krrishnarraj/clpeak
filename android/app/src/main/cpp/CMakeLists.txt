cmake_minimum_required(VERSION 3.20)

# Build OpenCL stub
add_subdirectory(libopencl-stub)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
get_filename_component(CLPEAK_GIT_ROOT "${CLPEAK_ROOT}" ABSOLUTE)
include(${CLPEAK_ROOT}/cmake/version.cmake)

# ---- Source files (formerly from cmake/sources.cmake) --------------------
set(CLPEAK_COMMON_SOURCES
    ${CLPEAK_ROOT}/src/common/peak.cpp
    ${CLPEAK_ROOT}/src/common/common.cpp
    ${CLPEAK_ROOT}/src/common/backend_gating.cpp
    ${CLPEAK_ROOT}/src/common/calibrate.cpp
    ${CLPEAK_ROOT}/src/common/inventory.cpp
    ${CLPEAK_ROOT}/src/common/options.cpp
    ${CLPEAK_ROOT}/src/common/result_store.cpp
)

set(CLPEAK_OPENCL_SOURCES
    ${CLPEAK_ROOT}/src/opencl/cl_peak.cpp
    ${CLPEAK_ROOT}/src/opencl/cl_common.cpp
    ${CLPEAK_ROOT}/src/opencl/global_bandwidth.cpp
    ${CLPEAK_ROOT}/src/opencl/local_bandwidth.cpp
    ${CLPEAK_ROOT}/src/opencl/image_bandwidth.cpp
    ${CLPEAK_ROOT}/src/opencl/atomic_throughput.cpp
    ${CLPEAK_ROOT}/src/opencl/transfer_bandwidth.cpp
    ${CLPEAK_ROOT}/src/opencl/kernel_latency.cpp
)

set(CLPEAK_ANDROID_SOURCES
    ${CLPEAK_COMMON_SOURCES}
    ${CLPEAK_OPENCL_SOURCES}
    logger_android.cpp
    entry_android.cpp
    enumerate_backends.cpp
)

# Vulkan is part of the NDK since API 24; libvulkan.so and headers ship with
# every modern NDK. glslc for shader compilation lives at
# ${ANDROID_NDK}/shader-tools/<host>. We enable Vulkan unconditionally here;
# at runtime vkPeak::initInstance() fails gracefully on devices without a
# Vulkan driver.
find_library(ANDROID_VULKAN_LIB vulkan)
if(ANDROID_VULKAN_LIB)
    list(APPEND CLPEAK_ANDROID_SOURCES ${CLPEAK_ROOT}/src/vulkan/vk_peak.cpp)
endif()

add_library(clpeak SHARED ${CLPEAK_ANDROID_SOURCES})

clpeak_setup_version(clpeak GIT_ROOT ${CLPEAK_GIT_ROOT})

target_include_directories(clpeak PRIVATE
    libopencl-stub/include
    ${CLPEAK_ROOT}/include
    ${CLPEAK_ROOT}/include/common
    ${CLPEAK_ROOT}/src/opencl
)

# Target OpenCL 3.0 headers so INT8 dot-product and other extension tests are
# reachable. The stub dlopens the device's libOpenCL at runtime, and each
# test checks its extension string before running -- unsupported devices skip
# cleanly.
target_compile_definitions(clpeak PRIVATE ANDROID_LOGGER CL_TARGET_OPENCL_VERSION=300)
target_compile_options(clpeak PRIVATE -fPIC -O2 -fexceptions)
set_target_properties(clpeak PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_link_libraries(clpeak log atomic dl OpenCL)

if(ANDROID_VULKAN_LIB)
    target_compile_definitions(clpeak PRIVATE ENABLE_VULKAN)
    target_link_libraries(clpeak ${ANDROID_VULKAN_LIB})

    # Compile GLSL shaders to SPIR-V and embed as C++ arrays
    set(CLPEAK_VK_SHADERS
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_sp_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_sp_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_sp_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_hp_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_hp_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_hp_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_dp_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_dp_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_dp_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_int32_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_int32_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_int32_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/global_bandwidth_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/global_bandwidth_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/global_bandwidth_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_int8_dp_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_int8_dp_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_int8_dp_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_mp_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_mp_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_mp_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_int4_packed_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_bf16_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_bf16_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/compute_bf16_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/coopmat_fp16.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/coopmat_bf16.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/coopmat_int8.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/coopmat_int8_k32.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/coopmat_fp8_e4m3.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/coopmat_fp8_e5m2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/coopmat_fp32.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/local_bandwidth_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/local_bandwidth_v2.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/local_bandwidth_v4.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/image_bandwidth_v1.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/atomic_throughput_global.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/atomic_throughput_global_float.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/atomic_throughput_global_uint64.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/atomic_throughput_local.comp
        ${CLPEAK_ROOT}/src/vulkan/shaders/kernel_latency.comp
    )
    include(${CLPEAK_ROOT}/src/vulkan/cmake/CompileShaders.cmake)
    compile_shaders(
        TARGET clpeak
        SHADERS ${CLPEAK_VK_SHADERS}
    )
endif()
