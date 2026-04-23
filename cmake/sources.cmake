# cmake/sources.cmake
# Single source of truth for clpeak core source files.
# Set CLPEAK_ROOT before including this file.

set(CLPEAK_CORE_SOURCES
    ${CLPEAK_ROOT}/src/common.cpp
    ${CLPEAK_ROOT}/src/result_store.cpp
    ${CLPEAK_ROOT}/src/clpeak.cpp
    ${CLPEAK_ROOT}/src/options.cpp
    ${CLPEAK_ROOT}/src/global_bandwidth.cpp
    ${CLPEAK_ROOT}/src/local_bandwidth.cpp
    ${CLPEAK_ROOT}/src/image_bandwidth.cpp
    ${CLPEAK_ROOT}/src/atomic_throughput.cpp
    ${CLPEAK_ROOT}/src/transfer_bandwidth.cpp
    ${CLPEAK_ROOT}/src/kernel_latency.cpp
)

set(CLPEAK_CORE_INCLUDE_DIRS
    ${CLPEAK_ROOT}/include
    ${CLPEAK_ROOT}/src/kernels
)

# Vulkan backend sources -- included when a Vulkan loader is available.
set(CLPEAK_VK_SOURCES
    ${CLPEAK_ROOT}/src/vk_peak.cpp
    ${CLPEAK_ROOT}/src/entry_vk.cpp
)

# SPIR-V compute shaders shared by both the main CLI build and the Android
# build.  compile_shaders() turns each .comp into an embedded C++ array and
# defines CLPEAK_VK_HAS_<NAME> for per-shader gating in vk_peak.cpp.
set(CLPEAK_VK_SHADERS
    ${CLPEAK_ROOT}/src/shaders/compute_sp_v1.comp
    ${CLPEAK_ROOT}/src/shaders/global_bandwidth_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_int8_dp_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_mp_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_int4_packed_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_bf16_v1.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_fp16.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_bf16.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_int8.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_fp8_e4m3.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_fp8_e5m2.comp
)
