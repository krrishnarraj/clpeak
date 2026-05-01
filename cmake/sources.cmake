# cmake/sources.cmake
# Single source of truth for clpeak core source files.
# Set CLPEAK_ROOT before including this file.

set(CLPEAK_CORE_SOURCES
    ${CLPEAK_ROOT}/src/common.cpp
    ${CLPEAK_ROOT}/src/result_store.cpp
    ${CLPEAK_ROOT}/src/clpeak.cpp
    ${CLPEAK_ROOT}/src/inventory.cpp
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

# CUDA backend sources -- included when CUDA Toolkit (driver API + NVRTC)
# is available.  Kernels are NVRTC-compiled at runtime; .cu files are
# embedded as C++ string literals via embed_cuda_kernels().
set(CLPEAK_CUDA_SOURCES
    ${CLPEAK_ROOT}/src/cuda_peak.cpp
    ${CLPEAK_ROOT}/src/cuda_blas.cpp
    ${CLPEAK_ROOT}/src/entry_cuda.cpp
)

# Metal backend sources -- enabled on APPLE only.  Apple silicon (M1+)
# is the only supported runtime target; the build still compiles on Intel
# Macs but the runtime refuses non-Apple-silicon devices.
set(CLPEAK_MTL_SOURCES
    ${CLPEAK_ROOT}/src/mtl_peak.mm
    ${CLPEAK_ROOT}/src/mtl_blas.mm
    ${CLPEAK_ROOT}/src/entry_mtl.mm
)

set(CLPEAK_MTL_KERNELS
    ${CLPEAK_ROOT}/src/mtl_kernels/compute_sp.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/compute_hp.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/compute_mp.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/compute_int8_dp.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/compute_int4_packed.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/global_bandwidth.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/kernel_latency.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/simdgroup_matrix_fp16.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/simdgroup_matrix_bf16.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/local_bandwidth.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/image_bandwidth.metal
    ${CLPEAK_ROOT}/src/mtl_kernels/atomic_throughput.metal
)

set(CLPEAK_CUDA_KERNELS
    ${CLPEAK_ROOT}/src/cuda_kernels/compute_sp.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/compute_hp.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/compute_dp.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/compute_mp.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/compute_bf16.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/compute_int8_dp.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/compute_int4_packed.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/global_bandwidth.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/kernel_latency.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/wmma_fp16.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/wmma_bf16.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/wmma_int8.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/wmma_int8_k32.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/wmma_fp8_e4m3.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/wmma_fp8_e5m2.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/local_bandwidth.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/image_bandwidth.cu
    ${CLPEAK_ROOT}/src/cuda_kernels/atomic_throughput.cu
)

# SPIR-V compute shaders shared by both the main CLI build and the Android
# build.  compile_shaders() turns each .comp into an embedded C++ array and
# defines CLPEAK_VK_HAS_<NAME> for per-shader gating in vk_peak.cpp.
set(CLPEAK_VK_SHADERS
    ${CLPEAK_ROOT}/src/shaders/compute_sp_v1.comp
    ${CLPEAK_ROOT}/src/shaders/global_bandwidth_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_int8_dp_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_int8_dp_v2.comp
    ${CLPEAK_ROOT}/src/shaders/compute_int8_dp_v4.comp
    ${CLPEAK_ROOT}/src/shaders/compute_mp_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_mp_v2.comp
    ${CLPEAK_ROOT}/src/shaders/compute_mp_v4.comp
    ${CLPEAK_ROOT}/src/shaders/compute_int4_packed_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_bf16_v1.comp
    ${CLPEAK_ROOT}/src/shaders/compute_bf16_v2.comp
    ${CLPEAK_ROOT}/src/shaders/compute_bf16_v4.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_fp16.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_bf16.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_int8.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_int8_k32.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_fp8_e4m3.comp
    ${CLPEAK_ROOT}/src/shaders/coopmat_fp8_e5m2.comp
    ${CLPEAK_ROOT}/src/shaders/local_bandwidth_v1.comp
    ${CLPEAK_ROOT}/src/shaders/local_bandwidth_v2.comp
    ${CLPEAK_ROOT}/src/shaders/local_bandwidth_v4.comp
    ${CLPEAK_ROOT}/src/shaders/local_bandwidth_v8.comp
    ${CLPEAK_ROOT}/src/shaders/image_bandwidth_v1.comp
    ${CLPEAK_ROOT}/src/shaders/atomic_throughput_global.comp
    ${CLPEAK_ROOT}/src/shaders/atomic_throughput_local.comp
    ${CLPEAK_ROOT}/src/shaders/kernel_latency.comp
)
