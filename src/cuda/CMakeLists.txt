if(NOT CLPEAK_ENABLE_CUDA OR NOT CUDAToolkit_FOUND)
  return()
endif()

add_library(peak_cuda STATIC
    cuda_peak.cpp
    cuda_device.cpp
    compute_kernel.cpp
    cuda_blas.cpp
    compute_float.cpp
    compute_int.cpp
    wmma.cpp
    global_bandwidth.cpp
    local_bandwidth.cpp
    image_bandwidth.cpp
    transfer_bandwidth.cpp
    kernel_latency.cpp
)
target_link_libraries(peak_cuda PUBLIC
    peak_common
    CUDA::cuda_driver
)
# cuBLASLt is loaded at run time via dlopen (cuda_blas.cpp), not linked, so the
# binary runs without it present; we still need its headers (from the toolkit
# include dirs above) for the GEMM types and the FP4 feature probe below.
target_include_directories(peak_cuda PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Kernels are precompiled to fatbins at build time (see below) and loaded via
# the CUDA driver at run time -- NVRTC and the toolkit include dir are no longer
# needed by the shipped binary.
target_compile_definitions(peak_cuda PRIVATE ENABLE_CUDA)
target_compile_definitions(peak_common PRIVATE ENABLE_CUDA)

# Probe whether the installed cuBLASLt exposes the block-scaled FP4 GEMM API
# (CUDA 12.8+, Blackwell).  The symbol names settled only in recent toolkits, so
# feature-test by compilation rather than a version number -- keeps the build
# green on older CUDA while auto-enabling the mxfp4/nvfp4 path where it exists.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
check_cxx_source_compiles("
#include <cublasLt.h>
#include <library_types.h>
int main() {
  cublasLtMatmulMatrixScale_t s0 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  cublasLtMatmulMatrixScale_t s1 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  cublasLtMatmulDescAttributes_t a = CUBLASLT_MATMUL_DESC_A_SCALE_MODE;
  cudaDataType_t t = CUDA_R_4F_E2M1;
  (void)s0; (void)s1; (void)a; (void)t;
  return 0;
}" CLPEAK_CUBLASLT_HAS_FP4)
unset(CMAKE_REQUIRED_INCLUDES)
if(CLPEAK_CUBLASLT_HAS_FP4)
  target_compile_definitions(peak_cuda PRIVATE CLPEAK_CUBLASLT_HAS_FP4)
endif()

# Ahead-of-time compile each .cu to a multi-arch fatbin and embed the bytes.
# Kernels are grouped by the compute capabilities they are valid for so nvcc is
# never asked to target an arch a kernel cannot build for (e.g. fp8 < sm_89,
# int4 removed on sm_90+, fp4 only on sm_120a).
set(CLPEAK_CUDA_ARCHS "70;75;80;86;89;90;100;120" CACHE STRING
    "CUDA compute capabilities to bake into the fatbins (ascending)")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedCudaKernels.cmake)

# Broad group: generic compute + bandwidth + fp16 wmma -- valid sm_70+.
embed_cuda_kernels(TARGET peak_cuda MASTER_ARCHS "${CLPEAK_CUDA_ARCHS}" MIN_ARCH 70
    KERNELS compute_sp compute_hp compute_dp compute_mp compute_int32
            compute_int8_dp global_bandwidth local_bandwidth image_bandwidth
            kernel_latency wmma_fp16)

# INT8 wmma fragment tile (m16n16k16) -- sm_72+.
embed_cuda_kernels(TARGET peak_cuda MASTER_ARCHS "${CLPEAK_CUDA_ARCHS}" MIN_ARCH 72
    KERNELS wmma_int8)

# INT4 mma.sync -- sm_75..sm_89 only (removed on Hopper+).
embed_cuda_kernels(TARGET peak_cuda MASTER_ARCHS "${CLPEAK_CUDA_ARCHS}" MIN_ARCH 75 MAX_ARCH 89
    KERNELS wmma_int4)

# Ampere+ tensor cores: bf16 / tf32 / fp64 / int8 m16n8k32 + int8 2:4 sparse
# (all inline mma.sync m16n8k32, sm_80+).
embed_cuda_kernels(TARGET peak_cuda MASTER_ARCHS "${CLPEAK_CUDA_ARCHS}" MIN_ARCH 80
    KERNELS compute_bf16 wmma_bf16 wmma_tf32 wmma_fp64 wmma_int8_k32 wmma_int8_sparse)

# FP8 mma.sync (E4M3 / E5M2) -- sm_89+.
embed_cuda_kernels(TARGET peak_cuda MASTER_ARCHS "${CLPEAK_CUDA_ARCHS}" MIN_ARCH 89
    KERNELS wmma_fp8_e4m3 wmma_fp8_e5m2)

# FP4 / MXFP4 / NVFP4 mma.sync (dense + 2:4 sparse) -- Blackwell sm_120a.
embed_cuda_kernels(TARGET peak_cuda MASTER_ARCHS "${CLPEAK_CUDA_ARCHS}" MIN_ARCH 999
    EXTRA_ARCHS "120a"
    KERNELS wmma_fp4_e2m1 wmma_mxf4_e2m1 wmma_nvf4_e2m1
            wmma_mxf4_sparse wmma_nvf4_sparse)
