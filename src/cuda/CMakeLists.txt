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
    atomic_throughput.cpp
)
target_link_libraries(peak_cuda PUBLIC
    peak_common
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::cublasLt
)
target_include_directories(peak_cuda PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# NVRTC needs the toolkit include dir to resolve <mma.h>, <cuda_fp16.h>
# etc. at runtime.  Embed it as a string literal the runtime hands to
# nvrtcCompileProgram via "-I<dir>".
list(GET CUDAToolkit_INCLUDE_DIRS 0 _clpeak_cuda_inc)
target_compile_definitions(peak_cuda PRIVATE
    ENABLE_CUDA
    CLPEAK_CUDA_INCLUDE_DIR="${_clpeak_cuda_inc}"
)
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

# Embed .cu kernel sources as C++ raw-string literals
set(CLPEAK_CUDA_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/compute_sp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/compute_hp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/compute_dp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/compute_mp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/compute_bf16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/compute_int8_dp.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/compute_int32.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/global_bandwidth.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/kernel_latency.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_fp16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_bf16.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_int8.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_int8_k32.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_int8_sparse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_fp8_e4m3.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_fp8_e5m2.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_fp4_e2m1.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_mxf4_e2m1.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_nvf4_e2m1.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_mxf4_sparse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_nvf4_sparse.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_tf32.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_fp64.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_int4.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/wmma_bmma_b1.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/local_bandwidth.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/image_bandwidth.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/atomic_throughput.cu
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedCudaKernels.cmake)
embed_cuda_kernels(
    TARGET peak_cuda
    KERNELS ${CLPEAK_CUDA_KERNELS}
)
