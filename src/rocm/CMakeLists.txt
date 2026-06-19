if(NOT CLPEAK_ENABLE_ROCM OR NOT hip_FOUND OR NOT CLPEAK_HIPCC_FOUND)
  return()
endif()

add_library(peak_rocm STATIC
    rocm_peak.cpp
    rocm_device.cpp
    compute_kernel.cpp
    compute_float.cpp
    compute_int.cpp
    wmma.cpp
    rocwmma.cpp
    mfma.cpp
    sparse_mfma.cpp
    rocblas.cpp
    hipblaslt_gemm.cpp
    global_bandwidth.cpp
    local_bandwidth.cpp
    image_bandwidth.cpp
    transfer_bandwidth.cpp
    kernel_latency.cpp
)

target_link_libraries(peak_rocm PUBLIC
    peak_common
    hip::host
)
# rocBLAS / hipBLASLt are loaded at run time via dlopen (rocblas.cpp,
# hipblaslt_gemm.cpp), not linked, so the binary runs without them present.  We
# still compile against their headers (resolved via the HIP include dir that
# hip::host exposes) and keep the CLPEAK_ROCM_HAS_* defines for the GEMM types.
if(rocblas_FOUND AND TARGET roc::rocblas)
  target_compile_definitions(peak_rocm PRIVATE CLPEAK_ROCM_HAS_ROCBLAS)
endif()
if(hipblaslt_FOUND AND TARGET roc::hipblaslt)
  target_compile_definitions(peak_rocm PRIVATE CLPEAK_ROCM_HAS_HIPBLASLT)
endif()
target_include_directories(peak_rocm PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_compile_features(peak_rocm PRIVATE cxx_std_14)

set(_clpeak_rocm_inc "/opt/rocm/include")
if(TARGET hip::host)
  get_target_property(_clpeak_hip_incs hip::host INTERFACE_INCLUDE_DIRECTORIES)
  if(_clpeak_hip_incs)
    foreach(_clpeak_hip_inc IN LISTS _clpeak_hip_incs)
      if(NOT _clpeak_hip_inc MATCHES "^\\$<")
        set(_clpeak_rocm_inc "${_clpeak_hip_inc}")
        break()
      endif()
    endforeach()
  endif()
endif()

target_compile_definitions(peak_rocm PRIVATE ENABLE_ROCM)

# _clpeak_rocm_inc above is used only for the configure-time hipBLASLt header
# probe below; it is no longer baked into the binary (kernels are AOT-compiled).

# Probe whether the installed hipBLASLt exposes the block-scaled FP4 GEMM API
# (gfx950 / MI350, recent ROCm).  Feature-test by compilation so the build stays
# green on ROCm versions without it and auto-enables the mxfp4 path where present.
if(hipblaslt_FOUND AND TARGET roc::hipblaslt)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES "${_clpeak_rocm_inc}")
  set(CMAKE_REQUIRED_DEFINITIONS "-D__HIP_PLATFORM_AMD__")
  check_cxx_source_compiles("
#include <hipblaslt/hipblaslt.h>
int main() {
  hipblasLtMatmulMatrixScale_t s = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  hipblasLtMatmulDescAttributes_t a = HIPBLASLT_MATMUL_DESC_A_SCALE_MODE;
  hipDataType t = HIP_R_4F_E2M1;
  (void)s; (void)a; (void)t;
  return 0;
}" CLPEAK_HIPBLASLT_HAS_FP4)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_DEFINITIONS)
  if(CLPEAK_HIPBLASLT_HAS_FP4)
    target_compile_definitions(peak_rocm PRIVATE CLPEAK_HIPBLASLT_HAS_FP4)
  endif()
endif()
if(CLPEAK_ROCWMMA_FOUND)
  target_compile_definitions(peak_rocm PRIVATE CLPEAK_ROCM_HAS_ROCWMMA)
endif()
target_compile_definitions(peak_common PRIVATE ENABLE_ROCM)

# Ahead-of-time compile each .hip to a bundled code object and embed the bytes.
# Kernels are grouped by the gfx families they are valid for so hipcc is never
# asked to target an arch a kernel cannot build for (MFMA is CDNA-only, WMMA is
# RDNA3+, fp8/mxfp4 are newest-gen only).  The candidate arches are intersected
# with what the installed hipcc actually supports.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedRocmKernels.cmake)

set(CLPEAK_ROCM_CDNA   "gfx908;gfx90a;gfx942;gfx950")   # MFMA-capable CDNA
set(CLPEAK_ROCM_RDNAW  "gfx1100;gfx1101;gfx1102;gfx1200;gfx1201") # WMMA RDNA3+
set(CLPEAK_ROCM_ALL    "gfx906;${CLPEAK_ROCM_CDNA};gfx1030;${CLPEAK_ROCM_RDNAW}")
# Texture/image sampling is a graphics feature: Vega (gfx906) and all RDNA parts
# have it, but the compute-only CDNA dies (gfx908/90a/942/950) do not -- the HIP
# tex2D API is marked unavailable there, so image_bandwidth must skip CDNA or the
# AOT compile hard-fails.
set(CLPEAK_ROCM_IMG    "gfx906;gfx1030;${CLPEAK_ROCM_RDNAW}")

# Generic compute + bandwidth -- every supported gfx.
embed_rocm_kernels(TARGET peak_rocm ARCHS "${CLPEAK_ROCM_ALL}"
    KERNELS compute_sp compute_hp compute_dp compute_mp compute_bf16
            compute_int32 compute_int8_dp
            global_bandwidth local_bandwidth kernel_latency)

# Image-sampling bandwidth -- texture-capable parts only (Vega + RDNA).
embed_rocm_kernels(TARGET peak_rocm ARCHS "${CLPEAK_ROCM_IMG}"
    KERNELS image_bandwidth)

# MFMA matrix core.  The builtins gate per sub-target with no fallback, so each
# is grouped to exactly the archs that select it:
#   fp16 (mfma_f32_16x16x16f16)        -- all CDNA
#   bf16 (mfma_f32_16x16x16bf16_1k)    -- the "1k" form is gfx90a+ (not gfx908)
#   int8 (mfma_i32_16x16x32_i8)        -- the K=32 form is CDNA3+ (gfx942/950)
embed_rocm_kernels(TARGET peak_rocm ARCHS "${CLPEAK_ROCM_CDNA}"
    KERNELS mfma_fp16)
embed_rocm_kernels(TARGET peak_rocm ARCHS "gfx90a;gfx942;gfx950"
    KERNELS mfma_bf16)
embed_rocm_kernels(TARGET peak_rocm ARCHS "gfx942;gfx950"
    KERNELS mfma_int8)
# fp8 MFMA -- CDNA3+ (gfx942/950).
embed_rocm_kernels(TARGET peak_rocm ARCHS "gfx942;gfx950"
    KERNELS mfma_fp8)
# mxfp4 scaled MFMA -- CDNA4 (gfx950).
embed_rocm_kernels(TARGET peak_rocm ARCHS "gfx950"
    KERNELS mfma_mxfp4)
# Sparse MFMA (2:4) -- CDNA3+ (gfx942/950).
embed_rocm_kernels(TARGET peak_rocm ARCHS "gfx942;gfx950"
    KERNELS smfmac_fp16 smfmac_bf16 smfmac_int8 smfmac_fp8)

# WMMA matrix core -- RDNA3+ (gfx11/gfx12).
embed_rocm_kernels(TARGET peak_rocm ARCHS "${CLPEAK_ROCM_RDNAW}"
    KERNELS wmma_fp16 wmma_bf16 wmma_int8)
# fp8 WMMA -- RDNA4 (gfx12).
embed_rocm_kernels(TARGET peak_rocm ARCHS "gfx1200;gfx1201"
    KERNELS wmma_fp8)

# rocWMMA library kernels (header-only templates, need C++17 + the rocWMMA
# include).  Only buildable when the headers are present; otherwise emit stubs
# (host code is #ifdef CLPEAK_ROCM_HAS_ROCWMMA gated and skips them).
if(CLPEAK_ROCWMMA_FOUND)
  embed_rocm_kernels(TARGET peak_rocm
      ARCHS "${CLPEAK_ROCM_CDNA};${CLPEAK_ROCM_RDNAW}" CXX17 ROCWMMA
      KERNELS rocwmma_fp16 rocwmma_int8)
else()
  embed_rocm_kernels(TARGET peak_rocm ARCHS "" KERNELS rocwmma_fp16 rocwmma_int8)
endif()
