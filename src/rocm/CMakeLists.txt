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

# Fallback for the header probe below when hip::host exposes no usable include
# dir.  CLPEAK_ROCM_INCLUDE_HINTS is set by the root CMakeLists (HIP_PATH /
# ROCM_PATH / /opt/rocm) and inherited into this scope.
list(GET CLPEAK_ROCM_INCLUDE_HINTS 0 _clpeak_rocm_inc)
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

# Every GPU the ROCm runtime can drive, member by member.  A GPU missing here
# gets no code object, and every kernel below then fails to load on it -- the
# Radeon 890M (gfx1150) did exactly that while only the discrete RDNA3 dies
# were listed -- so each family names its APUs and the parts ROCm does not
# officially support too.  CLPEAK_ROCM_ALL is the build's coverage: every group
# below is a subset of it, and rocm_device.cpp reads it back out of
# compute_sp's bundle to tell "no code for this GPU" from "no such instruction".
set(CLPEAK_ROCM_GCN5  "gfx900;gfx902;gfx904;gfx906;gfx909;gfx90c") # Vega; Raven..Renoir APUs
set(CLPEAK_ROCM_CDNA  "gfx908;gfx90a;gfx942;gfx950")                # MFMA-capable CDNA
set(CLPEAK_ROCM_RDNA1 "gfx1010;gfx1011;gfx1012;gfx1013")
set(CLPEAK_ROCM_RDNA2 "gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036")
set(CLPEAK_ROCM_RDNA3 "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153") # + RDNA3.5 APUs
set(CLPEAK_ROCM_RDNA4 "gfx1200;gfx1201")
set(CLPEAK_ROCM_RDNA  "${CLPEAK_ROCM_RDNA1};${CLPEAK_ROCM_RDNA2};${CLPEAK_ROCM_RDNA3};${CLPEAK_ROCM_RDNA4}")
set(CLPEAK_ROCM_ALL   "${CLPEAK_ROCM_GCN5};${CLPEAK_ROCM_CDNA};${CLPEAK_ROCM_RDNA}")
set(CLPEAK_ROCM_RDNAW "${CLPEAK_ROCM_RDNA3};${CLPEAK_ROCM_RDNA4}")             # WMMA RDNA3+
# v_dot4_i32_i8 / v_dot4_i32_iu8 (dot1- / dot8-insts): Vega 20 and every CDNA,
# the RDNA1 dies after Navi 10, all of RDNA2+.  Vega 10, the GCN5 APUs, Navi 10
# (gfx1010) and Cyan Skillfish (gfx1013) have neither, so the int8 dot kernel
# cannot build for them.
set(CLPEAK_ROCM_DOT4  "gfx906;${CLPEAK_ROCM_CDNA};gfx1011;gfx1012;${CLPEAK_ROCM_RDNA2};${CLPEAK_ROCM_RDNAW}")
# Texture/image sampling is a graphics feature: GCN5 and all RDNA parts have
# it, but the compute-only CDNA dies (gfx908/90a/942/950) do not -- the HIP
# tex2D API is marked unavailable there, so image_bandwidth must skip CDNA or the
# AOT compile hard-fails.
set(CLPEAK_ROCM_IMG   "${CLPEAK_ROCM_GCN5};${CLPEAK_ROCM_RDNA}")

# Generic compute + bandwidth -- every supported gfx.
embed_rocm_kernels(TARGET peak_rocm ARCHS "${CLPEAK_ROCM_ALL}"
    KERNELS compute_sp compute_hp compute_dp compute_mp compute_bf16
            compute_int32
            global_bandwidth local_bandwidth kernel_latency)

# INT8 dot product -- parts with a packed int8 dot instruction.
embed_rocm_kernels(TARGET peak_rocm ARCHS "${CLPEAK_ROCM_DOT4}"
    KERNELS compute_int8_dp)

# Image-sampling bandwidth -- texture-capable parts only (GCN5 + RDNA).
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
embed_rocm_kernels(TARGET peak_rocm ARCHS "${CLPEAK_ROCM_RDNA4}"
    KERNELS wmma_fp8)

# rocWMMA library kernels (header-only templates, need C++17 + the rocWMMA
# include).  Only buildable when the headers are present; otherwise emit stubs
# (host code is #ifdef CLPEAK_ROCM_HAS_ROCWMMA gated and skips them).  Offered
# every matrix-core arch; which of them the installed rocWMMA takes is probed,
# because its config.hpp static_asserts on any arch its version does not list
# (6.4 lacked gfx950; 7.2 has gfx1150/1151 but not gfx1103/1152/1153).
set(CLPEAK_ROCM_ROCWMMA "${CLPEAK_ROCM_CDNA};${CLPEAK_ROCM_RDNAW}")
if(CLPEAK_ROCWMMA_FOUND)
  embed_rocm_kernels(TARGET peak_rocm
      ARCHS "${CLPEAK_ROCM_ROCWMMA}" CXX17 ROCWMMA
      KERNELS rocwmma_fp16 rocwmma_int8)
else()
  embed_rocm_kernels(TARGET peak_rocm ARCHS "" KERNELS rocwmma_fp16 rocwmma_int8)
endif()
