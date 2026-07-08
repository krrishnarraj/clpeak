#ifndef CPU_KERNELS_IMPL_H
#define CPU_KERNELS_IMPL_H

// ===========================================================================
// Per-TU kernel aggregation.  This header is compiled once per feature TU
// (cpu_kernels_tu.cpp, built N times with different -m/-arch flags); it pulls in
// the kernel bodies -- grouped into cohesive headers under kernels/ -- and emits
// this TU's CpuKernelTable from whatever kernels its build flags enabled.
//
//   kernels/base_compute.h    fp32 / fp64 / int32 chains + streaming read (all TUs)
//   kernels/lowp_compute.h    fp16 / bf16-dot / mixed-precision / int8-dot / bf16-FMA
//   kernels/matrix_compute.h  AMX (int8/bf16/fp16/tf32/fp8) + NEON SMMLA/BFMMLA
//   kernels/sve_compute.h     ARM SVE compute + SVE bf16/i8mm matrix
//
// cpu_simd.h selects the base SIMD path from the TU's flags, and each advanced
// kernel is #if-gated on the compile-feature macros those flags define, so a
// given TU compiles in only the kernels it can run.  Everything lives in an
// anonymous namespace so each TU gets its own internal copy (no ODR clash, no
// cross-TU inlining).  To add a new TU: give it a kernel body in the matching
// kernels/ header (a new #if block + CPU_HAS_<X>/CPU_MAT_<X> define) and a table
// slot below; then register the TU in CMakeLists.txt + cpu_dispatch.cpp.
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#include "kernels/base_compute.h"
#include "kernels/lowp_compute.h"
#include "kernels/matrix_compute.h"
#include "kernels/sve_compute.h"

#ifndef CLPEAK_ISA_NAME_STR
#define CLPEAK_ISA_NAME_STR "scalar"
#endif

namespace clpeak_cpu {
namespace {

using namespace cpu_simd;

// ---- Table builder (this TU's offered kernels) ----------------------------
static const CpuKernelTable *tuTable()
{
  static CpuKernelTable t = [] {
    CpuKernelTable t{};
    t.fp32  = {runFp32Chain,  (double)INNER * F32_NACC * F32_LANES * 2.0};
    t.fp64  = {runFp64Chain,  (double)INNER * F64_NACC * F64_LANES * 2.0};
    t.int32 = {runInt32Chain, (double)INNER * I32_NACC * I32_LANES * 2.0};
    t.readsum = readBufferChecksum;
#ifdef CPU_HAS_FP16_KERNEL
    t.fp16 = {runFp16Chain, (double)INNER * FP16_NACC * FP16_LANES * 2.0};
#endif
#ifdef CPU_HAS_BF16_KERNEL
    t.bf16 = {runBf16Chain, (double)INNER * BF16_NACC * BF16_FLOPS_PER_INSTR};
#endif
#ifdef CPU_HAS_MP_KERNEL
    t.mp = {runMpChain, (double)INNER * MP_NACC * MP_OPS_PER_INSTR};
#endif
#ifdef CPU_HAS_INT8DP_KERNEL
    t.int8dp = {runInt8DpChain, (double)INNER * I8_NACC * I8_OPS_PER_INSTR};
#endif
#ifdef CPU_MAT_INT8_KERNEL
    t.mat_int8 = {runMatInt8Chain, (double)INNER * MAT_I8_OPS_PER_K};
#endif
#ifdef CPU_MAT_FP_KERNEL
    t.mat_fp = {runMatFpChain, (double)INNER * MAT_FP_OPS_PER_K};
#endif
#ifdef CPU_MAT_FP16_KERNEL
    t.mat_fp16 = {runMatFp16Chain, (double)INNER * MAT_FP16_OPS_PER_K};
#endif
#ifdef CPU_MAT_TF32_KERNEL
    t.mat_tf32 = {runMatTf32Chain, (double)INNER * MAT_TF32_OPS_PER_K};
#endif
#ifdef CPU_MAT_FP8_KERNEL
    t.mat_fp8 = {runMatFp8Chain, (double)INNER * MAT_FP8_OPS_PER_K};
#endif
#ifdef CPU_HAS_BF16FMA_KERNEL
    t.bf16fma = {runBf16FmaChain, (double)INNER * BF16FMA_NACC * BF16FMA_LANES * 2.0};
#endif
    // SVE overrides (vector-length-agnostic; ops derived from the runtime VL).
    // These win over the NEON base/advanced kernels the SVE TUs also compile:
    // a +sve TU still carries the NEON f32v path, but we want the SVE variant.
#ifdef CPU_HAS_SVE_KERNELS
    {
      const double w = (double)svcntw();   // 32-bit lanes at the running VL
      const double d = (double)svcntd();   // 64-bit lanes
      // bytes -- only used by the bf16/i8mm matrix ops-per-instr below, which the
      // plain `sve` TU doesn't compile, so mark maybe_unused to avoid a warning.
      [[maybe_unused]] const double bcnt = (double)svcntb();
      t.sveVLBytes = (int)svcntb();
      t.fp32   = {runSveFp32Chain,   (double)INNER * SVE_NACC_FP  * w * 2.0};
      t.fp64   = {runSveFp64Chain,   (double)INNER * SVE_NACC_FP  * d * 2.0};
      t.int32  = {runSveInt32Chain,  (double)INNER * SVE_NACC_FP  * w * 2.0};
      t.int8dp = {runSveInt8DpChain, (double)INNER * SVE_NACC_DOT * w * 8.0};
#ifdef CPU_HAS_SVE_BF16_KERNEL
      t.bf16   = {runSveBf16Chain,    (double)INNER * SVE_NACC_DOT * w * 4.0};
      t.mat_fp = {runSveMatBf16Chain, (double)INNER * SVE_NACC_DOT * bcnt * 2.0};
#endif
#ifdef CPU_HAS_SVE_I8MM_KERNEL
      t.mat_int8 = {runSveMatInt8Chain, (double)INNER * SVE_NACC_DOT * bcnt * 4.0};
#endif
    }
#endif
    t.isaName = CLPEAK_ISA_NAME_STR;
    return t;
  }();
  return &t;
}

} // anonymous namespace
} // namespace clpeak_cpu

#endif // CPU_KERNELS_IMPL_H
