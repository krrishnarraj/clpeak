#ifndef CPU_KERN_SME_COMPUTE_H
#define CPU_KERN_SME_COMPUTE_H

#ifdef ENABLE_CPU

// ===========================================================================
// ARM SME (Scalable Matrix Extension) kernels: ZA-tile outer products (the
// matrix-engine peak path) + streaming-SVE vector chains.  Covers Apple M4+
// (SME2, SVL=512b -- Apple has NO non-streaming SVE, so this is its only
// scalable-vector path) and Qualcomm Oryon Gen 3 (Snapdragon X2 Elite /
// 8 Elite Gen 5, per-cluster SME units).
//
// SME specifics the fixed-width/SVE kernels don't have:
//  - Each kernel body is `__arm_locally_streaming` (enters/exits streaming mode
//    on entry/return) and, when it touches ZA, `__arm_new("za")` (fresh ZA
//    state).  Callers need no attributes, so the fn-ptr dispatch is unchanged.
//  - The FMOPA/SMOPA intrinsics are opaque ZA-state updates (like AMX tiles),
//    so the accumulate loop can't be scalar-evolved/collapsed; inputs are zero
//    on purpose (throughput is data-independent, and zeros can't overflow).
//  - ZA holds 4 fp32 (za0-3.s) / 8 fp64 (za0-7.d) accumulator tiles; ALL are
//    issued to (and read back) so none is dead -- the AMX 4-tile pattern.
//  - opsPerIter is SVL-dependent: computed from svcntsw()/svcntsd()/svcntsb()
//    (the *streaming* VL, readable from non-streaming code) at table-build
//    time, and reported as "SME (SVL=512b)" in the device header.
//  - The SME unit is SHARED PER CLUSTER on every implementation so far (Apple:
//    1/cluster; X2 Elite: 3; 8 Elite Gen 5: 2).  Expect MT ~= #clusters x unit
//    peak, NOT #cores x ST -- that is real hardware behaviour, not a bug.
//
// Gated on __ARM_FEATURE_SME + CLPEAK_CORE_ONLY-excluded; owns the one
// #include <arm_sme.h> (which provides the streaming-compatible SVE ops too).
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#if defined(__ARM_FEATURE_SME) && !defined(CLPEAK_CORE_ONLY)
#include <arm_sme.h>

namespace clpeak_cpu {
namespace {

#define CPU_HAS_SME_KERNELS 1

// Expand X(i) for i in [0, N) -- SVE/SME Z types are sizeless (no arrays), so
// accumulator chains are individual named registers, like the SVE kernels.
#define SME_REP16(X) X(0)X(1)X(2)X(3)X(4)X(5)X(6)X(7)X(8)X(9)X(10)X(11)X(12)X(13)X(14)X(15)
#define SME_REP24(X) SME_REP16(X) X(16)X(17)X(18)X(19)X(20)X(21)X(22)X(23)
static constexpr int SME_NACC_FP = 24;   // streaming-SVE MAD chains (32 Z regs)

// ---- ZA outer products ------------------------------------------------------
// fp32: za32 FMOPA, 4 accumulator tiles.  (SVL/32)^2 FMAs per instruction.
__arm_locally_streaming __arm_new("za")
static double runSmeMatFp32Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  const svfloat32_t a = svdup_f32(0.0f), b = svdup_f32(0.0f);
  svzero_za();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      svmopa_za32_f32_m(0, pg, pg, a, b);
      svmopa_za32_f32_m(1, pg, pg, a, b);
      svmopa_za32_f32_m(2, pg, pg, a, b);
      svmopa_za32_f32_m(3, pg, pg, a, b);
    }
  svfloat32_t r = svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 0, 0);
  r = svadd_f32_x(pg, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 1, 0));
  r = svadd_f32_x(pg, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 2, 0));
  r = svadd_f32_x(pg, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pg, 3, 0));
  return (double)svaddv_f32(pg, r);
}

// bf16: za32 BFMOPA (widening 2-way bf16 dot per fp32 element).
__arm_locally_streaming __arm_new("za")
static double runSmeMatBf16Chain(uint64_t outer)
{
  const svbool_t ph = svptrue_b16();
  const svbfloat16_t a = svreinterpret_bf16_u16(svdup_u16(0)), b = a;
  svzero_za();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      svmopa_za32_bf16_m(0, ph, ph, a, b);
      svmopa_za32_bf16_m(1, ph, ph, a, b);
      svmopa_za32_bf16_m(2, ph, ph, a, b);
      svmopa_za32_bf16_m(3, ph, ph, a, b);
    }
  const svbool_t pw = svptrue_b32();
  svfloat32_t r = svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 0, 0);
  r = svadd_f32_x(pw, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 1, 0));
  r = svadd_f32_x(pw, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 2, 0));
  r = svadd_f32_x(pw, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 3, 0));
  return (double)svaddv_f32(pw, r);
}

// fp16: za32 FMOPA widening (2-way fp16 dot per fp32 element; base SME).
__arm_locally_streaming __arm_new("za")
static double runSmeMatFp16Chain(uint64_t outer)
{
  const svbool_t ph = svptrue_b16();
  const svfloat16_t a = svdup_f16((float16_t)0.0f), b = a;
  svzero_za();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      svmopa_za32_f16_m(0, ph, ph, a, b);
      svmopa_za32_f16_m(1, ph, ph, a, b);
      svmopa_za32_f16_m(2, ph, ph, a, b);
      svmopa_za32_f16_m(3, ph, ph, a, b);
    }
  const svbool_t pw = svptrue_b32();
  svfloat32_t r = svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 0, 0);
  r = svadd_f32_x(pw, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 1, 0));
  r = svadd_f32_x(pw, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 2, 0));
  r = svadd_f32_x(pw, r, svread_hor_za32_f32_m(svdup_f32(0.0f), pw, 3, 0));
  return (double)svaddv_f32(pw, r);
}

// int8: za32 SMOPA (widening 4-way int8 dot per int32 element).
__arm_locally_streaming __arm_new("za")
static double runSmeMatInt8Chain(uint64_t outer)
{
  const svbool_t pb = svptrue_b8();
  const svint8_t a = svdup_s8(0), b = svdup_s8(0);
  svzero_za();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      svmopa_za32_s8_m(0, pb, pb, a, b);
      svmopa_za32_s8_m(1, pb, pb, a, b);
      svmopa_za32_s8_m(2, pb, pb, a, b);
      svmopa_za32_s8_m(3, pb, pb, a, b);
    }
  const svbool_t pw = svptrue_b32();
  svint32_t r = svread_hor_za32_s32_m(svdup_s32(0), pw, 0, 0);
  r = svadd_s32_x(pw, r, svread_hor_za32_s32_m(svdup_s32(0), pw, 1, 0));
  r = svadd_s32_x(pw, r, svread_hor_za32_s32_m(svdup_s32(0), pw, 2, 0));
  r = svadd_s32_x(pw, r, svread_hor_za32_s32_m(svdup_s32(0), pw, 3, 0));
  return (double)svaddv_s32(pw, r);
}

// fp64: za64 FMOPA (FEAT_SME_F64F64; Apple M4+).  8 fp64 tiles (za0-7.d) --
// the first native fp64 matrix engine on any CPU we benchmark.  Gated on the
// ACLE macro OR the CMake-passed CLPEAK_SME_F64F64: AppleClang 21 enables the
// +sme-f64f64 *feature* (the intrinsic compiles) without defining the macro,
// so CMake proves the intrinsic builds and passes the define itself.
#if defined(__ARM_FEATURE_SME_F64F64) || defined(CLPEAK_SME_F64F64)
#define CPU_MAT_F64_KERNEL 1
__arm_locally_streaming __arm_new("za")
static double runSmeMatFp64Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b64();
  const svfloat64_t a = svdup_f64(0.0), b = svdup_f64(0.0);
  svzero_za();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      svmopa_za64_f64_m(0, pg, pg, a, b);
      svmopa_za64_f64_m(1, pg, pg, a, b);
      svmopa_za64_f64_m(2, pg, pg, a, b);
      svmopa_za64_f64_m(3, pg, pg, a, b);
      svmopa_za64_f64_m(4, pg, pg, a, b);
      svmopa_za64_f64_m(5, pg, pg, a, b);
      svmopa_za64_f64_m(6, pg, pg, a, b);
      svmopa_za64_f64_m(7, pg, pg, a, b);
    }
  svfloat64_t r = svread_hor_za64_f64_m(svdup_f64(0.0), pg, 0, 0);
  r = svadd_f64_x(pg, r, svread_hor_za64_f64_m(svdup_f64(0.0), pg, 1, 0));
  r = svadd_f64_x(pg, r, svread_hor_za64_f64_m(svdup_f64(0.0), pg, 2, 0));
  r = svadd_f64_x(pg, r, svread_hor_za64_f64_m(svdup_f64(0.0), pg, 3, 0));
  r = svadd_f64_x(pg, r, svread_hor_za64_f64_m(svdup_f64(0.0), pg, 4, 0));
  r = svadd_f64_x(pg, r, svread_hor_za64_f64_m(svdup_f64(0.0), pg, 5, 0));
  r = svadd_f64_x(pg, r, svread_hor_za64_f64_m(svdup_f64(0.0), pg, 6, 0));
  r = svadd_f64_x(pg, r, svread_hor_za64_f64_m(svdup_f64(0.0), pg, 7, 0));
  return (double)svaddv_f64(pg, r);
}
#endif // __ARM_FEATURE_SME_F64F64 || CLPEAK_SME_F64F64

// ---- Streaming-SVE vector chains --------------------------------------------
// Same acc = acc*b + c MAD chains as the SVE kernels, but locally streaming, so
// they run at the STREAMING VL on the SME unit.  On Apple this is the only
// scalable-vector compute (no plain SVE); on all current cores it measures the
// shared per-cluster SME unit, a genuinely different number from the NEON rows.
__arm_locally_streaming
static double runSsveFp32Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  volatile float vb = 0.999999f, vc = 0.000001f;
  const svfloat32_t b = svdup_f32(vb), c = svdup_f32(vc);
#define DECL(i) svfloat32_t a##i = svdup_f32(0.1f * (float)((i) + 1));
  SME_REP24(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) a##i = svmad_f32_x(pg, a##i, b, c);
      SME_REP24(STEP)
#undef STEP
    }
  svfloat32_t s = svdup_f32(0.0f);
#define RED(i) s = svadd_f32_x(pg, s, a##i);
  SME_REP24(RED)
#undef RED
  return (double)svaddv_f32(pg, s);
}

__arm_locally_streaming
static double runSsveFp64Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b64();
  volatile double vb = 0.999999, vc = 0.000001;
  const svfloat64_t b = svdup_f64(vb), c = svdup_f64(vc);
#define DECL(i) svfloat64_t a##i = svdup_f64(0.1 * (double)((i) + 1));
  SME_REP24(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) a##i = svmad_f64_x(pg, a##i, b, c);
      SME_REP24(STEP)
#undef STEP
    }
  svfloat64_t s = svdup_f64(0.0);
#define RED(i) s = svadd_f64_x(pg, s, a##i);
  SME_REP24(RED)
#undef RED
  return (double)svaddv_f64(pg, s);
}

} // anonymous namespace
} // namespace clpeak_cpu

#endif // __ARM_FEATURE_SME && !CLPEAK_CORE_ONLY

#endif // ENABLE_CPU
#endif // CPU_KERN_SME_COMPUTE_H
