#ifndef CPU_KERN_SVE_COMPUTE_H
#define CPU_KERN_SVE_COMPUTE_H

#ifdef ENABLE_CPU

// ===========================================================================
// ARM SVE (vector-length-agnostic) compute + matrix kernels.  SVE registers are
// "sizeless": svfloat32_t etc. can't be array/struct members, so the NACC
// independent accumulator chains are declared as individual named registers via
// the SVE_REP macros instead of the acc[NACC] loop the fixed-width kernels use.
// The kernels are VL-agnostic (one binary runs 128-bit Oryon/Vera, 256-bit
// Graviton3, ...); the op count is therefore VL-dependent and computed from
// svcntw()/svcntd()/svcntb() at table-build time (only ever reached when SVE is
// the running host's ISA).  b/c coefficients ride in loop-invariant Z registers,
// so each svmad is one back-to-back MAD with no reload -- the SVE analogue of the
// NEON FMLA chain.
//
// The whole set is #if-gated on __ARM_FEATURE_SVE and excluded under
// CLPEAK_CORE_ONLY, so this header only contributes kernels in an SVE TU.
// Also excluded when the TU has SME (__ARM_FEATURE_SME): these are
// NON-streaming SVE kernels, and an SME TU must never pick them up -- on Apple
// Silicon (SME without any non-streaming SVE) they would SIGILL if selected.
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#if defined(__ARM_FEATURE_SVE) && !defined(__ARM_FEATURE_SME) && !defined(CLPEAK_CORE_ONLY)
#include <arm_sve.h>

namespace clpeak_cpu {
namespace {

#define CPU_HAS_SVE_KERNELS 1

// Expand X(i) for i in [0, N).  32 architectural Z registers give headroom for
// 24 fp accumulators + the 2 coefficient regs (same latency-hiding rationale as
// the NEON NACC=24; revisit with a sweep when validating on real SVE silicon).
#define SVE_REP16(X) X(0)X(1)X(2)X(3)X(4)X(5)X(6)X(7)X(8)X(9)X(10)X(11)X(12)X(13)X(14)X(15)
#define SVE_REP24(X) SVE_REP16(X) X(16)X(17)X(18)X(19)X(20)X(21)X(22)X(23)
static constexpr int SVE_NACC_FP  = 24;   // fp32/fp64/int32 MAD chains
static constexpr int SVE_NACC_DOT = 16;   // SDOT / BFDOT / *MMLA (wider per-instr)

// acc = acc*b + c  (svmad: MAD Zdn = Zdn*Zm + Za, destructive on the accumulator)
static double runSveFp32Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  volatile float vb = 0.999999f, vc = 0.000001f;
  const svfloat32_t b = svdup_f32(vb), c = svdup_f32(vc);
#define DECL(i) svfloat32_t a##i = svdup_f32(0.1f * (float)((i) + 1));
  SVE_REP24(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) a##i = svmad_f32_x(pg, a##i, b, c);
      SVE_REP24(STEP)
#undef STEP
    }
  svfloat32_t s = svdup_f32(0.0f);
#define RED(i) s = svadd_f32_x(pg, s, a##i);
  SVE_REP24(RED)
#undef RED
  return (double)svaddv_f32(pg, s);
}

static double runSveFp64Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b64();
  volatile double vb = 0.999999, vc = 0.000001;
  const svfloat64_t b = svdup_f64(vb), c = svdup_f64(vc);
#define DECL(i) svfloat64_t a##i = svdup_f64(0.1 * (double)((i) + 1));
  SVE_REP24(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) a##i = svmad_f64_x(pg, a##i, b, c);
      SVE_REP24(STEP)
#undef STEP
    }
  svfloat64_t s = svdup_f64(0.0);
#define RED(i) s = svadd_f64_x(pg, s, a##i);
  SVE_REP24(RED)
#undef RED
  return (double)svaddv_f64(pg, s);
}

static double runSveInt32Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  volatile int vmul = 1664525;
  const svint32_t b = svdup_s32(vmul), c = svdup_s32(1013904223);
#define DECL(i) svint32_t a##i = svdup_s32((i) + 1);
  SVE_REP24(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) a##i = svmad_s32_x(pg, a##i, b, c);
      SVE_REP24(STEP)
#undef STEP
    }
  svint32_t s = svdup_s32(0);
#define RED(i) s = svadd_s32_x(pg, s, a##i);
  SVE_REP24(RED)
#undef RED
  return (double)svaddv_s32(pg, s);
}

// int8 4-way dot product into int32 lanes (SDOT: base SVE, no SVE2 needed).
static double runSveInt8DpChain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  const svint8_t a = svdup_s8(3), b = svdup_s8(5);
#define DECL(i) svint32_t acc##i = svdup_s32(0);
  SVE_REP16(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) acc##i = svdot_s32(acc##i, a, b);
      SVE_REP16(STEP)
#undef STEP
    }
  svint32_t s = svdup_s32(0);
#define RED(i) s = svadd_s32_x(pg, s, acc##i);
  SVE_REP16(RED)
#undef RED
  return (double)svaddv_s32(pg, s);
}

// SVE BF16: BFDOT (2-way bf16 dot -> fp32) and BFMMLA (2x2 matrix accumulate).
#if defined(__ARM_FEATURE_SVE_BF16)
#define CPU_HAS_SVE_BF16_KERNEL 1
static double runSveBf16Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  // 0x3F81 ~= 1.0039 in bf16 (nonzero, distinct from an exact 1.0 collapse).
  const svbfloat16_t a = svreinterpret_bf16_u16(svdup_u16(0x3F81));
  const svbfloat16_t b = a;
#define DECL(i) svfloat32_t acc##i = svdup_f32(0.0f);
  SVE_REP16(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) acc##i = svbfdot_f32(acc##i, a, b);
      SVE_REP16(STEP)
#undef STEP
    }
  svfloat32_t s = svdup_f32(0.0f);
#define RED(i) s = svadd_f32_x(pg, s, acc##i);
  SVE_REP16(RED)
#undef RED
  return (double)svaddv_f32(pg, s);
}
static double runSveMatBf16Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  const svbfloat16_t a = svreinterpret_bf16_u16(svdup_u16(0x3F81));
  const svbfloat16_t b = a;
#define DECL(i) svfloat32_t acc##i = svdup_f32(0.0f);
  SVE_REP16(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) acc##i = svbfmmla_f32(acc##i, a, b);
      SVE_REP16(STEP)
#undef STEP
    }
  svfloat32_t s = svdup_f32(0.0f);
#define RED(i) s = svadd_f32_x(pg, s, acc##i);
  SVE_REP16(RED)
#undef RED
  return (double)svaddv_f32(pg, s);
}
#endif // __ARM_FEATURE_SVE_BF16

// SVE I8MM: SMMLA (int8 2x2 matrix accumulate into int32).
#if defined(__ARM_FEATURE_SVE_MATMUL_INT8)
#define CPU_HAS_SVE_I8MM_KERNEL 1
static double runSveMatInt8Chain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  const svint8_t a = svdup_s8(3), b = svdup_s8(5);
#define DECL(i) svint32_t acc##i = svdup_s32(0);
  SVE_REP16(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) acc##i = svmmla_s32(acc##i, a, b);
      SVE_REP16(STEP)
#undef STEP
    }
  svint32_t s = svdup_s32(0);
#define RED(i) s = svadd_s32_x(pg, s, acc##i);
  SVE_REP16(RED)
#undef RED
  return (double)svaddv_s32(pg, s);
}
#endif // __ARM_FEATURE_SVE_MATMUL_INT8

// SVE2 FP8: 4-way fp8 dot -> fp32 lanes (FEAT_FP8DOT4 SVE form; NVIDIA Vera).
// Same FPMR/fpm_t setup as the NEON fp8 kernel in lowp_compute.h; the msr is
// hoisted out of the loop (verified via objdump).
#if defined(__ARM_FEATURE_FP8DOT4) && defined(__ARM_FEATURE_SVE2)
#define CPU_HAS_SVE_FP8DP_KERNEL 1
static double runSveFp8DpChain(uint64_t outer)
{
  const svbool_t pg = svptrue_b32();
  const fpm_t fpm = __arm_fpm_init();
  const svmfloat8_t a = svreinterpret_mf8_u8(svdup_u8(0x38));  // 0.5 in e4m3
  const svmfloat8_t b = a;
#define DECL(i) svfloat32_t acc##i = svdup_f32(0.0f);
  SVE_REP16(DECL)
#undef DECL
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
#define STEP(i) acc##i = svdot_f32_mf8_fpm(acc##i, a, b, fpm);
      SVE_REP16(STEP)
#undef STEP
    }
  svfloat32_t s = svdup_f32(0.0f);
#define RED(i) s = svadd_f32_x(pg, s, acc##i);
  SVE_REP16(RED)
#undef RED
  return (double)svaddv_f32(pg, s);
}
#endif // __ARM_FEATURE_FP8DOT4 && __ARM_FEATURE_SVE2

} // anonymous namespace
} // namespace clpeak_cpu

#endif // __ARM_FEATURE_SVE && !CLPEAK_CORE_ONLY

#endif // ENABLE_CPU
#endif // CPU_KERN_SVE_COMPUTE_H
