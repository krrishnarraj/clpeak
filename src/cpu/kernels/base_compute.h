#ifndef CPU_KERN_BASE_COMPUTE_H
#define CPU_KERN_BASE_COMPUTE_H

#ifdef ENABLE_CPU

// ===========================================================================
// Base compute chains present in EVERY TU: fp32 / fp64 / int32 FMA-chains and
// the streaming XOR read used by the bandwidth path.  All of these go through
// the cpu_simd.h vector abstraction (or its scalar fallback), so this header
// compiles at any ISA level -- the SIMD width is chosen by the TU's build flags.
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#include <cstring>

namespace clpeak_cpu {
namespace {

using namespace cpu_simd;

// ---- FP32 / FP64 FMA-chains -----------------------------------------------
#if defined(__aarch64__) || defined(_M_ARM64)
// NEON-specific chain shape.  AArch64 NEON has no FMAD-form instruction:
// FMLA is destructive on the ADDEND (Vd = Vd + Vn*Vm), so the generic affine
// chain acc = acc*b + c makes the compiler re-materialise the loop-invariant
// addend c with one MOV per FMLA.  Apple cores hide that (zero-cycle move
// elimination in rename) but Neoverse/Oryon don't: NEON fp32 measured ~2.4x
// below the SVE rows on Neoverse N2 (mov + fmla = 2 vector slots per FMA).
//
// Fix: a SELF-QUADRATIC recurrence, acc = acc + acc*acc, which maps to a
// single `fmla v,v,v` -- the destructive operand IS the accumulator, so the
// hot loop is pure back-to-back FMLA with no coefficient registers at all.
// It is genuinely nonlinear, so neither -ffast-math factoring (x + x*C ->
// x*(C+1)) nor scalar evolution has a closed form to collapse it with (the
// same argument as the FMLAL `mp` chain fix).  Dynamics: from acc0 in (-1,0)
// it decays HARMONICALLY (acc_n ~ -1/n) toward 0 and freezes once acc^2 drops
// below half an ulp of acc (~-1e-10 for fp32) -- values stay normal for the
// whole run: no denormal, inf, or NaN phase.  The volatile seed keeps the
// start value opaque so the recurrence can't be constant-evaluated.
static double runFp32Chain(uint64_t outer)
{
  float32x4_t acc[F32_NACC];
  volatile float vseed = -0.5f;
  const float seed = vseed;
  for (int j = 0; j < F32_NACC; j++) acc[j] = vdupq_n_f32(seed + 0.01f * (float)j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < F32_NACC; j++) acc[j] = vfmaq_f32(acc[j], acc[j], acc[j]);
    }
  float32x4_t s = acc[0];
  for (int j = 1; j < F32_NACC; j++) s = vaddq_f32(s, acc[j]);
  return (double)vaddvq_f32(s);
}

static double runFp64Chain(uint64_t outer)
{
  float64x2_t acc[F64_NACC];
  volatile double vseed = -0.5;
  const double seed = vseed;
  for (int j = 0; j < F64_NACC; j++) acc[j] = vdupq_n_f64(seed + 0.01 * (double)j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < F64_NACC; j++) acc[j] = vfmaq_f64(acc[j], acc[j], acc[j]);
    }
  float64x2_t s = acc[0];
  for (int j = 1; j < F64_NACC; j++) s = vaddq_f64(s, acc[j]);
  return (double)vaddvq_f64(s);
}
#else // !aarch64: generic affine chains through the cpu_simd.h abstraction
static double runFp32Chain(uint64_t outer)
{
  f32v acc[F32_NACC];
  // Seed the coefficients through volatile so the compiler can't prove their
  // values and close the affine recurrence acc=acc*b+c.  On non-FMA targets
  // (SSE2 / scalar) f32_fma is a transparent mul+add; with -ffast-math the
  // CPU_UNROLL_K-unrolled chain would otherwise fold N steps into one
  // (acc*b^N + const), deleting most of the work while opsPerIter still counts
  // it -> an impossible peak (SSE2 reported > AVX2).  Same guard as the int32
  // chain.  No-op on FMA targets (b/c become runtime operands loaded once).
  volatile float vb = 0.999999f, vc = 0.000001f;
  const f32v b = f32_set(vb);
  const f32v c = f32_set(vc);
  for (int j = 0; j < F32_NACC; j++) acc[j] = f32_set(0.1f * (float)(j + 1));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < F32_NACC; j++) acc[j] = f32_fma(acc[j], b, c);
    }
  f32v s = acc[0];
  for (int j = 1; j < F32_NACC; j++) s = f32_add(s, acc[j]);
  return (double)f32_hsum(s);
}

static double runFp64Chain(uint64_t outer)
{
  f64v acc[F64_NACC];
  // Volatile-seed the coefficients to keep the affine chain from collapsing on
  // non-FMA (SSE2 / scalar) targets under -ffast-math.  See runFp32Chain.
  volatile double vb = 0.999999, vc = 0.000001;
  const f64v b = f64_set(vb);
  const f64v c = f64_set(vc);
  for (int j = 0; j < F64_NACC; j++) acc[j] = f64_set(0.1 * (double)(j + 1));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < F64_NACC; j++) acc[j] = f64_fma(acc[j], b, c);
    }
  f64v s = acc[0];
  for (int j = 1; j < F64_NACC; j++) s = f64_add(s, acc[j]);
  return (double)f64_hsum(s);
}
#endif // aarch64 NEON vs generic

// ---- INT32 multiply-accumulate chain --------------------------------------
static double runInt32Chain(uint64_t outer)
{
  i32v acc[I32_NACC];
  // Opaque multiplier: a compile-time-constant multiplier lets the compiler
  // strength-reduce `acc*k` into shifts/adds (measuring the shifter, not the
  // integer multiplier, and varying with -mtune/inlining).  Reading it through
  // volatile forces a real vpmulld / vmlaq — a consistent, honest IMAD peak.
  volatile int vmul = 1664525;
  const i32v b = i32_set((int)vmul);
  const i32v c = i32_set(1013904223);
  for (int j = 0; j < I32_NACC; j++) acc[j] = i32_set(j + 1);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I32_NACC; j++) acc[j] = i32_madd(acc[j], b, c);
    }
  i32v s = acc[0];
  for (int j = 1; j < I32_NACC; j++) s = i32_add(s, acc[j]);
  return (double)i32_hsum(s);
}

// ---- Streaming read (integer XOR checksum) --------------------------------
static uint64_t readBufferChecksum(const float *p, size_t M, uint64_t iters)
{
#if defined(__AVX512F__)
  constexpr size_t W = 16;
  const size_t step = 8 * W;
  __m512i x0 = _mm512_setzero_si512(), x1 = x0, x2 = x0, x3 = x0;
  __m512i x4 = x0, x5 = x0, x6 = x0, x7 = x0;
  uint64_t tail = 0;
  for (uint64_t it = 0; it < iters; it++)
  {
    size_t i = 0;
    for (; i + step <= M; i += step)
    {
      x0 = _mm512_xor_si512(x0, _mm512_castps_si512(_mm512_loadu_ps(p + i + 0 * W)));
      x1 = _mm512_xor_si512(x1, _mm512_castps_si512(_mm512_loadu_ps(p + i + 1 * W)));
      x2 = _mm512_xor_si512(x2, _mm512_castps_si512(_mm512_loadu_ps(p + i + 2 * W)));
      x3 = _mm512_xor_si512(x3, _mm512_castps_si512(_mm512_loadu_ps(p + i + 3 * W)));
      x4 = _mm512_xor_si512(x4, _mm512_castps_si512(_mm512_loadu_ps(p + i + 4 * W)));
      x5 = _mm512_xor_si512(x5, _mm512_castps_si512(_mm512_loadu_ps(p + i + 5 * W)));
      x6 = _mm512_xor_si512(x6, _mm512_castps_si512(_mm512_loadu_ps(p + i + 6 * W)));
      x7 = _mm512_xor_si512(x7, _mm512_castps_si512(_mm512_loadu_ps(p + i + 7 * W)));
    }
    for (; i < M; i++) { uint32_t v; std::memcpy(&v, p + i, sizeof(v)); tail ^= v; }
  }
  __m512i x = _mm512_xor_si512(_mm512_xor_si512(_mm512_xor_si512(x0, x1), _mm512_xor_si512(x2, x3)),
                               _mm512_xor_si512(_mm512_xor_si512(x4, x5), _mm512_xor_si512(x6, x7)));
  alignas(64) uint64_t tmp[8]; _mm512_store_si512((__m512i *)tmp, x);
  for (uint64_t v : tmp) tail ^= v;
  return tail;
#elif defined(__AVX2__)
  constexpr size_t W = 8;
  const size_t step = 8 * W;
  __m256i x0 = _mm256_setzero_si256(), x1 = x0, x2 = x0, x3 = x0;
  __m256i x4 = x0, x5 = x0, x6 = x0, x7 = x0;
  uint64_t tail = 0;
  for (uint64_t it = 0; it < iters; it++)
  {
    size_t i = 0;
    for (; i + step <= M; i += step)
    {
      x0 = _mm256_xor_si256(x0, _mm256_castps_si256(_mm256_loadu_ps(p + i + 0 * W)));
      x1 = _mm256_xor_si256(x1, _mm256_castps_si256(_mm256_loadu_ps(p + i + 1 * W)));
      x2 = _mm256_xor_si256(x2, _mm256_castps_si256(_mm256_loadu_ps(p + i + 2 * W)));
      x3 = _mm256_xor_si256(x3, _mm256_castps_si256(_mm256_loadu_ps(p + i + 3 * W)));
      x4 = _mm256_xor_si256(x4, _mm256_castps_si256(_mm256_loadu_ps(p + i + 4 * W)));
      x5 = _mm256_xor_si256(x5, _mm256_castps_si256(_mm256_loadu_ps(p + i + 5 * W)));
      x6 = _mm256_xor_si256(x6, _mm256_castps_si256(_mm256_loadu_ps(p + i + 6 * W)));
      x7 = _mm256_xor_si256(x7, _mm256_castps_si256(_mm256_loadu_ps(p + i + 7 * W)));
    }
    for (; i < M; i++) { uint32_t v; std::memcpy(&v, p + i, sizeof(v)); tail ^= v; }
  }
  __m256i x = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(x0, x1), _mm256_xor_si256(x2, x3)),
                               _mm256_xor_si256(_mm256_xor_si256(x4, x5), _mm256_xor_si256(x6, x7)));
  alignas(32) uint64_t tmp[4]; _mm256_store_si256((__m256i *)tmp, x);
  for (uint64_t v : tmp) tail ^= v;
  return tail;
#elif defined(__SSE2__)
  constexpr size_t W = 4;
  const size_t step = 8 * W;
  __m128i x0 = _mm_setzero_si128(), x1 = x0, x2 = x0, x3 = x0;
  __m128i x4 = x0, x5 = x0, x6 = x0, x7 = x0;
  uint64_t tail = 0;
  for (uint64_t it = 0; it < iters; it++)
  {
    size_t i = 0;
    for (; i + step <= M; i += step)
    {
      x0 = _mm_xor_si128(x0, _mm_castps_si128(_mm_loadu_ps(p + i + 0 * W)));
      x1 = _mm_xor_si128(x1, _mm_castps_si128(_mm_loadu_ps(p + i + 1 * W)));
      x2 = _mm_xor_si128(x2, _mm_castps_si128(_mm_loadu_ps(p + i + 2 * W)));
      x3 = _mm_xor_si128(x3, _mm_castps_si128(_mm_loadu_ps(p + i + 3 * W)));
      x4 = _mm_xor_si128(x4, _mm_castps_si128(_mm_loadu_ps(p + i + 4 * W)));
      x5 = _mm_xor_si128(x5, _mm_castps_si128(_mm_loadu_ps(p + i + 5 * W)));
      x6 = _mm_xor_si128(x6, _mm_castps_si128(_mm_loadu_ps(p + i + 6 * W)));
      x7 = _mm_xor_si128(x7, _mm_castps_si128(_mm_loadu_ps(p + i + 7 * W)));
    }
    for (; i < M; i++) { uint32_t v; std::memcpy(&v, p + i, sizeof(v)); tail ^= v; }
  }
  __m128i x = _mm_xor_si128(_mm_xor_si128(_mm_xor_si128(x0, x1), _mm_xor_si128(x2, x3)),
                            _mm_xor_si128(_mm_xor_si128(x4, x5), _mm_xor_si128(x6, x7)));
  alignas(16) uint32_t tmp[4]; _mm_store_si128((__m128i *)tmp, x);
  for (uint32_t v : tmp) tail ^= v;
  return tail;
#elif defined(__aarch64__) || defined(_M_ARM64)
  // AArch64 only (matches cpu_simd.h, incl. the MSVC ARM64 _M_ARM64 alias):
  // 32-bit ARMv7 doesn't pull in arm_neon.h, so it uses the scalar read
  // fallback below.
  constexpr size_t W = 4;
  const size_t step = 8 * W;
  uint32x4_t x0 = vdupq_n_u32(0), x1 = x0, x2 = x0, x3 = x0;
  uint32x4_t x4 = x0, x5 = x0, x6 = x0, x7 = x0;
  uint64_t tail = 0;
  for (uint64_t it = 0; it < iters; it++)
  {
    size_t i = 0;
    for (; i + step <= M; i += step)
    {
      x0 = veorq_u32(x0, vreinterpretq_u32_f32(vld1q_f32(p + i + 0 * W)));
      x1 = veorq_u32(x1, vreinterpretq_u32_f32(vld1q_f32(p + i + 1 * W)));
      x2 = veorq_u32(x2, vreinterpretq_u32_f32(vld1q_f32(p + i + 2 * W)));
      x3 = veorq_u32(x3, vreinterpretq_u32_f32(vld1q_f32(p + i + 3 * W)));
      x4 = veorq_u32(x4, vreinterpretq_u32_f32(vld1q_f32(p + i + 4 * W)));
      x5 = veorq_u32(x5, vreinterpretq_u32_f32(vld1q_f32(p + i + 5 * W)));
      x6 = veorq_u32(x6, vreinterpretq_u32_f32(vld1q_f32(p + i + 6 * W)));
      x7 = veorq_u32(x7, vreinterpretq_u32_f32(vld1q_f32(p + i + 7 * W)));
    }
    for (; i < M; i++) { uint32_t v; std::memcpy(&v, p + i, sizeof(v)); tail ^= v; }
  }
  uint32x4_t x = veorq_u32(veorq_u32(veorq_u32(x0, x1), veorq_u32(x2, x3)),
                           veorq_u32(veorq_u32(x4, x5), veorq_u32(x6, x7)));
  alignas(16) uint32_t tmp[4]; vst1q_u32(tmp, x);
  for (uint32_t v : tmp) tail ^= v;
  return tail;
#else
  uint64_t acc = 0;
  for (uint64_t it = 0; it < iters; it++)
    for (size_t i = 0; i < M; i++) { uint32_t v; std::memcpy(&v, p + i, sizeof(v)); acc ^= v; }
  return acc;
#endif
}

} // anonymous namespace
} // namespace clpeak_cpu

#endif // ENABLE_CPU
#endif // CPU_KERN_BASE_COMPUTE_H
