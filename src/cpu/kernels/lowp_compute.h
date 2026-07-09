#ifndef CPU_KERN_LOWP_COMPUTE_H
#define CPU_KERN_LOWP_COMPUTE_H

#ifdef ENABLE_CPU

// ===========================================================================
// Low-/mixed-precision compute chains: fp16 FMA, bf16 dot, mixed-precision
// (fp16xfp16+fp32 FMLAL), int8 dot product, and the AVX10.2 native bf16 vector
// FMA.  Each kernel is #if-gated on a specific compile feature, so a dedicated
// feature TU enables exactly one path per dtype.  Guarded as a whole by
// CLPEAK_CORE_ONLY (passed to MSVC cl.exe TUs, whose coarse /arch:AVX512 can't
// isolate sub-features) so a core-AVX-512 binary never emits VNNI/BF16/FP16 it
// might lack at runtime.  To add a new low-precision dtype: add its #if block +
// a CPU_HAS_<X>_KERNEL define here, then wire the slot in the table builder.
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#include <cstring>

#ifndef CLPEAK_CORE_ONLY

namespace clpeak_cpu {
namespace {

// ---- FP16 native FMA -------------------------------------------------------
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define CPU_HAS_FP16_KERNEL 1
static constexpr int FP16_LANES = 8, FP16_NACC = 16;
static double runFp16Chain(uint64_t outer)
{
  float16x8_t acc[FP16_NACC];
  const float16x8_t b = vdupq_n_f16((float16_t)0.9995f);
  const float16x8_t c = vdupq_n_f16((float16_t)0.001f);
  for (int j = 0; j < FP16_NACC; j++) acc[j] = vdupq_n_f16((float16_t)(0.1f * (j + 1)));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < FP16_NACC; j++) acc[j] = vfmaq_f16(c, acc[j], b);
    }
  float16x8_t s = acc[0];
  for (int j = 1; j < FP16_NACC; j++) s = vaddq_f16(s, acc[j]);
  float16_t tmp[8]; vst1q_f16(tmp, s);
  double r = 0.0; for (int k = 0; k < 8; k++) r += (double)tmp[k];
  return r;
}
#elif defined(__AVX512FP16__)
#define CPU_HAS_FP16_KERNEL 1
static constexpr int FP16_LANES = 32, FP16_NACC = 16;
static double runFp16Chain(uint64_t outer)
{
  __m512h acc[FP16_NACC];
  const __m512h b = _mm512_set1_ph((_Float16)0.9995f);
  const __m512h c = _mm512_set1_ph((_Float16)0.001f);
  for (int j = 0; j < FP16_NACC; j++) acc[j] = _mm512_set1_ph((_Float16)(0.1f * (j + 1)));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < FP16_NACC; j++) acc[j] = _mm512_fmadd_ph(acc[j], b, c);
    }
  __m512h s = acc[0];
  for (int j = 1; j < FP16_NACC; j++) s = _mm512_add_ph(s, acc[j]);
  _Float16 tmp[32]; _mm512_storeu_ph(tmp, s);
  double r = 0.0; for (int k = 0; k < 32; k++) r += (double)tmp[k];
  return r;
}
#endif

// ---- BF16 dot --------------------------------------------------------------
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__ARM_FEATURE_BF16)
#define CPU_HAS_BF16_KERNEL 1
static constexpr int BF16_NACC = 16, BF16_FLOPS_PER_INSTR = 16;
static double runBf16Chain(uint64_t outer)
{
  float32x4_t acc[BF16_NACC];
  const bfloat16x4_t lo = vcvt_bf16_f32(vdupq_n_f32(1.0001f));
  const bfloat16x8_t a  = vcombine_bf16(lo, lo);
  const bfloat16x8_t b  = vcombine_bf16(lo, lo);
  for (int j = 0; j < BF16_NACC; j++) acc[j] = vdupq_n_f32(0.0f);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < BF16_NACC; j++) acc[j] = vbfdotq_f32(acc[j], a, b);
    }
  float32x4_t s = acc[0];
  for (int j = 1; j < BF16_NACC; j++) s = vaddq_f32(s, acc[j]);
  return (double)vaddvq_f32(s);
}
#elif defined(__AVX512BF16__)
#define CPU_HAS_BF16_KERNEL 1
static constexpr int BF16_NACC = 16, BF16_FLOPS_PER_INSTR = 64;
static double runBf16Chain(uint64_t outer)
{
  __m512 acc[BF16_NACC];
  const __m512bh a = _mm512_cvtne2ps_pbh(_mm512_set1_ps(1.0001f), _mm512_set1_ps(1.0001f));
  const __m512bh b = a;
  for (int j = 0; j < BF16_NACC; j++) acc[j] = _mm512_setzero_ps();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < BF16_NACC; j++) acc[j] = _mm512_dpbf16_ps(acc[j], a, b);
    }
  __m512 s = acc[0];
  for (int j = 1; j < BF16_NACC; j++) s = _mm512_add_ps(s, acc[j]);
  return (double)_mm512_reduce_add_ps(s);
}
#endif

// ---- Mixed precision (fp16 mul -> fp32 acc, widening FMLA) ------------------
#if (defined(__ARM_FEATURE_FP16FML) || defined(__ARM_FEATURE_FP16_FML)) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define CPU_HAS_MP_KERNEL 1
static constexpr int MP_NACC = 16, MP_OPS_PER_INSTR = 16;  // 2 FMLAL = 8 mul + 8 add
static double runMpChain(uint64_t outer)
{
  float32x4_t acc[MP_NACC];
  // FMLAL can only *add* to the fp32 accumulator (its operands are fp16), so a
  // pure-FMLAL chain `acc += m*b` is a LINEAR function of the iteration count and
  // -ffast-math closes it into `acc += N*m*b`, deleting every FMLAL -> a peak that
  // beats the impossible fp16 ceiling (mp must be <= ~0.5x fp16, FMLAL doing half
  // the flops/instr).  A live-operand recurrence, distinct per-chain coefficients,
  // and even an asm("+w") barrier all FAILED on the aarch64 server clang -- the
  // work stays *linear*, so it's always close-form-able.  The robust fix is the
  // fp16 chain's trick: feed the accumulator back as a MULTIPLICAND so the
  // recurrence is genuinely NONLINEAR (no closed form exists).  FMLAL multiplies
  // fp16, so narrow acc -> fp16 each step:
  //   acc <- acc + narrow(acc)*(-decay) + 1*refill   ->  fixed point refill/decay.
  // The vcvt is what makes it nonlinear (and uncollapsible); the constant refill
  // keeps acc off zero so the feedback term stays meaningful.
  const float16x8_t decay  = vdupq_n_f16((float16_t)-0.000977f);  // <0: contracts
  const float16x8_t refill = vdupq_n_f16((float16_t) 0.001953f);
  const float16x8_t one    = vdupq_n_f16((float16_t) 1.0f);
  for (int j = 0; j < MP_NACC; j++) acc[j] = vdupq_n_f32(1.0f + 0.01f * j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < MP_NACC; j++)
      {
        float16x4_t h  = vcvt_f16_f32(acc[j]);          // narrow: nonlinear feedback
        float16x8_t hh = vcombine_f16(h, h);
        acc[j] = vfmlalq_low_f16(acc[j], hh, decay);    // acc += narrow(acc)*(-decay)
        acc[j] = vfmlalq_high_f16(acc[j], one, refill); // acc += 1*refill (off-zero)
      }
    }
  float32x4_t s = acc[0];
  for (int j = 1; j < MP_NACC; j++) s = vaddq_f32(s, acc[j]);
  return (double)vaddvq_f32(s);
}
#endif

// ---- INT8 dot product ------------------------------------------------------
#if defined(__AVX512VNNI__)
#define CPU_HAS_INT8DP_KERNEL 1
static constexpr int I8_NACC = 16, I8_OPS_PER_INSTR = 128;
static double runInt8DpChain(uint64_t outer)
{
  __m512i acc[I8_NACC];
  const __m512i a = _mm512_set1_epi8((char)3);
  const __m512i b = _mm512_set1_epi8((char)5);
  for (int j = 0; j < I8_NACC; j++) acc[j] = _mm512_setzero_si512();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I8_NACC; j++) acc[j] = _mm512_dpbusd_epi32(acc[j], a, b);
    }
  __m512i s = acc[0];
  for (int j = 1; j < I8_NACC; j++) s = _mm512_add_epi32(s, acc[j]);
  return (double)_mm512_reduce_add_epi32(s);
}
#elif defined(__AVXVNNI__)
#define CPU_HAS_INT8DP_KERNEL 1
static constexpr int I8_NACC = 12, I8_OPS_PER_INSTR = 64;
static double runInt8DpChain(uint64_t outer)
{
  __m256i acc[I8_NACC];
  const __m256i a = _mm256_set1_epi8((char)3);
  const __m256i b = _mm256_set1_epi8((char)5);
  for (int j = 0; j < I8_NACC; j++) acc[j] = _mm256_setzero_si256();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I8_NACC; j++) acc[j] = _mm256_dpbusd_epi32(acc[j], a, b);
    }
  __m256i sAll = acc[0];
  for (int j = 1; j < I8_NACC; j++) sAll = _mm256_add_epi32(sAll, acc[j]);
  __m128i lo = _mm256_castsi256_si128(sAll);
  __m128i hi = _mm256_extracti128_si256(sAll, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return (double)_mm_cvtsi128_si32(lo);
}
#elif defined(__AVXVNNIINT8__)
// 256-bit AVX-VNNI-INT8: signed×signed int8 dot (VPDPBSSD).  Same throughput as
// AVX-VNNI's VPDPBUSD but a distinct ISA row (Zen 6, Lunar Lake, Arrow Lake+).
#define CPU_HAS_INT8DP_KERNEL 1
static constexpr int I8_NACC = 12, I8_OPS_PER_INSTR = 64;
static double runInt8DpChain(uint64_t outer)
{
  __m256i acc[I8_NACC];
  const __m256i a = _mm256_set1_epi8((char)3);
  const __m256i b = _mm256_set1_epi8((char)5);
  for (int j = 0; j < I8_NACC; j++) acc[j] = _mm256_setzero_si256();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I8_NACC; j++) acc[j] = _mm256_dpbssd_epi32(acc[j], a, b);
    }
  __m256i sAll = acc[0];
  for (int j = 1; j < I8_NACC; j++) sAll = _mm256_add_epi32(sAll, acc[j]);
  __m128i lo = _mm256_castsi256_si128(sAll);
  __m128i hi = _mm256_extracti128_si256(sAll, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return (double)_mm_cvtsi128_si32(lo);
}
#elif defined(__ARM_FEATURE_DOTPROD)
#define CPU_HAS_INT8DP_KERNEL 1
static constexpr int I8_NACC = 16, I8_OPS_PER_INSTR = 32;
static double runInt8DpChain(uint64_t outer)
{
  int32x4_t acc[I8_NACC];
  const int8x16_t a = vdupq_n_s8(3);
  const int8x16_t b = vdupq_n_s8(5);
  for (int j = 0; j < I8_NACC; j++) acc[j] = vdupq_n_s32(0);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I8_NACC; j++) acc[j] = vdotq_s32(acc[j], a, b);
    }
  int32x4_t s = acc[0];
  for (int j = 1; j < I8_NACC; j++) s = vaddq_s32(s, acc[j]);
  return (double)vaddvq_s32(s);
}
#endif

// ---- INT16 dot product (x86) ------------------------------------------------
// VPDPWSSD (2-way signed int16 dot -> int32) has shipped with VNNI since
// Cascade Lake / Alder Lake; AVX-VNNI-INT16 (Diamond Rapids, Nova Lake) adds
// the mixed-sign forms (VPDPWSUD etc.) as a separate CPUID feature.
//
// Unlike the int8 VPDPBUSD (opaque to the optimizer), clang MODELS the int16
// dot's semantics: with constant operands `acc += dot(a,b)` is linear in the
// iteration count and instcombine strength-reduced 4 unrolled 512-bit VPDPWSSD
// into 1 VPDPWSSD + 3 VPADDD (observed) -- the same collapse class as the ARM
// FMLAL `mp` chain.  Same fix: make the recurrence NONLINEAR by feeding the
// accumulator back as the multiplicand, `acc = dp(acc, acc, b)` (no closed
// form; int wraparound keeps it bounded and is well-defined for intrinsics),
// with a volatile-seeded initial acc so it can't constant-propagate.
#if defined(__AVX512VNNI__)
#define CPU_HAS_INT16DP_KERNEL 1
static constexpr int I16_NACC = 16, I16_OPS_PER_INSTR = 64;
static double runInt16DpChain(uint64_t outer)
{
  __m512i acc[I16_NACC];
  volatile int vseed = 3;
  const int seed = vseed;
  const __m512i b = _mm512_set1_epi16(5);
  for (int j = 0; j < I16_NACC; j++) acc[j] = _mm512_set1_epi32(seed + j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I16_NACC; j++) acc[j] = _mm512_dpwssd_epi32(acc[j], acc[j], b);
    }
  __m512i s = acc[0];
  for (int j = 1; j < I16_NACC; j++) s = _mm512_add_epi32(s, acc[j]);
  return (double)_mm512_reduce_add_epi32(s);
}
#elif defined(__AVXVNNI__)
#define CPU_HAS_INT16DP_KERNEL 1
static constexpr int I16_NACC = 12, I16_OPS_PER_INSTR = 32;
static double runInt16DpChain(uint64_t outer)
{
  __m256i acc[I16_NACC];
  volatile int vseed = 3;
  const int seed = vseed;
  const __m256i b = _mm256_set1_epi16(5);
  for (int j = 0; j < I16_NACC; j++) acc[j] = _mm256_set1_epi32(seed + j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I16_NACC; j++) acc[j] = _mm256_dpwssd_avx_epi32(acc[j], acc[j], b);
    }
  __m256i sAll = acc[0];
  for (int j = 1; j < I16_NACC; j++) sAll = _mm256_add_epi32(sAll, acc[j]);
  __m128i lo = _mm256_castsi256_si128(sAll);
  __m128i hi = _mm256_extracti128_si256(sAll, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return (double)_mm_cvtsi128_si32(lo);
}
#elif defined(__AVXVNNIINT16__)
// The AVX-VNNI-INT16 TU: mixed-sign VPDPWSUD (its distinct contribution --
// the ssd form already rides in the VNNI TUs above).  Same rate, own ISA row.
#define CPU_HAS_INT16DP_KERNEL 1
static constexpr int I16_NACC = 12, I16_OPS_PER_INSTR = 32;
static double runInt16DpChain(uint64_t outer)
{
  __m256i acc[I16_NACC];
  volatile int vseed = 3;
  const int seed = vseed;
  const __m256i b = _mm256_set1_epi16(5);
  for (int j = 0; j < I16_NACC; j++) acc[j] = _mm256_set1_epi32(seed + j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I16_NACC; j++) acc[j] = _mm256_dpwsud_epi32(acc[j], acc[j], b);
    }
  __m256i sAll = acc[0];
  for (int j = 1; j < I16_NACC; j++) sAll = _mm256_add_epi32(sAll, acc[j]);
  __m128i lo = _mm256_castsi256_si128(sAll);
  __m128i hi = _mm256_extracti128_si256(sAll, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return (double)_mm_cvtsi128_si32(lo);
}
#endif

// ---- FP8 dot product (ARM FEAT_FP8DOT4) --------------------------------------
// Native fp8 4-way dot -> fp32 (FDOT), first shipped by NVIDIA Vera.  The fpm_t
// operand programs the FPMR format register; the compiler hoists the msr out of
// the loop (verified), so the hot loop is back-to-back fdot.  Inputs are a
// nonzero constant (0x38 = 0.5 in e4m3); throughput is format-independent, and
// like the int8/bf16 dot chains the intrinsic is opaque enough that the
// accumulate loop isn't scalar-evolved -- re-verify with objdump on new clangs.
#if defined(__ARM_FEATURE_FP8DOT4)
#define CPU_HAS_FP8DP_KERNEL 1
static constexpr int FP8_NACC = 16, FP8_OPS_PER_INSTR = 32;  // 4 lanes x (4 mul + 4 add)
static double runFp8DpChain(uint64_t outer)
{
  const fpm_t fpm = __arm_fpm_init();
  const mfloat8x16_t a = vreinterpretq_mf8_u8(vdupq_n_u8(0x38));
  const mfloat8x16_t b = a;
  float32x4_t acc[FP8_NACC];
  for (int j = 0; j < FP8_NACC; j++) acc[j] = vdupq_n_f32(0.0f);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < FP8_NACC; j++) acc[j] = vdotq_f32_mf8_fpm(acc[j], a, b, fpm);
    }
  float32x4_t s = acc[0];
  for (int j = 1; j < FP8_NACC; j++) s = vaddq_f32(s, acc[j]);
  return (double)vaddvq_f32(s);
}
#endif

// ---- AVX10.2 native bf16 vector FMA (full-rate, not a dot) ------------------
// Distinct from the AVX512-BF16 dot path: VFMADD*PBF16 does a real bf16 multiply-
// add at full 512-bit vector rate (32 lanes).  Same affine chain as fp16; b must
// not round to 1.0 in bf16 (0.98828 is exact and distinct).
#if defined(__AVX10_2_512__)
#define CPU_HAS_BF16FMA_KERNEL 1
static constexpr int BF16FMA_LANES = 32, BF16FMA_NACC = 16;
static double runBf16FmaChain(uint64_t outer)
{
  __m512bh acc[BF16FMA_NACC];
  const __m512bh b = _mm512_set1_pbh((__bf16)0.98828f);
  const __m512bh c = _mm512_set1_pbh((__bf16)0.0117f);
  for (int j = 0; j < BF16FMA_NACC; j++) acc[j] = _mm512_set1_pbh((__bf16)(0.1f * (j + 1)));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < BF16FMA_NACC; j++) acc[j] = _mm512_fmadd_pbh(acc[j], b, c);
    }
  __m512bh s = acc[0];
  for (int j = 1; j < BF16FMA_NACC; j++) s = _mm512_add_pbh(s, acc[j]);
  // Reduce via a raw bf16->fp32 widening (no cast intrinsic takes __m512bh whole).
  uint16_t tmp[32]; std::memcpy(tmp, &s, sizeof(tmp));
  double r = 0.0;
  for (int i = 0; i < 32; i++)
  { uint32_t u = (uint32_t)tmp[i] << 16; float f; std::memcpy(&f, &u, sizeof(f)); r += (double)f; }
  return r;
}
#endif

} // anonymous namespace
} // namespace clpeak_cpu

#endif // CLPEAK_CORE_ONLY

#endif // ENABLE_CPU
#endif // CPU_KERN_LOWP_COMPUTE_H
