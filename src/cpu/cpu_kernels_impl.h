#ifndef CPU_KERNELS_IMPL_H
#define CPU_KERNELS_IMPL_H

// ===========================================================================
// All compute / read kernel bodies, plus a per-TU table builder.  This header
// is compiled once per feature TU (cpu_kernels_tu.cpp, built N times with
// different -m/-arch flags); cpu_simd.h selects the SIMD path from those flags,
// and the #if-gated advanced kernels compile in only when the TU's flags enable
// the feature.  Everything lives in an anonymous namespace so each TU gets its
// own internal copy (no ODR clash, no cross-TU inlining).
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#include <cstring>

#ifndef CLPEAK_ISA_NAME_STR
#define CLPEAK_ISA_NAME_STR "scalar"
#endif

#if (defined(__AMX_INT8__) || defined(__AMX_BF16__)) && defined(__linux__)
#include <immintrin.h>
#endif

namespace clpeak_cpu {
namespace {

using namespace cpu_simd;

// ---- FP32 / FP64 FMA-chains -----------------------------------------------
static double runFp32Chain(uint64_t outer)
{
  f32v acc[F32_NACC];
  const f32v b = f32_set(0.999999f);
  const f32v c = f32_set(0.000001f);
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
  const f64v b = f64_set(0.999999);
  const f64v c = f64_set(0.000001);
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

// ---- INT32 multiply-accumulate chain --------------------------------------
static double runInt32Chain(uint64_t outer)
{
  i32v acc[I32_NACC];
  const i32v b = i32_set(1664525);
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
#elif defined(__ARM_NEON) || defined(__aarch64__)
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

// Advanced-dtype kernels.  Each is gated on a specific compile feature, so a
// dedicated feature TU enables exactly one.  CLPEAK_CORE_ONLY (passed to MSVC
// TUs, whose coarse /arch:AVX512 can't isolate sub-features) keeps these out so
// a core-AVX-512 binary never emits VNNI/BF16 it might lack at runtime.
#ifndef CLPEAK_CORE_ONLY

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
static constexpr int MP_NACC = 24, MP_OPS_PER_INSTR = 16;  // 2 FMLAL = 8 mul + 8 add
static double runMpChain(uint64_t outer)
{
  float32x4_t acc[MP_NACC];
  const float16x8_t a = vdupq_n_f16((float16_t)0.9995f);
  const float16x8_t b = vdupq_n_f16((float16_t)0.001f);
  for (int j = 0; j < MP_NACC; j++) acc[j] = vdupq_n_f32(1.0f + 0.01f * j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < MP_NACC; j++)
      {
        acc[j] = vfmlalq_low_f16(acc[j], a, b);
        acc[j] = vfmlalq_high_f16(acc[j], a, b);
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

// ---- Matrix engine: int8 (AMX / SMMLA) and bf16 (AMX / BFMMLA) -------------
// ops are PER-k; the table multiplies by INNER (the chain loops INNER per outer).
#if defined(__AMX_INT8__) && defined(__linux__)
#define CPU_MAT_INT8_KERNEL 1
static constexpr double MAT_I8_OPS_PER_K = 4.0 * 16 * 16 * 64 * 2;
static thread_local bool g_amxI8Cfg = false;
static alignas(64) int8_t  g_amxA[16 * 64];
static alignas(64) int8_t  g_amxB[16 * 64];
static alignas(64) int32_t g_amxC[16 * 16];
static void amxConfigTiles()
{
  struct { uint8_t palette, start_row, reserved[14]; uint16_t colsb[16]; uint8_t rows[16]; } cfg = {};
  cfg.palette = 1;
  for (int t = 0; t < 6; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
  _tile_loadconfig(&cfg);
}
static double runMatInt8Chain(uint64_t outer)
{
  if (!g_amxI8Cfg) { amxConfigTiles(); g_amxI8Cfg = true; }
  _tile_loadd(4, g_amxA, 64);
  _tile_loadd(5, g_amxB, 64);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  for (uint64_t o = 0; o < outer; o++)
    for (int k = 0; k < INNER; k++)
    { _tile_dpbssd(0, 4, 5); _tile_dpbssd(1, 4, 5); _tile_dpbssd(2, 4, 5); _tile_dpbssd(3, 4, 5); }
  double sink = 0.0;
  _tile_stored(0, g_amxC, 64); sink += g_amxC[0];
  _tile_stored(1, g_amxC, 64); sink += g_amxC[0];
  _tile_stored(2, g_amxC, 64); sink += g_amxC[0];
  _tile_stored(3, g_amxC, 64); sink += g_amxC[0];
  return sink;
}
#elif defined(__ARM_FEATURE_MATMUL_INT8)
#define CPU_MAT_INT8_KERNEL 1
static constexpr double MAT_I8_OPS_PER_K = 16.0 * 64.0;   // MM_NACC * 64
static double runMatInt8Chain(uint64_t outer)
{
  constexpr int NACC = 16;
  int32x4_t acc[NACC];
  const int8x16_t a = vdupq_n_s8(3);
  const int8x16_t b = vdupq_n_s8(5);
  for (int j = 0; j < NACC; j++) acc[j] = vdupq_n_s32(0);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < NACC; j++) acc[j] = vmmlaq_s32(acc[j], a, b);
    }
  int32x4_t s = acc[0];
  for (int j = 1; j < NACC; j++) s = vaddq_s32(s, acc[j]);
  return (double)vaddvq_s32(s);
}
#endif

#if defined(__AMX_BF16__) && defined(__linux__)
#define CPU_MAT_FP_KERNEL 1
static constexpr double MAT_FP_OPS_PER_K = 4.0 * 16 * 16 * 32 * 2;
static thread_local bool g_amxBf16Cfg = false;
static alignas(64) uint16_t g_amxAb[16 * 32];
static alignas(64) uint16_t g_amxBb[16 * 32];
static alignas(64) float    g_amxCf[16 * 16];
static void amxConfigTilesBf16()
{
  struct { uint8_t palette, start_row, reserved[14]; uint16_t colsb[16]; uint8_t rows[16]; } cfg = {};
  cfg.palette = 1;
  for (int t = 0; t < 6; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
  _tile_loadconfig(&cfg);
}
static double runMatFpChain(uint64_t outer)
{
  if (!g_amxBf16Cfg) { amxConfigTilesBf16(); g_amxBf16Cfg = true; }
  _tile_loadd(4, g_amxAb, 64);
  _tile_loadd(5, g_amxBb, 64);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  for (uint64_t o = 0; o < outer; o++)
    for (int k = 0; k < INNER; k++)
    { _tile_dpbf16ps(0, 4, 5); _tile_dpbf16ps(1, 4, 5); _tile_dpbf16ps(2, 4, 5); _tile_dpbf16ps(3, 4, 5); }
  double sink = 0.0;
  _tile_stored(0, g_amxCf, 64); sink += g_amxCf[0];
  _tile_stored(1, g_amxCf, 64); sink += g_amxCf[0];
  _tile_stored(2, g_amxCf, 64); sink += g_amxCf[0];
  _tile_stored(3, g_amxCf, 64); sink += g_amxCf[0];
  return sink;
}
#elif defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
// BFMMLA is part of FEAT_BF16 (no separate I8MM needed).
#define CPU_MAT_FP_KERNEL 1
static constexpr double MAT_FP_OPS_PER_K = 16.0 * 32.0;   // BMM_NACC * 32
static double runMatFpChain(uint64_t outer)
{
  constexpr int NACC = 16;
  float32x4_t acc[NACC];
  const bfloat16x4_t lo = vcvt_bf16_f32(vdupq_n_f32(1.0001f));
  const bfloat16x8_t a  = vcombine_bf16(lo, lo);
  const bfloat16x8_t b  = vcombine_bf16(lo, lo);
  for (int j = 0; j < NACC; j++) acc[j] = vdupq_n_f32(0.0f);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < NACC; j++) acc[j] = vbfmmlaq_f32(acc[j], a, b);
    }
  float32x4_t s = acc[0];
  for (int j = 1; j < NACC; j++) s = vaddq_f32(s, acc[j]);
  return (double)vaddvq_f32(s);
}
#endif

#endif // CLPEAK_CORE_ONLY

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
    t.isaName = CLPEAK_ISA_NAME_STR;
    return t;
  }();
  return &t;
}

} // anonymous namespace
} // namespace clpeak_cpu

#endif // CPU_KERNELS_IMPL_H
