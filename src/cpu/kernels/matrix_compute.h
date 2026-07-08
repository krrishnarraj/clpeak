#ifndef CPU_KERN_MATRIX_COMPUTE_H
#define CPU_KERN_MATRIX_COMPUTE_H

#ifdef ENABLE_CPU

// ===========================================================================
// CPU matrix-engine kernels (the tensor-core analog): Intel AMX tile ops on x86
// (int8 / bf16 / fp16 / tf32 / fp8) and the ARM NEON matrix instructions SMMLA
// (int8) + BFMMLA (bf16).  ops are PER-k; the table builder multiplies by INNER.
// Feature-gated + CLPEAK_CORE_ONLY-excluded like the other advanced kernels.  To
// add a new AMX dtype: reuse amxConfig16x64(), add a #if block + a
// CPU_MAT_<X>_KERNEL define, and wire the slot in the table builder.
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#if (defined(__AMX_INT8__) || defined(__AMX_BF16__)) && (defined(__x86_64__) || defined(_M_X64))
#include <immintrin.h>
#endif

#ifndef CLPEAK_CORE_ONLY

namespace clpeak_cpu {
namespace {

// ---- Matrix engine: int8 (AMX / SMMLA) and bf16 (AMX / BFMMLA) -------------
#if defined(__AMX_INT8__) && (defined(__x86_64__) || defined(_M_X64))
#define CPU_MAT_INT8_KERNEL 1
static constexpr double MAT_I8_OPS_PER_K = 4.0 * 16 * 16 * 64 * 2;
static thread_local bool g_amxI8Cfg = false;
alignas(64) static int8_t  g_amxA[16 * 64];   // inputs: read-only, shared is fine
alignas(64) static int8_t  g_amxB[16 * 64];
// Output is written by every MT worker; thread_local avoids a data race and the
// cross-thread cache-line invalidation that shared storage would cause.
alignas(64) static thread_local int32_t g_amxC[16 * 16];
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

#if defined(__AMX_BF16__) && (defined(__x86_64__) || defined(_M_X64))
#define CPU_MAT_FP_KERNEL 1
static constexpr double MAT_FP_OPS_PER_K = 4.0 * 16 * 16 * 32 * 2;
static thread_local bool g_amxBf16Cfg = false;
alignas(64) static uint16_t g_amxAb[16 * 32];   // inputs: read-only, shared is fine
alignas(64) static uint16_t g_amxBb[16 * 32];
// Output is written by every MT worker; thread_local avoids a data race (see int8).
alignas(64) static thread_local float g_amxCf[16 * 16];
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

// ---- Newer x86 AMX matrix dtypes: fp16 / tf32 / fp8 ------------------------
// Same 4-accumulator-tile config as the int8/bf16 AMX kernels (opaque tiles, so
// the accumulate loop can't be scalar-evolved; inputs zero => data-independent
// throughput; output tiles thread_local so every MT worker gets its own).  The
// TILECFG is the canonical palette-1 / 16-row / 64-colsb layout for all of them;
// only the element width (hence K per tile) and the DP intrinsic differ.
#if (defined(__AMX_FP16__) || defined(__AMX_TF32__) || defined(__AMX_FP8__)) && \
    (defined(__x86_64__) || defined(_M_X64))
static void amxConfig16x64()
{
  struct { uint8_t palette, start_row, reserved[14]; uint16_t colsb[16]; uint8_t rows[16]; } cfg = {};
  cfg.palette = 1;
  for (int t = 0; t < 6; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
  _tile_loadconfig(&cfg);
}
#endif

#if defined(__AMX_FP16__) && (defined(__x86_64__) || defined(_M_X64))
#define CPU_MAT_FP16_KERNEL 1
static constexpr double MAT_FP16_OPS_PER_K = 4.0 * 16 * 16 * 32 * 2;  // K=32 fp16/row
static thread_local bool g_amxFp16Cfg = false;
alignas(64) static uint16_t g_amxAf16[16 * 32];
alignas(64) static uint16_t g_amxBf16m[16 * 32];
alignas(64) static thread_local float g_amxCf16[16 * 16];
static double runMatFp16Chain(uint64_t outer)
{
  if (!g_amxFp16Cfg) { amxConfig16x64(); g_amxFp16Cfg = true; }
  _tile_loadd(4, g_amxAf16, 64);
  _tile_loadd(5, g_amxBf16m, 64);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  for (uint64_t o = 0; o < outer; o++)
    for (int k = 0; k < INNER; k++)
    { _tile_dpfp16ps(0, 4, 5); _tile_dpfp16ps(1, 4, 5); _tile_dpfp16ps(2, 4, 5); _tile_dpfp16ps(3, 4, 5); }
  double sink = 0.0;
  _tile_stored(0, g_amxCf16, 64); sink += g_amxCf16[0];
  _tile_stored(1, g_amxCf16, 64); sink += g_amxCf16[0];
  _tile_stored(2, g_amxCf16, 64); sink += g_amxCf16[0];
  _tile_stored(3, g_amxCf16, 64); sink += g_amxCf16[0];
  return sink;
}
#endif

#if defined(__AMX_TF32__) && (defined(__x86_64__) || defined(_M_X64))
#define CPU_MAT_TF32_KERNEL 1
static constexpr double MAT_TF32_OPS_PER_K = 4.0 * 16 * 16 * 16 * 2;  // K=16 tf32(fp32-stored)/row
static thread_local bool g_amxTf32Cfg = false;
alignas(64) static float g_amxAt32[16 * 16];
alignas(64) static float g_amxBt32[16 * 16];
alignas(64) static thread_local float g_amxCt32[16 * 16];
static double runMatTf32Chain(uint64_t outer)
{
  if (!g_amxTf32Cfg) { amxConfig16x64(); g_amxTf32Cfg = true; }
  _tile_loadd(4, g_amxAt32, 64);
  _tile_loadd(5, g_amxBt32, 64);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  for (uint64_t o = 0; o < outer; o++)
    for (int k = 0; k < INNER; k++)
    { _tile_mmultf32ps(0, 4, 5); _tile_mmultf32ps(1, 4, 5); _tile_mmultf32ps(2, 4, 5); _tile_mmultf32ps(3, 4, 5); }
  double sink = 0.0;
  _tile_stored(0, g_amxCt32, 64); sink += g_amxCt32[0];
  _tile_stored(1, g_amxCt32, 64); sink += g_amxCt32[0];
  _tile_stored(2, g_amxCt32, 64); sink += g_amxCt32[0];
  _tile_stored(3, g_amxCt32, 64); sink += g_amxCt32[0];
  return sink;
}
#endif

#if defined(__AMX_FP8__) && (defined(__x86_64__) || defined(_M_X64))
#define CPU_MAT_FP8_KERNEL 1
static constexpr double MAT_FP8_OPS_PER_K = 4.0 * 16 * 16 * 64 * 2;  // K=64 fp8/row
static thread_local bool g_amxFp8Cfg = false;
alignas(64) static uint8_t g_amxAf8[16 * 64];
alignas(64) static uint8_t g_amxBf8[16 * 64];
alignas(64) static thread_local float g_amxCf8[16 * 16];
static double runMatFp8Chain(uint64_t outer)
{
  if (!g_amxFp8Cfg) { amxConfig16x64(); g_amxFp8Cfg = true; }
  _tile_loadd(4, g_amxAf8, 64);
  _tile_loadd(5, g_amxBf8, 64);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  // hf8 = e4m3 (the common ML fp8); throughput is format-independent.
  for (uint64_t o = 0; o < outer; o++)
    for (int k = 0; k < INNER; k++)
    { _tile_dphf8ps(0, 4, 5); _tile_dphf8ps(1, 4, 5); _tile_dphf8ps(2, 4, 5); _tile_dphf8ps(3, 4, 5); }
  double sink = 0.0;
  _tile_stored(0, g_amxCf8, 64); sink += g_amxCf8[0];
  _tile_stored(1, g_amxCf8, 64); sink += g_amxCf8[0];
  _tile_stored(2, g_amxCf8, 64); sink += g_amxCf8[0];
  _tile_stored(3, g_amxCf8, 64); sink += g_amxCf8[0];
  return sink;
}
#endif

} // anonymous namespace
} // namespace clpeak_cpu

#endif // CLPEAK_CORE_ONLY

#endif // ENABLE_CPU
#endif // CPU_KERN_MATRIX_COMPUTE_H
