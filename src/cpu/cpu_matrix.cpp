#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_simd.h"
#include "compute_common.h"

// ===========================================================================
// CPU matrix engine (the tensor-core analog).
//   x86 : Intel AMX tile matmul (int8 + bf16) on Sapphire Rapids and later.
//   ARM : SMMLA (FEAT_I8MM, int8) and BFMMLA (FEAT_BF16, bf16) matrix multiply.
// Reported under Benchmark::Amx, run in both the fp (bf16) and int (int8)
// phases, mirroring how the GPU tensor tests (wmma / joint_matrix) split.
// ===========================================================================

#if defined(__AMX_INT8__) && defined(__linux__)
#define CPU_AMX_INT8 1
#endif
#if defined(__AMX_BF16__) && defined(__linux__)
#define CPU_AMX_BF16 1
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
#define CPU_I8MM 1
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) && (defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_BF16))
// bfmmla is part of the BF16 extension; gate on the vector-arith macro plus the
// presence of the matmul feature family.
#define CPU_BFMMLA 1
#endif

#if defined(CPU_AMX_INT8) || defined(CPU_AMX_BF16)
#include <immintrin.h>
#include <unistd.h>
#include <sys/syscall.h>
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM 0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA 18
#endif

struct amx_tilecfg {
  uint8_t  palette;
  uint8_t  start_row;
  uint8_t  reserved[14];
  uint16_t colsb[16];
  uint8_t  rows[16];
};

static bool amxRequestPerm()
{
  // Process-wide; safe to call repeatedly.
  long r = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  return r == 0;
}

static void amxConfigTiles()
{
  amx_tilecfg cfg = {};
  cfg.palette = 1;
  // 6 tiles: tmm0..3 accumulators (16 rows x 64 bytes = 16x16 int32),
  // tmm4 = A (16x64 bytes int8), tmm5 = B (16x64 bytes int8, VNNI-packed).
  for (int t = 0; t < 6; t++) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
  _tile_loadconfig(&cfg);
}
#endif

// --------------------------------------------------------------------------
#if defined(CPU_AMX_INT8)
// 4 accumulator tiles, each _tile_dpbssd = 16*16*64 MAC = 32768 ops*2.
static constexpr double AMX_I8_OPS_PER_ITER = 4.0 * 16 * 16 * 64 * 2;
static thread_local bool g_amxI8Cfg = false;
static alignas(64) int8_t  g_amxA[16 * 64];
static alignas(64) int8_t  g_amxB[16 * 64];
static alignas(64) int32_t g_amxC[16 * 16];
static double runAmxInt8Chain(uint64_t outer)
{
  if (!g_amxI8Cfg) { amxConfigTiles(); g_amxI8Cfg = true; }
  _tile_loadd(4, g_amxA, 64);
  _tile_loadd(5, g_amxB, 64);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  for (uint64_t o = 0; o < outer; o++)
    for (int k = 0; k < cpu_simd::INNER; k++)
    {
      _tile_dpbssd(0, 4, 5);
      _tile_dpbssd(1, 4, 5);
      _tile_dpbssd(2, 4, 5);
      _tile_dpbssd(3, 4, 5);
    }
  _tile_stored(0, g_amxC, 64);
  return (double)g_amxC[0];
}
#endif

#if defined(CPU_AMX_BF16)
static constexpr double AMX_BF16_OPS_PER_ITER = 4.0 * 16 * 16 * 32 * 2;
static thread_local bool g_amxBf16Cfg = false;
static alignas(64) uint16_t g_amxAb[16 * 32];
static alignas(64) uint16_t g_amxBb[16 * 32];
static alignas(64) float    g_amxCf[16 * 16];
static double runAmxBf16Chain(uint64_t outer)
{
  if (!g_amxBf16Cfg) { amxConfigTiles(); g_amxBf16Cfg = true; }
  _tile_loadd(4, g_amxAb, 64);
  _tile_loadd(5, g_amxBb, 64);
  _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
  for (uint64_t o = 0; o < outer; o++)
    for (int k = 0; k < cpu_simd::INNER; k++)
    {
      _tile_dpbf16ps(0, 4, 5);
      _tile_dpbf16ps(1, 4, 5);
      _tile_dpbf16ps(2, 4, 5);
      _tile_dpbf16ps(3, 4, 5);
    }
  _tile_stored(0, g_amxCf, 64);
  return (double)g_amxCf[0];
}
#endif

// --------------------------------------------------------------------------
#if defined(CPU_I8MM)
// SMMLA: 2x2 int32 += 2x8 int8 * 8x2 int8 = 32 MAC*2 = 64 ops / instr.
static constexpr int   MM_NACC = 16;
static constexpr double I8MM_OPS_PER_ITER = (double)MM_NACC * 64.0;
static double runI8mmChain(uint64_t outer)
{
  int32x4_t acc[MM_NACC];
  const int8x16_t a = vdupq_n_s8(3);
  const int8x16_t b = vdupq_n_s8(5);
  for (int j = 0; j < MM_NACC; j++) acc[j] = vdupq_n_s32(0);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < cpu_simd::INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < MM_NACC; j++)
        acc[j] = vmmlaq_s32(acc[j], a, b);
    }
  return (double)vaddvq_s32(acc[0]);
}
#endif

#if defined(CPU_BFMMLA)
// BFMMLA: 2x2 fp32 += 2x4 bf16 * 4x2 bf16 = 16 MAC*2 = 32 ops / instr.
static constexpr int   BMM_NACC = 16;
static constexpr double BFMMLA_OPS_PER_ITER = (double)BMM_NACC * 32.0;
static double runBfmmlaChain(uint64_t outer)
{
  float32x4_t acc[BMM_NACC];
  const bfloat16x4_t lo = vcvt_bf16_f32(vdupq_n_f32(1.0001f));
  const bfloat16x8_t a  = vcombine_bf16(lo, lo);
  const bfloat16x8_t b  = vcombine_bf16(lo, lo);
  for (int j = 0; j < BMM_NACC; j++) acc[j] = vdupq_n_f32(0.0f);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < cpu_simd::INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < BMM_NACC; j++)
        acc[j] = vbfmmlaq_f32(acc[j], a, b);
    }
  return (double)vaddvq_f32(acc[0]);
}
#endif

int CpuPeak::runCpuMatrix(benchmark_config_t &cfg, Category category)
{
  if (category == Category::FpCompute)
  {
    auto test = currentDeviceScope->beginTest(
      {"cpu_matrix_fp", "CPU matrix engine (bf16)", "gflops"});
#if defined(CPU_AMX_BF16)
    if (info.hasAMX && amxRequestPerm())
      emitCompute(*this, test, "amx_bf16", AMX_BF16_OPS_PER_ITER, runAmxBf16Chain, cfg);
    else
      test.skip("amx_bf16", ResultStatus::Unsupported, "AMX tile permission unavailable");
#elif defined(CPU_BFMMLA)
    emitCompute(*this, test, "bfmmla_bf16", BFMMLA_OPS_PER_ITER, runBfmmlaChain, cfg);
#else
    test.skip("matrix_bf16", ResultStatus::Unsupported,
              "no CPU bf16 matrix engine (AMX / BFMMLA) on this target");
#endif
    return 0;
  }

  // IntCompute
  auto test = currentDeviceScope->beginTest(
    {"cpu_matrix_int", "CPU matrix engine (int8)", "gops"});
#if defined(CPU_AMX_INT8)
  if (info.hasAMX && amxRequestPerm())
    emitCompute(*this, test, "amx_int8", AMX_I8_OPS_PER_ITER, runAmxInt8Chain, cfg);
  else
    test.skip("amx_int8", ResultStatus::Unsupported, "AMX tile permission unavailable");
#elif defined(CPU_I8MM)
  emitCompute(*this, test, "i8mm_int8", I8MM_OPS_PER_ITER, runI8mmChain, cfg);
#else
  test.skip("matrix_int8", ResultStatus::Unsupported,
            "no CPU int8 matrix engine (AMX / I8MM) on this target");
#endif
  return 0;
}

#endif // ENABLE_CPU
