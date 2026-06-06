#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_simd.h"
#include "compute_common.h"

using namespace cpu_simd;

// int32 multiply-accumulate chain (acc = acc*b + c).  Counts the multiply and
// the add as 2 ops; integer wrap is modular at the intrinsic level (no UB).
static double runInt32Chain(uint64_t outer)
{
  i32v acc[I32_NACC];
  const i32v b = i32_set(1664525);     // LCG-style constants keep the chain live
  const i32v c = i32_set(1013904223);
  for (int j = 0; j < I32_NACC; j++) acc[j] = i32_set(j + 1);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I32_NACC; j++)
        acc[j] = i32_madd(acc[j], b, c);
    }
  i32v s = acc[0];
  for (int j = 1; j < I32_NACC; j++) s = i32_add(s, acc[j]);
  return (double)i32_hsum(s);
}

int CpuPeak::runComputeInt32(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"integer_compute", "Integer compute", "gops"});
  double ops = (double)INNER * I32_NACC * I32_LANES * 2.0;
  emitCompute(*this, test, "int", ops, runInt32Chain, cfg);
  return 0;
}

// ---------------------------------------------------------------------------
// int8 dot product (int8xint8 -> int32 accumulate): the CPU DP4a analog.
// AVX512-VNNI dpbusd (16 lanes x4) / AVX-VNNI (8 lanes x4) / ARM dotprod (4x4).
// ---------------------------------------------------------------------------
#if defined(__AVX512VNNI__)
#define CPU_HAS_INT8DP_KERNEL 1
static constexpr int I8_NACC = 16, I8_OPS_PER_INSTR = 128;  // 64 mul + 64 add
static double runInt8DpChain(uint64_t outer)
{
  __m512i acc[I8_NACC];
  const __m512i a = _mm512_set1_epi8((char)3);   // treated as u8 by dpbusd
  const __m512i b = _mm512_set1_epi8((char)5);   // treated as s8
  for (int j = 0; j < I8_NACC; j++) acc[j] = _mm512_setzero_si512();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < I8_NACC; j++)
        acc[j] = _mm512_dpbusd_epi32(acc[j], a, b);
    }
  __m512i s = acc[0];
  for (int j = 1; j < I8_NACC; j++) s = _mm512_add_epi32(s, acc[j]);
  return (double)_mm512_reduce_add_epi32(s);
}
#elif defined(__AVXVNNI__)
#define CPU_HAS_INT8DP_KERNEL 1
static constexpr int I8_NACC = 12, I8_OPS_PER_INSTR = 64;   // 32 mul + 32 add
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
      for (int j = 0; j < I8_NACC; j++)
        acc[j] = _mm256_dpbusd_epi32(acc[j], a, b);
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
static constexpr int I8_NACC = 16, I8_OPS_PER_INSTR = 32;   // 16 mul + 16 add
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
      for (int j = 0; j < I8_NACC; j++)
        acc[j] = vdotq_s32(acc[j], a, b);
    }
  int32x4_t s = acc[0];
  for (int j = 1; j < I8_NACC; j++) s = vaddq_s32(s, acc[j]);
  return (double)vaddvq_s32(s);
}
#endif

// ---------------------------------------------------------------------------
// Atomic throughput: std::atomic fetch_add under three contention regimes.
//   uncontended : one thread on its own counter (peak single-core atomic rate)
//   contended   : all cores hammering one shared cache line (coherence-bound)
//   sharded     : all cores each on a private padded counter (scales)
// ---------------------------------------------------------------------------
namespace {
struct alignas(64) PaddedAtomic {
  std::atomic<uint64_t> v{0};
  char pad[64 - sizeof(std::atomic<uint64_t>)];
};
constexpr int AT_INNER = 4096;
}

int CpuPeak::runAtomicThroughput(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"atomic_throughput", "Atomic throughput (fetch_add)", "gops"});

  const int maxT = pool->maxThreads();
  PaddedAtomic *shared  = new PaddedAtomic[1];
  PaddedAtomic *sharded = new PaddedAtomic[(size_t)maxT];

  unsigned int forced = forceIters ? specifiedIters : 0;
  auto gops = [](double ops, int n, double meanUs) -> float {
    return meanUs > 0.0 ? (float)(ops * (double)n / (meanUs * 1e3)) : -1.0f;
  };
  const double opsPerIter = (double)AT_INNER;

  // uncontended (1 thread)
  {
    Workload body = [&](int, uint64_t iters) {
      for (uint64_t o = 0; o < iters; o++)
        for (int k = 0; k < AT_INNER; k++)
          shared[0].v.fetch_add(1, std::memory_order_relaxed);
    };
    double us = runWorkload(1, body, cfg.targetTimeUs, forced);
    test.emit("uncontended", gops(opsPerIter, 1, us));
  }
  // contended (all threads, one line)
  {
    Workload body = [&](int, uint64_t iters) {
      for (uint64_t o = 0; o < iters; o++)
        for (int k = 0; k < AT_INNER; k++)
          shared[0].v.fetch_add(1, std::memory_order_relaxed);
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("contended MT", gops(opsPerIter, maxT, us));
  }
  // sharded (all threads, private lines)
  {
    Workload body = [&](int tid, uint64_t iters) {
      for (uint64_t o = 0; o < iters; o++)
        for (int k = 0; k < AT_INNER; k++)
          sharded[(size_t)tid].v.fetch_add(1, std::memory_order_relaxed);
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("sharded MT", gops(opsPerIter, maxT, us));
  }

  delete[] shared;
  delete[] sharded;
  return 0;
}

int CpuPeak::runComputeInt8DP(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"int8_dot_product_compute", "INT8 dot-product compute", "gops"});
#ifdef CPU_HAS_INT8DP_KERNEL
  if (!info.hasInt8DP)
  {
    test.skip("int8_dp", ResultStatus::Unsupported, "no int8 dot instruction on this CPU");
    return 0;
  }
  double ops = (double)INNER * I8_NACC * I8_OPS_PER_INSTR;
  emitCompute(*this, test, "int8_dp", ops, runInt8DpChain, cfg);
#else
  test.skip("int8_dp", ResultStatus::Unsupported,
            "int8 dot not compiled for this target ISA");
#endif
  return 0;
}

#endif // ENABLE_CPU
