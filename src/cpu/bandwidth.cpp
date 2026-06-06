#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/result_store.h>
#include "cpu_simd.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// Streaming read with explicit SIMD loads and integer XOR sinks.  This keeps
// the loads observable without turning the read-bandwidth test into an FP-add
// throughput test.
//
// NOTE on why the loads aren't optimized away: XOR is self-inverse, so XORing
// the same address twice cancels — a compiler that proved the buffer is
// loop-invariant could in principle strength-reduce the outer `iters` loop to a
// single pass.  It does not here because (a) the working set is always far
// larger than the register file, so a whole pass can't be hoisted, and (b) the
// returned checksum feeds a volatile sink in the caller.  If this is ever
// refactored to a tiny (register-sized) buffer, re-check the generated asm.
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
  alignas(64) uint64_t tmp[8]; _mm512_store_si512((__m512i*)tmp, x);
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
  alignas(32) uint64_t tmp[4]; _mm256_store_si256((__m256i*)tmp, x);
  for (uint64_t v : tmp) tail ^= v;
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
    for (size_t i = 0; i < M; i++)
    {
      uint32_t v;
      std::memcpy(&v, p + i, sizeof(v));
      acc ^= v;
    }
  return acc;
#endif
}

// ---------------------------------------------------------------------------
// Cache bandwidth: read-only streaming.  1T uses one resident working set.
// MT keeps private levels resident per thread, but splits shared levels across
// all threads so the aggregate working set remains inside the target cache.
// ---------------------------------------------------------------------------
int CpuPeak::runCacheBandwidth(benchmark_config_t &cfg)
{
  logger::TestSpec spec{"cache_bandwidth", "Cache bandwidth (read)", "gbps",
                        Category::Bandwidth};
  auto test = currentDeviceScope->beginTest(spec);

  const int maxT = pool->maxThreads();
  const bool appleCpu = info.vendor == "Apple" || info.name.rfind("Apple", 0) == 0;
  const uint64_t cap = 32ull * 1024 * 1024;  // bound the per-thread allocation
  // Per-thread buffer must hold the largest single-thread working set we stream.
  // That is usually the L3 set, but on Apple Silicon the per-cluster L2 (e.g.
  // 12 MB) can exceed the reported/last-level cache, so size to the max of the
  // L2 and L3 sets, capped so the NT allocation stays bounded.
  uint64_t largestLevel = std::max<uint64_t>(info.l2CacheBytes / 2, info.l3CacheBytes / 2);
  uint64_t allocBytes = std::min<uint64_t>(std::max<uint64_t>(largestLevel, 65536), cap);
  size_t allocFloats = (size_t)(allocBytes / sizeof(float));
  if (allocFloats < 1024) allocFloats = 1024;

  std::vector<std::vector<float>> bufs((size_t)maxT);
  for (auto &b : bufs) { b.resize(allocFloats); populate(b.data(), allocFloats); }

  std::vector<uint64_t> sink((size_t)maxT, 0);

  struct Level { const char *name; uint64_t bytes; bool sharedForMt; };
  const Level levels[] = {
    {"L1", std::max<uint64_t>(info.l1dCacheBytes / 2, 4096), false},
    {"L2", std::max<uint64_t>(info.l2CacheBytes  / 2, 16384), appleCpu},
    {"L3", std::min<uint64_t>(std::max<uint64_t>(info.l3CacheBytes / 2, 65536), allocBytes), true},
  };

  unsigned int forced = forceIters ? specifiedIters : 0;

  for (const auto &lvl : levels)
  {
    size_t M1 = (size_t)(lvl.bytes / sizeof(float));
    if (M1 > allocFloats) M1 = allocFloats;
    if (M1 < 64) M1 = 64;

    uint64_t mtBytes = lvl.sharedForMt
      ? std::max<uint64_t>(lvl.bytes / (uint64_t)maxT, 4096)
      : lvl.bytes;
    size_t MN = (size_t)(mtBytes / sizeof(float));
    if (MN > allocFloats) MN = allocFloats;
    if (MN < 64) MN = 64;

    Workload body1 = [&](int tid, uint64_t iters) {
      sink[(size_t)tid] ^= readBufferChecksum(bufs[(size_t)tid].data(), M1, iters);
    };
    Workload bodyN = [&](int tid, uint64_t iters) {
      sink[(size_t)tid] ^= readBufferChecksum(bufs[(size_t)tid].data(), MN, iters);
    };

    double us1 = runWorkload(1,    body1, cfg.targetTimeUs, forced);
    double usN = runWorkload(maxT, bodyN, cfg.targetTimeUs, forced);

    double stPassBytes = (double)M1 * sizeof(float);
    double mtPassBytes = (double)MN * sizeof(float) * (double)maxT;
    auto gbps = [](double bytes, double meanUs) -> float {
      return meanUs > 0.0 ? (float)(bytes / (meanUs * 1e3)) : -1.0f;
    };

    if (us1 > 0) test.emit(std::string(lvl.name) + " ST", gbps(stPassBytes, us1));
    else         test.skip(std::string(lvl.name) + " ST", ResultStatus::Error, "read failed");
    if (usN > 0) test.emit(std::string(lvl.name) + " MT", gbps(mtPassBytes, usN));
    else         test.skip(std::string(lvl.name) + " MT", ResultStatus::Error, "read failed");
  }

  volatile uint64_t keep = 0;
  for (uint64_t s : sink) keep ^= s;
  (void)keep;
  return 0;
}

// ---------------------------------------------------------------------------
// DRAM bandwidth: STREAM-style read / copy / triad over shared arrays far
// larger than the LLC, partitioned across all cores.
// ---------------------------------------------------------------------------
int CpuPeak::runDramBandwidth(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"global_memory_bandwidth", "DRAM bandwidth", "gbps"});

  const int maxT = pool->maxThreads();
  uint64_t arrayBytes = std::max<uint64_t>(info.l3CacheBytes * 4, 64ull * 1024 * 1024);
  arrayBytes = std::min<uint64_t>(arrayBytes, 512ull * 1024 * 1024);
  size_t N = (size_t)(arrayBytes / sizeof(float));
  N = (N / (size_t)maxT) * (size_t)maxT;     // even split
  if (N < (size_t)maxT) N = (size_t)maxT;

  std::vector<float> A(N), B(N), C(N);
  populate(B.data(), N);
  populate(C.data(), N);
  std::vector<uint64_t> sink((size_t)maxT, 0);

  auto chunk = [&](int tid, size_t &lo, size_t &hi) {
    size_t per = N / (size_t)maxT;
    lo = (size_t)tid * per;
    hi = (tid == maxT - 1) ? N : lo + per;
  };

  unsigned int forced = forceIters ? specifiedIters : 0;
  auto gbps = [](double bytes, double meanUs) -> float {
    return meanUs > 0.0 ? (float)(bytes / (meanUs * 1e3)) : -1.0f;
  };

  // read: N*4 bytes
  {
    Workload body = [&](int tid, uint64_t iters) {
      size_t lo, hi; chunk(tid, lo, hi);
      sink[(size_t)tid] ^= readBufferChecksum(A.data() + lo, hi - lo, iters);
    };
    populate(A.data(), N);
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("read", gbps((double)N * sizeof(float), us));
  }
  // copy: A=C -> N*4 read + N*4 write
  {
    Workload body = [&](int tid, uint64_t iters) {
      size_t lo, hi; chunk(tid, lo, hi);
      for (uint64_t it = 0; it < iters; it++)
        std::memcpy(A.data() + lo, C.data() + lo, (hi - lo) * sizeof(float));
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("copy", gbps(2.0 * N * sizeof(float), us));
  }
  // triad: A = B + s*C -> 2 read + 1 write
  {
    const float s = 1.5f;
    Workload body = [&](int tid, uint64_t iters) {
      size_t lo, hi; chunk(tid, lo, hi);
      for (uint64_t it = 0; it < iters; it++)
        for (size_t i = lo; i < hi; i++)
          A[i] = B[i] + s * C[i];
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("triad", gbps(3.0 * N * sizeof(float), us));
  }

  volatile uint64_t keep = 0;
  for (uint64_t v : sink) keep ^= v;
  (void)keep;
  return 0;
}

// ---------------------------------------------------------------------------
// memcpy bandwidth (libc memcpy over a large shared buffer).
// ---------------------------------------------------------------------------
int CpuPeak::runMemcpyBandwidth(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"transfer_bandwidth", "memcpy bandwidth", "gbps"});

  const int maxT = pool->maxThreads();
  uint64_t arrayBytes = std::min<uint64_t>(std::max<uint64_t>(info.l3CacheBytes * 4, 64ull * 1024 * 1024),
                                           256ull * 1024 * 1024);
  size_t N = (size_t)(arrayBytes / sizeof(float));
  N = (N / (size_t)maxT) * (size_t)maxT;
  if (N < (size_t)maxT) N = (size_t)maxT;

  std::vector<float> src(N), dst(N);
  populate(src.data(), N);

  unsigned int forced = forceIters ? specifiedIters : 0;
  Workload body = [&](int tid, uint64_t iters) {
    size_t per = N / (size_t)maxT;
    size_t lo  = (size_t)tid * per;
    size_t hi  = (tid == maxT - 1) ? N : lo + per;
    for (uint64_t it = 0; it < iters; it++)
      std::memcpy(dst.data() + lo, src.data() + lo, (hi - lo) * sizeof(float));
  };
  double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
  double bytes = 2.0 * N * sizeof(float);   // read + write
  test.emit("memcpy", us > 0.0 ? (float)(bytes / (us * 1e3)) : -1.0f);
  return 0;
}

#endif // ENABLE_CPU
