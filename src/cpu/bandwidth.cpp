#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/result_store.h>
#include "cpu_simd.h"

#include <algorithm>
#include <cstring>
#include <vector>

// Streaming read with 8 explicit SIMD vector accumulators so the loop is
// load-issue / bandwidth bound (not scalar compute bound).  Accumulators carry
// across the iters loop; the returned sum is only a sink to keep the loads live.
static double sumBuffer(const float *p, size_t M, uint64_t iters)
{
  using namespace cpu_simd;
  const size_t W = (size_t)F32_LANES;
  const size_t step = 8 * W;
  f32v s0 = f32_set(0), s1 = f32_set(0), s2 = f32_set(0), s3 = f32_set(0);
  f32v s4 = f32_set(0), s5 = f32_set(0), s6 = f32_set(0), s7 = f32_set(0);
  float tail = 0.0f;

  for (uint64_t it = 0; it < iters; it++)
  {
    size_t i = 0;
    for (; i + step <= M; i += step)
    {
      s0 = f32_add(s0, f32_load(p + i + 0 * W));
      s1 = f32_add(s1, f32_load(p + i + 1 * W));
      s2 = f32_add(s2, f32_load(p + i + 2 * W));
      s3 = f32_add(s3, f32_load(p + i + 3 * W));
      s4 = f32_add(s4, f32_load(p + i + 4 * W));
      s5 = f32_add(s5, f32_load(p + i + 5 * W));
      s6 = f32_add(s6, f32_load(p + i + 6 * W));
      s7 = f32_add(s7, f32_load(p + i + 7 * W));
    }
    for (; i < M; i++) tail += p[i];
  }
  f32v s = f32_add(f32_add(f32_add(s0, s1), f32_add(s2, s3)),
                   f32_add(f32_add(s4, s5), f32_add(s6, s7)));
  return (double)f32_hsum(s) + (double)tail;
}

// ---------------------------------------------------------------------------
// Cache bandwidth: per-core private buffers sized to each level, read-only
// streaming.  1T = single-core peak; NT = aggregate across all cores (each
// core streaming its own resident buffer).  Buffers are capped at 8 MB so the
// NT allocation stays bounded and the L3 working set stays cache-resident.
// ---------------------------------------------------------------------------
int CpuPeak::runCacheBandwidth(benchmark_config_t &cfg)
{
  logger::TestSpec spec{"cache_bandwidth", "Cache bandwidth (read)", "gbps",
                        Category::Bandwidth};
  auto test = currentDeviceScope->beginTest(spec);

  const int maxT = pool->maxThreads();
  const uint64_t cap = 8ull * 1024 * 1024;   // bound the per-thread allocation
  // Per-thread buffer must hold the largest level we stream (the L3 working
  // set), capped so the NT allocation and the L3 set stay cache-resident.
  uint64_t allocBytes = std::min<uint64_t>(std::max<uint64_t>(info.l3CacheBytes / 2, 65536), cap);
  size_t allocFloats = (size_t)(allocBytes / sizeof(float));
  if (allocFloats < 1024) allocFloats = 1024;

  std::vector<std::vector<float>> bufs((size_t)maxT);
  for (auto &b : bufs) { b.resize(allocFloats); populate(b.data(), allocFloats); }

  std::vector<double> sink((size_t)maxT, 0.0);

  struct Level { const char *name; uint64_t bytes; };
  const Level levels[] = {
    {"L1", std::max<uint64_t>(info.l1dCacheBytes / 2, 4096)},
    {"L2", std::max<uint64_t>(info.l2CacheBytes  / 2, 16384)},
    {"L3", std::min<uint64_t>(std::max<uint64_t>(info.l3CacheBytes / 2, 65536), allocBytes)},
  };

  unsigned int forced = forceIters ? specifiedIters : 0;

  for (const auto &lvl : levels)
  {
    size_t M = (size_t)(lvl.bytes / sizeof(float));
    if (M > allocFloats) M = allocFloats;
    if (M < 64) M = 64;

    Workload body = [&](int tid, uint64_t iters) {
      sink[(size_t)tid] += sumBuffer(bufs[(size_t)tid].data(), M, iters);
    };

    double us1 = runWorkload(1,    body, cfg.targetTimeUs, forced);
    double usN = runWorkload(maxT, body, cfg.targetTimeUs, forced);

    double perPassBytes = (double)M * sizeof(float);
    auto gbps = [](double bytes, double meanUs) -> float {
      return meanUs > 0.0 ? (float)(bytes / (meanUs * 1e3)) : -1.0f;
    };

    if (us1 > 0) test.emit(std::string(lvl.name) + " ST", gbps(perPassBytes, us1));
    else         test.skip(std::string(lvl.name) + " ST", ResultStatus::Error, "read failed");
    if (usN > 0) test.emit(std::string(lvl.name) + " MT", gbps(perPassBytes * maxT, usN));
    else         test.skip(std::string(lvl.name) + " MT", ResultStatus::Error, "read failed");
  }

  volatile double keep = 0.0;
  for (double s : sink) keep += s;
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
  std::vector<double> sink((size_t)maxT, 0.0);

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
      sink[(size_t)tid] += sumBuffer(A.data() + lo, hi - lo, iters);
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

  volatile double keep = 0.0;
  for (double v : sink) keep += v;
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
