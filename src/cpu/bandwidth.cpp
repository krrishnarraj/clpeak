#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/result_store.h>
#include "cpu_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// The streaming-read kernel is ISA-dispatched (compiled per-ISA in
// cpu_kernels_tu.cpp).  Forward to the selected variant.
static inline uint64_t readBufferChecksum(const float *p, size_t M, uint64_t iters)
{
  return clpeak_cpu::kernels().readsum(p, M, iters);
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

// Number of floats per STREAM array.  Must exceed the *aggregate* LLC (on
// multi-CCX/CCD AMD the per-instance L3 is only a slice — sizing off it would
// let the whole array sit in cache and report cache, not DRAM, bandwidth), and
// is bounded so we don't hog memory.  Even split across threads.
static size_t pickStreamFloats(const cpu_device_info_t &info, int maxT)
{
  uint64_t llc = std::max(info.l3TotalBytes, info.l3CacheBytes);
  uint64_t arrayBytes = std::max<uint64_t>(llc * 4, 64ull << 20);
  uint64_t cap = info.totalMemBytes ? std::min<uint64_t>(512ull << 20, info.totalMemBytes / 16)
                                    : (512ull << 20);
  if (cap < llc * 2) cap = llc * 2;            // always large enough to miss the LLC
  arrayBytes = std::min(arrayBytes, cap);
  size_t N = (size_t)(arrayBytes / sizeof(float));
  N = (N / (size_t)maxT) * (size_t)maxT;
  if (N < (size_t)maxT) N = (size_t)maxT;
  return N;
}

// ---------------------------------------------------------------------------
// DRAM bandwidth: STREAM-style read / copy / triad over shared arrays far
// larger than the LLC, partitioned across all cores.  Arrays are allocated
// untouched and first-touched in parallel so their pages land on the NUMA node
// of the thread that will use them (single-threaded init would place every page
// on one node and cripple bandwidth on multi-socket / multi-CCD systems).
// ---------------------------------------------------------------------------
int CpuPeak::runDramBandwidth(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"global_memory_bandwidth", "DRAM bandwidth", "gbps"});

  const int maxT = pool->maxThreads();
  const size_t N = pickStreamFloats(info, maxT);

  auto chunk = [&](int tid, size_t &lo, size_t &hi) {
    size_t per = N / (size_t)maxT;
    lo = (size_t)tid * per;
    hi = (tid == maxT - 1) ? N : lo + per;
  };

  // `new float[N]` leaves the pages untouched (floats are not value-initialized),
  // so the parallel populate below is the first touch.
  float *A = new float[N];
  float *B = new float[N];
  float *C = new float[N];
  pool->run(maxT, [&](int tid) {
    size_t lo, hi; chunk(tid, lo, hi);
    populate(A + lo, hi - lo);
    populate(B + lo, hi - lo);
    populate(C + lo, hi - lo);
  });

  std::vector<uint64_t> sink((size_t)maxT, 0);
  unsigned int forced = forceIters ? specifiedIters : 0;
  auto gbps = [](double bytes, double meanUs) -> float {
    return meanUs > 0.0 ? (float)(bytes / (meanUs * 1e3)) : -1.0f;
  };

  {
    Workload body = [&](int tid, uint64_t iters) {
      size_t lo, hi; chunk(tid, lo, hi);
      sink[(size_t)tid] ^= readBufferChecksum(A + lo, hi - lo, iters);
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("read", gbps((double)N * sizeof(float), us));
  }
  {
    Workload body = [&](int tid, uint64_t iters) {
      size_t lo, hi; chunk(tid, lo, hi);
      for (uint64_t it = 0; it < iters; it++)
        std::memcpy(A + lo, C + lo, (hi - lo) * sizeof(float));
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("copy", gbps(2.0 * N * sizeof(float), us));
  }
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
  delete[] A; delete[] B; delete[] C;
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
  const size_t N = pickStreamFloats(info, maxT);

  auto chunk = [&](int tid, size_t &lo, size_t &hi) {
    size_t per = N / (size_t)maxT;
    lo = (size_t)tid * per;
    hi = (tid == maxT - 1) ? N : lo + per;
  };

  float *src = new float[N];
  float *dst = new float[N];
  pool->run(maxT, [&](int tid) {
    size_t lo, hi; chunk(tid, lo, hi);
    populate(src + lo, hi - lo);
    populate(dst + lo, hi - lo);
  });

  unsigned int forced = forceIters ? specifiedIters : 0;
  Workload body = [&](int tid, uint64_t iters) {
    size_t lo, hi; chunk(tid, lo, hi);
    for (uint64_t it = 0; it < iters; it++)
      std::memcpy(dst + lo, src + lo, (hi - lo) * sizeof(float));
  };
  double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
  double bytes = 2.0 * N * sizeof(float);
  test.emit("memcpy", us > 0.0 ? (float)(bytes / (us * 1e3)) : -1.0f);
  delete[] src; delete[] dst;
  return 0;
}

#endif // ENABLE_CPU
