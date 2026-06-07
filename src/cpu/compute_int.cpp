#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

#include <atomic>
#include <string>

using clpeak_cpu::kernels;

int CpuPeak::runComputeInt32(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest({"integer_compute", "Integer compute", "gops"});
  const auto &v = kernels().int32;
  if (v.fn) emitCompute(*this, test, "int", v.opsPerIter, v.fn, cfg);
  else      test.skip("int", ResultStatus::Unsupported, "no SIMD int32 path for this CPU");
  return 0;
}

int CpuPeak::runComputeInt8DP(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"int8_dot_product_compute", "INT8 dot-product compute", "gops"});
  const auto &v = kernels().int8dp;
  if (v.fn) emitCompute(*this, test, "int8_dp", v.opsPerIter, v.fn, cfg);
  else      test.skip("int8_dp", ResultStatus::Unsupported, "no int8 dot instruction on this CPU");
  return 0;
}

// ---------------------------------------------------------------------------
// Atomic throughput: std::atomic fetch_add under three contention regimes.
// (ISA-neutral, so not part of the dispatched kernel set.)
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

  {
    Workload body = [&](int, uint64_t iters) {
      for (uint64_t o = 0; o < iters; o++)
        for (int k = 0; k < AT_INNER; k++)
          shared[0].v.fetch_add(1, std::memory_order_relaxed);
    };
    double us = runWorkload(1, body, cfg.targetTimeUs, forced);
    test.emit("uncontended", gops(opsPerIter, 1, us));
  }
  {
    Workload body = [&](int, uint64_t iters) {
      for (uint64_t o = 0; o < iters; o++)
        for (int k = 0; k < AT_INNER; k++)
          shared[0].v.fetch_add(1, std::memory_order_relaxed);
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("contended MT", gops(opsPerIter, maxT, us));
  }
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

#endif // ENABLE_CPU
