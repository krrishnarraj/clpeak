#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/result_store.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

// Microarchitecture probes (Category::Latency): atomic-RMW cost and branch-
// mispredict penalty.  Both report nanoseconds -- these are per-operation
// costs, not throughput peaks, and they are the two numbers synchronization-
// heavy (databases, browsers) and branchy (interpreters, parsers) code lives
// and dies by.

// ---------------------------------------------------------------------------
// Atomic fetch-add: uncontended (1 thread, line stays in its L1, measures the
// core's atomic-RMW cost) and contended (every core hammering ONE cache line,
// measures how gracefully the coherence fabric serializes ownership).  The
// contended row is system-wide: wall time per completed op across all cores.
// Relaxed ordering on purpose -- we measure the RMW itself, not fence cost.
// ---------------------------------------------------------------------------
int CpuPeak::runAtomics(benchmark_config_t &cfg)
{
  logger::TestSpec spec{"atomics", "Atomic fetch-add latency", "ns",
                        Category::Latency};
  auto test = currentDeviceScope->beginTest(spec);

  struct alignas(64) PaddedAtomic { std::atomic<uint64_t> v{0}; };
  static PaddedAtomic ctr;

  Workload body = [](int, uint64_t iters) {
    for (uint64_t i = 0; i < iters; i++)
      ctr.v.fetch_add(1, std::memory_order_relaxed);
  };

  const int maxT = pool->maxThreads();
  unsigned int forced = forceIters ? specifiedIters : 0;
  double us1 = runWorkload(1,    body, cfg.targetTimeUs, forced);
  double usN = runWorkload(maxT, body, cfg.targetTimeUs, forced);

  if (us1 > 0) test.emit("uncontended ST", (float)(us1 * 1e3));
  else         test.skip("uncontended ST", ResultStatus::Error, "workload failed");
  // Each of the maxT threads completes one op per mean iteration, so the
  // system-wide time between completions is wall / maxT.
  if (usN > 0) test.emit("contended MT", (float)(usN * 1e3 / (double)maxT));
  else         test.skip("contended MT", ResultStatus::Error, "workload failed");
  return 0;
}

// ---------------------------------------------------------------------------
// Branch mispredict penalty: the classic sorted-vs-shuffled data-dependent
// branch.  Same data, same instruction stream -- sorted input predicts
// perfectly, shuffled input is a coin flip the predictor cannot learn (the
// pattern period, N elements, is far beyond any TAGE history), so the time
// delta divided by the mispredict count is the pipeline-refill cost.  That
// cost is the single biggest determinant of interpreter/parser speed.
//
// The taken arm writes through a volatile pointer: a conditional volatile
// store cannot be speculated or if-converted, so the compiler MUST emit a
// real conditional branch (clang/GCC turn a plain two-sided += into
// csel/cmov, which has no misprediction to measure).  The store executes for
// the same elements in both runs (same data), so its cost cancels in the
// delta.  Verify with objdump: the loop body must contain a conditional
// branch, not csel/cmov.
// ---------------------------------------------------------------------------
namespace {

static double branchPassNs(const uint8_t *v, size_t n, uint64_t passes)
{
  using clock = std::chrono::steady_clock;
  uint64_t s1 = 0, s2 = 0;
  volatile uint64_t sink = 0;
  auto t0 = clock::now();
  for (uint64_t p = 0; p < passes; p++)
    for (size_t i = 0; i < n; i++)
    {
      if (v[i] < 128) { s1 += v[i]; sink = s1; }
      else            { s2 += v[i]; }
    }
  double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                  clock::now() - t0).count();
  sink = s1 + s2;
  return ns / (double)(passes * n);
}

static void branchPenalty(double &predictedNs, double &randomNs, double &penaltyNs)
{
  const size_t n = 1u << 18;              // 256 K elements: period >> any history
  std::vector<uint8_t> data(n);
  std::mt19937 rng(0xC0FFEE);
  for (size_t i = 0; i < n; i++) data[i] = (uint8_t)(rng() & 0xFF);

  std::vector<uint8_t> sorted = data;
  std::sort(sorted.begin(), sorted.end());

  // Size the run to ~200 ms per variant off a shuffled probe.
  double probeNs = branchPassNs(data.data(), n, 4);
  uint64_t passes = (uint64_t)(200e6 / (probeNs * (double)n));
  passes = std::min<uint64_t>(std::max<uint64_t>(passes, 8), 1u << 20);

  double tSorted = 1e30, tRandom = 1e30;
  for (int rep = 0; rep < 3; rep++)
  {
    tSorted = std::min(tSorted, branchPassNs(sorted.data(), n, passes));
    tRandom = std::min(tRandom, branchPassNs(data.data(), n, passes));
  }

  predictedNs = tSorted;
  randomNs    = tRandom;
  // 50/50 random data against any predictor -> ~0.5 mispredicts per branch.
  penaltyNs   = (tRandom - tSorted) / 0.5;
}

} // anonymous namespace

int CpuPeak::runBranchPenalty(benchmark_config_t &cfg)
{
  (void)cfg;
  logger::TestSpec spec{"branch_mispredict", "Branch mispredict penalty", "ns",
                        Category::Latency};
  auto test = currentDeviceScope->beginTest(spec);

  // Pinned single-thread, like the pointer-chase.
  double pred = -1.0, rnd = -1.0, pen = -1.0;
  pool->run(1, [&](int) { branchPenalty(pred, rnd, pen); });

  if (pred > 0 && rnd > 0)
  {
    test.emit("predicted", (float)pred);      // ns/branch, sorted input
    test.emit("random", (float)rnd);          // ns/branch, 50/50 input
    test.emit("penalty", (float)std::max(pen, 0.0));  // ns per mispredict
  }
  else
  {
    test.skip("penalty", ResultStatus::Error, "branch benchmark failed");
  }
  return 0;
}

#endif // ENABLE_CPU
