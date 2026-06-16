#ifndef CPU_COMPUTE_COMMON_H
#define CPU_COMPUTE_COMMON_H

#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/result_store.h>
#include "cpu_kernels.h"

#include <cctype>
#include <string>
#include <vector>

// Run one compute variant single-threaded (1T) and across all logical cores
// (NT), emitting both metrics.  `chain(iters)` performs `iters` outer
// iterations of the kernel and returns a sink value (kept live so the compiler
// can't elide the work); `opsPerIterPerThread` is the op count one thread
// performs in one outer iteration (flops for FP, ops for INT).  `unitDivider`
// is 1e3 for giga-units (the emitted unit is gflops / gops).
template <class ChainFn>
static void emitCompute(CpuPeak &peak, logger::TestScope &test,
                        const std::string &label,
                        double opsPerIterPerThread,
                        ChainFn chain, benchmark_config_t &cfg)
{
  const int maxT = peak.pool->maxThreads();
  std::vector<double> sink((size_t)maxT, 0.0);

  CpuPeak::Workload body = [&](int tid, uint64_t iters) {
    sink[(size_t)tid] += chain(iters);
  };

  unsigned int forced = peak.forceIters ? peak.specifiedIters : 0;
  double us1 = peak.runWorkload(1,    body, cfg.targetTimeUs, forced);
  double usN = peak.runWorkload(maxT, body, cfg.targetTimeUs, forced);

  // Keep the accumulated work observable so -O3 can't delete the kernels.
  volatile double keep = 0.0;
  for (int t = 0; t < maxT; t++) keep += sink[(size_t)t];
  (void)keep;

  auto giga = [](double opsPerIter, int n, double meanUs) -> float {
    if (meanUs <= 0.0) return -1.0f;
    return (float)(opsPerIter * (double)n / (meanUs * 1e3));
  };

  if (us1 > 0.0) test.emit(label + " ST", giga(opsPerIterPerThread, 1, us1));
  else           test.skip(label + " ST", ResultStatus::Error, "workload failed");

  if (usN > 0.0) test.emit(label + " MT", giga(opsPerIterPerThread, maxT, usN));
  else           test.skip(label + " MT", ResultStatus::Error, "workload failed");
}

// Canonical ISA label -> tag suffix, e.g. "AVX-512" -> "avx_512",
// "AVX2+FMA" -> "avx2_fma".  Lowercased; each run of non-alphanumeric chars
// becomes a single '_'.  Appended to the test tag so each ISA's rows stay
// unique in the dump / baseline files.
static inline std::string isaSlug(const std::string &isa)
{
  std::string s;
  bool pendingSep = false;
  for (char c : isa)
  {
    if (std::isalnum((unsigned char)c))
    {
      if (pendingSep && !s.empty()) s += '_';
      pendingSep = false;
      s += (char)std::tolower((unsigned char)c);
    }
    else
      pendingSep = true;
  }
  return s;
}

// Run EVERY supported ISA variant of one compute kernel: a separate test (with
// its own header) per variant, the ISA appended to the display name and slugged
// into the tag.  If no variant is supported, emit one untagged Unsupported test.
static void emitVariants(CpuPeak &peak, const logger::TestSpec &base,
                         const std::string &metric,
                         const std::vector<clpeak_cpu::IsaVariant> &vars,
                         const char *unsupReason, benchmark_config_t &cfg)
{
  if (vars.empty())
  {
    auto test = peak.currentDeviceScope->beginTest(base);
    test.skip(metric + " ST", ResultStatus::Unsupported, unsupReason);
    return;
  }
  for (const auto &iv : vars)
  {
    logger::TestSpec spec = base;
    spec.tag     = base.tag + "_" + isaSlug(iv.isa);
    spec.display = base.display + " (" + iv.isa + ")";
    auto test = peak.currentDeviceScope->beginTest(spec);
    emitCompute(peak, test, metric, iv.v.opsPerIter, iv.v.fn, cfg);
  }
}

#endif // ENABLE_CPU
#endif // CPU_COMPUTE_COMMON_H
