#ifndef CPU_COMPUTE_COMMON_H
#define CPU_COMPUTE_COMMON_H

#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/run_document.h>
#include "cpu_kernels.h"

#include <string>
#include <vector>

// Run one compute variant single-threaded (1T) and across all logical cores
// (NT), emitting both metrics.  `chain(iters)` performs `iters` outer
// iterations of the kernel and returns a sink value (kept live so the compiler
// can't elide the work); `opsPerIterPerThread` is the op count one thread
// performs in one outer iteration (flops for FP, ops for INT).  `unit`, when
// non-empty, overrides the test's for these two readings -- what lets an int8
// row live inside an otherwise floating-point test.
template <class ChainFn>
static void emitCompute(CpuPeak &peak, logger::TestScope &test,
                        const std::string &label,
                        double opsPerIterPerThread,
                        ChainFn chain, benchmark_config_t &cfg,
                        const char *description = nullptr,
                        const char *unit = nullptr)
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

  // ST/MT mean the same thing in every test routed through this runner, so the
  // notes live here rather than at each call site.  A reading that names a
  // data type as well ("bf16 ST") keeps that in its label; the note only has
  // to explain the thread count.
  static const char *stNote = "One thread, running on a single core.";
  static const char *mtNote = "Every hardware thread at once -- the whole chip.";

  auto opts = [&](const char *threadNote) {
    logger::EmitOptions o;
    o.description = description ? std::string(description) + "  " + threadNote
                                : std::string(threadNote);
    if (unit) o.unit = unit;
    return o;
  };

  if (us1 > 0.0) test.emit(label + " ST", giga(opsPerIterPerThread, 1, us1), opts(stNote));
  else           test.skip(label + " ST", ResultStatus::Error, "workload failed",
                           opts(stNote).description);

  if (usN > 0.0) test.emit(label + " MT", giga(opsPerIterPerThread, maxT, usN), opts(mtNote));
  else           test.skip(label + " MT", ResultStatus::Error, "workload failed",
                           opts(mtNote).description);
}

// Run EVERY supported ISA variant of one compute kernel.  Each ISA is its own
// test -- comparing SSE2 against AVX-512 is the point of running both -- but
// they share one tag and are told apart by `variant`, so the ISA never gets
// slugged into the tag.  That keeps the tag identical across machines, which
// is what makes `--compare` work between them.
static void emitVariants(CpuPeak &peak, const logger::TestSpec &base,
                         const std::string &metric,
                         const std::vector<clpeak_cpu::IsaVariant> &vars,
                         const char *unsupReason, benchmark_config_t &cfg)
{
  logger::TestSpec spec = base;
  // Every test routed through emitCompute reports the same kernel at one thread
  // and at all of them.  That is homogeneous: the larger reading is the chip's
  // real peak for this kernel, not a number invented by picking the biggest of
  // several unrelated ones, which is the case the distinction exists to catch.
  // The per-core figure is one tap away, exactly as float2 is under float4.
  //
  // smt_scaling is the deliberate exception, and opens its own scope: there the
  // comparison between the two thread counts IS the result, so collapsing to
  // the larger would delete the finding.
  //
  // Set here rather than at thirty call sites because it is a property of this
  // runner -- every test it drives has exactly this pair of readings.
  spec.shape = TestShape::Homogeneous;
  if (spec.axis.empty()) spec.axis = "threads";

  if (vars.empty())
  {
    auto test = peak.currentDeviceScope->beginTest(spec);
    // No thread-count suffix: there is no ST/MT pair to distinguish when the
    // kernel does not exist on this host at all.
    test.skip(metric, ResultStatus::Unsupported, unsupReason);
    return;
  }
  for (const auto &iv : vars)
  {
    spec.variant = iv.isa;
    auto test = peak.currentDeviceScope->beginTest(spec);
    emitCompute(peak, test, metric, iv.v.opsPerIter, iv.v.fn, cfg);
  }
}

// One row of a family that shares a test: a data type, or an operation.
struct FamilyRow {
  const char *metric;        // "bf16", "fdiv fp32"
  const char *description;   // what this row measures, beyond the family
  const char *unsupReason;   // shown when this host has no variant for it
  const std::vector<clpeak_cpu::IsaVariant> *vars;
  const char *unit = nullptr;  // nullptr = the test's unit
};

// Run a family of related kernels as ONE test per ISA, instead of one test per
// row.  The CPU matrix engine is six data types on one unit and the divide /
// sqrt rows are four operations on one unit; as separate tests they were six
// and four near-identical lines whose names carried the only difference.
//
// Rows are grouped by ISA rather than the other way round because the ISA is
// what makes two readings incomparable: bf16 on AMX and bf16 on SME are
// different hardware, while bf16 and fp16 on the same AMX are the same unit
// asked for a different format.  A row the host cannot run at all still
// appears, as a skip, so the reader sees which formats the engine lacks.
static void emitFamily(CpuPeak &peak, const logger::TestSpec &base,
                       const std::vector<FamilyRow> &rows,
                       benchmark_config_t &cfg)
{
  logger::TestSpec spec = base;
  spec.shape = TestShape::Heterogeneous;

  // Ordered union of the ISAs any row supports, first-seen order (the menus
  // are built baseline-first, so this stays low-ISA to high-ISA).
  std::vector<const char *> isas;
  for (const FamilyRow &r : rows)
    for (const auto &iv : *r.vars)
    {
      bool known = false;
      for (const char *seen : isas)
        if (std::string(seen) == iv.isa) { known = true; break; }
      if (!known) isas.push_back(iv.isa);
    }

  // Nothing on this host runs any of it: one test, one skip per row, each
  // saying why that particular format or operation is missing.
  if (isas.empty())
  {
    auto test = peak.currentDeviceScope->beginTest(spec);
    for (const FamilyRow &r : rows)
      test.skip(r.metric, ResultStatus::Unsupported, r.unsupReason,
                r.description ? r.description : "");
    return;
  }

  for (const char *isa : isas)
  {
    spec.variant = isa;
    auto test = peak.currentDeviceScope->beginTest(spec);
    for (const FamilyRow &r : rows)
    {
      const clpeak_cpu::IsaVariant *match = nullptr;
      for (const auto &iv : *r.vars)
        if (std::string(iv.isa) == isa) { match = &iv; break; }

      if (!match)
      {
        test.skip(r.metric, ResultStatus::Unsupported, r.unsupReason,
                  r.description ? r.description : "");
        continue;
      }
      emitCompute(peak, test, r.metric, match->v.opsPerIter, match->v.fn, cfg,
                  r.description, r.unit);
    }
  }
}

#endif // ENABLE_CPU
#endif // CPU_COMPUTE_COMMON_H
