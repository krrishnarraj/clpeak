#ifndef CPU_COMPUTE_COMMON_H
#define CPU_COMPUTE_COMMON_H

#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/form_race.h>
#include <common/run_document.h>
#include "cpu_kernels.h"

#include <cstdio>
#include <string>
#include <vector>

// Run one compute variant single-threaded (1T, on the fastest core) and across
// all logical cores (NT), emitting both metrics.  `v.fn(iters)` performs
// `iters` outer iterations of the kernel and returns a sink value (kept live so
// the compiler can't elide the work); `v.opsPerIter` is the op count one
// thread performs in one outer iteration (flops for FP, ops for INT).  `unit`,
// when non-empty, overrides the test's for these two readings -- what lets an
// int8 row live inside an otherwise floating-point test.
//
// A variant with other spellings (ChainVariant::alts) races them in each
// reading: every form runs for a quarter of the budget, and the fastest -- the
// earlier on a tie (kFormRaceTie) -- is timed again over the whole of it, so
// the reading is one measurement rather than the best of several noisy ones.
// Its note names the form; --verbose logs every form's rate.
[[maybe_unused]] static void emitCompute(CpuPeak &peak, logger::TestScope &test,
                                         const std::string &label,
                                         const clpeak_cpu::ChainVariant &v,
                                         benchmark_config_t &cfg,
                                         const char *description = nullptr,
                                         const char *unit = nullptr)
{
  const int maxT = peak.pool->maxThreads();
  std::vector<double> sink((size_t)maxT, 0.0);
  unsigned int forced = peak.forceIters ? peak.specifiedIters : 0;

  // Ops per second of one form on `nThreads`, timed over `budgetUs`; <= 0 on failure.
  auto rate = [&](const clpeak_cpu::ChainVariant &f, int nThreads, unsigned int budgetUs) {
    CpuPeak::Workload body = [&](int tid, uint64_t iters) {
      sink[(size_t)tid] += f.fn(iters);
    };
    return f.opsPerIter * peak.runWorkload(nThreads, body, budgetUs, forced);
  };

  std::vector<const clpeak_cpu::ChainVariant *> forms{&v};
  for (int i = 0; i < v.nAlts; i++)
    forms.push_back(&v.alts[i]);

  struct Reading { double rate; const clpeak_cpu::ChainVariant *form; };
  auto measure = [&](int nThreads, const char *threads) -> Reading {
    if (forms.size() == 1)
      return {rate(v, nThreads, cfg.targetTimeUs), &v};
    // A discarded warm-up first: the first probe otherwise runs on a core
    // still ramping (an M1 Pro's read 312 against its twin's 408), and with
    // ties going to the earlier form it is the one that loses.
    rate(v, nThreads, cfg.targetTimeUs / 8);
    std::vector<double> probe(forms.size());
    size_t best = 0;
    for (size_t i = 0; i < forms.size(); i++)
    {
      probe[i] = rate(*forms[i], nThreads, cfg.targetTimeUs / 4);
      if (probe[i] > probe[best] * clpeak::kFormRaceTie)
        best = i;
    }
    if (clpeak::verboseEnabled())
    {
      std::string line;
      for (size_t i = 0; i < forms.size(); i++)
      {
        char r[48];
        snprintf(r, sizeof(r), " %.4g G/s", probe[i] * 1e-9);
        line += (i ? "; " : "") + std::string(forms[i]->form) + r;
      }
      CLPEAK_VLOG("[cpu] %s %s race: %s -- timing %s\n", label.c_str(), threads,
                  line.c_str(), forms[best]->form);
    }
    return {rate(*forms[best], nThreads, cfg.targetTimeUs), forms[best]};
  };

  const Reading st = measure(1, "ST");
  const Reading mt = measure(maxT, "MT");

  // Keep the accumulated work observable so -O3 can't delete the kernels.
  volatile double keep = 0.0;
  for (int t = 0; t < maxT; t++) keep += sink[(size_t)t];
  (void)keep;

  // ST/MT mean the same thing in every test routed through this runner, so the
  // notes live here rather than at each call site.  A reading that names a
  // data type as well ("bf16 ST") keeps that in its label; the note only has
  // to explain the thread count.
  static const char *stNote = "One thread, on the fastest core.";
  static const char *mtNote = "Every hardware thread at once -- the whole chip, each core "
                              "doing as much as it can.";

  // A race names its winner inside the thread note, so the reading keeps to
  // two sentences.
  auto opts = [&](const char *threadNote, const Reading &r) {
    logger::EmitOptions o;
    std::string note = threadNote;
    if (forms.size() > 1)
    {
      note.pop_back();   // the full stop
      note += " (fastest of " + std::to_string(forms.size()) + " spellings: " +
              r.form->form + ").";
    }
    o.description = description ? std::string(description) + "  " + note : note;
    if (unit) o.unit = unit;
    return o;
  };

  if (st.rate > 0.0) test.emit(label + " ST", (float)st.rate, opts(stNote, st));
  else               test.skip(label + " ST", ResultStatus::Error, "workload failed",
                               opts(stNote, st));

  if (mt.rate > 0.0) test.emit(label + " MT", (float)mt.rate, opts(mtNote, mt));
  else               test.skip(label + " MT", ResultStatus::Error, "workload failed",
                               opts(mtNote, mt));
}

// Why `slot` has no variant on this host.  `cpuLacks` is the call site's
// reason, right when the CPU lacks every feature that would give one; when the
// CPU has one but this binary cannot run it (clpeak_cpu::MissingKernel), the
// skip names whose gap it is instead -- one clause per cause:
//   "this CPU has FEAT_SME, but this build has no kernel for it (its compiler
//    could not build one)"
[[maybe_unused]] static std::string unsupportedReason(const clpeak_cpu::MenuSlot &slot,
                                                      const char *cpuLacks)
{
  using clpeak_cpu::MissingKernel;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  const char *arch = "x86";
#else
  const char *arch = "Arm";
#endif
  std::string out;
  for (MissingKernel::Why why : {MissingKernel::NotBuilt, MissingKernel::TileStateRefused,
                                 MissingKernel::NotWritten})
  {
    std::vector<const char *> feats;
    for (const MissingKernel &k : slot.missing)
      if (k.why == why) feats.push_back(k.feature);
    if (feats.empty()) continue;

    std::string list = feats[0];
    for (size_t i = 1; i < feats.size(); i++)
      list += std::string(i + 1 < feats.size() ? ", " : " and ") + feats[i];
    const bool one = feats.size() == 1;
    const char *it = one ? "it" : "them";

    if (!out.empty()) out += "; ";
    out += "this CPU has " + list + ", but ";
    switch (why)
    {
    case MissingKernel::NotBuilt:
      out += std::string("this build has no kernel for ") + it +
             " (its compiler could not build " + (one ? "one" : "any") + ")";
      break;
    case MissingKernel::TileStateRefused:
      out += std::string("the OS refused ") + (one ? "its" : "their") + " tile state";
      break;
    case MissingKernel::NotWritten:
      out += std::string("clpeak has no ") + arch + " kernel for " + it + " yet";
      break;
    }
  }
  return out.empty() ? std::string(cpuLacks) : out;
}

// Run EVERY supported ISA variant of one compute kernel.  Each ISA is its own
// test -- comparing SSE2 against AVX-512 is the point of running both -- but
// they share one tag and are told apart by `variant`, so the ISA never gets
// slugged into the tag.  That keeps the tag identical across machines, which
// is what makes `--compare` work between them.  `unsupReason` is the skip when
// the CPU lacks the kernel's feature (see unsupportedReason()).
[[maybe_unused]] static void emitVariants(CpuPeak &peak, const logger::TestSpec &base,
                         const std::string &metric,
                         const clpeak_cpu::MenuSlot &slot,
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

  if (slot.vars.empty())
  {
    auto test = peak.currentDeviceScope->beginTest(spec);
    // No thread-count suffix: there is no ST/MT pair to distinguish when the
    // kernel does not exist on this host at all.
    test.skip(metric, ResultStatus::Unsupported, unsupportedReason(slot, unsupReason));
    return;
  }
  for (const auto &iv : slot.vars)
  {
    spec.variant = iv.isa;
    auto test = peak.currentDeviceScope->beginTest(spec);
    emitCompute(peak, test, metric, iv.v, cfg);
  }
}

// One row of a family that shares a test: a data type, or an operation.
struct FamilyRow {
  const char *metric;        // "bf16", "fdiv fp32"
  const char *description;   // what this row measures, beyond the family
  const char *unsupReason;   // shown when the CPU lacks every variant of it
  const clpeak_cpu::MenuSlot *slot;
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
// appears, as a skip, so the reader sees which formats the engine lacks; a
// row another ISA runs skips in this ISA's test naming that ISA instead.
[[maybe_unused]] static void emitFamily(CpuPeak &peak, const logger::TestSpec &base,
                       const std::vector<FamilyRow> &rows,
                       benchmark_config_t &cfg)
{
  logger::TestSpec spec = base;
  spec.shape = TestShape::Heterogeneous;

  // When every row of THIS call shares one unit override, it is the unit of
  // the readings this opening produces, so it heads them.  A mixed-unit call
  // keeps the family's unit and the rows carry their own.
  {
    const char *shared = rows.empty() ? nullptr : rows.front().unit;
    bool uniform = shared != nullptr;
    for (const FamilyRow &r : rows)
      if (!r.unit || std::string(r.unit) != shared) { uniform = false; break; }
    if (uniform) spec.unit = shared;
  }

  // A row that never ran still names the unit it would have been measured in:
  // an unsupported int8 row inside a floating-point test must not read as
  // flops.
  auto rowOpts = [](const FamilyRow &r) {
    logger::EmitOptions o;
    if (r.description) o.description = r.description;
    if (r.unit) o.unit = r.unit;
    return o;
  };

  // Why row `r` has no reading in the `isa` test.  One that runs under another
  // ISA is missing here because this ISA has no form of it, not because the
  // CPU lacks it, which is all the row's own reason can say.
  auto rowReason = [](const FamilyRow &r, const char *isa) {
    if (r.slot->vars.empty())
      return unsupportedReason(*r.slot, r.unsupReason);
    std::string runs;
    for (const auto &iv : r.slot->vars)
      runs += (runs.empty() ? "" : ", ") + std::string(iv.isa);
    return std::string(isa) + " has no " + r.metric + " form; this CPU runs it as " + runs;
  };

  // Ordered union of the ISAs any row supports, first-seen order (the menus
  // are built baseline-first, so this stays low-ISA to high-ISA).
  std::vector<const char *> isas;
  for (const FamilyRow &r : rows)
    for (const auto &iv : r.slot->vars)
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
      test.skip(r.metric, ResultStatus::Unsupported,
                unsupportedReason(*r.slot, r.unsupReason), rowOpts(r));
    return;
  }

  for (const char *isa : isas)
  {
    spec.variant = isa;
    auto test = peak.currentDeviceScope->beginTest(spec);
    for (const FamilyRow &r : rows)
    {
      const clpeak_cpu::IsaVariant *match = nullptr;
      for (const auto &iv : r.slot->vars)
        if (std::string(iv.isa) == isa) { match = &iv; break; }

      if (!match)
      {
        test.skip(r.metric, ResultStatus::Unsupported, rowReason(r, isa), rowOpts(r));
        continue;
      }
      emitCompute(peak, test, r.metric, match->v, cfg,
                  r.description, r.unit);
    }
  }
}

#endif // ENABLE_CPU
#endif // CPU_COMPUTE_COMMON_H
