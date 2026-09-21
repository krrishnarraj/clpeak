#ifndef CLPEAK_COREML_BENCH_H
#define CLPEAK_COREML_BENCH_H

// The measurement shape every timed test in this backend follows -- the one
// the GPU backends use and the ONNX backend documents (src/onnx/AGENTS.md,
// "Three phases per rung"):
//
//   1. warmup   1 + warmupCount untimed predictions.  The extra one is not a
//               spare: the Neural Engine finishes specialising on the first
//               prediction, so run one is a different event from run two.
//   2. probe    exactly one timed prediction, to size the batch.
//   3. timed    pickIters(probe, budget, forced, kCoremlMaxIters) predictions;
//               when the budget affords only one, the probe already is the
//               measurement.

#include <coreml/coreml_peak.h>
#include "coreml_session.h"

#include <chrono>
#include <cstring>
#include <string>

struct CoremlMeasurement
{
  double meanUs = -1.0;     // mean per prediction; negative on failure
  double probeUs = -1.0;    // the single calibration prediction
  unsigned iters = 0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

inline CoremlMeasurement coremlMeasure(CoremlSession &s, unsigned warmupCount,
                                       unsigned budgetUs, bool forceIters, unsigned forced,
                                       unsigned maxIters = kCoremlMaxIters)
{
  CoremlMeasurement m;
  if (s.timeRuns(1 + warmupCount, m.error) < 0.0)
  {
    m.status = ResultStatus::Error;
    if (m.error.empty())
      m.error = "prediction failed";
    return m;
  }
  m.probeUs = s.timeRuns(1, m.error);
  if (m.probeUs <= 0.0)
  {
    m.status = ResultStatus::Error;
    if (m.error.empty())
      m.error = "prediction failed";
    return m;
  }
  m.iters = pickIters(m.probeUs, budgetUs, forceIters ? forced : 0, maxIters);
  m.meanUs = (m.iters > 1) ? s.timeRuns(m.iters, m.error) : m.probeUs;
  if (m.meanUs <= 0.0)
  {
    m.status = ResultStatus::Error;
    if (m.error.empty())
      m.error = "prediction failed";
  }
  return m;
}

// Bind the runtime scalar `s` that keeps a throughput graph live.  A hair
// above one, so the scaled values stay inside the range the fills chose.
inline bool coremlBindScalar(CoremlSession &s, const char *name, int dtype, std::string &error)
{
  const size_t es = (size_t)coremlElemBytes(dtype, 1);
  void *p = s.bindInput(name, dtype, {1}, es, error);
  if (!p)
    return false;
  const std::string v = coremlFloatScalar(1.0009765625f, dtype);
  std::memcpy(p, v.data(), es);
  return true;
}

// The reason a row is unsupported when the compute plan put any of its
// operations somewhere other than the device it was created for.  Two
// different things land on the CPU: an operation the unit cannot run in
// this form, and one the planner judged too small to be worth sending --
// and the second is as real for an application as the first, since no Core
// ML configuration can force a unit.
inline std::string coremlOffDeviceReason(const coreml_device_info_t &dev, const CoremlSession &s)
{
  const std::string unit = coremlKindName(dev.kind);
  if (s.offDeviceCapable())
    return "Core ML's planner sends " + s.offDevice() + " at this size to another compute unit "
           "rather than the " + unit + " -- it could run there, but the framework does not "
           "send work this small, so a model would not see the " + unit + " for it either";
  return "the " + unit + " cannot run " + s.offDevice() + " in this form -- Core ML would run it "
         "on another compute unit, so this would not be a " + unit + " number";
}

// A sentence for a row's description when a passing session still had
// negligible glue -- a closing scalar multiply, a reshape -- placed on
// another unit; empty when everything ran on the device.
inline std::string coremlGlueNote(const CoremlSession &s)
{
  const std::string g = s.glue();
  if (g.empty())
    return std::string();
  return "  Core ML placed " + g + " on another compute unit, at under 5% of the "
         "plan's estimated cost.";
}

// Session creation cost, for the compile-time gates.
inline double coremlCreateUs(const CoremlSession &s)
{
  return s.compileUs + s.loadUs + s.planUs;
}

#endif // CLPEAK_COREML_BENCH_H
