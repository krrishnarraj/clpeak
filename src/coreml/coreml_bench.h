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
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

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

// Whether this device's models store their weights as [N, K], the transpose
// of the W[K, N] they multiply by, read with transpose_y (coremlEmitWeight):
// the gemm chain's, the block's projections' and the accuracy matmul's,
// so an accuracy row reads the layout its rates do.  The Neural Engine's
// compiler takes that layout as it is and rearranges a [K, N] weight first:
// on an M1 Pro a sixteen-layer chain of 2048-wide fp16 layers loads in 1.2 s
// against 15, the gemm test takes 94 s against 332 and the block's 23
// sessions compile in 36 s against 247, every rate within 2%.  The GPU and
// CPU compile either in seconds, and there the layout moves readings
// instead -- the M1 Pro's GPU runs [N, K] int8 per-channel weights at 0.4
// TFLOPS against 4.7, its CPU blockwise int4 at 5.0 against 0.7 -- so they
// keep [K, N].  The arithmetic is the same on every unit; only the order a
// weight's bytes are stored in differs.
inline bool coremlTransposedWeights(const coreml_device_info_t &dev)
{
  return dev.kind == CoremlDeviceKind::NeuralEngine;
}

// What the timed runs computed, read once they are over: a NaN or an
// infinity in the row `output` reduces to withholds the timing as an error,
// the ONNX backend's rule (onnxNonFiniteReason, src/onnx/onnx_session.cpp).
// Nothing in these graphs can overflow -- the gemm chains keep each layer's
// magnitude and the block's largest value sits 31x under fp16's -- so either
// one says the unit computed something other than the graph.  Only the
// reduced row is read, so a wrong but finite answer passes: a tripwire, not
// an accuracy test.  `dtype` is the output's (fp16 or fp32); empty when the
// row is finite or cannot be read.
inline std::string coremlNonFiniteReason(CoremlSession &s, const std::string &output, int dtype,
                                         const std::string &where)
{
  std::vector<uint8_t> raw;
  std::string err;
  const size_t es = (size_t)coremlElemBytes(dtype, 1);
  if (!s.outputBytes(output, raw, err) || raw.empty() || raw.size() % es != 0)
    return std::string();
  // Every exponent bit set is an infinity or a NaN; read as bits, so no
  // floating-point mode can compile the test away.
  const size_t n = raw.size() / es;
  size_t bad = 0;
  for (size_t i = 0; i < n; i++)
  {
    if (es == 4)
    {
      uint32_t x;
      std::memcpy(&x, &raw[4 * i], 4);
      bad += (x & 0x7f800000u) == 0x7f800000u;
    }
    else
    {
      uint16_t h;
      std::memcpy(&h, &raw[2 * i], 2);
      bad += (h & 0x7c00u) == 0x7c00u;
    }
  }
  if (bad == 0)
    return std::string();
  const std::string count = std::to_string(n);
  return "Core ML returned NaN or infinity " + where + " (" +
         (bad == n ? "all " + count : std::to_string(bad) + " of the " + count) +
         " values of the row the graph reduces to), and every value in this graph stays far "
         "inside its type's range: the compute unit computed something other than this graph, "
         "so its timing is withheld";
}

#endif // CLPEAK_COREML_BENCH_H
