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

#include <common/form_race.h>
#include <coreml/coreml_peak.h>
#include "coreml_session.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// What one session of a program over resident constants holds at once, for
// a test's memory gate: two copies of the constants -- the program's weight
// blob, alive until the session has compiled and loaded it, and the compute
// unit's own; Core ML maps the compiled weights from disk, and the Neural
// Engine's compiler runs in a process of its own, so there is no third --
// and every intermediate tensor, which the runtimes behind the other
// backends keep rather than reusing two (src/litert/litert_bench.h).
inline uint64_t coremlHeldBytes(uint64_t constants, uint64_t activations)
{
  return 2 * constants + activations;
}

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

// A clause for a row's description ("; ..."), when a passing session still
// had glue -- a closing scalar multiply, a reshape, under kCoremlOffDeviceShare
// of the plan's cost -- placed on another unit; empty when everything ran on
// the device.
inline std::string coremlGlueNote(const CoremlSession &s)
{
  const std::string g = s.glue();
  if (g.empty())
    return std::string();
  return "; negligible " + g + " ran elsewhere";
}

// Session creation cost, for the compile-time gates.
inline double coremlCreateUs(const CoremlSession &s)
{
  return s.compileUs + s.loadUs + s.planUs;
}

// The two orders a model can store a weight in (coremlEmitWeight): W[K, N]
// as it multiplies, [in, out], or its transpose [N, K] -- the [out, in]
// layout a converted Linear layer carries -- read with transpose_y.
inline const char *coremlLayoutName(bool transposed)
{
  return transposed ? "[out, in]" : "[in, out]";
}

// Which of the two a compute unit runs faster depends on the unit, the
// format and the shape, in ways no rule written here would carry to the
// next chip.  On an M1 Pro the CPU runs blockwise int4 at 5.0 TFLOPS
// stored [out, in] and at 0.74 stored [in, out]; the GPU runs per-channel
// int8 at 4.7 stored [in, out] and at 0.4 stored [out, in], yet decodes a
// blockwise int4 block at 48 GB/s stored [out, in] against 20; the Neural
// Engine runs the two at one rate, and compiles [in, out] ten times
// slower, rearranging every weight first.  So the gemm chain races the two
// at every width and the block at every point, each row reporting the
// faster and saying which, and the accuracy matmul reads the faster one --
// `false` is [in, out], `true` [out, in], and clpeak::FormRace has the rule
// that closes a race.
using CoremlLayoutRace = clpeak::FormRace;

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
