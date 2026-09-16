#ifndef CLPEAK_LITERT_BENCH_H
#define CLPEAK_LITERT_BENCH_H

// The measurement shape every timed test in this backend follows -- the one
// the GPU backends use and the ONNX backend documents (src/onnx/AGENTS.md,
// "Three phases per rung"):
//
//   1. warmup   1 + warmupCount untimed inferences.  The extra one is not a
//               spare: XNNPACK packs weights and the GPU accelerator finishes
//               compiling and uploading on the first run, so run one is a
//               different event from run two.
//   2. probe    exactly one timed inference, to size the batch.
//   3. timed    pickIters(probe, budget, forced, kLitertMaxIters) inferences;
//               when the budget affords only one, the probe already is the
//               measurement.
//
// Every batch, the probe included, ends with a wait for the accelerator: the
// GPU accelerator's OpenCL path returns from a run as soon as the work is
// submitted, and a probe that timed a submission would size the batch for
// a run a hundred times shorter than the real one -- on a phone that turned
// a one-second budget into ten.  `syncEach` waits after every run instead,
// for the latency rows, where a result read back is the unit of work.

#include <litert/litert_peak.h>
#include "litert_model.h"
#include "litert_session.h"

#include <cstring>
#include <string>

// A kernel that declines a type at prepare time ("input->type !=
// kTfLiteFloat32", "failed to prepare") is the runtime saying it has no
// kernel for the format, which is a capability and not a failure.
inline ResultStatus litertFailureStatus(const std::string &error)
{
  if (error.find("failed to prepare") != std::string::npos ||
      error.find("not supported") != std::string::npos ||
      error.find("Unsupported") != std::string::npos ||
      error.find("unsupported") != std::string::npos)
    return ResultStatus::Unsupported;
  return ResultStatus::Error;
}

struct LitertMeasurement
{
  double meanUs = -1.0;     // mean per inference; negative on failure
  double probeUs = -1.0;    // the single calibration inference
  unsigned iters = 0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

inline LitertMeasurement litertMeasure(LitertSession &s, unsigned warmupCount, unsigned budgetUs,
                                       bool forceIters, unsigned forced,
                                       unsigned maxIters = kLitertMaxIters, bool syncEach = false)
{
  LitertMeasurement m;
  if (s.timeRuns(1 + warmupCount, m.error, syncEach) < 0.0)
  {
    if (m.error.empty())
      m.error = "inference failed";
    m.status = litertFailureStatus(m.error);
    return m;
  }
  m.probeUs = s.timeRuns(1, m.error, syncEach);
  if (m.probeUs <= 0.0)
  {
    m.status = ResultStatus::Error;
    if (m.error.empty())
      m.error = "inference failed";
    return m;
  }
  m.iters = pickIters(m.probeUs, budgetUs, forceIters ? forced : 0, maxIters);
  m.meanUs = (m.iters > 1) ? s.timeRuns(m.iters, m.error, syncEach) : m.probeUs;
  if (m.meanUs <= 0.0)
  {
    m.status = ResultStatus::Error;
    if (m.error.empty())
      m.error = "inference failed";
  }
  return m;
}

// Write the runtime scalar that keeps a throughput graph live into input 0.
// A hair above one for the float types, so the scaled values stay inside
// the range the fills chose; exactly one for the integer types, where the
// scalar's own scale (1/127, 1/32767) puts one at the top code.
inline bool litertBindScalar(LitertSession &s, const LitertPlan &p, std::string &error)
{
  const bool integer = (p.act == clpeak_tflite::TfType::I8 || p.act == clpeak_tflite::TfType::I16);
  const std::string v = litertScalarBytes(p.act, integer ? 1.0f : 1.0009765625f);
  return s.writeInput(0, v.data(), v.size(), error);
}

// The session config a plan asks for on a device.
inline LitertSessionConfig litertConfigFor(const LitertPlan &p, bool profile = false)
{
  LitertSessionConfig cfg;
  cfg.gpuPrecision = p.gpuPrecision;
  cfg.gpuAllowQuantized = p.gpuAllowQuantized;
  cfg.profile = profile;
  return cfg;
}

#endif // CLPEAK_LITERT_BENCH_H
