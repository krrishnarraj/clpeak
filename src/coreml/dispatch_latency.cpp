#ifdef ENABLE_COREML

// coreml-dispatch-latency: what it costs to ask this compute unit to do
// anything at all.
//
// A Core ML prediction goes through the framework, a driver and -- on the
// Neural Engine -- a request to a coprocessor, and every layer charges a
// toll per submission.  That toll is why a chip advertising tens of TOPS
// can lose to the CPU on small work, and no throughput row can show it.
//
// There is a complication the ONNX backend's version of this test does not
// have: Core ML's planner will not send trivial work to an accelerator.  A
// 64-element multiply, a 256-cube matmul, an 8 MB matrix-vector product are
// all kept on the CPU under a cpuAndNeuralEngine configuration, and the
// compute plan says so.  So each row here climbs a ladder of sizes and
// reports the *smallest* the planner actually sends to this unit, naming
// the size in the description.  Two things are learned at once: how small
// a piece of work this unit is ever given, and what one prediction of it
// costs.  On the CPU the ladders end at their first rung.
//
// Model creation is timed too -- on the Neural Engine it is a compiler run,
// and it is why cold-start behaviour differs so sharply from steady state.
// The trivial model is salted per process so that Core ML's cache of
// compiled programs cannot hand back a previous run's work.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace
{

// Elementwise sizes: 64 values up to 4M (8 MB of fp16), as [rows, 4096]
// once they are wide enough -- the Neural Engine's compiler refuses a
// single four-million-wide row outright, while an 8 MB [1024, 4096] tensor
// is the smallest elementwise model the planner sends it.
const int64_t kTrivialWidths[] = {64, 4096, 262144, 4194304};
constexpr int64_t kTrivialCols = 4096;
// Square matmuls with a live input, 33 MFLOP up to 2 GFLOP.
const int64_t kMatMulDims[] = {256, 512, 1024};

struct Timed
{
  double perRunUs = -1.0;
  double createUs = -1.0;
  double compileUs = -1.0;
  double loadUs = -1.0;
  int64_t size = 0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

// Time one prediction of `prog`, with `inName` an fp16 input of rows x cols.
// A model the planner keeps off this unit comes back Unsupported with the
// planner's reason, and `createUs` still filled in.
Timed timeGraph(const coreml_device_info_t &dev, const CoremlProgram &prog, const char *inName,
                int64_t rows, int64_t cols, unsigned warmup, bool forceIters, unsigned forced)
{
  Timed t;
  std::string err;
  auto s = CoremlSession::create(dev, prog, err);
  if (!s)
  {
    t.error = err;
    t.status = ResultStatus::Unsupported;
    return t;
  }
  t.createUs = s->compileUs + s->loadUs;   // the plan is this backend's own bookkeeping
  t.compileUs = s->compileUs;
  t.loadUs = s->loadUs;
  if (!s->onDevice())
  {
    t.error = coremlOffDeviceReason(dev, *s);
    t.status = ResultStatus::Unsupported;
    return t;
  }
  void *x = s->bindInput(inName, CML_FP16, {rows, cols}, (size_t)rows * cols * 2, err);
  if (!x)
  {
    t.error = err;
    t.status = ResultStatus::Error;
    return t;
  }
  {
    const uint16_t half = coremlFloatToHalf(0.5f);
    uint16_t *p = static_cast<uint16_t *>(x);
    for (int64_t i = 0; i < rows * cols; i++)
      p[i] = half;
  }
  if (s->timeRuns(1 + warmup, err) < 0.0)
  {
    t.error = err;
    t.status = ResultStatus::Error;
    return t;
  }
  // Five runs, where every other test probes with one: this is the one
  // measurement that *is* the per-submission overhead, and five of them
  // cost nothing.
  const double probe = s->timeRuns(5, err);
  if (probe <= 0.0)
  {
    t.error = err;
    t.status = ResultStatus::Error;
    return t;
  }
  const unsigned iters = pickIters(probe, 500000u, forceIters ? forced : 0, 200000u);
  t.perRunUs = s->timeRuns(iters, err);
  if (t.perRunUs <= 0.0)
  {
    t.error = err;
    t.status = ResultStatus::Error;
  }
  return t;
}

std::string mbOrValues(int64_t width)
{
  if (width >= (1 << 20))
    return std::to_string(width >> 19) + " MB of";
  return std::to_string(width);
}

} // namespace

int CoreMLPeak::runDispatchLatency(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();
  const uint32_t salt = (uint32_t)std::random_device{}();

  // The trivial ladder.  model_create is the creation cost of the rung the
  // planner accepted, so that on an accelerator it is that unit's compiler
  // being timed; when no rung is accepted it falls back to the first.
  Timed trivial, first;
  for (int64_t w : kTrivialWidths)
  {
    if (clpeak::cancelRequested())
      break;
    const int64_t rows = w > kTrivialCols ? w / kTrivialCols : 1;
    const int64_t cols = w > kTrivialCols ? kTrivialCols : w;
    Timed t = timeGraph(dev, coremlTrivialModel(spec, rows, cols, salt), "x", rows, cols, warmupCount,
                        forceIters, specifiedIters);
    t.size = w;
    if (first.createUs < 0.0)
      first = t;
    trivial = t;
    if (t.perRunUs > 0.0 || t.status != ResultStatus::Unsupported)
      break;
    CLPEAK_VLOG("coreml-dispatch[%s]: %lld-wide multiply: %s\n", dev.displayName.c_str(), (long long)w,
                t.error.c_str());
  }

  Timed matmul;
  for (int64_t d : kMatMulDims)
  {
    if (clpeak::cancelRequested())
      break;
    Timed t = timeGraph(dev, coremlSmallMatMulModel(spec, d), "x", d, d, warmupCount, forceIters,
                        specifiedIters);
    t.size = d;
    matmul = t;
    if (t.perRunUs > 0.0 || t.status != ResultStatus::Unsupported)
      break;
    CLPEAK_VLOG("coreml-dispatch[%s]: %lld-cube matmul: %s\n", dev.displayName.c_str(), (long long)d,
                t.error.c_str());
  }

  auto test = currentDeviceScope->beginTest(
      {"coreml_dispatch_latency", "Core ML dispatch latency", "s", Category::Latency,
       "The fixed cost of handing one prediction to this compute unit, on the "
       "smallest piece of work Core ML's planner will send it -- an "
       "accelerator is never given trivial work, so each row names the size "
       "it settled on -- and the cost of preparing a model for it, a compiler "
       "run on the Neural Engine.  It is why a chip advertising tens of TOPS "
       "can still lose to the host on small work, and no throughput row can "
       "show it.",
       TestShape::Heterogeneous, "what is submitted"});

  if (trivial.perRunUs > 0.0)
  {
    std::string note = "One elementwise multiply over " + mbOrValues(trivial.size) +
                       " values, the smallest such model the planner sends to this unit";
    if (trivial.size <= kTrivialWidths[0])
      note += " -- as close to doing nothing as a model can get, so almost all of this "
              "is the cost of asking.";
    else
      note += ".  Nothing smaller is ever given to it, so this is the floor a model "
              "actually sees: the submission plus moving that much data in and out.";
    test.emit("trivial_op", (float)(trivial.perRunUs * 1e-6), note.c_str());
  }
  else
    test.skip("trivial_op", trivial.status,
              "no elementwise model up to 8 MB was sent to this unit: " + trivial.error,
              "One elementwise multiply, at the smallest size the planner sends to this unit.");

  if (matmul.perRunUs > 0.0)
  {
    const double gflop = 2.0 * (double)matmul.size * (double)matmul.size * (double)matmul.size / 1.0e9;
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.2f", gflop);
    std::string note = "A " + std::to_string(matmul.size) + "-cube fp16 matrix multiply with a live " +
                       "input (" + buf + " GFLOP), the smallest the planner sends to this unit";
    if (matmul.size <= kMatMulDims[0])
      note += " -- which it finishes in well under a millisecond, so whatever this reads "
              "above the row before it is still mostly overhead.";
    else
      note += ".  Divide the arithmetic by the matmul peak row to see how much of this "
              "is work; the rest is the submission and the input copy.";
    test.emit("small_matmul", (float)(matmul.perRunUs * 1e-6), note.c_str());
  }
  else
    test.skip("small_matmul", matmul.status,
              "no matrix multiply up to 1024-cube was sent to this unit: " + matmul.error,
              "A small matrix multiply, at the smallest size the planner sends to this unit.");

  const Timed &created = (trivial.perRunUs > 0.0 && trivial.createUs > 0.0) ? trivial : first;
  if (created.createUs > 0.0)
  {
    const std::string note =
        "Compiling and loading the elementwise model above (" + mbOrValues(created.size) +
        " values), fresh -- it is salted so Core ML's cache cannot answer from a previous "
        "run: " + std::to_string((long long)(created.compileUs / 1000.0)) +
        " ms to compile the package and " + std::to_string((long long)(created.loadUs / 1000.0)) +
        " ms to load it for this configuration, which is where a unit's own compiler runs.";
    test.emit("model_create", (float)(created.createUs * 1e-6), note.c_str());
  }
  else
    test.skip("model_create", created.status, created.error,
              "Compiling and loading a trivial model for this compute unit.");

  test.end();
  return 0;
}

#endif // ENABLE_COREML
