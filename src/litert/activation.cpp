#ifdef ENABLE_LITERT

// litert-activation: how fast the operations *between* the matrix multiplies
// run -- softmax, layer normalisation, the SiLU gate.  None of them does
// meaningful arithmetic: they read a tensor and write one back, so their
// ceiling is memory bandwidth and their rate is reported as the bandwidth
// they achieve.  On an NPU these are the operations most likely to be
// handed back to the CPU, which the fallback guard reports.
//
// Each rate is net of a reference graph that reads, scales and reduces the
// same constant with no operation applied, timed once per size and shared
// by the three operations.  Every rung is reported at the three working-set
// sizes litert-tensor-bw always measures, so the two ladders divide row for
// row; see src/onnx/activation.cpp for why a single number per operation
// cannot be made honest, and src/coreml/activation.cpp for the two-session
// minimum and noise floor used here.

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kCols = 4096;

struct Size
{
  uint64_t bytes;
  const char *label;
};
const Size kSizes[] = {
    {8ull << 20, "8mb"},
    {32ull << 20, "32mb"},
    {128ull << 20, "128mb"},
};

uint64_t maxTensorBytes() { return clpeak::memoryBudget(1ull << 30); }
constexpr uint64_t kCopiesAtPeak = 4;   // the constant, its scaled copy, the result, the runtime's own
constexpr unsigned int kSizeBudgetUs = 1000000;

// The share of the work the operation itself has to account for before the
// difference is worth reporting, and the factor by which it has to clear
// the graphs' own session-to-session spread.
constexpr double kMinOpShare = 0.10;
constexpr double kNoiseFactor = 3.0;

struct Variant
{
  LitertActivation act;
  const char *label;
  const char *note;
};

const Variant kVariants[] = {
    {LitertActivation::Silu, "silu",
     "x times sigmoid(x) -- the gate in the feed-forward network of most "
     "current language models, two operators in a .tflite."},
    {LitertActivation::Softmax, "softmax",
     "Softmax across the row, at the heart of attention: two passes and a "
     "maximum before it can divide."},
    {LitertActivation::LayerNorm, "layernorm",
     "Mean and variance across each row, then rescale -- every transformer "
     "layer does this at least twice, and a .tflite spells it as seven "
     "operators."},
};

struct Run
{
  double us = -1.0;         // the faster of the two sessions
  double spreadUs = 0.0;    // how far apart they were
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Run measureOnce(const LitertRuntime &rt, const litert_device_info_t &dev, const LitertPlan &plan,
                int64_t rows, LitertActivation act, unsigned warmup, bool forceIters, unsigned forced)
{
  Run r;
  std::string err;
  auto s = LitertSession::create(rt, dev, litertActivationModel(plan, rows, kCols, act),
                                 litertConfigFor(plan), err);
  if (!s)
  {
    r.error = err;
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (!s->onDevice())
  {
    r.error = s->offDevice();
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (!litertBindScalar(*s, plan, err))
  {
    r.error = err;
    r.status = ResultStatus::Error;
    return r;
  }
  auto m = litertMeasure(*s, warmup, kSizeBudgetUs, forceIters, forced);
  if (m.meanUs <= 0.0)
  {
    r.error = m.error;
    r.status = m.status;
    return r;
  }
  r.us = m.meanUs;
  return r;
}

Run measure(const LitertRuntime &rt, const litert_device_info_t &dev, const LitertPlan &plan,
            int64_t rows, LitertActivation act, unsigned warmup, bool forceIters, unsigned forced)
{
  Run a = measureOnce(rt, dev, plan, rows, act, warmup, forceIters, forced);
  if (a.us <= 0.0)
    return a;
  Run b = measureOnce(rt, dev, plan, rows, act, warmup, forceIters, forced);
  if (b.us <= 0.0)
    return b;
  Run r;
  r.us = std::min(a.us, b.us);
  r.spreadUs = std::fabs(a.us - b.us);
  return r;
}

} // namespace

int LitertPeak::runActivation(const LitertRuntime &rt, const litert_device_info_t &dev,
                              benchmark_config_t &cfg)
{
  (void)cfg;
  const LitertPlan plan = litertPlanFor(LitertFormat::Fp16, dev.accel);
  const uint64_t elem = litertElemBytes(plan.act, 1);

  auto test = currentDeviceScope->beginTest(
      {"litert_activation", "LiteRT activation throughput", "bps", Category::Bandwidth,
       "How fast this accelerator applies the operations between the matrix "
       "multiplies -- the feed-forward gate, softmax, layer normalisation -- "
       "as the bandwidth each achieves over a transformer-shaped tensor, at "
       "the same three working-set sizes the resident-weight rows use.  Each "
       "is net of reading, scaling and reducing the same tensor with nothing "
       "applied, so it is the operation's own cost.",
       TestShape::Heterogeneous, "operation and working set"});

  struct Ref
  {
    bool tried = false;
    Run run;
  };
  Ref refs[sizeof(kSizes) / sizeof(kSizes[0])];

  for (const Variant &v : kVariants)
  {
    for (size_t si = 0; si < sizeof(kSizes) / sizeof(kSizes[0]); si++)
    {
      if (clpeak::cancelRequested())
        break;
      const Size &sz = kSizes[si];
      const std::string metric = std::string(v.label) + "_" + sz.label;
      const std::string note = std::string(sz.label) + " of activations -- " + v.note +
                               "  Net of the read, scale and reduction around it.";
      // The working set is sized in the bytes the accelerator streams; the
      // GPU's fp16 policy keeps an fp32 graph's activations in half.
      const int64_t rows = (int64_t)(sz.bytes / (2ull * (uint64_t)kCols));
      const uint64_t graphBytes = (uint64_t)rows * (uint64_t)kCols * elem;

      if (graphBytes * kCopiesAtPeak > maxTensorBytes())
      {
        test.skip(metric, ResultStatus::Unsupported, "not enough memory for this working set", note);
        continue;
      }

      Ref &ref = refs[si];
      if (!ref.tried)
      {
        ref.tried = true;
        ref.run = measure(rt, dev, plan, rows, LitertActivation::None, warmupCount, forceIters, specifiedIters);
        CLPEAK_VLOG("litert-activation[%s]: %s reference %.1f us (spread %.1f us)\n",
                    dev.displayName.c_str(), sz.label, ref.run.us, ref.run.spreadUs);
      }
      if (ref.run.us <= 0.0)
      {
        test.skip(metric, ref.run.status,
                  "reference graph failed: " + (ref.run.error.empty() ? std::string("run failed") : ref.run.error),
                  note);
        continue;
      }

      Run full = measure(rt, dev, plan, rows, v.act, warmupCount, forceIters, specifiedIters);
      if (full.us <= 0.0)
      {
        test.skip(metric, full.status, full.error.empty() ? "run failed" : full.error, note);
        continue;
      }
      const double opUs = full.us - ref.run.us;
      const double noiseUs = std::max(full.spreadUs, ref.run.spreadUs);
      if (opUs < kMinOpShare * full.us || opUs < kNoiseFactor * noiseUs)
      {
        CLPEAK_VLOG("litert-activation[%s]: %s %s: %.1f us against a %.1f us reference "
                    "(spread %.1f us), too close to measure\n",
                    dev.displayName.c_str(), v.label, sz.label, full.us, ref.run.us, noiseUs);
        test.skip(metric, ResultStatus::Error,
                  opUs < kMinOpShare * full.us
                      ? "the operation costs less than a tenth of the reference graph around it, "
                        "too close to the noise to report"
                      : "the operation's cost (" + std::to_string((long long)opUs) +
                            " us) is within the graphs' own session-to-session spread (" +
                            std::to_string((long long)noiseUs) +
                            " us), so it cannot be resolved -- on an accelerator that fuses it "
                            "into the passes around it, it costs nothing",
                  note);
        continue;
      }
      // Read once, written once.
      const double bps = 2.0 * (double)sz.bytes / (opUs * 1.0e-6);
      CLPEAK_VLOG("litert-activation[%s]: %s %s -> %.2f GB/s (%.1f us, ref %.1f us, spread %.1f us)\n",
                  dev.displayName.c_str(), v.label, sz.label, bps / 1.0e9, full.us, ref.run.us, noiseUs);
      // The reference reads, writes and reads the tensor again in refUs; an
      // operation far above that rate was fused into the passes around it
      // or stayed in cache -- see src/coreml/activation.cpp.
      const double refStreamBps = 3.0 * (double)sz.bytes / (ref.run.us * 1.0e-6);
      std::string row = note;
      if (bps > 2.0 * refStreamBps)
        row += "  Faster than this accelerator streamed the same tensor in the reference "
               "graph, so the operation was fused into the passes around it or the working "
               "set stayed in cache: this is the rate of its arithmetic, not of memory.";
      test.emit(metric, (float)bps, row.c_str());
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
