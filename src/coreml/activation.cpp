#ifdef ENABLE_COREML

// coreml-activation: how fast the operations *between* the matrix multiplies
// run -- softmax, layer normalisation, the SiLU gate.  None of them does
// meaningful arithmetic: they read a tensor and write one back, so their
// ceiling is memory bandwidth and their rate is reported as the bandwidth
// they achieve.  These are also the operations the Neural Engine is known
// to apply at a fraction of the rate it streams weights, and the ones most
// likely to be handed to another compute unit -- which the plan reports.
//
// Each rate is net of a reference graph that reads, scales and reduces the
// same constant with no operation applied, timed once per size and shared
// by the three operations.  Every rung is reported at the three working-set
// sizes coreml-tensor-bw always measures, so the two ladders divide row for
// row; see src/onnx/activation.cpp for why a single number per operation
// cannot be made honest.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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
constexpr uint64_t kCopiesAtPeak = 3;
constexpr unsigned int kSizeBudgetUs = 1000000;

// The share of the work the operation itself has to account for before the
// difference is worth reporting; see the ONNX test for the calibration.
constexpr double kMinOpShare = 0.10;

// And the difference has to clear the graphs' own session-to-session spread
// by this factor.  The share alone is not enough: the GPU fuses SiLU into
// the passes around it, so its true cost is nothing, and two sessions of
// the same 128 MB graph differ by 15-30% on an M1 Pro (1336 us against
// 1627 us for one and the same model) -- a difference that clears a tenth
// by luck divides 268 MB by a few hundred microseconds and publishes 1.35
// TB/s for an operation that cost nothing.  So every graph, reference and
// operation alike, is built and timed twice: the *minimum* of each pair is
// the estimate (a slow session is discarded, and a spuriously fast one can
// only shrink the difference), and the larger of the two spreads is the
// noise floor the difference has to rise above.
constexpr double kNoiseFactor = 3.0;

struct Variant
{
  CoremlActivation act;
  const char *label;
  const char *note;
};

const Variant kVariants[] = {
    {CoremlActivation::Silu, "silu",
     "x times sigmoid(x) -- the gate in the feed-forward network of most "
     "current language models."},
    {CoremlActivation::Softmax, "softmax",
     "Softmax across the row, at the heart of attention: two passes and a "
     "maximum before it can divide."},
    {CoremlActivation::LayerNorm, "layernorm",
     "Mean and variance across each row, then rescale -- every transformer "
     "layer does this at least twice."},
};

struct Run
{
  double us = -1.0;         // the faster of the two sessions
  double spreadUs = 0.0;    // how far apart they were
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Run measureOnce(const coreml_device_info_t &dev, int spec, int64_t rows, CoremlActivation act,
            unsigned warmup, bool forceIters, unsigned forced)
{
  Run r;
  std::string err;
  auto s = CoremlSession::create(dev, coremlActivationModel(spec, rows, kCols, act), err);
  if (!s)
  {
    r.error = err;
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (!s->onDevice())
  {
    r.error = coremlOffDeviceReason(dev, *s);
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (!coremlBindScalar(*s, "s", CML_FP16, err))
  {
    r.error = err;
    r.status = ResultStatus::Error;
    return r;
  }
  auto m = coremlMeasure(*s, warmup, kSizeBudgetUs, forceIters, forced);
  if (m.meanUs <= 0.0)
  {
    r.error = m.error;
    r.status = m.status;
    return r;
  }
  r.us = m.meanUs;
  return r;
}

// Two sessions of the same graph; see kNoiseFactor.
Run measure(const coreml_device_info_t &dev, int spec, int64_t rows, CoremlActivation act,
            unsigned warmup, bool forceIters, unsigned forced)
{
  Run a = measureOnce(dev, spec, rows, act, warmup, forceIters, forced);
  if (a.us <= 0.0)
    return a;
  Run b = measureOnce(dev, spec, rows, act, warmup, forceIters, forced);
  if (b.us <= 0.0)
    return b;
  Run r;
  r.us = std::min(a.us, b.us);
  r.spreadUs = std::fabs(a.us - b.us);
  return r;
}

} // namespace

int CoreMLPeak::runActivation(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_activation", "Core ML activation throughput", "bps", Category::Bandwidth,
       "How fast this compute unit applies the operations between the matrix "
       "multiplies -- the feed-forward gate, softmax, layer normalisation -- "
       "as the bandwidth each achieves over a transformer-shaped tensor, at "
       "the same three working-set sizes the resident-weight rows use.  Each "
       "is net of reading, scaling and reducing the same tensor with nothing "
       "applied, so it is the operation's own cost.",
       TestShape::Heterogeneous, "operation and working set"});

  // Reference per size, shared by the three operations.
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
      const int64_t rows = (int64_t)(sz.bytes / (2ull * (uint64_t)kCols));

      if (sz.bytes * kCopiesAtPeak > maxTensorBytes())
      {
        test.skip(metric, ResultStatus::Unsupported, "not enough memory for this working set", note);
        continue;
      }

      Ref &ref = refs[si];
      if (!ref.tried)
      {
        ref.tried = true;
        ref.run = measure(dev, spec, rows, CoremlActivation::None, warmupCount, forceIters, specifiedIters);
        CLPEAK_VLOG("coreml-activation[%s]: %s reference %.1f us (spread %.1f us)\n",
                    dev.displayName.c_str(), sz.label, ref.run.us, ref.run.spreadUs);
      }
      if (ref.run.us <= 0.0)
      {
        test.skip(metric, ref.run.status,
                  "reference graph failed: " + (ref.run.error.empty() ? std::string("run failed") : ref.run.error),
                  note);
        continue;
      }

      Run full = measure(dev, spec, rows, v.act, warmupCount, forceIters, specifiedIters);
      if (full.us <= 0.0)
      {
        test.skip(metric, full.status, full.error.empty() ? "run failed" : full.error, note);
        continue;
      }
      const double opUs = full.us - ref.run.us;
      const double noiseUs = std::max(full.spreadUs, ref.run.spreadUs);
      if (opUs < kMinOpShare * full.us || opUs < kNoiseFactor * noiseUs)
      {
        CLPEAK_VLOG("coreml-activation[%s]: %s %s: %.1f us against a %.1f us reference "
                    "(spread %.1f us), too close to measure\n",
                    dev.displayName.c_str(), v.label, sz.label, full.us, ref.run.us, noiseUs);
        test.skip(metric, ResultStatus::Error,
                  opUs < kMinOpShare * full.us
                      ? "the operation costs less than a tenth of the reference graph around it, "
                        "too close to the noise to report"
                      : "the operation's cost (" + std::to_string((long long)opUs) +
                            " us) is within the graphs' own session-to-session spread (" +
                            std::to_string((long long)noiseUs) +
                            " us), so it cannot be resolved -- on a unit that fuses it into "
                            "the passes around it, it costs nothing",
                  note);
        continue;
      }
      // Read once, written once.
      const double bps = 2.0 * (double)sz.bytes / (opUs * 1.0e-6);
      CLPEAK_VLOG("coreml-activation[%s]: %s %s -> %.2f GB/s (%.1f us, ref %.1f us, spread %.1f us)\n",
                  dev.displayName.c_str(), v.label, sz.label, bps / 1.0e9, full.us, ref.run.us, noiseUs);
      // The reference reads, writes and reads the tensor again in refUs, so
      // three passes over it in that time is a rate this unit was actually
      // seen to stream at.  An operation reporting far above it did not
      // stream two passes of its own: the unit fused it into the passes
      // around it (the M1 Pro's GPU applies SiLU to 128 MB in ~130 us,
      // which as a bandwidth would be two terabytes a second), or the
      // working set stayed in cache.  The number is still the rate at which
      // the data got the operation applied, and is reported; the note says
      // what it is not.
      const double refStreamBps = 3.0 * (double)sz.bytes / (ref.run.us * 1.0e-6);
      std::string row = note;
      if (bps > 2.0 * refStreamBps)
        row += "  Faster than this unit streamed the same tensor in the reference graph, "
               "so the operation was fused into the passes around it or the working set "
               "stayed in cache: this is the rate of its arithmetic, not of memory.";
      test.emit(metric, (float)bps, row.c_str());
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
