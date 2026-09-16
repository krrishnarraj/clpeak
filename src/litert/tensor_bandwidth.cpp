#ifdef ENABLE_LITERT

// litert-tensor-bw: how fast this accelerator streams weights out of its own
// memory.
//
// One matrix-vector product, y = x * W, against a resident fp16 weight
// matrix -- the operation generating a token performs and the one every
// runtime tunes hardest.  W is a model constant, so it sits wherever the
// accelerator keeps weights after the model compiles (XNNPACK's packed
// copy, the GPU's buffer); the vector is one row, so the return trip is
// free; at two operations per weight the arithmetic cannot be the limit.
// The size climbs until the rate stops falling, which is where the working
// set has left fast local memory, and every rung is named for its size so
// it means the same thing on every device.
//
// Submission overhead is subtracted: the same graph is timed once with a
// weight small enough that its transfer cannot matter, and that floor is
// taken off every rung.  The design and its reasons are the ONNX backend's
// (src/onnx/tensor_bandwidth.cpp).

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <cstring>
#include <string>
#include <vector>

namespace
{

struct Size
{
  int64_t dim;
  const char *label;
  const char *note;
};

const Size kSizes[] = {
    {2048, "8mb",
     "Eight megabytes of weights -- small enough to sit in fast local memory "
     "on most devices, so this is usually the fastest rung."},
    {4096, "32mb", "Thirty-two megabytes -- around the size of a large cache or a device's fast local memory."},
    {8192, "128mb", "128 megabytes -- past the cache of most devices, though not all."},
    {16384, "512mb",
     "512 megabytes -- beyond any cache shipping today, so this is main memory "
     "on anything that still shows a falling rate by this point."},
    {32768, "2gb",
     "Two gigabytes.  Only reached by a device whose rate was still dropping at "
     "512 MB, meaning a cache larger than anything current."},
};

constexpr double kFallingRatio = 0.9;
constexpr size_t kAlwaysMeasured = 3;
constexpr int64_t kFloorDim = 256;
constexpr unsigned int kSizeBudgetUs = 1000000;

struct Result
{
  double us = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Result measure(const LitertRuntime &rt, const litert_device_info_t &dev, const LitertPlan &plan,
               int64_t d, unsigned warmup, bool forceIters, unsigned forced, unsigned budgetUs,
               unsigned maxIters, bool enforcePlacement)
{
  Result r;
  std::string err;
  auto s = LitertSession::create(rt, dev, litertGemvModel(plan, d, d, 0x243f6a88u), litertConfigFor(plan), err);
  if (!s)
  {
    r.error = err;
    r.status = ResultStatus::Unsupported;
    return r;
  }
  CLPEAK_VLOG("litert-tensor-bw[%s]: %lld dim create %.3f s\n", dev.displayName.c_str(), (long long)d,
              s->createUs / 1.0e6);
  if (enforcePlacement && !s->onDevice())
  {
    r.error = s->offDevice();
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (s->createUs > kLitertMaxCreateUs)
  {
    r.error = "model creation took " + std::to_string((long long)(s->createUs / 1.0e6)) +
              " s, exceeds " + std::to_string((long long)(kLitertMaxCreateUs / 1.0e6)) +
              " s compilation budget";
    r.status = ResultStatus::Unsupported;
    return r;
  }
  std::string x;
  for (int64_t i = 0; i < d; i++)
    x += litertScalarBytes(plan.act, 0.5f);
  if (!s->writeInput(0, x.data(), x.size(), err))
  {
    r.error = err;
    r.status = ResultStatus::Error;
    return r;
  }
  auto m = litertMeasure(*s, warmup, budgetUs, forceIters, forced, maxIters);
  if (m.meanUs <= 0.0)
  {
    r.error = m.error;
    r.status = m.status;
    return r;
  }
  r.us = m.meanUs;
  return r;
}

} // namespace

int LitertPeak::runTensorBandwidth(const LitertRuntime &rt, const litert_device_info_t &dev,
                                   benchmark_config_t &cfg)
{
  (void)cfg;
  const LitertPlan plan = litertPlanFor(LitertFormat::Fp16, dev.accel);

  auto test = currentDeviceScope->beginTest(
      {"litert_tensor_bw", "LiteRT resident-weight bandwidth", "bps", Category::Bandwidth,
       "How fast this accelerator streams a weight matrix it already holds "
       "through a matrix-vector product -- the shape of generating one token -- "
       "at growing sizes, net of the cost of asking.  Where the rate drops is "
       "where a model stopped fitting in fast local memory, and it decides "
       "whether a model streams from main memory on every token.",
       TestShape::Heterogeneous, "working set"});

  // The floor is the smallest graph's whole time.
  Result floor = measure(rt, dev, plan, kFloorDim, warmupCount, forceIters, specifiedIters, 200000, 200, false);
  const double floorUs = floor.us > 0.0 ? floor.us : 0.0;
  CLPEAK_VLOG("litert-tensor-bw[%s]: per-dispatch floor %.1f us\n", dev.displayName.c_str(), floorUs);

  double prevBps = 0.0;
  bool stillFalling = true;
  size_t index = 0;
  for (const Size &s : kSizes)
  {
    if (clpeak::cancelRequested())
      break;
    const size_t rung = index++;
    if (rung >= kAlwaysMeasured && prevBps > 0.0 && !stillFalling)
      break;
    // The bytes streamed per pass: the fp16 plan's weights are half on
    // every accelerator (the GPU's stores them as half under its fp16
    // policy, and the model now carries them as half too).
    const uint64_t bytes = (uint64_t)s.dim * (uint64_t)s.dim * 2;
    // The weights exist in the model and again in the accelerator's copy.
    const uint64_t peak = litertWeightBytes(plan, s.dim, s.dim) + bytes;
    if (peak > clpeak::memoryBudget(3ull << 30))
    {
      if (rung < kAlwaysMeasured)
        test.skip(s.label, ResultStatus::Unsupported, "not enough memory for this working set", s.note);
      break;
    }
    Result r = measure(rt, dev, plan, s.dim, warmupCount, forceIters, specifiedIters, kSizeBudgetUs,
                       kLitertMaxIters, true);
    if (r.us <= 0.0)
    {
      test.skip(s.label, r.status, r.error.empty() ? "run failed" : r.error, s.note);
      break;
    }
    const double netUs = r.us - floorUs;
    if (netUs <= 0.0)
    {
      test.skip(s.label, ResultStatus::Error,
                "took no longer than the per-dispatch floor, so nothing measurable was streamed", s.note);
      break;
    }
    const double bps = (double)bytes / (netUs * 1.0e-6);
    CLPEAK_VLOG("litert-tensor-bw[%s]: %s -> %.2f GB/s (%.1f us net of %.1f)\n", dev.displayName.c_str(),
                s.label, bps / 1.0e9, netUs, floorUs);
    stillFalling = (prevBps <= 0.0) || (bps < prevBps * kFallingRatio);
    prevBps = bps;
    test.emit(s.label, (float)bps, s.note);
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
