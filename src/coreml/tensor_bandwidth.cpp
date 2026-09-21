#ifdef ENABLE_COREML

// coreml-tensor-bw: how fast this compute unit streams weights out of its
// own memory.
//
// One matrix-vector product, y = x * W, against a resident fp16 weight
// matrix -- the operation generating a token performs and the one every
// runtime tunes hardest.  W is a model constant, so it sits wherever Core ML
// keeps weights after the model loads; the vector is one row, so the return
// trip is free; at two operations per weight the arithmetic cannot be the
// limit.  The size climbs until the rate stops falling, which is where the
// working set has left fast local memory -- on the Neural Engine, where a
// model stopped fitting in its SRAM -- and every rung is named for its size
// so it means the same thing on every device.
//
// Submission overhead is subtracted: the same graph is timed once with a
// weight small enough that its transfer cannot matter, and that floor is
// taken off every rung.  The design and its reasons are the ONNX backend's
// (src/onnx/tensor_bandwidth.cpp).

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

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

Result measure(const coreml_device_info_t &dev, int spec, int64_t d, unsigned warmup,
               bool forceIters, unsigned forced)
{
  Result r;
  std::string err;
  auto s = CoremlSession::create(dev, coremlGemvModel(spec, d, d, 0x243f6a88u), err);
  if (!s)
  {
    r.error = err;
    r.status = ResultStatus::Unsupported;
    return r;
  }
  const double createUs = coremlCreateUs(*s);
  CLPEAK_VLOG("coreml-tensor-bw[%s]: %lld dim create %.2f s\n", dev.displayName.c_str(),
              (long long)d, createUs / 1.0e6);
  if (!s->onDevice())
  {
    r.error = coremlOffDeviceReason(dev, *s);
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (createUs > kCoremlMaxCreateUs)
  {
    r.error = "model creation took " + std::to_string((long long)(createUs / 1.0e6)) +
              " s, exceeds " + std::to_string((long long)(kCoremlMaxCreateUs / 1.0e6)) +
              " s compilation budget";
    r.status = ResultStatus::Unsupported;
    return r;
  }
  void *x = s->bindInput("x", CML_FP16, {1, d}, (size_t)d * 2, err);
  if (!x)
  {
    r.error = err;
    r.status = ResultStatus::Error;
    return r;
  }
  {
    const uint16_t half = coremlFloatToHalf(0.5f);
    uint16_t *p = static_cast<uint16_t *>(x);
    for (int64_t i = 0; i < d; i++)
      p[i] = half;
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

// The per-dispatch floor: the same graph at 256, on whichever unit the
// planner gives it -- placement is not enforced here, since the point is the
// cost of asking and a refused row would leave every rung gross of it.
Result measureFloor(const coreml_device_info_t &dev, int spec, unsigned warmup,
                    bool forceIters, unsigned forced)
{
  Result r;
  std::string err;
  auto s = CoremlSession::create(dev, coremlGemvModel(spec, kFloorDim, kFloorDim, 0x243f6a88u), err);
  if (!s)
  {
    r.error = err;
    return r;
  }
  void *x = s->bindInput("x", CML_FP16, {1, kFloorDim}, (size_t)kFloorDim * 2, err);
  if (!x)
    return r;
  const uint16_t half = coremlFloatToHalf(0.5f);
  for (int64_t i = 0; i < kFloorDim; i++)
    static_cast<uint16_t *>(x)[i] = half;
  auto m = coremlMeasure(*s, warmup, 200000, forceIters, forced, 200);
  r.us = m.meanUs;
  return r;
}

} // namespace

int CoreMLPeak::runTensorBandwidth(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_tensor_bw", "Core ML resident-weight bandwidth", "bps", Category::Bandwidth,
       "How fast this compute unit streams a weight matrix it already holds "
       "through a matrix-vector product -- the shape of generating one token -- "
       "at growing sizes, net of the cost of asking.  Where the rate drops is "
       "where a model stopped fitting in fast local memory: on the Neural "
       "Engine that is its on-chip SRAM, and it decides whether a model "
       "streams from main memory on every token.",
       TestShape::Heterogeneous, "working set"});

  // The floor is the smallest graph's whole time.  The planner may keep a
  // 128 KB product on the CPU whatever unit was asked for, in which case the
  // floor is the CPU's and slightly understates an accelerator's own -- a
  // bias of tens of microseconds against rungs that take milliseconds.
  Result floor = measureFloor(dev, spec, warmupCount, forceIters, specifiedIters);
  const double floorUs = floor.us > 0.0 ? floor.us : 0.0;
  CLPEAK_VLOG("coreml-tensor-bw[%s]: per-dispatch floor %.1f us\n", dev.displayName.c_str(), floorUs);

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
    const uint64_t bytes = (uint64_t)s.dim * (uint64_t)s.dim * 2;
    // The weights exist twice while the model is built and written.
    if (bytes * 2 > clpeak::memoryBudget(3ull << 30))
    {
      if (rung < kAlwaysMeasured)
        test.skip(s.label, ResultStatus::Unsupported, "not enough memory for this working set", s.note);
      break;
    }
    Result r = measure(dev, spec, s.dim, warmupCount, forceIters, specifiedIters);
    if (r.us <= 0.0)
    {
      test.skip(s.label, r.status, r.error.empty() ? "run failed" : r.error, s.note);
      // A small rung the planner keeps off this unit says nothing about the
      // larger ones -- Core ML sends the Neural Engine a 128 MB matrix-vector
      // product and keeps an 8 MB one on the CPU -- so the base rungs are
      // each tried; past them a failure ends the ladder.
      if (rung + 1 < kAlwaysMeasured && r.status == ResultStatus::Unsupported)
        continue;
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
    CLPEAK_VLOG("coreml-tensor-bw[%s]: %s -> %.2f GB/s (%.1f us net of %.1f)\n", dev.displayName.c_str(),
                s.label, bps / 1.0e9, netUs, floorUs);
    stillFalling = (prevBps <= 0.0) || (bps < prevBps * kFallingRatio);
    prevBps = bps;
    test.emit(s.label, (float)bps, s.note);
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
