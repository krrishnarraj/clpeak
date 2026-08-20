#ifdef ENABLE_ONNX

// onnx-tensor-bw: how fast this execution provider can stream weights out of
// its own memory.
//
// The graph is a chain of matrix-vector products, y <- y * W, against one
// square fp16 weight matrix held as an initializer.  Several properties make
// that the right probe.  W being a constant means it sits wherever the
// provider keeps weights after session creation, so this measures the
// provider's read path and not the host handing a buffer over; the vector is
// one row, so the return trip is free; and at two operations per weight the
// arithmetic cannot be the limit, which leaves memory as the only thing
// being measured.
//
// One matmul per dispatch, deliberately.  Chaining several against the same
// weights would amortise submission overhead -- but the Apple Neural Engine
// refuses a chained program outright (ANEProgramProcessRequestDirect fails)
// while running the single-op form happily, so the portable shape wins.
//
// Submission overhead is therefore subtracted instead.  The same graph is
// timed once with a weight matrix small enough that its transfer cannot
// matter, and that floor is taken off every rung.  Without it the ladder
// reads backwards on hardware that is slow to accept work: an RTX 5060
// charges ~17 us per dispatch, which is half the time eight megabytes takes
// to move, and it reported 219 / 265 / 392 GB/s across a ladder whose small
// end should be the fastest.
//
// It has to be a matmul rather than something simpler.  An elementwise-plus-
// reduction graph measures the reduction: on this Mac it reads ~22 GB/s
// against a machine that does roughly ten times that, because reductions are
// not what any provider optimises.  A GEMV is the exact operation generating
// a token performs, and every provider tunes it hard.
//
// Sweeping the size is what makes this worth having.  Accelerators put a
// small, fast local memory in front of DRAM -- SRAM on an NPU, cache on a
// CPU or GPU -- and the size where the rate falls away is where a model
// stopped fitting in it.  That cliff decides whether a given model streams
// its weights from DRAM on every single token.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace
{

struct Size
{
  int64_t     dim;         // square weight matrix [dim, dim]; bytes = dim^2 * 2
  const char *label;
  const char *note;
};

// The ladder climbs until the rate stops falling, which is where the working
// set has left the last level of cache and main memory is all that is left.
//
// It is open-ended on purpose.  A fixed top rung has to be raised as caches
// grow -- 128 MB is already inside a single AMD Infinity Cache -- and raising
// it turns "the main-memory rate" into a different measurement under the same
// name.  Climbing until the curve flattens finds main memory on any device,
// now and later, and every rung stays comparable because a rung is named for
// its working-set size rather than for its position in the list.
const Size kSizes[] = {
  {2048, "8mb",
   "Eight megabytes of weights -- small enough to sit in fast local memory on "
   "most devices, so this is usually the fastest rung."},
  {4096, "32mb",
   "Thirty-two megabytes -- around the size of a large cache or an NPU's local memory."},
  {8192, "128mb",
   "128 megabytes -- past the cache of most devices, though not all."},
  {16384, "512mb",
   "512 megabytes -- beyond any cache shipping today, so this is main memory "
   "on anything that still shows a falling rate by this point."},
  {32768, "2gb",
   "Two gigabytes.  Only reached by a device whose rate was still dropping at "
   "512 MB, meaning a cache larger than anything current."},
};

// A rung is only worth trying if the one before it was still meaningfully
// faster; once the curve flattens, main memory has been found.
constexpr double kFallingRatio = 0.9;

// One matrix-vector product against a square fp16 weight matrix.
std::string streamModel(int64_t d)
{
  std::string w((size_t)d * d * 2, '\0');
  {
    // Uniform over +/-sqrt(3/d) so the result keeps the vector's magnitude:
    // a d-deep fp16 dot product of larger values would saturate.
    const float lim = std::sqrt(3.0f / (float)d);
    uint16_t *h = reinterpret_cast<uint16_t *>(&w[0]);
    uint32_t s = 0x243f6a88u;
    for (int64_t i = 0; i < d * d; i++)
    {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      h[i] = floatToHalf(((float)(s >> 8) / 16777216.0f - 0.5f) * 2.0f * lim);
    }
  }

  return onnxMatMulModel(1, d, d, ONNX_DT_FLOAT16, w);
}

struct Result
{
  double us = -1.0;              // mean time for one dispatch of this size
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

// Weight matrix used to measure the per-dispatch floor: 128 KB moves in well
// under a microsecond on anything here, so essentially all of its time is the
// cost of asking.
constexpr int64_t kFloorDim = 256;

Result measure(const OrtRuntime &rt, const onnx_ep_info_t &ep, int64_t d,
               unsigned int warmup, bool forceIters, unsigned int forced)
{
  Result r;

  OrtSession *session = nullptr;
  {
    std::string model = streamModel(d);
    auto ses = onnxCreateSession(rt, ep, model);
    model.clear();
    model.shrink_to_fit();
    if (!ses.session)
    {
      r.error  = ses.error;
      r.status = ResultStatus::Unsupported;
      return r;
    }
    session = ses.session;
  }

  std::vector<uint16_t> xBuf((size_t)d, floatToHalf(0.5f));
  std::vector<uint16_t> yBuf((size_t)d, 0);

  OrtMemoryInfo *mi = nullptr;
  OrtValue *inVal = nullptr, *outVal = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  const int64_t inShape[2]  = {1, d};
  const int64_t outShape[2] = {1, d};
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, xBuf.data(), xBuf.size() * 2, inShape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, yBuf.data(), yBuf.size() * 2, outShape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &outVal);
  if (mi) rt.api->ReleaseMemoryInfo(mi);

  auto run = [&](unsigned int n) -> double {
    static const char *ins[]  = {"A"};
    static const char *outs[] = {"C"};
    auto a = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n; i++)
    {
      OrtStatus *rs = rt.api->Run(session, nullptr, ins,
                                  (const OrtValue *const *)&inVal, 1,
                                  outs, 1, &outVal);
      if (rs)
      {
        r.error = onnxStatusText(rt, rs);
        return -1.0;
      }
    }
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - a).count() / (double)n;
  };

  if (!st && run(1 + warmup) > 0.0)
  {
    double probe = run(3);
    if (probe > 0.0)
    {
      unsigned int iters = pickIters(probe, 1000000u, forceIters ? forced : 0);
      r.us = run(iters);
    }
  }
  if (st)
    r.error = onnxStatusText(rt, st);
  if (r.us <= 0.0 && r.status == ResultStatus::Ok)
    r.status = ResultStatus::Error;

  if (inVal)  rt.api->ReleaseValue(inVal);
  if (outVal) rt.api->ReleaseValue(outVal);
  rt.api->ReleaseSession(session);
  return r;
}

} // namespace

int OnnxPeak::runTensorBandwidth(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                                 benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx-tensor-bw", "ONNX resident-tensor bandwidth", "gbps",
       Category::Bandwidth,
       "How fast this provider streams model weights out of its own memory, "
       "measured on the exact operation that generating a token performs -- "
       "one row of numbers multiplied through a weight matrix -- at three "
       "weight sizes.  Producing each word of text requires reading every "
       "weight, so this is the ceiling on how fast words can appear.  The "
       "size at which the rate drops is the size at which a model stopped "
       "fitting in the device's fast local memory; past that point its "
       "weights come from main memory on every single token.  The fixed cost "
       "of handing work to the device is measured separately and subtracted, "
       "so these are transfer rates rather than round-trip times."});

  // The floor first: every rung is reported net of it.
  Result floor = measure(rt, ep, kFloorDim, warmupCount, forceIters,
                         specifiedIters);
  const double floorUs = (floor.us > 0.0) ? floor.us : 0.0;
  CLPEAK_VLOG("onnx-tensor-bw[%s]: dispatch floor %.2f us\n",
              ep.providerKey.c_str(), floorUs);

  double prevGbps = 0.0;
  bool   stillFalling = true;
  for (const Size &s : kSizes)
  {
    if (clpeak::cancelRequested())
      break;

    // Stop once the curve has flattened: the rungs above only exist for
    // devices whose caches are larger than the ones below.
    if (prevGbps > 0.0 && !stillFalling)
    {
      CLPEAK_VLOG("onnx-tensor-bw[%s]: rate flattened, stopping below %s\n",
                  ep.providerKey.c_str(), s.label);
      break;
    }

    Result r = measure(rt, ep, s.dim, warmupCount, forceIters, specifiedIters);
    if (r.us <= 0.0)
    {
      // A rung past the standard three is exploratory: a device that cannot
      // allocate it has simply run out of ladder, which is not a failure
      // worth a row.
      if (s.dim > 8192)
      {
        CLPEAK_VLOG("onnx-tensor-bw[%s]: %s unavailable (%s)\n",
                    ep.providerKey.c_str(), s.label, r.error.c_str());
        break;
      }
      test.skip(s.label, r.status, r.error.empty() ? "run failed" : r.error,
                s.note);
      continue;
    }

    const double netUs = r.us - floorUs;
    if (netUs <= 0.0)
    {
      // The transfer is lost inside the cost of asking; no honest rate to
      // report, and reporting the raw one would be a submission benchmark
      // wearing a bandwidth label.
      test.skip(s.label, ResultStatus::Error,
                "too small to measure against this provider's dispatch cost",
                s.note);
      continue;
    }
    const double bytes = (double)s.dim * (double)s.dim * 2.0;
    const double gbps  = bytes / (netUs * 1.0e-6) / 1.0e9;
    test.emit(s.label, (float)gbps, s.note);

    stillFalling = (prevGbps <= 0.0) || (gbps < prevGbps * kFallingRatio);
    prevGbps = gbps;
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
