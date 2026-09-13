#ifdef ENABLE_ONNX

// onnx-tensor-bw: how fast this execution provider can stream weights out of
// its own memory.
//
// The graph is one matrix-vector product, y = x * W, against a weight matrix
// held as an initializer.  Several properties make that the right probe.  W
// being a constant means it sits wherever the provider keeps weights after
// session creation, so this measures the provider's read path and not the
// host handing a buffer over; the vector is one row, so the return trip is
// free; and at two operations per weight the arithmetic cannot be the limit,
// which leaves memory as the only thing being measured.
//
// The element width is whichever of fp16 and fp32 the provider streams
// faster (onnxStreamDtype), not fp16 by assumption.  A provider without a
// native fp16 kernel for the operation converts every fp16 tensor on the way
// in and reports the conversion under this heading -- ONNX Runtime's CPU EP
// read a flat 2 GB/s at every size on a Threadripper whose fp32 decode row
// streams 38.  The rungs are named for their byte size, so a fp32 matrix has
// half the columns and every reading still means what its name says.
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
#include "onnx_probe.h"
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
  int64_t     dim;         // [dim, dim] in fp16, [dim, dim/2] in fp32: dim^2 * 2 bytes
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
   "Thirty-two megabytes -- around the size of a large cache or a device's fast local memory."},
  {8192, "128mb",
   "128 megabytes -- past the cache of most devices, though not all."},
  {16384, "512mb",
   "512 megabytes -- beyond any cache shipping today, so this is main memory "
   "on anything that still shows a falling rate by this point."},
  {32768, "2gb",
   "Two gigabytes.  Only reached by a device whose rate was still dropping at "
   "512 MB, meaning a cache larger than anything current."},
};

// A rung beyond the base three is only worth trying if the one before it was
// still meaningfully faster; once the curve flattens, main memory has been
// found.
constexpr double kFallingRatio = 0.9;

// The first three rungs always run.  The stop rule reads a flat curve as
// "main memory reached", which is true when the operation is limited by
// memory and false when it is limited by arithmetic -- and a provider slow
// enough for the latter is exactly the one whose curve is flat from the
// start.  A stock CPU build of ONNX Runtime 1.30 runs this at 2.2 GB/s at
// every size, and stopping after two rungs would have dropped the only
// reading taken at a size no cache could hold.
constexpr size_t kAlwaysMeasured = 3;

// Columns of the weight matrix for `d` rows: d^2 * 2 bytes either way.
int64_t streamCols(int64_t d, int dtype)
{
  return dtype == ONNX_DT_FLOAT ? d / 2 : d;
}

// One matrix-vector product against a [d, cols] weight matrix of `dtype`.
std::string streamModel(int64_t d, int dtype)
{
  const int64_t cols = streamCols(d, dtype);
  std::string w((size_t)onnxElemBytes(dtype, d * cols), '\0');
  {
    // Uniform over +/-sqrt(3/d) so the result keeps the vector's magnitude:
    // a d-deep fp16 dot product of larger values would saturate.
    const float lim = std::sqrt(3.0f / (float)d);
    float *f = reinterpret_cast<float *>(&w[0]);
    uint16_t *h = reinterpret_cast<uint16_t *>(&w[0]);
    uint32_t s = 0x243f6a88u;
    for (int64_t i = 0; i < d * cols; i++)
    {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      const float v = ((float)(s >> 8) / 16777216.0f - 0.5f) * 2.0f * lim;
      if (dtype == ONNX_DT_FLOAT)
        f[i] = v;
      else
        h[i] = floatToHalf(v);
    }
  }

  return onnxMatMulModel(1, d, cols, dtype, w);
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
               int dtype, unsigned int warmup, bool forceIters,
               unsigned int forced)
{
  Result r;
  const size_t es = (size_t)onnxElemBytes(dtype, 1);
  const int64_t cols = streamCols(d, dtype);

  OrtSession *session = nullptr;
  {
    std::string model = streamModel(d, dtype);
    auto createStart = std::chrono::steady_clock::now();
    auto ses = onnxCreateSession(rt, ep, model);
    auto createEnd = std::chrono::steady_clock::now();
    double createUs = std::chrono::duration<double, std::micro>(
                          createEnd - createStart).count();
    CLPEAK_VLOG("onnx-tensor-bw[%s]: %lld dim create %.1f s\n",
                ep.providerKey.c_str(), (long long)d, createUs / 1.0e6);
    model.clear();
    model.shrink_to_fit();
    if (!ses.session)
    {
      r.error  = ses.error;
      r.status = ResultStatus::Unsupported;
      return r;
    }
    if (createUs > kOnnxMaxCreateUs)
    {
      CLPEAK_VLOG("onnx-tensor-bw[%s]: %lld dim create %.1f s > %.1f s, skipping\n",
                  ep.providerKey.c_str(), (long long)d,
                  createUs / 1.0e6, kOnnxMaxCreateUs / 1.0e6);
      r.error = "session creation took " +
                std::to_string((long long)(createUs / 1.0e6)) +
                " s, exceeds " +
                std::to_string((long long)(kOnnxMaxCreateUs / 1.0e6)) +
                " s compilation budget";
      r.status = ResultStatus::Unsupported;
      rt.api->ReleaseSession(ses.session);
      return r;
    }
    session = ses.session;
  }

  std::vector<uint8_t> xBuf((size_t)d * es, 0);
  std::vector<uint8_t> yBuf((size_t)cols * es, 0);
  {
    const std::string half = onnxFloatScalar(0.5f, dtype);
    for (int64_t i = 0; i < d; i++)
      std::memcpy(&xBuf[(size_t)i * es], half.data(), es);
  }

  OrtMemoryInfo *mi = nullptr;
  OrtValue *inVal = nullptr, *outVal = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  const int64_t inShape[2]  = {1, d};
  const int64_t outShape[2] = {1, cols};
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, xBuf.data(), xBuf.size(), inShape, 2,
        (ONNXTensorElementDataType)dtype, &inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, yBuf.data(), yBuf.size(), outShape, 2,
        (ONNXTensorElementDataType)dtype, &outVal);
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
    double probe = run(1);
    if (probe > 0.0)
    {
      unsigned int iters = pickIters(probe, 1000000u, forceIters ? forced : 0,
                                     kOnnxMaxIters);
      // The probe was one whole pass; when the budget affords only one, it
      // already is the measurement.
      r.us = (iters > 1) ? run(iters) : probe;
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
      {"onnx_tensor_bw", "ONNX resident-tensor bandwidth", "bps",
       Category::Bandwidth,
       "How fast this provider streams weights out of its own memory, on the "
       "operation generating a token performs: one row of numbers multiplied "
       "through a weight matrix, in whichever of fp16 and fp32 it streams "
       "faster.  The size at which the rate drops is where a model stops "
       "fitting in fast local memory, and the cost of handing work over is "
       "subtracted, so these are transfer rates rather than round trips.",
       // The point of the ladder is where the rate drops, so the rungs are
       // not interchangeable and the fastest of them is not the answer.
       TestShape::Heterogeneous, "weight size"});

  // The width this provider streams fastest, decided once per provider and
  // shared with the activation rows so the two ladders divide row for row.
  const int dtype = onnxStreamDtype(rt, ep);
  const char *widthNote =
      (dtype == ONNX_DT_FLOAT)
          ? "  Measured in fp32, which this provider streams faster than fp16."
          : "";

  // The floor first: every rung is reported net of it.
  Result floor = measure(rt, ep, kFloorDim, dtype, warmupCount, forceIters,
                         specifiedIters);
  const double floorUs = (floor.us > 0.0) ? floor.us : 0.0;
  CLPEAK_VLOG("onnx-tensor-bw[%s]: dispatch floor %.2f us\n",
              ep.providerKey.c_str(), floorUs);

  double prevBps = 0.0;
  bool   stillFalling = true;
  size_t index = 0;
  for (const Size &s : kSizes)
  {
    const size_t rung = index++;
    if (clpeak::cancelRequested())
      break;

    // Stop once the curve has flattened, but only past the base rungs: those
    // exist for devices whose caches are larger, while the base three are
    // what every reading is compared against.
    if (rung >= kAlwaysMeasured && prevBps > 0.0 && !stillFalling)
    {
      CLPEAK_VLOG("onnx-tensor-bw[%s]: rate flattened, stopping below %s\n",
                  ep.providerKey.c_str(), s.label);
      break;
    }

    // Three times the weight matrix is what exists at peak: the raw values and
    // the model embedding them overlap while the model is built, and the model
    // and ORT's copy overlap while the session is created.  The first three
    // rungs are otherwise unconditional, and on a phone an over-optimistic
    // estimate is a kill rather than a failed allocation.
    if ((uint64_t)s.dim * (uint64_t)s.dim * 2ull * 3ull >
        clpeak::memoryBudget(2ull << 30))
    {
      CLPEAK_VLOG("onnx-tensor-bw[%s]: %s exceeds this machine's memory "
                  "budget, stopping\n", ep.providerKey.c_str(), s.label);
      break;
    }

    const std::string note = std::string(s.note) + widthNote;
    Result r = measure(rt, ep, s.dim, dtype, warmupCount, forceIters,
                       specifiedIters);
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
                note.c_str());
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
                note.c_str());
      continue;
    }
    const double bytes = (double)s.dim * (double)s.dim * 2.0;
    const double bps  = bytes / (netUs * 1.0e-6);
    test.emit(s.label, (float)bps, note.c_str());

    stillFalling = (prevBps <= 0.0) || (bps < prevBps * kFallingRatio);
    prevBps = bps;
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
