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
// weights would amortise submission overhead and let the small rungs measure
// on-chip memory properly -- but the Apple Neural Engine refuses a chained
// program outright (ANEProgramProcessRequestDirect fails) while running the
// single-op form happily, so the portable shape wins.
//
// The cost of that choice is that the smallest rung still carries one
// dispatch: on a provider with expensive submission -- Core ML charges tens
// of microseconds, see onnx-dispatch-latency -- a couple of megabytes moves
// in about the time one empty dispatch takes, so that rung reads low and
// says more about overhead than about memory.  Compare it against the
// dispatch row before drawing conclusions from it.  The largest rung is
// always dominated by the transfer and is the one to trust.
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

// Three rungs: comfortably inside on-chip memory, around the size of a large
// last-level cache or NPU SRAM, and unambiguously main memory.
const Size kSizes[] = {
  {2048, "8mb",
   "Eight megabytes of weights -- may still sit in fast local memory.  On a "
   "provider that is slow to accept work this rung also carries one "
   "submission, so check it against the dispatch-latency row."},
  {4096, "32mb",
   "Thirty-two megabytes -- around the size of a large cache or an NPU's local memory."},
  {8192, "128mb",
   "128 megabytes -- too big for any on-chip memory, so this is the main-memory rate, "
   "and the reading that sets how fast a large model can generate text."},
};

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
  double gbps = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Result measure(const OrtRuntime &rt, const onnx_ep_info_t &ep, int64_t d,
               unsigned int warmup, bool forceIters, unsigned int forced)
{
  Result r;
  const int64_t bytes = d * d * 2;

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
      double mean_us = run(iters);
      if (mean_us > 0.0)
        r.gbps = (double)bytes / (mean_us * 1.0e-6) / 1.0e9;
    }
  }
  if (st)
    r.error = onnxStatusText(rt, st);
  if (r.gbps <= 0.0 && r.status == ResultStatus::Ok)
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
       "weights come from main memory on every single token."});

  for (const Size &s : kSizes)
  {
    if (clpeak::cancelRequested())
      break;
    Result r = measure(rt, ep, s.dim, warmupCount, forceIters, specifiedIters);
    if (r.gbps > 0.0)
      test.emit(s.label, (float)r.gbps, s.note);
    else
      test.skip(s.label, r.status, r.error.empty() ? "run failed" : r.error,
                s.note);
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
