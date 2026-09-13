#ifdef ENABLE_ONNX

// onnx-dispatch-latency: what it costs to ask this execution provider to do
// anything at all.
//
// NPUs are reached through a runtime, a driver and often an ahead-of-time
// compiled graph, and every one of those layers charges a toll per
// submission.  That toll is why a device advertising tens of TOPS can still
// lose to a CPU on small work, and it is invisible in every throughput row
// in this backend.  Two readings bracket it: the smallest graph that can be
// expressed at all, and a 256-cube matmul -- 34 MFLOP, which any accelerator
// here should finish in well under a millisecond, so whatever the row shows
// above that floor is overhead rather than arithmetic.
//
// Session creation is timed too.  On the NPU providers it is a compiler run,
// not a bookkeeping call, and it is the reason cold-start behaviour differs
// so sharply from steady-state throughput.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_probe.h"
#include "onnx_session.h"

#include <chrono>
#include <cstring>
#include <vector>

namespace
{

constexpr int64_t kTrivialWidth = 64;    // smallest graph worth expressing
constexpr int64_t kSmallMatMul  = 256;   // 34 MFLOP: arithmetic is negligible

double nowUs()
{
  return std::chrono::duration<double, std::micro>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The one-node elementwise graph lives in onnx_model as onnxTrivialMulModel
// (shared with the viability probe, which compiles the cheapest possible
// session rather than a matmul).  Y = X * K over [1, width] fp16; K is a
// full-size constant rather than a scalar on purpose -- see there.

// In whichever width the provider streams faster (onnxStreamDtype): the
// arithmetic is meant to be negligible, and on a provider without native
// fp16 kernels a fp16 matmul is not -- ONNX Runtime's CPU EP took 1.1 ms for
// this graph in fp16 on a Threadripper, twenty times its fp32 time, all of it
// arithmetic rather than the overhead the row is named for.
std::string smallMatMulModel(int64_t d, int dtype)
{
  std::string w((size_t)onnxElemBytes(dtype, d * d), '\0');
  float *f = reinterpret_cast<float *>(&w[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&w[0]);
  uint32_t s = 0x243f6a88u;
  for (int64_t i = 0; i < d * d; i++)
  {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    const float v = (float)(s >> 8) / 16777216.0f - 0.5f;
    if (dtype == ONNX_DT_FLOAT)
      f[i] = v;
    else
      h[i] = floatToHalf(v);
  }
  return onnxMatMulModel(d, d, d, dtype, w);
}

struct Timed
{
  double perRunUs = -1.0;
  double createUs = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Timed timeGraph(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                const std::string &model,
                const char *inName, const char *outName,
                int64_t rows, int64_t cols, int dtype,
                unsigned int warmup, bool forceIters, unsigned int forced)
{
  Timed t;

  const double t0 = nowUs();
  auto ses = onnxCreateSession(rt, ep, model);
  t.createUs = nowUs() - t0;
  if (!ses.session)
  {
    t.error  = ses.error;
    t.status = ResultStatus::Unsupported;
    t.createUs = -1.0;
    return t;
  }

  const size_t es = (size_t)onnxElemBytes(dtype, 1);
  std::vector<uint8_t> in((size_t)rows * cols * es, 0);
  std::vector<uint8_t> out((size_t)rows * cols * es, 0);
  {
    const std::string half = onnxFloatScalar(0.5f, dtype);
    for (int64_t i = 0; i < rows * cols; i++)
      std::memcpy(&in[(size_t)i * es], half.data(), es);
  }

  OrtMemoryInfo *mi = nullptr;
  OrtValue *inVal = nullptr, *outVal = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  const int64_t shape[2] = {rows, cols};
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, in.data(), in.size(), shape, 2,
        (ONNXTensorElementDataType)dtype, &inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, out.data(), out.size(), shape, 2,
        (ONNXTensorElementDataType)dtype, &outVal);
  if (mi) rt.api->ReleaseMemoryInfo(mi);

  auto run = [&](unsigned int n) -> double {
    const char *ins[]  = {inName};
    const char *outs[] = {outName};
    const double a = nowUs();
    for (unsigned int i = 0; i < n; i++)
    {
      OrtStatus *rs = rt.api->Run(ses.session, nullptr, ins,
                                  (const OrtValue *const *)&inVal, 1,
                                  outs, 1, &outVal);
      if (rs)
      {
        t.error = onnxStatusText(rt, rs);
        return -1.0;
      }
    }
    return (nowUs() - a) / (double)n;
  };

  if (!st && run(1 + warmup) > 0.0)
  {
    // Five runs, where every other test here probes with one: this is the
    // one measurement that *is* the per-submission overhead, a submission
    // costs microseconds, and five of them cost nothing.  Probing with one
    // would buy no time and size the batch off a single noisy sample.
    double probe = run(5);
    if (probe > 0.0)
    {
      // A trivial graph runs in microseconds, so the batch has to be large
      // or the clock itself becomes the measurement.
      unsigned int iters = pickIters(probe, 500000u, forceIters ? forced : 0,
                                     200000u);
      t.perRunUs = run(iters);
    }
  }
  if (st)
    t.error = onnxStatusText(rt, st);
  if (t.perRunUs <= 0.0)
    t.status = ResultStatus::Error;

  if (inVal)  rt.api->ReleaseValue(inVal);
  if (outVal) rt.api->ReleaseValue(outVal);
  rt.api->ReleaseSession(ses.session);
  return t;
}

} // namespace

int OnnxPeak::runDispatchLatency(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                                 benchmark_config_t &cfg)
{
  (void)cfg;

  Timed trivial = timeGraph(rt, ep, onnxTrivialMulModel(kTrivialWidth), "X", "Y",
                            1, kTrivialWidth, ONNX_DT_FLOAT16, warmupCount,
                            forceIters, specifiedIters);
  const int mmDtype = onnxStreamDtype(rt, ep);
  Timed matmul;
  if (!clpeak::cancelRequested())
    matmul = timeGraph(rt, ep, smallMatMulModel(kSmallMatMul, mmDtype), "A", "C",
                       kSmallMatMul, kSmallMatMul, mmDtype, warmupCount,
                       forceIters, specifiedIters);

  auto test = currentDeviceScope->beginTest(
      {"onnx_dispatch_latency", "ONNX dispatch latency", "s",
       Category::Latency,
       "The fixed cost of handing one piece of work to this provider, with "
       "the arithmetic made deliberately negligible.  It is why a chip "
       "advertising tens of TOPS can still lose to the host on small work, "
       "and no throughput row can show it.",
       TestShape::Heterogeneous, "what is submitted"});

  if (trivial.perRunUs > 0.0)
    test.emit("trivial_op", (float)(trivial.perRunUs * 1e-6),
              "One multiply over 64 values -- as close to doing nothing as a "
              "graph can get, so almost all of this is overhead.");
  else
    test.skip("trivial_op", trivial.status, trivial.error,
              "One multiply over 64 values.");

  const std::string mmNote =
      std::string("A 256x256x256 matrix multiply in ") +
      (mmDtype == ONNX_DT_FLOAT ? "fp32" : "fp16") +
      ": 34 million operations, which any accelerator here should finish in "
      "well under a millisecond.  Whatever this reads above the row before "
      "it is still mostly overhead.";
  if (matmul.perRunUs > 0.0)
    test.emit("matmul_256", (float)(matmul.perRunUs * 1e-6), mmNote.c_str());
  else
    test.skip("matmul_256", matmul.status, matmul.error,
              "A 256x256x256 matrix multiply.");

  if (trivial.createUs > 0.0)
    test.emit("session_create", (float)(trivial.createUs * 1e-6),
              "Preparing that trivial graph for execution.  On providers "
              "that compile ahead of time this runs a compiler, not "
              "bookkeeping.");
  else
    test.skip("session_create", trivial.status, trivial.error,
              "Preparing the trivial graph for execution.");

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
