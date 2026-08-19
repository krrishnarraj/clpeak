#ifdef ENABLE_ONNX

// onnx-gemm: single-node MatMul peak through an ONNX Runtime execution
// provider.  C[D,D] = A[D,D] x B[D,D] with B an embedded constant, so the EP
// treats it as model weights (pre-packed / kept device-side), which is the
// weight-stationary GEMM inference is made of.  The identical in-memory
// model runs on every EP, making NPU / GPU / CPU rows directly comparable.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{

// Probe size and the per-iteration time the chosen size should land at.
// 20 ms/iter with the 5 s BLAS-style budget gives a ~250-dispatch timed
// phase -- long enough to ride out NPU/GPU clock ramps.
constexpr int64_t kProbeDim       = 1024;
constexpr double  kTargetIterUs   = 20000.0;
constexpr int64_t kMinDim         = 1024;
constexpr int64_t kMaxDim         = 4096;   // 4096^3 keeps EP graph compiles sane

struct GemmSetup
{
  OrtSession *session = nullptr;
  OrtValue   *inVal   = nullptr;
  OrtValue   *outVal  = nullptr;
  std::vector<uint8_t> inBuf, outBuf;
  std::string error;
};

size_t dtypeSize(int dtype) { return dtype == ONNX_DT_FLOAT ? 4 : 2; }

// Deterministic small values in [-0.5, 0.5): keeps fp16 accumulations far
// from overflow and avoids the NaN/denormal slow paths raw bit patterns hit.
void fillWeights(std::string &raw, int dtype, int64_t count)
{
  uint32_t s = 0x243f6a88u;
  raw.resize((size_t)count * dtypeSize(dtype));
  float    *f = reinterpret_cast<float *>(&raw[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float v = (float)(s >> 8) / 16777216.0f - 0.5f;
    if (dtype == ONNX_DT_FLOAT)        f[i] = v;
    else if (dtype == ONNX_DT_FLOAT16) h[i] = floatToHalf(v);
    else                               h[i] = floatToBf16(v);
  }
}

// Releases the ORT handles and drops the buffers.  Deliberately leaves
// `error` intact: callers tear a failed setup down and then report its
// message.
void destroySetup(const OrtRuntime &rt, GemmSetup &g)
{
  if (g.inVal)   rt.api->ReleaseValue(g.inVal);
  if (g.outVal)  rt.api->ReleaseValue(g.outVal);
  if (g.session) rt.api->ReleaseSession(g.session);
  g.inVal   = nullptr;
  g.outVal  = nullptr;
  g.session = nullptr;
  g.inBuf.clear();  g.inBuf.shrink_to_fit();
  g.outBuf.clear(); g.outBuf.shrink_to_fit();
}

// Build model + session + bound input/output tensors for one (dtype, D).
GemmSetup makeSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                    int dtype, int64_t D)
{
  GemmSetup g;

  std::string weights;
  fillWeights(weights, dtype, D * D);
  std::string modelBytes = onnxMatMulModel(D, D, D, dtype, weights);
  weights.clear();
  weights.shrink_to_fit();

  auto ses = onnxCreateSession(rt, ep, modelBytes);
  if (!ses.session)
  {
    g.error = ses.error;
    return g;
  }
  g.session = ses.session;

  const size_t es = dtypeSize(dtype);
  g.inBuf.resize((size_t)D * D * es);
  g.outBuf.resize((size_t)D * D * es);
  {
    std::string a;
    fillWeights(a, dtype, D * D);
    std::memcpy(g.inBuf.data(), a.data(), a.size());
  }

  OrtMemoryInfo *mi = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  if (st)
  {
    g.error = onnxStatusText(rt, st);
    destroySetup(rt, g);
    return g;
  }

  const int64_t shape[2] = {D, D};
  st = rt.api->CreateTensorWithDataAsOrtValue(
      mi, g.inBuf.data(), g.inBuf.size(), shape, 2,
      (ONNXTensorElementDataType)dtype, &g.inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, g.outBuf.data(), g.outBuf.size(), shape, 2,
        (ONNXTensorElementDataType)dtype, &g.outVal);
  rt.api->ReleaseMemoryInfo(mi);
  if (st)
  {
    g.error = onnxStatusText(rt, st);
    destroySetup(rt, g);
  }
  return g;
}

// Mean microseconds per Run() over n runs; negative on failure.
double timeRuns(const OrtRuntime &rt, GemmSetup &g, unsigned int n)
{
  static const char *inNames[]  = {"A"};
  static const char *outNames[] = {"C"};

  auto t0 = std::chrono::steady_clock::now();
  for (unsigned int i = 0; i < n; i++)
  {
    OrtStatus *st = rt.api->Run(g.session, nullptr,
                                inNames, (const OrtValue *const *)&g.inVal, 1,
                                outNames, 1, &g.outVal);
    if (st)
    {
      g.error = onnxStatusText(rt, st);
      return -1.0;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
}

} // namespace

int OnnxPeak::runGemm(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                      benchmark_config_t &cfg, Category category)
{
  (void)cfg;
  (void)category;

  auto test = currentDeviceScope->beginTest(
      {"onnx-gemm-fp", "ONNX MatMul peak", "tflops", Category::Unknown,
       "Matrix-multiply speed through ONNX Runtime on this execution "
       "provider, using a single-operation model with constant weights.  "
       "The identical model runs on every provider, so NPU, GPU and CPU "
       "rows are directly comparable -- and the gap against a vendor's "
       "advertised TOPS is real, not an artifact of different test code.  "
       "Providers that cannot run an operation entirely on their device "
       "report it as unsupported instead of quietly measuring the CPU."});

  const char *fp32Note = "Full 32-bit precision.  Many NPUs cannot run this "
                         "at all, or route it away from the matrix hardware "
                         "-- that is a finding, not a failure.";
  const char *fp16Note = "16-bit floats, the native currency of most NPU "
                         "matrix hardware.";

  struct Variant { int dtype; const char *label; const char *note; };
  const Variant variants[] = {
      {ONNX_DT_FLOAT,   "fp32", fp32Note},
      {ONNX_DT_FLOAT16, "fp16", fp16Note},
  };

  // ---- Size probe -------------------------------------------------------
  // One size per device, chosen from a small fp32 (or fp16) probe run and
  // bucketed coarsely so a device doesn't drift between sizes run to run.
  int64_t D = 0;
  for (const Variant &v : variants)
  {
    GemmSetup probe = makeSetup(rt, ep, v.dtype, kProbeDim);
    if (!probe.session)
      continue;
    timeRuns(rt, probe, 1);                       // discard: first-run compile
    double t0 = timeRuns(rt, probe, 3);
    destroySetup(rt, probe);
    if (t0 <= 0.0)
      continue;
    D = (int64_t)((double)kProbeDim * std::cbrt(kTargetIterUs / t0));
    D = ((D + 512) / 1024) * 1024;
    if (D < kMinDim) D = kMinDim;
    if (D > kMaxDim) D = kMaxDim;
    CLPEAK_VLOG("onnx-gemm[%s]: probe %.0f us/iter at %lld -> D=%lld\n",
                ep.providerKey.c_str(), t0, (long long)kProbeDim, (long long)D);
    break;
  }
  if (D == 0)
  {
    // Not even the probe session could be built on this EP.
    GemmSetup why = makeSetup(rt, ep, ONNX_DT_FLOAT, kProbeDim);
    std::string reason = why.error.empty() ? "no supported datatype" : why.error;
    destroySetup(rt, why);
    test.skipAll({"fp32", "fp16"}, ResultStatus::Unsupported, reason);
    return 0;
  }

  const double flops = 2.0 * (double)D * (double)D * (double)D;
  const std::string dims = std::to_string(D) + "x" + std::to_string(D) +
                           "x" + std::to_string(D) + ".  ";

  // ---- Per-dtype measurement ---------------------------------------------
  for (const Variant &v : variants)
  {
    if (clpeak::cancelRequested())
      break;

    logger::EmitOptions o;
    o.description = dims + v.note;

    GemmSetup g = makeSetup(rt, ep, v.dtype, D);
    if (!g.session)
    {
      test.skip(v.label, ResultStatus::Unsupported, g.error, o.description);
      continue;
    }

    double per_iter_us = -1.0;
    if (timeRuns(rt, g, 1 + warmupCount) > 0.0)    // compile + warmup
      per_iter_us = timeRuns(rt, g, 3);            // calibration probe
    if (per_iter_us <= 0.0)
    {
      test.skip(v.label, ResultStatus::Error,
                g.error.empty() ? "run failed" : g.error, o.description);
      destroySetup(rt, g);
      continue;
    }

    unsigned int iters = pickIters(per_iter_us, 5000000u,
                                   forceIters ? specifiedIters : 0);
    double mean_us = timeRuns(rt, g, iters);
    destroySetup(rt, g);
    if (mean_us <= 0.0)
    {
      test.skip(v.label, ResultStatus::Error,
                g.error.empty() ? "run failed" : g.error, o.description);
      continue;
    }

    test.emit(v.label, (float)(flops * 1.0e6 / mean_us / 1.0e12), o);
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
