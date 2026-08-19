#ifdef ENABLE_ONNX

// onnx-gemm: single-node MatMul peak through an ONNX Runtime execution
// provider.  C = A x B with B an embedded constant, so the EP treats it as
// model weights (pre-packed / kept device-side) -- the weight-stationary
// GEMM inference is made of.  The identical in-memory model runs on every
// EP, making NPU / GPU / CPU rows directly comparable.
//
// Two scopes, because a test carries one unit: onnx-gemm-fp reports TFLOPS
// for the float dtypes, onnx-gemm-int reports TOPS for the int8 QDQ form.
// int8 is the dtype most NPUs are actually built for, so an NPU that only
// shows up in the int scope is the expected shape, not a gap.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{

// A fixed ladder, swept in full, best result reported -- deliberately not a
// size chosen from a timing probe.  Two reasons.
//
// A probe is unstable: the size it picks comes out of a cube root and then
// gets bucketed, so a couple of percent of timing noise can push the estimate
// across a bucket edge and change the size, and the size changes the answer.
// On this Mac that made the fp16 row alternate between 5.8 and 6.2 TFLOPS
// depending on nothing but which side of the boundary the probe landed.
//
// And no single size is right anyway: on the same device fp32 runs faster at
// 3072 while fp16 runs faster at 2048, because they are served by different
// engines with different fast-memory limits.  Sweeping asks each datatype
// where its own peak is; a fixed ladder also means two devices are always
// compared on identical work.
constexpr int64_t kDims[] = {1024, 2048, 4096};

// Per-size budget for the timed phase.  Lower than the 5 s a single-size test
// would use, so sweeping three sizes costs about what measuring one did.
constexpr unsigned int kSizeBudgetUs = 2000000;

struct Variant
{
  int         dtype;    // element type of the graph's input/output
  bool        qdq;      // build the quantized (DequantizeLinear/MatMul/Q) form
  const char *label;
  const char *note;
};

struct GemmSetup
{
  OrtSession *session = nullptr;
  OrtValue   *inVal   = nullptr;
  OrtValue   *outVal  = nullptr;
  std::vector<uint8_t> inBuf, outBuf;
  const char *inName  = "A";
  const char *outName = "C";
  std::string error;
};

size_t dtypeSize(int dtype)
{
  switch (dtype)
  {
  case ONNX_DT_FLOAT:                     return 4;
  case ONNX_DT_FLOAT16: case ONNX_DT_BFLOAT16: return 2;
  default:                                return 1;   // int8 / uint8
  }
}

// Deterministic values, generated once and reused for inputs and weights.
// Floats land in [-0.5, 0.5) and int8 in [-127, 127]: small magnitudes keep
// fp16 accumulation over a 4096-deep dot product far from overflow, and
// avoid the NaN/denormal slow paths raw random bit patterns would hit.
void fillTensor(std::string &raw, int dtype, int64_t count, uint32_t seed)
{
  uint32_t s = seed;
  raw.resize((size_t)count * dtypeSize(dtype));
  float    *f = reinterpret_cast<float *>(&raw[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  int8_t   *q = reinterpret_cast<int8_t *>(&raw[0]);
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float v = (float)(s >> 8) / 16777216.0f - 0.5f;      // [-0.5, 0.5)
    switch (dtype)
    {
    case ONNX_DT_FLOAT:    f[i] = v; break;
    case ONNX_DT_FLOAT16:  h[i] = floatToHalf(v); break;
    case ONNX_DT_BFLOAT16: h[i] = floatToBf16(v); break;
    default:               q[i] = (int8_t)(v * 254.0f); break;   // [-127, 127]
    }
  }
}

// Output scale for the QDQ form.  Each dequantized product is a pair of
// values in [-1, 1], so a K-deep dot product has standard deviation
// sqrt(K)/3; four sigma keeps nearly every output inside int8 without
// compressing the useful range into a handful of codes.
float qdqOutputScale(int64_t K)
{
  return (float)(4.0 * std::sqrt((double)K) / 3.0 / 127.0);
}

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
  // `error` is deliberately left intact: callers tear a failed setup down
  // and then report its message.
}

// Build model + session + bound input/output tensors for one (variant, D).
GemmSetup makeSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                    const Variant &v, int64_t D)
{
  GemmSetup g;

  std::string weights;
  fillTensor(weights, v.dtype, D * D, 0x243f6a88u);

  std::string modelBytes;
  if (v.qdq)
  {
    const float s = 1.0f / 127.0f;
    modelBytes = onnxQdqMatMulModel(D, D, D, weights, s, s, qdqOutputScale(D));
    g.inName  = "A_q";
    g.outName = "C_q";
  }
  else
  {
    modelBytes = onnxMatMulModel(D, D, D, v.dtype, weights);
  }
  weights.clear();
  weights.shrink_to_fit();

  auto ses = onnxCreateSession(rt, ep, modelBytes);
  if (!ses.session)
  {
    g.error = ses.error;
    return g;
  }
  g.session = ses.session;

  std::string a;
  fillTensor(a, v.dtype, D * D, 0x9e3779b9u);
  g.inBuf.assign(a.begin(), a.end());
  g.outBuf.resize((size_t)D * D * dtypeSize(v.dtype));

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
      (ONNXTensorElementDataType)v.dtype, &g.inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, g.outBuf.data(), g.outBuf.size(), shape, 2,
        (ONNXTensorElementDataType)v.dtype, &g.outVal);
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
  const char *inNames[]  = {g.inName};
  const char *outNames[] = {g.outName};

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
  const bool isInt = (category == Category::IntCompute);

  static const Variant kFpVariants[] = {
    {ONNX_DT_FLOAT,   false, "fp32",
     "Full 32-bit precision.  Many NPUs cannot run this at all, or route it "
     "away from the matrix hardware -- that is a finding, not a failure."},
    {ONNX_DT_FLOAT16, false, "fp16",
     "16-bit floats, the native currency of most NPU matrix hardware."},
  };
  static const Variant kIntVariants[] = {
    {ONNX_DT_INT8, true, "int8_qdq",
     "8-bit integers in QDQ form -- quantized in, quantized out, the shape "
     "quantized inference actually ships in.  This is what an NPU's headline "
     "TOPS figure is quoted for."},
  };

  const Variant *variants = isInt ? kIntVariants : kFpVariants;
  const size_t   nVariants = isInt ? 1 : 2;

  auto test = currentDeviceScope->beginTest(
      {isInt ? "onnx-gemm-int" : "onnx-gemm-fp",
       isInt ? "ONNX MatMul peak (int8)" : "ONNX MatMul peak",
       isInt ? "tops" : "tflops",
       Category::Unknown,
       "Matrix-multiply speed through ONNX Runtime on this execution "
       "provider, using a single-operation model with constant weights.  "
       "The identical model runs on every provider, so NPU, GPU and CPU "
       "rows are directly comparable -- and the gap against a vendor's "
       "advertised TOPS is real, not an artifact of different test code.  "
       "Providers that cannot run an operation entirely on their device "
       "report it as unsupported instead of quietly measuring the CPU."});

  // ---- Sweep every size, keep each datatype's best ----------------------
  for (size_t i = 0; i < nVariants; i++)
  {
    if (clpeak::cancelRequested())
      break;
    const Variant &v = variants[i];

    double      best     = 0.0;
    int64_t     bestDim  = 0;
    std::string firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;

    for (int64_t D : kDims)
    {
      if (clpeak::cancelRequested())
        break;

      GemmSetup g = makeSetup(rt, ep, v, D);
      if (!g.session)
      {
        if (firstErr.empty())
          firstErr = g.error;
        continue;
      }

      double per_iter_us = -1.0;
      if (timeRuns(rt, g, 1 + warmupCount) > 0.0)   // compile + warmup
        per_iter_us = timeRuns(rt, g, 3);           // calibration probe
      if (per_iter_us <= 0.0)
      {
        if (firstErr.empty())
        {
          firstErr  = g.error.empty() ? "run failed" : g.error;
          errStatus = ResultStatus::Error;
        }
        destroySetup(rt, g);
        continue;
      }

      unsigned int iters = pickIters(per_iter_us, kSizeBudgetUs,
                                     forceIters ? specifiedIters : 0);
      double mean_us = timeRuns(rt, g, iters);
      if (mean_us <= 0.0 && firstErr.empty())
      {
        firstErr  = g.error.empty() ? "run failed" : g.error;
        errStatus = ResultStatus::Error;
      }
      destroySetup(rt, g);
      if (mean_us <= 0.0)
        continue;

      const double ops  = 2.0 * (double)D * (double)D * (double)D;
      const double rate = ops * 1.0e6 / mean_us / 1.0e12;
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 -> %.3f\n", ep.providerKey.c_str(),
                  v.label, (long long)D, rate);
      if (rate > best)
      {
        best    = rate;
        bestDim = D;
      }
    }

    logger::EmitOptions o;
    if (best > 0.0)
    {
      o.description = "Best of 1024, 2048 and 4096 cubed; fastest at " +
                      std::to_string(bestDim) + ".  " + v.note;
      test.emit(v.label, (float)best, o);
    }
    else
    {
      o.description = std::string("Best of 1024, 2048 and 4096 cubed.  ") + v.note;
      test.skip(v.label, errStatus,
                firstErr.empty() ? "no supported datatype" : firstErr,
                o.description);
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
