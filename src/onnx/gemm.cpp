#ifdef ENABLE_ONNX

// onnx-gemm: MatMul peak through an ONNX Runtime execution provider.
//
// Both operands are model constants and the result is summed down to a single
// row, so nothing large crosses the host boundary per run.  That shape is
// forced by discrete GPUs: with A as a graph input and C returned to the
// host, an RTX 5060 reported 15 TFLOPS for fp16 while a whole transformer
// block -- whose weights are resident -- reached 28 on the same device.  The
// peak was measuring PCIe.  On unified-memory devices the difference is
// small, but the graph is identical everywhere so the rows stay comparable.
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
#include <algorithm>
#include <cstring>
#include <vector>

namespace
{

// The ladder doubles from 1024 until the rate stops improving, and the peak
// is reported along with the size that produced it.
//
// Reporting a peak rather than "the rate at 4096" is what keeps this number
// comparable over time.  A fixed size has to be raised as hardware grows --
// today's largest rung will one day be too small to saturate anything -- and
// the moment it is raised, every result recorded before becomes a different
// measurement wearing the same name.  An extending search has no such
// horizon: faster hardware simply climbs further, and "the best this device
// can do at any size" means the same thing in ten years as it does now.
//
// It is also not a size chosen from a timing probe, which was the previous
// design.  A probe is unstable -- the size comes out of a cube root and is
// then bucketed, so a couple of percent of timing noise can push the estimate
// across a bucket edge and change the answer.  On an M1 Pro that made the
// fp16 row alternate between 5.8 and 6.2 TFLOPS depending on nothing else.
// And no single size is right anyway: fp32 there peaks at 4096 while fp16
// peaks at 2048, because different engines serve them.
constexpr int64_t kMinDim = 1024;
constexpr int64_t kMaxDim = 32768;

// A size counts as an improvement only if it beats the best so far by this
// much; two failures in a row end the search.  The grace of one lets a curve
// dip at a single size and recover, which happens when one size lands badly
// against a cache but the next tiles better.
constexpr double kImproveFactor = 1.03;
constexpr int    kMaxStrikes    = 2;

// Ceilings that keep the search from running away on either axis.  The time
// bound is predicted from the previous size's measured rate, so a slow
// provider stops early instead of spending minutes on one matrix, and it
// scales itself: hardware fast enough to make a bigger size cheap is exactly
// the hardware that should try it.
constexpr double   kMaxIterUs      = 2.0e6;        // one iteration, predicted

// Both operands together, capped at a quarter of physical memory.  A fixed
// ceiling here would be a crash on a phone and a needless limit on a
// workstation; see clpeak::memoryBudget.
static uint64_t maxWeightBytes() { return clpeak::memoryBudget(3ull << 30); }

// Per-size budget for the timed phase.  Lower than the 5 s a single-size test
// would use, since the ladder measures several.
constexpr unsigned int kSizeBudgetUs = 2000000;

struct Variant
{
  int         dtype;    // element type of the graph's input/output
  bool        qdq;      // build the quantized (DequantizeLinear/MatMul/Q) form
  const char *label;
  const char *note;
};

// Quantization schemes, tried in order until one fuses.  There is no single
// choice that works everywhere: TensorRT rejects unsigned activations and
// demands a zero point of zero, while x86 MLAS without VNNI implements only
// the unsigned form and quietly declines to fuse the signed one.  Trying is
// the only way to know, and the fusion check is what decides.
struct QuantScheme
{
  int         actDtype;
  const char *name;
};

const QuantScheme kQuantSchemes[] = {
  {ONNX_DT_INT8,  "signed activations"},     // TensorRT, ARM
  {ONNX_DT_UINT8, "unsigned activations"},   // x86 without VNNI
};

struct GemmSetup
{
  OrtSession *session = nullptr;
  OrtValue   *inVal   = nullptr;   // the scalar that keeps the graph live
  OrtValue   *outVal  = nullptr;   // reduced row
  std::vector<uint8_t> inBuf, outBuf;
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
  uint8_t  *u = reinterpret_cast<uint8_t *>(&raw[0]);
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float v = (float)(s >> 8) / 16777216.0f - 0.5f;      // [-0.5, 0.5)
    switch (dtype)
    {
    case ONNX_DT_FLOAT:    f[i] = v; break;
    case ONNX_DT_FLOAT16:  h[i] = floatToHalf(v); break;
    case ONNX_DT_BFLOAT16: h[i] = floatToBf16(v); break;
    case ONNX_DT_UINT8:    u[i] = (uint8_t)(v * 254.0f + 128.0f); break;  // zp 128
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
                    const Variant &v, int64_t D, bool profile = false,
                    int actDtype = ONNX_DT_UINT8)
{
  GemmSetup g;

  std::string aRaw, bRaw;
  // Quantized: `actDtype` activations against int8 weights.
  fillTensor(aRaw, v.qdq ? actDtype : v.dtype, D * D, 0x9e3779b9u);
  fillTensor(bRaw, v.dtype, D * D, 0x243f6a88u);

  std::string modelBytes;
  if (v.qdq)
  {
    const float s = 1.0f / 127.0f;
    modelBytes = onnxResidentQdqMatMulModel(D, D, D, aRaw, bRaw, s, s,
                                            qdqOutputScale(D), actDtype);
  }
  else
  {
    modelBytes = onnxResidentMatMulModel(D, D, D, v.dtype, aRaw, bRaw);
  }
  aRaw.clear(); aRaw.shrink_to_fit();
  bRaw.clear(); bRaw.shrink_to_fit();

  // Both models hold their operands as constants and need folding held off,
  // or the whole multiply is evaluated once at load time.
  auto ses = onnxCreateSession(rt, ep, modelBytes,
                               /*keepConstantsUnfolded=*/true, profile);
  if (!ses.session)
  {
    g.error = ses.error;
    return g;
  }
  g.session = ses.session;

  // The QDQ graph reduces in float; the plain one keeps its own dtype.
  const int ioDtype = v.qdq ? ONNX_DT_FLOAT : v.dtype;
  const size_t es   = dtypeSize(ioDtype);
  {
    // Scales the reduced result; exists only so the graph depends on
    // something supplied at run time.
    std::string one;
    fillTensor(one, ioDtype, 1, 0x12345678u);
    g.inBuf.assign(one.begin(), one.end());
  }
  g.outBuf.assign((size_t)D * es, 0);

  OrtMemoryInfo *mi = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  if (st)
  {
    g.error = onnxStatusText(rt, st);
    destroySetup(rt, g);
    return g;
  }

  const int64_t outShape[1] = {D};
  st = rt.api->CreateTensorWithDataAsOrtValue(
      mi, g.inBuf.data(), g.inBuf.size(), nullptr, 0,
      (ONNXTensorElementDataType)ioDtype, &g.inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, g.outBuf.data(), g.outBuf.size(), outShape, 1,
        (ONNXTensorElementDataType)ioDtype, &g.outVal);
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
  static const char *inNames[]  = {"S"};
  static const char *outNames[] = {"Y"};

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
    {ONNX_DT_BFLOAT16, false, "bf16",
     "The 16-bit float with fp32's exponent range and three fewer mantissa "
     "bits.  Modern matrix hardware usually runs it at the fp16 rate; a "
     "provider that falls well short of its own fp16 row is emulating it, "
     "and one that refuses it outright has no bf16 path at all."},
  };
  static const Variant kIntVariants[] = {
    {ONNX_DT_INT8, true, "int8_qdq",
     "8-bit integers in QDQ form -- quantized in, quantized out, the shape "
     "quantized inference actually ships in.  This is what an NPU's headline "
     "TOPS figure is quoted for."},
  };

  const Variant *variants  = isInt ? kIntVariants : kFpVariants;
  const size_t   nVariants = isInt ? 1
                                   : sizeof(kFpVariants) / sizeof(kFpVariants[0]);

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

    // First and last timings with their sizes, to confirm the work actually
    // happened (see the folding check after the loop).
    double  firstUs = 0.0, lastUs = 0.0;
    int64_t firstDim = 0, lastDim = 0;
    double  lastRate = 0.0;
    int     strikes = 0;
    std::string ranAs;      // kernel the provider actually used (quantized only)
    int     actDtype = ONNX_DT_UINT8;
    const char *schemeName = "";

    // For the quantized variant, settle the scheme before sweeping anything.
    // ONNX Runtime rewrites graphs before running them, and a provider that
    // will not fuse dequantize/matmul/quantize into a quantized kernel
    // dequantizes the operands and multiplies in floating point instead --
    // a perfectly good number that is not an int8 number.  One profiled run
    // per scheme at the smallest size answers it, and doing it first means a
    // provider that fuses neither is not swept at all.
    if (v.qdq)
    {
      std::string tried;
      for (const QuantScheme &qs : kQuantSchemes)
      {
        if (clpeak::cancelRequested())
          break;
        GemmSetup probe = makeSetup(rt, ep, v, kMinDim, /*profile=*/true,
                                    qs.actDtype);
        if (!probe.session)
        {
          if (firstErr.empty())
            firstErr = probe.error;
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %s rejected: %s\n",
                      ep.providerKey.c_str(), v.label, qs.name,
                      probe.error.c_str());
          continue;
        }
        timeRuns(rt, probe, 1);
        auto ops = onnxCollectExecutedOps(rt, probe.session);
        destroySetup(rt, probe);

        std::string joined;
        for (const auto &o : ops)
          joined += (joined.empty() ? "" : ", ") + o;
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %s executed %s\n",
                    ep.providerKey.c_str(), v.label, qs.name, joined.c_str());

        if (onnxOpsRanIntegerMatMul(ops))
        {
          actDtype   = qs.actDtype;
          schemeName = qs.name;
          // A provider that compiles the whole subgraph names its kernel
          // after itself, and that name carries a hash of the graph -- it
          // would differ between runs and has no place in a saved result.
          const std::string named = onnxQuantizedKernelName(ops);
          ranAs = named.empty() ? "a kernel it compiled itself" : named;
          break;
        }
        if (!ops.empty())
          tried = joined;
      }

      if (ranAs.empty())
      {
        logger::EmitOptions o;
        o.description = std::string("Peak over a doubling sweep of square "
                                    "sizes.  ") + v.note;
        test.skip(v.label,
                  tried.empty() ? errStatus : ResultStatus::Unsupported,
                  tried.empty()
                      ? (firstErr.empty() ? "no supported quantization scheme"
                                          : firstErr)
                      : "provider did not fuse a quantized matmul with either "
                        "signed or unsigned activations -- it dequantized the "
                        "operands and multiplied in floating point, so this is "
                        "not an int8 rate (ran: " + tried + ")",
                  o.description);
        continue;
      }
    }

    for (int64_t D = kMinDim; D <= kMaxDim; D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      // Would this size fit, and would one iteration finish in reasonable
      // time at the rate the previous size managed?
      const uint64_t weightBytes =
          2ull * (uint64_t)D * (uint64_t)D * (uint64_t)dtypeSize(v.dtype);
      if (weightBytes > maxWeightBytes())
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 needs %llu MB of operands, "
                    "stopping\n", ep.providerKey.c_str(), v.label,
                    (long long)D, (unsigned long long)(weightBytes >> 20));
        break;
      }
      if (lastRate > 0.0)
      {
        const double predictedUs =
            2.0 * (double)D * (double)D * (double)D / lastRate;
        if (predictedUs > kMaxIterUs)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 would take ~%.1f s per "
                      "iteration, stopping\n", ep.providerKey.c_str(),
                      v.label, (long long)D, predictedUs / 1.0e6);
          break;
        }
      }

      GemmSetup g = makeSetup(rt, ep, v, D, /*profile=*/false, actDtype);
      if (!g.session)
      {
        if (firstErr.empty())
          firstErr = g.error;
        // Larger sizes need strictly more of everything, so nothing above
        // this one can succeed either.
        break;
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
        break;
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
        break;

      if (firstUs == 0.0)
      {
        firstUs  = mean_us;
        firstDim = D;
      }
      lastUs  = mean_us;
      lastDim = D;

      const double ops  = 2.0 * (double)D * (double)D * (double)D;
      const double rate = ops * 1.0e6 / mean_us / 1.0e12;
      // Rate per FLOP-count is what the ladder is searching on; the raw rate
      // in ops/us drives the time prediction for the next rung.
      lastRate = ops / mean_us;
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 -> %.3f\n", ep.providerKey.c_str(),
                  v.label, (long long)D, rate);

      if (rate > best * kImproveFactor)
      {
        strikes = 0;
        best    = rate;
        bestDim = D;
      }
      else
      {
        if (rate > best)
        {
          best    = rate;
          bestDim = D;
        }
        if (++strikes >= kMaxStrikes)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: no further gain past %lld^3\n",
                      ep.providerKey.c_str(), v.label, (long long)bestDim);
          break;
        }
      }
    }

    // Both operands are constants, so this test depends on ORT honouring the
    // request not to fold them; if it ever stopped, the matmul would be
    // evaluated once at load time and every timed run would measure an empty
    // graph.  Real work grows with the cube of the size -- 64x across this
    // ladder -- so anything close to flat means nothing was computed.  Better
    // an error than a spectacular number.
    // Folding collapses the graph to a constant, so what remains is dispatch
    // and the time stops tracking the size at all -- 143 us at 1024 against
    // 162 us at 8192.  Real work grows as the cube of the size divided by
    // whatever the rate gained along the way, and that gain can be large:
    // TensorRT's int8 rate improves 9.4x between 1024 and 16384, so its time
    // grows 433x where the work grew 4096x.  The tolerance has to clear that
    // comfortably or the guard discards the very measurements it exists to
    // protect -- a factor of 8 threw away a correct 124 TOPS reading.  A rate
    // improving 64x across one ladder has never been observed.
    double expectedGrowth = 1.0;
    for (int64_t d = firstDim; d > 0 && d < lastDim; d *= 2)
      expectedGrowth *= 8.0;                       // each doubling is 8x work
    if (best > 0.0 && firstUs > 0.0 && lastDim > firstDim &&
        lastUs < firstUs * expectedGrowth / 64.0)
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %.1f us at %lld vs %.1f us at %lld -- "
                  "work does not scale, constants were folded\n",
                  ep.providerKey.c_str(), v.label, firstUs,
                  (long long)firstDim, lastUs, (long long)lastDim);
      best      = 0.0;
      firstErr  = "this runtime folded the operands at load time: it accepted "
                  "the request to disable constant folding and ignored it, "
                  "which ONNX Runtime did before about 1.18, so the timings "
                  "do not scale with the problem size and mean nothing";
      errStatus = ResultStatus::Error;
    }

    logger::EmitOptions o;
    if (best > 0.0)
    {
      o.description = "Peak over a doubling sweep of square sizes; fastest at "
                      + std::to_string(bestDim) + " cubed.  " + v.note;
      if (!ranAs.empty())
        o.description += "  The provider ran the multiply as " + ranAs +
                         " with " + schemeName +
                         ", confirming it really was integer arithmetic.";
      test.emit(v.label, (float)best, o);
    }
    else
    {
      o.description = std::string("Peak over a doubling sweep of square sizes.  ")
                      + v.note;
      test.skip(v.label, errStatus,
                firstErr.empty() ? "no supported datatype" : firstErr,
                o.description);
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
