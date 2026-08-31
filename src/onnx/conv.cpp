#ifdef ENABLE_ONNX

// onnx-conv: 2-D convolution peak through an ONNX Runtime execution provider.
//
// Worth measuring separately from the matmul rows because neural accelerators
// were built for convolution first, and many still reach a higher fraction of
// their arithmetic peak on it than on a plain matrix multiply -- their fixed
// dataflow was designed around sliding a small kernel over a feature map.
// The gap between this and onnx-gemm is itself a comparable architectural
// number: a device that convolves far faster than it multiplies is telling
// you what its hardware was shaped for.
//
// Three variants cover the shapes real networks are made of: a 3x3
// convolution, a 1x1 (pointwise) one -- arithmetically a matmul over pixels,
// so it should track the GEMM rows -- and a depthwise 3x3, which has the same
// shape as the first but a fraction of the multiply-accumulates, and is where
// hardware built around dense arrays tends to fall over.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace
{

  // Channel count is fixed and the feature map grows, so every variant keeps
  // the same weights and only the work per weight changes.  256 channels is
  // wide enough to fill an accelerator's array and narrow enough that the 3x3
  // weights stay a couple of megabytes.
  constexpr int64_t kChannels = 256;
  constexpr int64_t kMinSpatial = 32;

  // No fixed upper size: the sweep stops when the rate stops improving, when
  // one pass would take too long, or when the feature map stops fitting -- the
  // same self-scaling bounds the matmul ladder uses, so a faster device climbs
  // further without the number changing meaning.
  constexpr int64_t kMaxSpatial = 4096;
  static uint64_t maxTensorBytes() { return clpeak::memoryBudget(1ull << 30); }

  constexpr double kImproveFactor = 1.03;
  constexpr int kMaxStrikes = 2;
  constexpr double kMaxIterUs = 2.0e6;
  constexpr unsigned int kSizeBudgetUs = 2000000;

  struct Variant
  {
    int64_t kernel;
    bool depthwise;
    const char *label;
    const char *note;
  };

  const Variant kVariants[] = {
      {3, false, "conv3x3",
       "A 3x3 convolution over 256 channels -- the shape most vision networks are "
       "built from, and the one neural accelerators were designed around."},
      {1, false, "conv1x1",
       "A 1x1 convolution: arithmetically a matrix multiply applied at every "
       "pixel, so it should land near the matmul rows.  Where it does not, the "
       "provider is handling the two shapes with different machinery."},
      {3, true, "depthwise3x3",
       "A depthwise 3x3 convolution -- the same shape as the first but with each "
       "channel kept separate, so there is far less arithmetic per value loaded.  "
       "Hardware built around dense multiply-accumulate arrays usually collapses "
       "here, which is why efficient mobile networks are often slower than their "
       "arithmetic suggests."},
  };

  // Multiply-accumulates x2 for one pass at this size.
  double convFlops(const Variant &v, int64_t spatial)
  {
    const double inPerGroup = v.depthwise ? 1.0 : (double)kChannels;
    return 2.0 * (double)kChannels * (double)spatial * (double)spatial *
           inPerGroup * (double)v.kernel * (double)v.kernel;
  }

  void fillHalf(std::string &raw, int64_t count, uint32_t seed)
  {
    raw.assign((size_t)count * 2, '\0');
    uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
    uint32_t s = seed;
    for (int64_t i = 0; i < count; i++)
    {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      h[i] = floatToHalf((float)(s >> 8) / 16777216.0f - 0.5f);
    }
  }

  struct ConvSetup
  {
    OrtSession *session = nullptr;
    OrtValue *inVal = nullptr;
    OrtValue *outVal = nullptr;
    std::vector<uint8_t> inBuf, outBuf;
    std::string error;
  };

  void destroySetup(const OrtRuntime &rt, ConvSetup &c)
  {
    if (c.inVal)
      rt.api->ReleaseValue(c.inVal);
    if (c.outVal)
      rt.api->ReleaseValue(c.outVal);
    if (c.session)
      rt.api->ReleaseSession(c.session);
    c.inVal = nullptr;
    c.outVal = nullptr;
    c.session = nullptr;
    c.inBuf.clear();
    c.inBuf.shrink_to_fit();
    c.outBuf.clear();
    c.outBuf.shrink_to_fit();
  }

  ConvSetup makeSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                      const Variant &v, int64_t spatial)
  {
    ConvSetup c;
    const int64_t group = v.depthwise ? kChannels : 1;
    const int64_t inPerGroup = kChannels / group;

    {
      std::string xRaw, wRaw;
      fillHalf(xRaw, kChannels * spatial * spatial, 0x9e3779b9u);
      fillHalf(wRaw, kChannels * inPerGroup * v.kernel * v.kernel, 0x243f6a88u);
      std::string model = onnxResidentConvModel(kChannels, spatial, v.kernel,
                                                group, ONNX_DT_FLOAT16,
                                                xRaw, wRaw);
      xRaw.clear();
      xRaw.shrink_to_fit();
      wRaw.clear();
      wRaw.shrink_to_fit();

      auto ses = onnxCreateSession(rt, ep, model, /*keepConstantsUnfolded=*/true);
      model.clear();
      model.shrink_to_fit();
      if (!ses.session)
      {
        c.error = ses.error;
        return c;
      }
      c.session = ses.session;
    }

    {
      uint16_t one = floatToHalf(1.0009765625f);
      c.inBuf.assign(2, 0);
      std::memcpy(c.inBuf.data(), &one, 2);
    }
    c.outBuf.assign((size_t)kChannels * 2, 0);

    OrtMemoryInfo *mi = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                OrtMemTypeDefault, &mi);
    const int64_t outShape[1] = {kChannels};
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, c.inBuf.data(), c.inBuf.size(), nullptr, 0,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &c.inVal);
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, c.outBuf.data(), c.outBuf.size(), outShape, 1,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &c.outVal);
    if (mi)
      rt.api->ReleaseMemoryInfo(mi);
    if (st)
    {
      c.error = onnxStatusText(rt, st);
      destroySetup(rt, c);
    }
    return c;
  }

  double timeRuns(const OrtRuntime &rt, ConvSetup &c, unsigned int n)
  {
    static const char *inNames[] = {"S"};
    static const char *outNames[] = {"R"};

    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n; i++)
    {
      OrtStatus *st = rt.api->Run(c.session, nullptr,
                                  inNames, (const OrtValue *const *)&c.inVal, 1,
                                  outNames, 1, &c.outVal);
      if (st)
      {
        c.error = onnxStatusText(rt, st);
        return -1.0;
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
  }

} // namespace

int OnnxPeak::runConv(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                      benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx_conv", "ONNX convolution peak", "tflops", Category::FpCompute,
       "Convolution speed in half precision, swept over feature-map sizes and "
       "reported at its best.  Neural accelerators were built for this "
       "operation before they were asked to do anything else, and many reach "
       "a higher share of their arithmetic peak here than on a plain matrix "
       "multiply.  Read alongside the matmul rows: the gap between them says "
       "what the hardware was shaped for.",
       TestShape::Heterogeneous, "convolution shape",
       Direction::FromUnit, "", /*streaming=*/true});

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;

    double best = 0.0;
    int64_t bestSpatial = 0;
    double lastRate = 0.0;
    int strikes = 0;
    std::string firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;

    for (int64_t sp = kMinSpatial; sp <= kMaxSpatial; sp *= 2)
    {
      if (clpeak::cancelRequested())
        break;
      const uint64_t tensorBytes =
          (uint64_t)kChannels * (uint64_t)sp * (uint64_t)sp * 2ull;
      if (tensorBytes > maxTensorBytes())
      {
        CLPEAK_VLOG("onnx-conv[%s/%s]: %lldx%lld needs %llu MB per tensor, "
                    "stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)sp, (long long)sp,
                    (unsigned long long)(tensorBytes >> 20));
        break;
      }
      if (lastRate > 0.0 && convFlops(v, sp) / lastRate > kMaxIterUs)
      {
        CLPEAK_VLOG("onnx-conv[%s/%s]: %lldx%lld would take too long, stopping\n",
                    ep.providerKey.c_str(), v.label, (long long)sp, (long long)sp);
        break;
      }

      ConvSetup c = makeSetup(rt, ep, v, sp);
      if (!c.session)
      {
        if (firstErr.empty())
          firstErr = c.error;
        break;
      }

      double per_iter_us = -1.0;
      if (timeRuns(rt, c, 1 + warmupCount) > 0.0)
        per_iter_us = timeRuns(rt, c, 1);
      if (per_iter_us <= 0.0)
      {
        if (firstErr.empty())
        {
          firstErr = c.error.empty() ? "run failed" : c.error;
          errStatus = ResultStatus::Error;
        }
        destroySetup(rt, c);
        break;
      }

      unsigned int iters = pickIters(per_iter_us, kSizeBudgetUs,
                                     forceIters ? specifiedIters : 0,
                                     kOnnxMaxIters);
      // The probe was one whole pass; when the budget affords only one, it
      // already is the measurement.
      double mean_us = (iters > 1) ? timeRuns(rt, c, iters) : per_iter_us;
      if (mean_us <= 0.0 && firstErr.empty())
      {
        firstErr = c.error.empty() ? "run failed" : c.error;
        errStatus = ResultStatus::Error;
      }
      destroySetup(rt, c);
      if (mean_us <= 0.0)
        break;

      const double flops = convFlops(v, sp);
      const double rate = flops * 1.0e6 / mean_us / 1.0e12;
      lastRate = flops / mean_us;
      CLPEAK_VLOG("onnx-conv[%s/%s]: %lldx%lld -> %.3f\n", ep.providerKey.c_str(),
                  v.label, (long long)sp, (long long)sp, rate);

      if (rate > best * kImproveFactor)
      {
        strikes = 0;
        best = rate;
        bestSpatial = sp;
      }
      else
      {
        if (rate > best)
        {
          best = rate;
          bestSpatial = sp;
        }
        if (++strikes >= kMaxStrikes)
          break;
      }

      // Measured, not predicted.  The gate at the top of the loop extrapolates
      // the next size's cost from this one's rate, which is blind to a
      // provider that falls off a cliff between rungs; this one asks only
      // whether the pass just timed took longer than a pass is allowed to,
      // and the next feature map is four times the work.
      if (per_iter_us > kMaxIterUs)
      {
        CLPEAK_VLOG("onnx-conv[%s/%s]: %lldx%lld measured %.1f s per pass, "
                    "stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)sp, (long long)sp, per_iter_us / 1.0e6);
        break;
      }
    }

    logger::EmitOptions o;
    if (best > 0.0)
    {
      o.description = "Peak over a doubling sweep of feature-map sizes; "
                      "fastest at " +
                      std::to_string(bestSpatial) + " square.  " + v.note;
      test.emit(v.label, (float)best, o);
    }
    else
    {
      o.description = std::string("Peak over a doubling sweep of feature-map sizes.  ") + v.note;
      test.skip(v.label, errStatus,
                firstErr.empty() ? "convolution unsupported" : firstErr,
                o.description);
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
