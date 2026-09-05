#ifdef ENABLE_ONNX

#include "onnx_probe.h"
#include "onnx_model.h"
#include "onnx_session.h"

#include <onnx/onnx_peak.h>
#include <common/common.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

// Copied from gemm.cpp - keep in sync with kFpVariants/kIntVariants there.
// Probe uses 32^3 so AOT compilation stays cheap (QNN HTP: 32^3 ~0.2s vs 1024^3 33s,
// TensorRT 32^3 ~0.8s vs 64^3 3.6s).
namespace
{
constexpr int64_t kProbeDim = 32;

  struct Variant
  {
    int dtype;
    bool qdq;
    const char *label;
    const char *note;
    int64_t blockSize;
    bool nvfp4;
  };

  constexpr float kNvfp4GlobalScale = 0.125f;

  const Variant kFpVariants[] = {
      {ONNX_DT_FLOAT, false, "fp32", "", 0, false},
      {ONNX_DT_FLOAT16, false, "fp16", "", 0, false},
      {ONNX_DT_BFLOAT16, false, "bf16", "", 0, false},
      {ONNX_DT_FLOAT8E4M3FN, true, "fp8_e4m3", "", 0, false},
      {ONNX_DT_FLOAT8E5M2, true, "fp8_e5m2", "", 0, false},
      {ONNX_DT_FLOAT4E2M1, true, "fp4_e2m1", "", 0, false},
      {ONNX_DT_FLOAT4E2M1, false, "nvfp4", "", 16, true},
      {ONNX_DT_FLOAT4E2M1, false, "fp4_weight", "", 32, false},
      {ONNX_DT_INT4, false, "int4_weight", "", 32, false},
  };
  const Variant kIntVariants[] = {
      {ONNX_DT_INT8, true, "int8_qdq", "", 0, false},
  };

  struct QuantScheme
  {
    int actDtype;
    int wDtype;
    const char *name;
  };
  const QuantScheme kInt8Schemes[] = {
      {ONNX_DT_INT8, ONNX_DT_INT8, "signed activations"},
      {ONNX_DT_UINT8, ONNX_DT_INT8, "unsigned activations"},
  };

  size_t schemesFor(const Variant &v, QuantScheme out[2])
  {
    if (v.nvfp4)
    {
      out[0] = {v.dtype, v.dtype, "both operands blocked, with a global scale"};
      return 1;
    }
    if (v.blockSize > 0)
    {
      out[0] = {ONNX_DT_FLOAT16, v.dtype, "16-bit activations against blocked weights"};
      return 1;
    }
    if (v.dtype == ONNX_DT_INT8)
    {
      out[0] = kInt8Schemes[0];
      out[1] = kInt8Schemes[1];
      return 2;
    }
    out[0] = {v.dtype, v.dtype, "matching activations and weights"};
    return 1;
  }

  size_t dtypeSize(int dtype)
  {
    switch (dtype)
    {
    case ONNX_DT_FLOAT:
      return 4;
    case ONNX_DT_FLOAT16:
    case ONNX_DT_BFLOAT16:
      return 2;
    default:
      return 1;
    }
  }

  void fillTensor(std::string &raw, int dtype, int64_t count, uint32_t seed)
  {
    uint32_t s = seed;
    raw.assign((size_t)onnxElemBytes(dtype, count), '\0');
    float *f = reinterpret_cast<float *>(&raw[0]);
    uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
    for (int64_t i = 0; i < count; i++)
    {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      float v = (float)(s >> 8) / 16777216.0f - 0.5f;
      switch (dtype)
      {
      case ONNX_DT_FLOAT:
        f[i] = v;
        break;
      case ONNX_DT_FLOAT16:
        h[i] = floatToHalf(v);
        break;
      case ONNX_DT_BFLOAT16:
        h[i] = floatToBf16(v);
        break;
      default:
        onnxStoreQuantElem(&raw[0], i, dtype, v * 2.0f);
        break;
      }
    }
  }

  float qdqOutputScale(int64_t K, int outDtype)
  {
    const double top = (outDtype == ONNX_DT_FLOAT4E2M1) ? 6.0 : 127.0;
    return (float)(4.0 * std::sqrt((double)K) / 3.0 / top);
  }

  // Minimal setup for probe - mirrors gemm.cpp::makeSetup but at kProbeDim.
  // Any change to gemm.cpp's GemmSetup/finishSetup/makeSetup/timeRuns must be
  // mirrored here (same graph, same inputs).
  struct GemmSetup
  {
    OrtSession *session = nullptr;
    OrtValue *inVal = nullptr;
    OrtValue *zaVal = nullptr;
    OrtValue *outVal = nullptr;
    std::vector<uint8_t> inBuf, zaBuf, outBuf;
    std::string error;
  };

  void destroySetup(const OrtRuntime &rt, GemmSetup &g)
  {
    if (g.inVal)
      rt.api->ReleaseValue(g.inVal);
    if (g.zaVal)
      rt.api->ReleaseValue(g.zaVal);
    if (g.outVal)
      rt.api->ReleaseValue(g.outVal);
    if (g.session)
      rt.api->ReleaseSession(g.session);
    g.inVal = nullptr;
    g.zaVal = nullptr;
    g.outVal = nullptr;
    g.session = nullptr;
  }

  void finishSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep, GemmSetup &g,
                   const std::string &modelBytes, int ioDtype, int64_t D, bool profile,
                   bool keepQdqUnfused, int zaDtype = 0)
  {
    auto ses = onnxCreateSession(rt, ep, modelBytes, true, profile, keepQdqUnfused);
    if (!ses.session)
    {
      g.error = ses.error;
      return;
    }
    g.session = ses.session;
    const size_t es = dtypeSize(ioDtype);
    {
      std::string one;
      fillTensor(one, ioDtype, 1, 0x12345678u);
      g.inBuf.assign(one.begin(), one.end());
    }
    g.outBuf.assign((size_t)D * es, 0);
    OrtMemoryInfo *mi = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mi);
    if (st)
    {
      g.error = onnxStatusText(rt, st);
      destroySetup(rt, g);
      return;
    }
    if (!st && zaDtype)
    {
      g.zaBuf.assign((size_t)onnxElemBytes(zaDtype, 1), 0);
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, g.zaBuf.data(), g.zaBuf.size(), nullptr, 0,
          (ONNXTensorElementDataType)zaDtype, &g.zaVal);
    }
    const int64_t outShape[1] = {D};
    if (!st)
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
  }

  GemmSetup makeSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep, const Variant &v,
                      int64_t D, bool profile, int actDtype, bool reduceInFloat,
                      int wgtDtype, bool unfoldActs)
  {
    GemmSetup g;
    std::string modelBytes;
    if (v.nvfp4)
    {
      std::string aPacked, aScales, bPacked, bScales;
      onnxFillNvfp4(aPacked, aScales, D, D, 1, v.blockSize, kNvfp4GlobalScale, 0x9e3779b9u);
      onnxFillNvfp4(bPacked, bScales, D, D, 0, v.blockSize, kNvfp4GlobalScale, 0x243f6a88u);
      modelBytes = onnxResidentNvfp4MatMulModel(D, D, D, v.blockSize, aPacked, aScales,
                                                bPacked, bScales, kNvfp4GlobalScale);
      finishSetup(rt, ep, g, modelBytes, ONNX_DT_FLOAT, D, profile, true);
      return g;
    }
    if (v.blockSize > 0)
    {
      std::string aRaw, wPacked, wScales;
      fillTensor(aRaw, ONNX_DT_FLOAT16, D * D, 0x9e3779b9u);
      onnxFillBlockedWeights(wPacked, wScales, D, D, v.blockSize, 0x243f6a88u, v.dtype);
      modelBytes = onnxResidentWeightOnlyMatMulModel(D, D, D, v.dtype, v.blockSize, aRaw,
                                                     wPacked, wScales);
      finishSetup(rt, ep, g, modelBytes, ONNX_DT_FLOAT16, D, profile, false);
      return g;
    }
    std::string aRaw, bRaw;
    const int wDtype = v.qdq ? wgtDtype : v.dtype;
    fillTensor(aRaw, v.qdq ? actDtype : v.dtype, D * D, 0x9e3779b9u);
    fillTensor(bRaw, wDtype, D * D, 0x243f6a88u);
    if (v.qdq)
    {
      modelBytes = onnxResidentQdqMatMulModel(D, D, D, aRaw, bRaw, onnxQuantScaleFor(actDtype),
                                              onnxQuantScaleFor(wDtype),
                                              qdqOutputScale(D, actDtype), actDtype, wDtype,
                                              unfoldActs);
    }
    else
    {
      modelBytes = onnxResidentMatMulModel(D, D, D, v.dtype, aRaw, bRaw, reduceInFloat);
    }
    const bool unfusable = !onnxQdqFusionIsLegal(actDtype) || !onnxQdqFusionIsLegal(wDtype);
    const int ioDtype = (v.qdq || reduceInFloat) ? ONNX_DT_FLOAT : v.dtype;
    // Second input exists exactly when the model builds Add-0.
    const int zaDtype = unfoldActs ? (v.qdq ? actDtype : v.dtype) : 0;
    finishSetup(rt, ep, g, modelBytes, ioDtype, D, profile, v.qdq && unfusable,
                zaDtype);
    return g;
  }

  double timeRuns(const OrtRuntime &rt, GemmSetup &g, unsigned int n)
  {
    static const char *inNames[] = {"S", "ZA"};
    static const char *outNames[] = {"Y"};
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n; i++)
    {
      const OrtValue *ins[] = {g.inVal, g.zaVal};
      OrtStatus *st = rt.api->Run(g.session, nullptr, inNames, ins,
                                  g.zaVal ? 2 : 1, outNames, 1, &g.outVal);
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

OnnxProbeCache onnxProbeGemmVariants(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  OnnxProbeCache out;
  // Probe is per-EP and per-variant at 32^3, so do it fresh each call.
  // The cached wrapper below memoizes this for the global once-per-EP probe.

  auto probeOne = [&](const Variant &v)
  {
    OnnxProbeResult r;
    r.actDtype = ONNX_DT_UINT8;
    r.wgtDtype = ONNX_DT_INT8;

    if (std::string why = onnxDtypeUnsupportedReason(rt, v.dtype); !why.empty())
    {
      r.ok = false;
      r.reason = why;
      out[v.label] = r;
      return;
    }

    auto hasCast = [](const std::vector<std::string> &ops) {
      for (const auto &o : ops)
        if (o == "Cast")
          return true;
      return false;
    };

    // One profiled 32^3 build: session + one run + executed ops.  `built` is
    // false when no session came back (`err` set); ops may still be empty on
    // a built session whose profile could not be read.
    auto probeBuild = [&](int actDtype, bool rif, bool unfold, int wgtDtype,
                          double &createUs, std::string &err, bool &built) {
      std::vector<std::string> ops;
      built = false;
      auto t0 = std::chrono::steady_clock::now();
      GemmSetup s = makeSetup(rt, ep, v, kProbeDim, /*profile=*/true, actDtype,
                              rif, wgtDtype, unfold);
      auto t1 = std::chrono::steady_clock::now();
      createUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
      if (!s.session)
      {
        err = s.error;
        return ops;
      }
      built = true;
      timeRuns(rt, s, 1);
      ops = onnxCollectExecutedOps(rt, s.session);
      destroySetup(rt, s);
      return ops;
    };

    // Plain matmul: old result-scaled shape, native then with the product
    // cast (the ladder's own retry order).  An operand Add-0 trick was tried
    // here to defeat load-time folding and reverted: the elementwise either
    // promotes (CPU fp16 profiles an inserted Cast and reads 0.41 against
    // fp32's 0.40) or compiles to a slower program (ANE peaks -20% against
    // these graphs back-to-back).  Folding EPs report an error via the
    // per-doubling timing guard instead.
    if (!v.qdq && v.blockSize == 0 && !v.nvfp4)
    {
      auto t0 = std::chrono::steady_clock::now();
      GemmSetup tiny = makeSetup(rt, ep, v, kProbeDim, false, r.actDtype,
                                 false, r.wgtDtype, false);
      bool triedFloatReduce = false;
      bool reduceInFloat = false;
      if (!tiny.session && !v.qdq && !triedFloatReduce)
      {
        std::string tinyErr = tiny.error;
        destroySetup(rt, tiny);
        tiny = makeSetup(rt, ep, v, kProbeDim, false, r.actDtype, true,
                         r.wgtDtype, false);
        if (tiny.session)
          reduceInFloat = true;
        else
          tiny.error = tinyErr;
      }
      auto t1 = std::chrono::steady_clock::now();
      r.createUs = std::chrono::duration<double, std::micro>(t1 - t0).count();
      r.reduceInFloat = reduceInFloat;
      r.unfoldActs = false;

      CLPEAK_VLOG("onnx-probe[%s/%s]: %lld^3 tiny create %.1f s\n",
                  ep.providerKey.c_str(), v.label, (long long)kProbeDim,
                  r.createUs / 1.0e6);

      if (!tiny.session)
      {
        r.ok = false;
        r.reason = tiny.error;
        out[v.label] = r;
        return;
      }
      if (r.createUs > kOnnxTinyMaxCreateUs)
      {
        CLPEAK_VLOG("onnx-probe[%s/%s]: tiny %.1f s > %.1f s, skipping\n",
                    ep.providerKey.c_str(), v.label, r.createUs / 1.0e6,
                    kOnnxTinyMaxCreateUs / 1.0e6);
        r.ok = false;
        r.reason = "session creation at " + std::to_string(kProbeDim) +
                   "^3 took " + std::to_string((long long)(r.createUs / 1.0e6)) +
                   " s, exceeds tiny budget";
        destroySetup(rt, tiny);
        out[v.label] = r;
        return;
      }
      destroySetup(rt, tiny);
      r.ok = true;
      out[v.label] = r;
      return;
    }

    // Quantized / weight-only / nvfp4: check fusion at 32^3 with profile.
    // The Add-0 trick is attempted only for the types QLinearMatMul can
    // carry (Add has no float8/float4 constraint); weight-only and nvfp4
    // keep the old shape.  Per scheme the trick shape goes first and the old
    // shape follows when the trick fails to build, promotes via a Cast, or
    // stays unfused (a strict matcher may balk at the extra Add).
    {
      std::string tried;
      std::string firstErr;
      QuantScheme schemes[2];
      size_t nSchemes = schemesFor(v, schemes);
      const bool trickOk = v.qdq && onnxQdqFusionIsLegal(v.dtype);
      for (size_t si = 0; si < nSchemes; si++)
      {
        const QuantScheme &qs = schemes[si];
        if (clpeak::cancelRequested())
          break;
        for (int attempt = 0; attempt < 2; attempt++)
        {
          const bool unfold = trickOk && (attempt == 0);
          if (!trickOk && attempt > 0)
            break;
          double cu = 0.0;
          std::string berr;
          bool built = false;
          std::vector<std::string> ops =
              probeBuild(qs.actDtype, false, unfold, qs.wDtype, cu, berr, built);
          CLPEAK_VLOG("onnx-probe[%s/%s]: %s%s tiny probe %lld^3 create %.1f s\n",
                      ep.providerKey.c_str(), v.label, qs.name,
                      unfold ? " unfold" : "", (long long)kProbeDim, cu / 1.0e6);
          if (!built)
          {
            if (firstErr.empty())
              firstErr = berr;
            continue;
          }
          std::string joined;
          for (auto &o : ops)
            joined += (joined.empty() ? "" : ", ") + o;
          CLPEAK_VLOG("onnx-probe[%s/%s]: %s%s executed %s\n",
                      ep.providerKey.c_str(), v.label, qs.name,
                      unfold ? " unfold" : "", joined.c_str());
          if (unfold && hasCast(ops))
          {
            CLPEAK_VLOG("onnx-probe[%s/%s]: trick session runs via an inserted "
                        "Cast, retrying the old shape\n",
                        ep.providerKey.c_str(), v.label);
            continue;
          }
          if (onnxOpsRanQuantizedMatMul(ops))
          {
            r.ok = true;
            r.actDtype = qs.actDtype;
            r.wgtDtype = qs.wDtype;
            r.schemeName = qs.name;
            for (auto &o : ops)
              if (o == "Cast")
                r.castedActs = true;
            r.ranAs = onnxQuantizedKernelName(ops);
            if (r.ranAs.empty())
              r.ranAs = "a kernel it compiled itself";
            r.createUs = cu;
            r.unfoldActs = unfold;
            out[v.label] = r;
            return;
          }
          if (!ops.empty())
            tried = joined;
        }
      }
      r.ok = false;
      r.reason = tried.empty() ? (firstErr.empty() ? "no fused quantized matmul" : firstErr)
                               : "provider did not fuse a quantized matmul (ran: " + tried + ")";
      out[v.label] = r;
      return;
    }
  };

  for (auto &v : kFpVariants)
    probeOne(v);
  for (auto &v : kIntVariants)
    probeOne(v);

  return out;
}

const OnnxProbeCache &onnxProbeGemmCache(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  static std::unordered_map<std::string, OnnxProbeCache> memo;
  static std::mutex mtx;
  // OpenVINO shares one providerKey across its NPU/GPU/CPU targets, which
  // compile and fuse independently -- key by target too, or the GPU row
  // would reuse the NPU probe.
  const std::string memoKey =
      ep.providerKey + '\x1f' + ep.epDevice;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = memo.find(memoKey);
  if (it != memo.end())
    return it->second;
  auto cache = onnxProbeGemmVariants(rt, ep);
  auto res = memo.emplace(memoKey, std::move(cache));
  return res.first->second;
}

bool onnxEpViable(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                  std::string &reason)
{
  // Memoized: listing (CLI, FFI catalog) and runAll() each ask, and the
  // answer cannot change while the runtime stays mapped (handles are
  // leaked on purpose, so the base pointer is a stable identity).
  static std::unordered_map<std::string, std::pair<bool, std::string>> memo;
  static std::mutex mtx;
  const std::string memoKey =
      std::to_string((uintptr_t)(const void *)rt.base) + '\x1f' +
      ep.providerKey + '\x1f' + ep.epDevice;
  {
    std::lock_guard<std::mutex> lk(mtx);
    auto it = memo.find(memoKey);
    if (it != memo.end())
    {
      reason = it->second.second;
      return it->second.first;
    }
  }

  // A provider that cannot create even one of these can run nothing here.
  // fp32 and fp16 are the plain matmuls every float-capable EP takes, and
  // int8 QDQ in both spellings covers an integer-only NPU with no float
  // path at all (signed fuses on ARM/TensorRT, unsigned on pre-VNNI x86).
  // All three spell at opset 17, so no runtime gate applies.
  //
  // Creation only, no runs: the failures this filters -- an absent device
  // (OpenVINO NPU on a box with no NPU) or a graph no EP backend takes
  // under the fallback guard (NNAPI declining everything) -- all happen
  // at creation, fast, with nothing compiled.
  //
  // Plain variants try the native reduction only: an EP missing just the
  // reduction still runs fp32, whose reduction is universal, so the cast
  // retry the ladder uses would add nothing to the verdict.
  const std::string tag =
      ep.providerKey + (ep.epDevice.empty() ? "" : "/" + ep.epDevice);
  std::string firstErr;
  auto tryBuild = [&](const Variant &v, int actDtype, int wgtDtype) {
    if (clpeak::cancelRequested())
      return false;
    GemmSetup s =
        makeSetup(rt, ep, v, kProbeDim, /*profile=*/false, actDtype,
                  /*reduceInFloat=*/false, wgtDtype, /*unfoldActs=*/false);
    const bool ok = (s.session != nullptr);
    if (!ok && firstErr.empty())
      firstErr = s.error;
    CLPEAK_VLOG("onnx-viable[%s]: %s%d%s (%s)\n", tag.c_str(),
                v.qdq ? "qdq-" : "", v.dtype,
                v.qdq ? (actDtype == ONNX_DT_UINT8 ? "-u8" : "-s8") : "",
                ok ? "ok" : s.error.c_str());
    destroySetup(rt, s);
    return ok;
  };

  static const Variant kFp32{ONNX_DT_FLOAT, false, "", "", 0, false};
  static const Variant kFp16{ONNX_DT_FLOAT16, false, "", "", 0, false};
  static const Variant kInt8{ONNX_DT_INT8, true, "", "", 0, false};
  const bool ok = tryBuild(kFp32, 0, 0) || tryBuild(kFp16, 0, 0) ||
                  tryBuild(kInt8, ONNX_DT_INT8, ONNX_DT_INT8) ||
                  tryBuild(kInt8, ONNX_DT_UINT8, ONNX_DT_INT8);

  reason = ok ? "" : (firstErr.empty() ? "the provider refused every probe graph"
                                       : firstErr);
  {
    std::lock_guard<std::mutex> lk(mtx);
    memo.emplace(memoKey, std::make_pair(ok, reason));
  }
  return ok;
}

#endif // ENABLE_ONNX
