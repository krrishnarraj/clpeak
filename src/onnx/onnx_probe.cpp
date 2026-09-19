#ifdef ENABLE_ONNX

#include "onnx_probe.h"
#include "gemm_setup.h"
#include "onnx_model.h"
#include "onnx_session.h"

#include <onnx/onnx_peak.h>
#include <common/common.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace onnxgemm;

namespace
{

double elapsedUs(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

const char *shapeName(OnnxLiveShape s)
{
  switch (s)
  {
  case OnnxLiveShape::ResultScaled:  return "result-scaled";
  case OnnxLiveShape::OperandScaled: return "operand-scaled";
  case OnnxLiveShape::Add0:          return "add-0";
  case OnnxLiveShape::QdqAdd0:       return "qdq-add-0";
  }
  return "?";
}

// One profiled 32^3 build: session, one run, the kernels that executed.
struct Build
{
  bool built = false;
  std::string error;
  double createUs = 0.0;
  std::vector<std::string> ops; // empty on a built session whose profile
                                // could not be read, or an opaque one
  std::string matmulInType;     // element type the MatMul kernel consumed
  size_t casts = 0;
  double runUs = 0.0;           // one run at 32^3: essentially dispatch
};

Build probeBuild(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                 const Variant &v, int actDtype, bool reduceInFloat,
                 int wgtDtype, OnnxLiveShape shape)
{
  Build b;
  auto t0 = std::chrono::steady_clock::now();
  GemmSetup s = makeSetup(rt, ep, v, kProbeDim, /*profile=*/true, actDtype,
                          reduceInFloat, wgtDtype, shape);
  b.createUs = elapsedUs(t0);
  if (!s.session)
  {
    b.error = s.error;
    return b;
  }
  b.built = true;
  // What the provider charges to accept a submission of this graph: at 32^3
  // the arithmetic is 65 kFLOP, so the time is all overhead.  The ladder
  // subtracts this, so over-measuring it is not harmless -- a first rung
  // then looks like it computed nothing and the row is called folded.
  //
  // Two warmups and the best of two batches, because a single run after a
  // single warmup catches whatever the provider defers to first use.  On
  // CUDA that read 199 us against a 17 us dispatch, which condemned four
  // rows that were working perfectly.
  timeRuns(rt, s, 2);
  const double a = timeRuns(rt, s, 5);
  const double b2 = timeRuns(rt, s, 5);
  const double runUs = (a > 0.0 && b2 > 0.0) ? std::min(a, b2)
                                             : std::max(a, b2);
  b.runUs = (runUs > 0.0) ? runUs : 0.0;
  if (runUs < 0.0)
  {
    // Built but will not run: as good as refused, and the run's own message
    // is the one worth keeping.
    b.built = false;
    b.error = s.error;
    destroySetup(rt, s);
    return b;
  }
  b.ops = onnxCollectExecutedOps(rt, s.session, &b.matmulInType);
  b.casts = onnxCountOp(b.ops, "Cast");
  destroySetup(rt, s);
  return b;
}

} // namespace

OnnxProbeCache onnxProbeGemmVariants(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  OnnxProbeCache out;

  // For every variant, per scheme: try each candidate shape in preference
  // order (result-scaled first) and keep every one that builds, fuses where
  // the row needs it, and runs the multiply at the row's own width.  The
  // ladder walks the kept list, using the first and dropping to the next only
  // when it catches that one folding -- so a non-folding provider measures the
  // fast result-scaled shape and a folder falls through to a live one.  The
  // The width check is a *preference*, not a veto.  It exists to stop a live
  // operand being chosen after ORT quietly widened the multiply behind it --
  // Apple's CPU EP keeps fp16 in the result-scaled shape and promotes it in
  // the operand-scaled one, and picking the wrong one there reports the fp32
  // rate under the fp16 label.  But where *no* shape keeps the width the
  // cause is the provider rather than the shape: ONNX Runtime 1.17's x86 CPU
  // EP has no fp16 MatMul at all and casts in every shape.  Refusing there
  // would delete the fp16 row and, through this cache, every fp16 row in conv
  // and the transformer block.  So widening shapes are kept as a second
  // choice and the row reports the width it actually ran at.
  //
  // The plain-float rows have a bare MatMul to read the type from; the
  // quantized ones dequantize to float by design and the weight-only ones
  // fuse to MatMulNBits, so neither is checked.
  auto probeOne = [&](const Variant &v)
  {
    OnnxProbeResult r;
    r.actDtype = ONNX_DT_UINT8;
    r.wgtDtype = ONNX_DT_INT8;

    if (std::string why = onnxDtypeUnsupportedReason(rt, v.dtype); !why.empty())
    {
      r.reason = why;
      out[v.label] = r;
      return;
    }
    // A graph this provider crashes on rather than declines is not built,
    // here or anywhere downstream of this cache.
    if (std::string why = onnxProviderFenceReason(ep, v.dtype, v.qdq); !why.empty())
    {
      CLPEAK_VLOG("onnx-probe[%s/%s]: not built: %s\n", ep.providerKey.c_str(),
                  v.label, why.c_str());
      r.reason = why;
      out[v.label] = r;
      return;
    }

    const bool needsFusion = v.qdq || v.blockSize > 0 || v.nvfp4;
    const std::vector<OnnxLiveShape> shapes = liveShapesFor(v);
    const char *want = onnxProfileTypeName(v.dtype);

    QuantScheme schemes[2];
    const size_t nSchemes = schemesFor(v, schemes);

    std::string firstErr;
    std::string tried; // executed ops of the last unfused build, for the skip
    // Shapes that built and ran, but at a wider datatype than the row names.
    // Kept as a second choice: see the note above.
    struct Widened
    {
      OnnxLiveShape shape;
      bool reduceInFloat;
      double createUs;
      double runUs;
      std::string ranAs;
    };
    std::vector<Widened> wide;

    for (size_t si = 0; si < nSchemes && !r.ok; si++)
    {
      const QuantScheme &qs = schemes[si];
      if (clpeak::cancelRequested())
        break;

      bool reduceInFloat = false;
      for (OnnxLiveShape shape : shapes)
      {
        if (clpeak::cancelRequested())
          break;

        // Plain rows retry with the product cast to fp32 when the native
        // reduction is refused (the ladder's own retry order); the quantized
        // rows already reduce a dequantized fp32 result.  A cast decided for
        // one shape carries to the rest.
        Build b;
        for (int rif = 0; rif < 2 && !b.built; rif++)
        {
          if (rif == 1 && needsFusion)
            break;
          b = probeBuild(rt, ep, v, qs.actDtype, reduceInFloat || rif == 1,
                         qs.wDtype, shape);
          if (b.built && rif == 1)
            reduceInFloat = true;
          CLPEAK_VLOG("onnx-probe[%s/%s]: %s %s%s %lld^3 create %.1f s (%s)\n",
                      ep.providerKey.c_str(), v.label, qs.name,
                      shapeName(shape),
                      (reduceInFloat || rif == 1) ? " fp32-reduce" : "",
                      (long long)kProbeDim, b.createUs / 1.0e6,
                      b.built ? onnxJoinOps(b.ops).c_str() : b.error.c_str());
        }
        if (!b.built)
        {
          if (firstErr.empty())
            firstErr = b.error;
          continue;
        }
        if (needsFusion && !onnxOpsRanQuantizedMatMul(b.ops))
        {
          if (!b.ops.empty())
            tried = onnxJoinOps(b.ops);
          continue;
        }
        const bool widened = !needsFusion && want[0] &&
                             !b.matmulInType.empty() && b.matmulInType != want;
        if (widened)
        {
          CLPEAK_VLOG("onnx-probe[%s/%s]: %s ran the multiply in %s, not %s\n",
                      ep.providerKey.c_str(), v.label, shapeName(shape),
                      b.matmulInType.c_str(), want);
          wide.push_back({shape, reduceInFloat, b.createUs, b.runUs,
                          b.matmulInType});
          continue;
        }

        // Viable at the right width.  The first such shape settles the scheme
        // and the reported provenance; later ones only extend the fallback
        // list.
        if (!r.ok)
        {
          r.ok = true;
          r.reduceInFloat = reduceInFloat;
          r.createUs = b.createUs;
          r.probeUs = b.runUs;
          r.actDtype = qs.actDtype;
          r.wgtDtype = qs.wDtype;
          if (needsFusion)
          {
            r.schemeName = qs.name;
            r.castedActs = b.casts > 0;
            r.ranAs = onnxQuantizedKernelName(b.ops);
            if (r.ranAs.empty())
              r.ranAs = "a kernel it compiled itself";
          }
        }
        r.shapes.push_back(shape);
      }
    }

    // Nothing kept the width, but something ran: this provider has no kernel
    // for the row's datatype and computed it wider.  Report that rather than
    // dropping the row.
    if (!r.ok && !wide.empty())
    {
      const Widened &w = wide.front();
      r.ok = true;
      r.reduceInFloat = w.reduceInFloat;
      r.createUs = w.createUs;
      r.probeUs = w.runUs;
      r.ranWider = w.ranAs;
      for (const Widened &x : wide)
        r.shapes.push_back(x.shape);
      CLPEAK_VLOG("onnx-probe[%s/%s]: no shape keeps %s; the provider computes "
                  "it in %s, measuring that\n",
                  ep.providerKey.c_str(), v.label, want, w.ranAs.c_str());
    }

    if (r.ok)
      CLPEAK_VLOG("onnx-probe[%s/%s]: shapes %zu (first %s)%s\n",
                  ep.providerKey.c_str(), v.label, r.shapes.size(),
                  shapeName(r.shapes.front()),
                  r.reduceInFloat ? ", product cast to fp32" : "");

    if (r.ok && r.createUs > kOnnxTinyMaxCreateUs)
    {
      CLPEAK_VLOG("onnx-probe[%s/%s]: tiny %.1f s > %.1f s, skipping\n",
                  ep.providerKey.c_str(), v.label, r.createUs / 1.0e6,
                  kOnnxTinyMaxCreateUs / 1.0e6);
      r.ok = false;
      r.reason = "session creation at " + std::to_string(kProbeDim) +
                 "^3 took " + std::to_string((long long)(r.createUs / 1.0e6)) +
                 " s, exceeds tiny budget";
    }
    if (!r.ok && r.reason.empty())
    {
      if (!tried.empty())
        r.reason = "provider did not fuse a quantized matmul (ran: " + tried + ")";
      else if (!firstErr.empty())
        r.reason = firstErr;
      else
        // Say what was actually being looked for.  "No fused quantized
        // matmul" on a plain fp16 row named the wrong question entirely.
        r.reason = needsFusion ? "no fused quantized matmul"
                               : "this provider built no usable graph for " +
                                     std::string(v.label);
    }
    out[v.label] = r;
  };

  for (size_t i = 0; i < kFpVariantCount; i++)
    probeOne(kFpVariants[i]);
  for (size_t i = 0; i < kIntVariantCount; i++)
    probeOne(kIntVariants[i]);

  return out;
}

const OnnxProbeCache &onnxProbeGemmCache(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  static std::unordered_map<std::string, OnnxProbeCache> memo;
  static std::mutex mtx;
  // OpenVINO shares one providerKey across its NPU/GPU/CPU targets, which
  // compile and fuse independently -- key by target too, or the GPU row
  // would reuse the NPU probe.  And by runtime, like the other memos here:
  // the GUI can point the backend at a different ONNX Runtime between runs,
  // and what a provider fuses or which widths it casts is that runtime's
  // answer, not the provider name's.  Handles are never unmapped, so the
  // base pointer is a stable identity, and the same path picked again
  // reuses its handle and so its probe.
  const std::string memoKey =
      std::to_string((uintptr_t)(const void *)rt.base) + '\x1f' +
      ep.providerKey + '\x1f' + ep.epDevice;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = memo.find(memoKey);
  if (it != memo.end())
    return it->second;
  auto cache = onnxProbeGemmVariants(rt, ep);
  auto res = memo.emplace(memoKey, std::move(cache));
  return res.first->second;
}

// ---------------------------------------------------------------------------
// Streaming width
// ---------------------------------------------------------------------------

// Both answers come from one sweep, so they are memoized together.
struct StreamProbe
{
  int dtype = ONNX_DT_FLOAT16;
  double bps = 0.0;
};

static StreamProbe onnxStreamProbe(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  static std::unordered_map<std::string, StreamProbe> memo;
  static std::mutex mtx;
  const std::string memoKey =
      std::to_string((uintptr_t)(const void *)rt.base) + '\x1f' +
      ep.providerKey + '\x1f' + ep.epDevice;
  {
    std::lock_guard<std::mutex> lk(mtx);
    auto it = memo.find(memoKey);
    if (it != memo.end())
      return it->second;
  }

  // Both the choice and the rate are taken net of a dispatch floor, measured
  // with a 256-square matrix whose 128 KB cannot matter -- exactly what
  // onnx-tensor-bw does.  Without it the figure is mostly the cost of asking:
  // DirectML charges 156 us a submission and read 48 GB/s raw where it
  // actually streams 717.
  //
  // The working set has to be large enough that the transfer outweighs that
  // charge, and eight megabytes is not.  A fast card moves them in about
  // five microseconds against thirty of dispatch, so the remainder is the
  // difference of two nearly equal numbers: CUDA read 2511 GB/s from a 3 us
  // net, and TensorRT's two widths came out 561 and 965 on noise alone.
  // Sixty-four megabytes is four times the dispatch on the fastest provider
  // here and still one session per width -- but it is also 64 MB of weights
  // three times over at peak, so a small device falls back to eight and
  // reports whatever it can.
  constexpr int64_t kK = 2048;
  constexpr int64_t kFloorDim = 256; // 128 KB either way: cannot matter
  const uint64_t kBytes = (clpeak::memoryBudget(1ull << 30) > (192ull << 20))
                              ? (64ull << 20)
                              : (8ull << 20);
  // Below this share the reading is dispatch, not bandwidth.  Reporting it
  // would put a meaningless ceiling under the activation rows -- too low
  // refuses every one of them, too high checks nothing -- so the probe says
  // it has no answer instead, and only the proportional guard applies.
  constexpr double kMinNetShare = 0.25;
  const int candidates[2] = {ONNX_DT_FLOAT16, ONNX_DT_FLOAT};
  StreamProbe best;

  // Time one [1,k]x[k,n] product of `dtype`; negative when it could not run.
  auto timeGemv = [&](int dtype, int64_t k, int64_t n) -> double {
    const size_t es = dtypeSize(dtype);
    std::string w;
    fillTensor(w, dtype, k * n, 0x243f6a88u);
    std::string model = onnxMatMulModel(1, k, n, dtype, w);
    std::string().swap(w);
    auto ses = onnxCreateSession(rt, ep, model);
    std::string().swap(model);
    if (!ses.session)
    {
      CLPEAK_VLOG("onnx-stream[%s]: %s refused (%s)\n", ep.providerKey.c_str(),
                  dtype == ONNX_DT_FLOAT ? "fp32" : "fp16", ses.error.c_str());
      return -1.0;
    }

    std::vector<uint8_t> x((size_t)k * es, 0), y((size_t)n * es, 0);
    {
      std::string one = onnxFloatScalar(0.5f, dtype);
      for (int64_t i = 0; i < k; i++)
        std::memcpy(&x[(size_t)i * es], one.data(), es);
    }
    OrtMemoryInfo *mi = nullptr;
    OrtValue *xv = nullptr, *yv = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                OrtMemTypeDefault, &mi);
    const int64_t xs[2] = {1, k}, ys[2] = {1, n};
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, x.data(), x.size(), xs, 2, (ONNXTensorElementDataType)dtype, &xv);
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, y.data(), y.size(), ys, 2, (ONNXTensorElementDataType)dtype, &yv);
    if (mi)
      rt.api->ReleaseMemoryInfo(mi);

    double us = -1.0;
    if (!st)
    {
      static const char *ins[] = {"A"};
      static const char *outs[] = {"C"};
      auto run = [&](unsigned int reps) -> double {
        auto t0 = std::chrono::steady_clock::now();
        for (unsigned int i = 0; i < reps; i++)
        {
          OrtStatus *rs = rt.api->Run(ses.session, nullptr, ins,
                                      (const OrtValue *const *)&xv, 1,
                                      outs, 1, &yv);
          if (rs)
          {
            rt.api->ReleaseStatus(rs);
            return -1.0;
          }
        }
        return elapsedUs(t0) / reps;
      };
      // Warm, then a short batch: this decides a width, not a rate.
      if (run(3) > 0.0)
      {
        double probe = run(1);
        if (probe > 0.0)
        {
          unsigned int reps = pickIters(probe, 200000u, 0, 50u);
          us = (reps > 1) ? run(reps) : probe;
        }
      }
    }
    else
      rt.api->ReleaseStatus(st);

    if (xv)
      rt.api->ReleaseValue(xv);
    if (yv)
      rt.api->ReleaseValue(yv);
    rt.api->ReleaseSession(ses.session);
    return us;
  };

  for (int dtype : candidates)
  {
    if (clpeak::cancelRequested())
      break;
    const size_t es = dtypeSize(dtype);
    // The floor has to be small in *both* dimensions.  Shrinking only the
    // columns leaves a [1,2048]x[2048,256] matrix -- 2 MB of the 8 being
    // measured -- and subtracting a quarter of the transfer reported Core ML
    // at 150 GB/s where onnx-tensor-bw, which subtracts a real floor, says 83.
    const double floorUs = timeGemv(dtype, kFloorDim, kFloorDim);
    const double us = timeGemv(dtype, kK, (int64_t)(kBytes / (kK * es)));
    if (us <= 0.0)
      continue;
    const double netUs = us - ((floorUs > 0.0) ? floorUs : 0.0);
    if (netUs < kMinNetShare * us)
    {
      CLPEAK_VLOG("onnx-stream[%s]: %s %.0f us against a %.0f us floor -- "
                  "dispatch, not bandwidth; no reading\n",
                  ep.providerKey.c_str(),
                  dtype == ONNX_DT_FLOAT ? "fp32" : "fp16", us, floorUs);
      continue;
    }
    const double bps = (double)kBytes / (netUs * 1.0e-6);
    CLPEAK_VLOG("onnx-stream[%s]: %s %.1f GB/s (%.0f us less a %.0f us floor)\n",
                ep.providerKey.c_str(),
                dtype == ONNX_DT_FLOAT ? "fp32" : "fp16", bps / 1.0e9,
                us, floorUs);
    if (bps > best.bps)
    {
      best.bps = bps;
      best.dtype = dtype;
    }
  }

  std::lock_guard<std::mutex> lk(mtx);
  memo.emplace(memoKey, best);
  return best;
}

int onnxStreamDtype(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  return onnxStreamProbe(rt, ep).dtype;
}

double onnxStreamBps(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  return onnxStreamProbe(rt, ep).bps;
}

// ---------------------------------------------------------------------------
// Folding record
// ---------------------------------------------------------------------------

namespace
{
std::string foldKey(const onnx_ep_info_t &ep)
{
  return ep.providerKey + '\x1f' + ep.epDevice;
}
struct FoldStore
{
  std::mutex mtx;
  std::unordered_map<std::string, std::unordered_set<std::string>> map;
};
FoldStore &foldStore()
{
  static FoldStore s;
  return s;
}
} // namespace

void onnxNoteGemmFolded(const onnx_ep_info_t &ep, const std::string &label)
{
  FoldStore &s = foldStore();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.map[foldKey(ep)].insert(label);
}

bool onnxGemmFolded(const onnx_ep_info_t &ep, const std::string &label)
{
  FoldStore &s = foldStore();
  std::lock_guard<std::mutex> lk(s.mtx);
  auto it = s.map.find(foldKey(ep));
  if (it == s.map.end())
    return false;
  return it->second.count(label) > 0;
}

void onnxClearGemmFolded(const onnx_ep_info_t &ep)
{
  FoldStore &s = foldStore();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.map.erase(foldKey(ep));
}

// ---------------------------------------------------------------------------
// Viability
// ---------------------------------------------------------------------------

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

  // Tier 0 -- attach, no model and no session.  A target the provider
  // cannot serve fails here, fast, with nothing compiled (OpenVINO NPU
  // with no NPU, a QNN backend_path with no library behind it).  Session
  // creation runs the same append first, so an attach failure means every
  // later attempt would fail identically: record it and stop.
  //
  // Tier 1 -- the trivial graph: one Mul over 64 fp16 values, the cheapest
  // session every ahead-of-time compiler here can build (the
  // dispatch-latency test's graph, shared via onnxTrivialMulModel).  A
  // live EP answers here and never pays for a matmul it was only ever
  // going to pass.  The output depends on a runtime input, so there is
  // nothing to fold and default session options match the latency test's
  // own session exactly.
  //
  // Tier 2 -- matmul legs: fp32 and fp16 plain plus int8 QDQ in both
  // spellings.  No known EP needs these to prove viable, but an
  // integer-only NPU with no float path would decline the trivial Mul
  // while fusing int8 QDQ, and only these legs catch that.  All three
  // spell at opset 17, so no runtime gate applies.
  //
  // Creation only, no runs: the failures this filters all happen at
  // creation -- an absent device or a graph no EP backend takes under the
  // fallback guard -- fast, with nothing compiled on the failure paths.
  const std::string tag =
      ep.providerKey + (ep.epDevice.empty() ? "" : "/" + ep.epDevice);
  std::string firstErr;

  {
    auto t0 = std::chrono::steady_clock::now();
    std::string attachErr = onnxProviderAttach(rt, ep);
    CLPEAK_VLOG("onnx-viable[%s]: attach %.1f ms (%s)\n", tag.c_str(),
                elapsedUs(t0) / 1000.0,
                attachErr.empty() ? "ok" : attachErr.c_str());
    if (!attachErr.empty())
    {
      reason = attachErr;
      std::lock_guard<std::mutex> lk(mtx);
      memo.emplace(memoKey, std::make_pair(false, reason));
      return false;
    }
  }

  {
    auto t0 = std::chrono::steady_clock::now();
    auto ses = onnxCreateSession(rt, ep, onnxTrivialMulModel());
    const double us = elapsedUs(t0);
    if (ses.session)
    {
      CLPEAK_VLOG("onnx-viable[%s]: trivial %.1f ms (ok)\n", tag.c_str(),
                  us / 1000.0);
      rt.api->ReleaseSession(ses.session);
      reason.clear();
      std::lock_guard<std::mutex> lk(mtx);
      memo.emplace(memoKey, std::make_pair(true, reason));
      return true;
    }
    CLPEAK_VLOG("onnx-viable[%s]: trivial %.1f ms (%s)\n", tag.c_str(),
                us / 1000.0, ses.error.c_str());
    firstErr = ses.error;
  }

  auto tryBuild = [&](const Variant &v, int actDtype, int wgtDtype) {
    if (clpeak::cancelRequested())
      return false;
    auto t0 = std::chrono::steady_clock::now();
    GemmSetup s =
        makeSetup(rt, ep, v, kProbeDim, /*profile=*/false, actDtype,
                  /*reduceInFloat=*/false, wgtDtype,
                  OnnxLiveShape::ResultScaled);
    const double us = elapsedUs(t0);
    const bool ok = (s.session != nullptr);
    if (!ok && firstErr.empty())
      firstErr = s.error;
    CLPEAK_VLOG("onnx-viable[%s]: %s%d%s %.1f ms (%s)\n", tag.c_str(),
                v.qdq ? "qdq-" : "", v.dtype,
                v.qdq ? (actDtype == ONNX_DT_UINT8 ? "-u8" : "-s8") : "",
                us / 1000.0, ok ? "ok" : s.error.c_str());
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
