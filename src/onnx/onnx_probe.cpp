#ifdef ENABLE_ONNX

#include "onnx_probe.h"
#include "gemm_setup.h"
#include "onnx_model.h"
#include "onnx_session.h"

#include <onnx/onnx_peak.h>
#include <common/common.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
  // Unverified placement (onnx_session.h): a 32-cube runs on Core ML's CPU
  // whatever unit was asked for, and this probe asks what builds and fuses,
  // not where the ladder's sizes will run.
  GemmSetup s = makeSetup(rt, ep, v, kProbeDim, /*profile=*/true, actDtype,
                          reduceInFloat, wgtDtype, shape,
                          /*verifyPlacement=*/false);
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
    // A blocked weight is spelled with DequantizeLinear's block_size, opset
    // 21, whatever its own type needs: int8 is opset 17 on its own and the
    // check above would pass it on a runtime that cannot load the graph.
    if (v.blockSize > 0)
    {
      const uint32_t needApi = onnxMinOrtApiForOpset(21);
      if (needApi && rt.apiVersion < needApi)
      {
        r.reason = "needs opset 21 for the blocked weight scales, which arrived in "
                   "ONNX Runtime 1." + std::to_string(needApi) + "; this runtime is " +
                   rt.versionString;
        out[v.label] = r;
        return;
      }
    }
    // A graph this provider crashes on rather than declines is not built,
    // here or anywhere downstream of this cache.
    if (std::string why = onnxProviderFenceReason(ep, v.dtype, v.qdq, v.blockSize);
        !why.empty())
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

// All three answers come from one sweep, so they are memoized together.
struct StreamProbe
{
  int dtype = ONNX_DT_FLOAT16;
  double bps = 0.0;
  bool fp32Narrowed = false;
};

// An fp32 matrix-vector product at full width lands within a few parts per
// million of a double-precision reference: 2048 terms of rounding at 2^-24
// is under 2 ppm on the values used here.  Holding the operands in 16 bits
// rounds each one at 2^-11 and lands near 300 ppm; bf16 near 2000.  20 ppm
// sits an order of magnitude from both.
constexpr double kFp32ExactRelErr = 2.0e-5;

// Narrowed storage streams the same bytes in half the time, so the fp32
// reading comes out near twice fp16's; full-width fp32 comes out near equal.
constexpr double kNarrowedRateRatio = 1.5;

// The same question, asked so that the answer does not rest on time scaling
// with work.  The ratio above sets fp16's product beside an fp32 one of half
// its columns, so where fp32 is held at 16 bits it compares two shapes doing
// different amounts of work -- and on an OpenVINO GPU (Arc A380) the two did
// not scale together: fp32 read 1.28x fp16's rate while moving half its
// credited bytes, because per byte it actually moved the narrower product
// streamed 56 GB/s against fp16's 87, where onnx-tensor-bw's second-long
// rungs read 102 to 127 on the same card.  So fp16 is also timed at the fp32
// product's own shape.  Held at half width, the two are one 16-bit kernel over
// the same bytes and take the same time whatever dispatch, clocks or shape
// do; at full width fp32 moves twice the bytes.  The same time is within this
// factor either way -- the block's decode rows read 0.99x and 1.01x on that
// card.  On a full-width provider this fires only where fp32 streams 1.6x
// (2 / 1.25) fp16's bytes per second, which the ratio above already calls
// narrowed.
constexpr double kSameShapeTimeRatio = 1.25;

// The same-shape pair runs in alternating batches and each keeps its best, so
// a clock that drifts while they run lands on both rather than on one.
constexpr int kSameShapeRounds = 3;

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
  // here and still one session per width, plus fp16 at fp32's shape where
  // fp32 comes back rounded (kSameShapeTimeRatio) -- but the fp32 product is
  // held while that one builds, 96 MB of weights counted as every gate counts
  // them (onnxHeldBytes), so a small device falls back to eight and reports
  // whatever it can.
  constexpr int64_t kK = 2048;
  constexpr int64_t kFloorDim = 256; // 128 KB either way: cannot matter
  const uint64_t kBytes =
      (clpeak::memoryBudget(1ull << 30) > onnxHeldBytes(ep, 96ull << 20, 0))
          ? (64ull << 20)
          : (8ull << 20);
  // Below this share the reading is dispatch, not bandwidth.  Reporting it
  // would put a meaningless ceiling under the activation rows -- too low
  // refuses every one of them, too high checks nothing -- so the probe says
  // it has no answer instead, and only the proportional guard applies.
  constexpr double kMinNetShare = 0.25;
  const int candidates[2] = {ONNX_DT_FLOAT16, ONNX_DT_FLOAT};
  StreamProbe best;

  // One [1,k]x[k,n] product of `dtype`, built, its input and output bound.
  // Every input element is 0.5, so an fp32 product's exact answer is half of
  // each column's sum -- of the fp32 values the model actually holds -- and
  // `ref` keeps it when the product is to be scored.
  struct Gemv
  {
    OrtSession *session = nullptr;
    OrtValue *xv = nullptr, *yv = nullptr;
    std::vector<uint8_t> x, y;
    std::vector<double> ref;
  };
  auto release = [&](Gemv &g) {
    if (g.xv)
      rt.api->ReleaseValue(g.xv);
    if (g.yv)
      rt.api->ReleaseValue(g.yv);
    if (g.session)
      rt.api->ReleaseSession(g.session);
    g = Gemv();
  };
  // Build one in `g`; false when it could not be.
  auto build = [&](Gemv &g, int dtype, int64_t k, int64_t n,
                   bool score) -> bool {
    const size_t es = dtypeSize(dtype);
    std::string w;
    fillTensor(w, dtype, k * n, 0x243f6a88u);
    if (score && dtype == ONNX_DT_FLOAT)
    {
      g.ref.assign((size_t)n, 0.0);
      const float *wf = reinterpret_cast<const float *>(w.data());
      for (int64_t i = 0; i < k; i++)
        for (int64_t j = 0; j < n; j++)
          g.ref[(size_t)j] += 0.5 * (double)wf[(size_t)(i * n + j)];
    }
    std::string model = onnxMatMulModel(1, k, n, dtype, w);
    std::string().swap(w);
    auto ses = onnxCreateSession(rt, ep, model);
    std::string().swap(model);
    if (!ses.session)
    {
      CLPEAK_VLOG("onnx-stream[%s]: %s refused (%s)\n", ep.providerKey.c_str(),
                  dtype == ONNX_DT_FLOAT ? "fp32" : "fp16", ses.error.c_str());
      release(g);
      return false;
    }
    g.session = ses.session;

    g.x.assign((size_t)k * es, 0);
    g.y.assign((size_t)n * es, 0);
    {
      std::string one = onnxFloatScalar(0.5f, dtype);
      for (int64_t i = 0; i < k; i++)
        std::memcpy(&g.x[(size_t)i * es], one.data(), es);
    }
    OrtMemoryInfo *mi = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                OrtMemTypeDefault, &mi);
    const int64_t xs[2] = {1, k}, ys[2] = {1, n};
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, g.x.data(), g.x.size(), xs, 2, (ONNXTensorElementDataType)dtype,
          &g.xv);
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, g.y.data(), g.y.size(), ys, 2, (ONNXTensorElementDataType)dtype,
          &g.yv);
    if (mi)
      rt.api->ReleaseMemoryInfo(mi);
    if (st)
    {
      rt.api->ReleaseStatus(st);
      release(g);
      return false;
    }
    return true;
  };
  // Run a built product `reps` times: the mean per run in microseconds, or
  // -1 when a run failed.
  auto run = [&](Gemv &g, unsigned int reps) -> double {
    static const char *ins[] = {"A"};
    static const char *outs[] = {"C"};
    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < reps; i++)
    {
      OrtStatus *rs = rt.api->Run(g.session, nullptr, ins,
                                  (const OrtValue *const *)&g.xv, 1, outs, 1,
                                  &g.yv);
      if (rs)
      {
        rt.api->ReleaseStatus(rs);
        return -1.0;
      }
    }
    return elapsedUs(t0) / reps;
  };
  // Warm, then a short batch: this decides a width, not a rate.
  auto timeBuilt = [&](Gemv &g) -> double {
    if (run(g, 3) <= 0.0)
      return -1.0;
    const double probe = run(g, 1);
    if (probe <= 0.0)
      return -1.0;
    const unsigned int reps = pickIters(probe, 200000u, 0, 50u);
    return (reps > 1) ? run(g, reps) : probe;
  };
  // Build, time and release one product; negative when it could not run.
  auto timeGemv = [&](int dtype, int64_t k, int64_t n) -> double {
    Gemv g;
    const double us = build(g, dtype, k, n, false) ? timeBuilt(g) : -1.0;
    release(g);
    return us;
  };
  // The relative RMS error of the row a scored product last returned, or -1.
  auto relErrOf = [&](const Gemv &g) -> double {
    if (g.ref.empty())
      return -1.0;
    const float *yf = reinterpret_cast<const float *>(g.y.data());
    double num = 0.0, den = 0.0;
    for (size_t j = 0; j < g.ref.size(); j++)
    {
      const double d = (double)yf[j] - g.ref[j];
      num += d * d;
      den += g.ref[j] * g.ref[j];
    }
    return (den > 0.0) ? std::sqrt(num / den) : -1.0;
  };

  double bpsOf[2] = {0.0, 0.0};   // indexed like `candidates`
  double floorOf[2] = {0.0, 0.0}; // what each width's reading is net of
  double fp32RelErr = -1.0;
  // The fp32 product outlives its own reading: where it comes back rounded,
  // it is timed again against fp16 at its shape (kSameShapeTimeRatio).
  Gemv g32;
  for (size_t ci = 0; ci < 2; ci++)
  {
    const int dtype = candidates[ci];
    if (clpeak::cancelRequested())
      break;
    const size_t es = dtypeSize(dtype);
    // The floor has to be small in *both* dimensions.  Shrinking only the
    // columns leaves a [1,2048]x[2048,256] matrix -- 2 MB of the 8 being
    // measured -- and subtracting a quarter of the transfer reported Core ML
    // at 150 GB/s where onnx-tensor-bw, which subtracts a real floor, says 83.
    const double floorUs = timeGemv(dtype, kFloorDim, kFloorDim);
    const int64_t n = (int64_t)(kBytes / (kK * es));
    double us = -1.0;
    if (dtype != ONNX_DT_FLOAT)
      us = timeGemv(dtype, kK, n);
    else if (build(g32, dtype, kK, n, /*score=*/true))
    {
      us = timeBuilt(g32);
      if (us > 0.0)
        fp32RelErr = relErrOf(g32);
    }
    if (us <= 0.0)
      continue;
    floorOf[ci] = (floorUs > 0.0) ? floorUs : 0.0;
    const double netUs = us - floorOf[ci];
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
    bpsOf[ci] = bps;
  }
  if (fp32RelErr >= 0.0)
    CLPEAK_VLOG("onnx-stream[%s]: fp32 result within %.1f ppm of a double-"
                "precision reference\n",
                ep.providerKey.c_str(), fp32RelErr * 1.0e6);

  // fp32 came back rounded like a 16-bit type, which is storage held at half
  // width or arithmetic rounded the way TF32 rounds it.  fp16 at the fp32
  // product's shape tells the two apart (kSameShapeTimeRatio), raced against
  // it after a fresh warm-up: the fp32 one sat idle while fp16 was built.
  const bool fp32Rounded = fp32RelErr > kFp32ExactRelErr;
  double sameShapeRatio = 0.0; // fp32's net time over fp16's at that shape
  if (fp32Rounded && bpsOf[0] > 0.0 && bpsOf[1] > 0.0 &&
      !clpeak::cancelRequested())
  {
    Gemv g16;
    double us32 = -1.0, us16 = -1.0;
    if (build(g16, ONNX_DT_FLOAT16, kK,
              (int64_t)(kBytes / (kK * dtypeSize(ONNX_DT_FLOAT))), false) &&
        run(g16, 3) > 0.0 && run(g32, 3) > 0.0)
    {
      const double p32 = run(g32, 1), p16 = run(g16, 1);
      if (p32 > 0.0 && p16 > 0.0)
      {
        const unsigned int r32 = pickIters(p32, 100000u, 0, 25u);
        const unsigned int r16 = pickIters(p16, 100000u, 0, 25u);
        for (int i = 0; i < kSameShapeRounds; i++)
        {
          const double a = run(g32, r32), b = run(g16, r16);
          if (a <= 0.0 || b <= 0.0)
          {
            us32 = us16 = -1.0;
            break;
          }
          us32 = (us32 > 0.0) ? std::min(us32, a) : a;
          us16 = (us16 > 0.0) ? std::min(us16, b) : b;
        }
      }
    }
    release(g16);
    const double net32 = us32 - floorOf[1], net16 = us16 - floorOf[0];
    if (us32 > 0.0 && us16 > 0.0 && net32 >= kMinNetShare * us32 &&
        net16 >= kMinNetShare * us16)
    {
      sameShapeRatio = net32 / net16;
      CLPEAK_VLOG("onnx-stream[%s]: at fp32's shape, fp32 %.0f us and fp16 "
                  "%.0f us less %.0f / %.0f us floors: %.2fx the time for "
                  "twice the bytes\n",
                  ep.providerKey.c_str(), us32, us16, floorOf[1], floorOf[0],
                  sameShapeRatio);
    }
    else if (us32 > 0.0 && us16 > 0.0)
      CLPEAK_VLOG("onnx-stream[%s]: at fp32's shape, fp32 %.0f us and fp16 "
                  "%.0f us against %.0f / %.0f us floors -- dispatch, not "
                  "bandwidth; no reading\n",
                  ep.providerKey.c_str(), us32, us16, floorOf[1], floorOf[0]);
  }
  release(g32);

  // fp32 came back rounded *and* moved half its credited bytes -- streaming
  // faster per credited byte, or taking fp16's own time at its shape: the
  // provider holds it at half width, and its reading counts bytes that never
  // moved (see onnxFp32Narrowed).
  const double bps16 = bpsOf[0], bps32 = bpsOf[1];
  const bool fasterPerByte = bps32 > kNarrowedRateRatio * bps16;
  const bool sameTime = sameShapeRatio >= 1.0 / kSameShapeTimeRatio &&
                        sameShapeRatio <= kSameShapeTimeRatio;
  best.fp32Narrowed = fp32Rounded && bps16 > 0.0 && (fasterPerByte || sameTime);
  if (best.fp32Narrowed)
  {
    char same[64] = "";
    if (sameShapeRatio > 0.0)
      std::snprintf(same, sizeof(same), ", %.2fx its time at the same shape",
                    sameShapeRatio);
    CLPEAK_VLOG("onnx-stream[%s]: fp32 is held at half width here (%.0f ppm, "
                "%.2fx fp16's rate per credited byte%s); streaming fp16\n",
                ep.providerKey.c_str(), fp32RelErr * 1.0e6, bps32 / bps16,
                same);
  }
  else if (sameShapeRatio > kSameShapeTimeRatio)
    CLPEAK_VLOG("onnx-stream[%s]: fp32 is rounded in the arithmetic, not the "
                "storage: it moves its full width\n",
                ep.providerKey.c_str());
  else if (sameShapeRatio > 0.0)
    CLPEAK_VLOG("onnx-stream[%s]: fp16 is the slower of the two at fp32's "
                "shape, which says nothing about how fp32 is held\n",
                ep.providerKey.c_str());

  // fp16 on a tie, and fp32 only where it is what it says it is.
  if (bps16 > 0.0)
  {
    best.dtype = ONNX_DT_FLOAT16;
    best.bps = bps16;
  }
  if (bps32 > best.bps && !best.fp32Narrowed)
  {
    best.dtype = ONNX_DT_FLOAT;
    best.bps = bps32;
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

bool onnxFp32Narrowed(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  return onnxStreamProbe(rt, ep).fp32Narrowed;
}

// ---------------------------------------------------------------------------
// Row-reduction view
// ---------------------------------------------------------------------------

namespace
{
struct Rank4Store
{
  std::mutex mtx;
  std::unordered_set<std::string> keys;
};
Rank4Store &rank4Store()
{
  static Rank4Store s;
  return s;
}
std::string rank4Key(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  return std::to_string((uintptr_t)(const void *)rt.base) + '\x1f' +
         ep.providerKey + '\x1f' + ep.epDevice;
}
} // namespace

void onnxNoteRank4Reduce(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  Rank4Store &s = rank4Store();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.keys.insert(rank4Key(rt, ep));
}

bool onnxPrefersRank4Reduce(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  Rank4Store &s = rank4Store();
  std::lock_guard<std::mutex> lk(s.mtx);
  return s.keys.count(rank4Key(rt, ep)) > 0;
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
  //
  // The refusals are expected, but ORT reports some at ERROR severity
  // (OpenVINO's missing NPU target), which with no LogSink installed
  // (--list-devices, the GUI catalog) would reach the terminal past the
  // console mute.  Demote the relay while probing; the reason is kept
  // from the returned statuses and the verbose lines below.
  const OnnxOrtRelayGuard relayGuard;
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
    // Placement is not the question here (onnx_session.h): Core ML runs a
    // 64-value multiply on the CPU under every configuration, and a
    // provider that builds it is alive.
    auto ses = onnxCreateSession(rt, ep, onnxTrivialMulModel(),
                                 /*keepConstantsUnfolded=*/false,
                                 /*profile=*/false, /*keepQdqUnfused=*/false,
                                 /*verifyPlacement=*/false);
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
                  OnnxLiveShape::ResultScaled, /*verifyPlacement=*/false);
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
