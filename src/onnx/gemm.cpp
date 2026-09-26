#ifdef ENABLE_ONNX

// onnx-gemm: MatMul peak through an ONNX Runtime execution provider.
//
// Both operands are model constants and the result is reduced to a single
// row, so nothing large crosses the host boundary per run.  That shape is
// forced by discrete GPUs: with A as a graph input and C returned to the
// host, an RTX 5060 reported 15 TFLOPS for fp16 while a whole transformer
// block -- whose weights are resident -- reached 28 on the same device.  The
// peak was measuring PCIe.  On unified-memory devices the difference is
// small, but the graph is identical everywhere so the rows stay comparable.
//
// A graph of constants is a constant expression, though, and a vendor
// compiler is entitled to evaluate it while it builds: QNN and OpenVINO
// both did, and their timed runs measured dispatch.  So a runtime scalar
// scales the activation operand *before* the multiply wherever the probe
// finds that costs nothing but one elementwise pass (see OnnxLiveShape and
// onnx_probe.cpp); the older result-scaled form survives only where a live
// operand would drag in a precision cast, and there the fold check below
// stands guard over it.
//
// One test, `onnx_gemm`: the same single-operation model on whichever
// formats the provider accepts.  The int8 QDQ reading is measured in ops
// rather than flops and carries that unit itself.
// int8 is the dtype most NPUs are actually built for, so an NPU whose only
// measured reading is the int8 one is the expected shape, not a gap.

#include <onnx/onnx_peak.h>
#include "gemm_setup.h"
#include "onnx_model.h"
#include "onnx_probe.h"
#include "onnx_session.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

using namespace onnxgemm;

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
  constexpr int kMaxStrikes = 2;

  // A rung has to do at least this much work, as a share of what the provider
  // charges to accept the submission, before it counts as having computed
  // anything.  The two populations are far apart: across an RTX 5060 on
  // DirectML, CUDA and TensorRT under three runtimes, the most dispatch-bound
  // *real* rung is Windows TensorRT's nvfp4 at 1024, which does 43% of its
  // submission cost, while every folded rung measured does under 4% or comes
  // out negative.  0.15 sits in that gap with room on both sides -- room the
  // threshold needs, because the charge is estimated by the 32^3 probe, whose
  // graph carries a small reduction the estimate slightly overstates.
  constexpr double kFoldWorkFloor = 0.15;

  // Ceilings that keep the search from running away on either axis.  The time
  // bound is predicted from the previous size's measured rate, so a slow
  // provider stops early instead of spending minutes on one matrix, and it
  // scales itself: hardware fast enough to make a bigger size cheap is exactly
  // the hardware that should try it.
  constexpr double kMaxIterUs = 2.0e6; // one iteration, predicted

  // Both operands together, capped at a quarter of physical memory.  A fixed
  // ceiling here would be a crash on a phone and a needless limit on a
  // workstation; see clpeak::memoryBudget.
  //
  // And capped again by protobuf.  An ONNX model is a protobuf message, whose
  // serialized size cannot exceed 2 GiB, and both operands live inside it as
  // initializers -- so a size the machine has memory for can still be
  // unbuildable.  fp32 at 16384 needs exactly 2 GiB of operands and ORT answers
  // "Model data size exceeds maximum supported size (2GB)", which is a property
  // of the format rather than of the device and does not belong in a memory
  // budget.  The slack leaves room for the graph around the weights.
  static uint64_t maxWeightBytes()
  {
    const uint64_t protobufCeiling = (2ull << 30) - (64ull << 20);
    return std::min(clpeak::memoryBudget(3ull << 30), protobufCeiling);
  }

  // Per-size budget for the timed phase.  Lower than the 5 s a single-size test
  // would use, since the ladder measures several.
  constexpr unsigned int kSizeBudgetUs = 2000000;

  const char *shapeNameFor(OnnxLiveShape s)
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

  // What the description says about where the runtime dependency entered.
  // Only the live forms need a sentence: they cost a pass over the operand
  // that is inside the figure, and a reader dividing rows should know it.
  const char *shapeNote(const Variant &v, OnnxLiveShape shape)
  {
    switch (shape)
    {
    case OnnxLiveShape::ResultScaled:
      return "";
    case OnnxLiveShape::OperandScaled:
      return v.qdq
                 ? "  The activations are scaled at run time and quantized on "
                   "device, so this provider's compiler cannot fold the "
                   "multiply away; that pass is inside the figure."
                 : "  The activations are scaled at run time, so this "
                   "provider's compiler cannot fold the multiply away; that "
                   "pass is inside the figure.";
    case OnnxLiveShape::Add0:
      return "  The activations take a runtime zero on the way in, so this "
             "provider's compiler cannot fold the multiply away; that pass is "
             "inside the figure.";
    case OnnxLiveShape::QdqAdd0:
      return "  The activations take a runtime zero as a quantized op, so this "
             "provider's compiler cannot fold the multiply away; that pass is "
             "inside the figure.";
    }
    return "";
  }

  // The compile gate predicts the next rung's session creation from the growth
  // it has seen rather than an assumed one.  Doubling D is 4x the weights and
  // 8x the work, and a compiler's time lands anywhere between: QNN's HTP grew
  // 5.5x and 6.2x a doubling at the top of its ladders under the plugin
  // runtime, 7x and 16x under the one before it.  Until two compiles are
  // large enough for their ratio to mean anything (kOnnxCreateGrowthFloor), 4x
  // is assumed; after that the ratio is used, bounded so that one noisy pair
  // cannot predict a runaway.  This replaces stopping on a fixed growth
  // factor, which ended the older runtime's int8 ladder at 2048 on a 1.2 ->
  // 8.8 s step whose next rung predicted 35 s.
  constexpr double kAssumedCreateGrowth = 4.0;
  constexpr double kMaxCreateGrowth = 16.0;

  // The chained rows.  Experimental: the single-multiply rows put one matmul
  // in a dispatch with a live-operand pass in front and a reduction behind,
  // and on a provider whose matrix unit is fast next to its memory and its
  // dispatch, those can be most of the time.  Sixteen distinct layers in one
  // graph spread all three over sixteen multiplies, which is also how a model
  // runs.  The ladder starts smaller because the work per rung is sixteen
  // times the square's.
  constexpr int kChainLayers = 16;
  constexpr int64_t kChainMinDim = 512;

  // A verbose run asks the provider's own profiler (onnxNativeProfilePath) to
  // watch one more build of the row's best size.  Where that size took longer
  // than this to compile, the largest measured size that did not is watched
  // instead: how the time splits between operations barely moves from one
  // size to the next, and the top rung can cost minutes.
  constexpr double kNativeProfileCreateCapUs = 90.0e6;

  // Three significant figures: the two int8 spellings can land within a few
  // percent of each other, and the sentence naming the slower one has to be
  // able to show that it was slower.
  std::string opsText(double rate)
  {
    const double v = (rate >= 1.0e12) ? rate / 1.0e12 : rate / 1.0e9;
    const char *unit = (rate >= 1.0e12) ? "TOPS" : "GOPS";
    char buf[32];
    std::snprintf(buf, sizeof buf, v >= 100.0 ? "%.0f %s"
                                   : v >= 10.0 ? "%.1f %s"
                                               : "%.2f %s",
                  v, unit);
    return buf;
  }

} // namespace

int OnnxPeak::runGemm(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                      benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx_gemm", "ONNX MatMul peak",
       "flops",
       Category::Unknown,
       "Matrix-multiply rate through this execution provider, one data type "
       "per row, swept over square sizes and reported at its best.  The same "
       "model runs on every provider, and one that cannot run it entirely on "
       "its own device reports unsupported rather than quietly measuring the "
       "host.",
       TestShape::Heterogeneous, "data type"});

  // runAll also clears per EP before dispatching; this entry clear keeps
  // direct runGemm callers (tests, future entry points) from inheriting a
  // stale record from an earlier run in the same process.
  onnxClearGemmFolded(ep);

  // Set once any ladder on this provider is caught folding its resident
  // product.  A compiler that evaluated one constant matmul at build time
  // evaluates the next, and on QNN learning that again costs a minute of
  // compiling per row, so later rows start on their first live shape.  A
  // live shape can never be the wrong answer, only a pass slower.
  bool providerFolds = false;

  // The scheme each single-multiply row settled on, by label, so its chained
  // row runs the same one.
  std::unordered_map<std::string, size_t> schemeOf;

  // One quantization scheme's slice of a probe result: the primary one lives
  // in the result's own fields, the others in moreSchemes.
  struct Scheme
  {
    int actDtype;
    int wgtDtype;
    const char *name;
    std::string ranAs;
    bool castedActs;
    double probeUs;
    std::vector<OnnxLiveShape> shapes;
  };
  auto schemesOf = [](const OnnxProbeResult &pr) {
    std::vector<Scheme> out;
    out.push_back({pr.actDtype, pr.wgtDtype, pr.schemeName, pr.ranAs,
                   pr.castedActs, pr.probeUs, pr.shapes});
    for (const OnnxProbeScheme &m : pr.moreSchemes)
      out.push_back({m.actDtype, m.wgtDtype, m.name, m.ranAs, m.castedActs,
                     m.probeUs, m.shapes});
    return out;
  };

  // What one ladder -- one variant, scheme and shape -- found.
  struct Ladder
  {
    double best = 0.0;
    int64_t bestDim = 0;
    OnnxLiveShape shape = OnnxLiveShape::ResultScaled;
    std::string err;
    ResultStatus errStatus = ResultStatus::Unsupported;
    // Set by the folding guards below; drives the paired suppression in
    // runNumericError (see onnx_probe.h).
    bool folded = false;
    int64_t firstDim = 0;
    // Sizes the provider's runtime sent elsewhere (onnx_session.h,
    // offDevice): below the first measured rung the ladder climbs past
    // them, above it they end the ladder, and the row says both.
    int offDeviceBelow = 0;
    int64_t offDeviceAbove = 0;
    // The first measured size reduced through the rank-4 view, or 0.
    int64_t rank4From = 0;
    // Session-creation time of every measured size, ascending.
    std::vector<std::pair<int64_t, double>> creates;
  };

  // One doubling ladder.  `tail` is the row's, and sticks: once a provider
  // refuses the 2-D reduction at some size it gets the rank-4 view for every
  // size after, in this ladder and the row's later ones.
  auto ladder = [&](const Variant &v, const std::string &label,
                    const OnnxProbeResult &pr, const Scheme &sc,
                    OnnxLiveShape shape, int layers, int64_t minDim,
                    OnnxGemmTail &tail) -> Ladder {
    Ladder L;
    L.shape = shape;
    const bool foldable = (shape == OnnxLiveShape::ResultScaled);
    const char *tag = label.c_str();

    int rungs = 0;
    // Why the ladder ended, for the single-rung verdict below.
    bool endedOnWork = false; // memory or measured time: real work happened

    // First and last timings with their sizes, to confirm the work actually
    // happened (see the folding check after the loop).
    double firstUs = 0.0, lastUs = 0.0;
    int64_t lastDim = 0;
    double lastRate = 0.0;
    double prevCreateUs = 0.0, prevPrevCreateUs = 0.0;
    double prevUs = 0.0;
    int64_t prevD = 0;
    int strikes = 0;

    for (int64_t D = minDim; D <= kMaxDim; D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      // Would this size fit, and would one iteration finish in reasonable
      // time at the rate the previous size managed?  The operand estimate
      // follows the shape -- the float-scaled QDQ form holds its activations
      // as fp32 -- and on a phone an under-estimate here is an out-of-memory
      // kill rather than a slow row.
      const uint64_t weightBytes = operandBytes(v, D, shape, layers);
      if (weightBytes > maxWeightBytes())
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 needs %llu MB of operands, "
                    "stopping\n",
                    ep.providerKey.c_str(), tag,
                    (long long)D, (unsigned long long)(weightBytes >> 20));
        endedOnWork = true;
        break;
      }
      const double ops = 2.0 * (double)D * (double)D * (double)D * layers;
      if (lastRate > 0.0)
      {
        const double predictedUs = ops / lastRate;
        if (predictedUs > kMaxIterUs)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 would take ~%.1f s per "
                      "iteration, stopping\n",
                      ep.providerKey.c_str(),
                      tag, (long long)D, predictedUs / 1.0e6);
          endedOnWork = true;
          break;
        }
      }
      // Predicted compilation gate: skip the next rung before paying its
      // build when the compiles so far predict it over budget.  The first
      // rung is always attempted -- its time seeds this prediction.
      if (D > minDim && prevCreateUs > 0.0)
      {
        double growth = kAssumedCreateGrowth;
        if (prevPrevCreateUs > kOnnxCreateGrowthFloor)
          growth = std::min(std::max(prevCreateUs / prevPrevCreateUs,
                                     kAssumedCreateGrowth),
                            kMaxCreateGrowth);
        if (prevCreateUs * growth > kOnnxMaxCreateUs)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 predicted create %.1f s "
                      "(prev %.1f s x%.1f) > %.1f s, stopping\n",
                      ep.providerKey.c_str(), tag, (long long)D,
                      prevCreateUs * growth / 1.0e6, prevCreateUs / 1.0e6,
                      growth, kOnnxMaxCreateUs / 1.0e6);
          break;
        }
      }

      auto build = [&](OnnxGemmTail t, double &createUs) {
        auto createStart = std::chrono::steady_clock::now();
        GemmSetup g = makeSetup(rt, ep, v, D, /*profile=*/false, sc.actDtype,
                                pr.reduceInFloat, sc.wgtDtype, shape,
                                /*verifyPlacement=*/true, t, layers);
        createUs = std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - createStart)
                       .count();
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 session create %.1f s%s\n",
                    ep.providerKey.c_str(), tag, (long long)D,
                    createUs / 1.0e6,
                    t == OnnxGemmTail::Rank4 ? " (rank-4 reduction)" : "");
        return g;
      };
      double createUs = 0.0;
      GemmSetup g = build(tail, createUs);

      // A provider can take the multiply and refuse the reduction behind it
      // at a real size -- the QNN Adreno backend did at every size from 1024,
      // leaving the multiply in a partition fed by constants alone, which it
      // then rejected as "Zero tensor size!".  The same values reduced
      // through a rank-4 view is the one other spelling worth a compile.
      if (!g.session && !g.offDevice && tail == OnnxGemmTail::Rows &&
          onnxFailureStatus(g.error) == ResultStatus::Unsupported &&
          !clpeak::cancelRequested())
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 refused (%s); retrying the "
                    "reduction through a rank-4 view\n",
                    ep.providerKey.c_str(), tag, (long long)D,
                    g.error.c_str());
        double retryUs = 0.0;
        GemmSetup r4 = build(OnnxGemmTail::Rank4, retryUs);
        if (r4.session)
        {
          // Every later size builds once, in this view, so this build's time
          // is the one the compile gate should extrapolate from.
          tail = OnnxGemmTail::Rank4;
          destroySetup(rt, g);
          g = std::move(r4);
          createUs = retryUs;
        }
        else
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 refused with the rank-4 view "
                      "too (%s)\n",
                      ep.providerKey.c_str(), tag, (long long)D,
                      r4.error.c_str());
          destroySetup(rt, r4);
        }
      }

      if (!g.session)
      {
        if (L.err.empty())
          L.err = g.error;
        if (g.offDevice)
        {
          // The runtime placed this size on another unit.  Below the first
          // rung that ran, too small for its planner is not too big for the
          // unit -- Core ML's Neural Engine declines a 1024-cube and takes
          // the 2048 -- so the ladder climbs on, for a few sizes: a shape
          // the unit cannot run at all will not become runnable by growing,
          // and each refused size still costs a compile.  Above a measured
          // rung it is the unit declining larger work, which ends the
          // ladder the way any other limit does.
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 %s\n", ep.providerKey.c_str(),
                      tag, (long long)D, g.error.c_str());
          if (rungs > 0)
          {
            L.offDeviceAbove = D;
            break;
          }
          prevPrevCreateUs = prevCreateUs;
          prevCreateUs = createUs;
          if (++L.offDeviceBelow < kOnnxOffDevicePatience)
            continue;
        }
        // Larger sizes need strictly more of everything, so nothing above
        // this one can succeed either.
        break;
      }
      if (tail == OnnxGemmTail::Rank4 && L.rank4From == 0)
        L.rank4From = D;

      double per_iter_us = -1.0;
      if (timeRuns(rt, g, 1 + warmupCount) > 0.0) // compile + warmup
        per_iter_us = timeRuns(rt, g, 1);         // calibration probe
      if (per_iter_us <= 0.0)
      {
        if (L.err.empty())
        {
          L.err = g.error.empty() ? "run failed" : g.error;
          L.errStatus = ResultStatus::Error;
        }
        destroySetup(rt, g);
        break;
      }

      unsigned int iters = pickIters(per_iter_us, kSizeBudgetUs,
                                     forceIters ? specifiedIters : 0,
                                     kOnnxMaxIters);
      // The probe was one whole iteration, so when the budget affords only
      // one, it already is the measurement -- and at that end of the ladder
      // repeating it is the most expensive thing the sweep does.
      double mean_us = (iters > 1) ? timeRuns(rt, g, iters) : per_iter_us;
      if (mean_us <= 0.0 && L.err.empty())
      {
        L.err = g.error.empty() ? "run failed" : g.error;
        L.errStatus = ResultStatus::Error;
      }
      destroySetup(rt, g);
      if (mean_us <= 0.0)
        break;

      rungs++;
      L.creates.push_back({D, createUs});
      if (firstUs == 0.0)
      {
        firstUs = mean_us;
        L.firstDim = D;
      }
      lastUs = mean_us;
      lastDim = D;

      const double rate = ops * 1.0e6 / mean_us;
      // Rate per FLOP-count is what the ladder is searching on; the raw rate
      // in ops/us drives the time prediction for the next rung.
      lastRate = ops / mean_us;
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 -> %.3f\n", ep.providerKey.c_str(),
                  tag, (long long)D, rate);

      // Per-doubling fold detector, for the foldable shape only.  A folded
      // graph leaves dispatch plus a reduction over D elements, so its time
      // barely moves while the work grows 8x: QNN read ~170 us at both 1024
      // and 2048.
      //
      // The comparison is on *work* -- the time left once the provider's
      // per-submission charge is taken off, which the 32^3 probe measured on
      // this same graph.  Raw times will not do.  Windows TensorRT charges
      // 114 us to accept anything, so its nvfp4 rungs read 163.5 and 325.9
      // us: 1.99x, a hair under the 2x a raw test demands, and the row was
      // refused as folded on a card that measures 215 TFLOPS there.  Net of
      // dispatch those rungs are 49.5 and 211.9 us -- 4.3x for 8x the work,
      // exactly the climb of a provider approaching its peak.  A rate test
      // is worse still for the same reason.
      //
      // A rung whose whole time is inside the dispatch charge computed
      // nothing at all, which is a fold by itself (DirectML: 153.6 us at
      // 1024 against a 156 us submission).  The live shapes cannot fold, so
      // their ladders are never second-guessed.
      const double dispatchUs = (sc.probeUs > 0.0) ? sc.probeUs : 0.0;
      const double work = mean_us - dispatchUs;
      const double prevWork = prevUs - dispatchUs;

      // Two questions, and the first needs only one rung.  A rung whose work
      // disappears into the cost of submitting it computed nothing: DirectML
      // spent 153.6 us on a 1024-cube it charges 156 us merely to accept.
      // That is a fold on its own, and catching it here rather than by
      // comparing rungs is what lets a provider whose compile budget affords
      // a single size still be judged.
      const bool computedNothing =
          foldable && dispatchUs > 0.0 && work < kFoldWorkFloor * dispatchUs;
      // And then: did the work grow with the size?  Eight times the arithmetic
      // has to cost at least twice the time however much the rate improves --
      // a folded graph's residue is the reduction, which grows with D rather
      // than D cubed.
      const bool workFlat = foldable && prevUs > 0.0 && D == prevD * 2 &&
                            prevWork > 0.0 && work < prevWork * 2.0;
      if (computedNothing || workFlat)
      {
        if (computedNothing)
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 took %.1f us against a %.1f us "
                      "submission -- it computed nothing, constants were "
                      "folded\n",
                      ep.providerKey.c_str(), tag, (long long)D, mean_us,
                      dispatchUs);
        else
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %.1f us at %lld vs %.1f us at %lld, "
                      "less a %.1f us submission -- %.2fx for 8x the work, "
                      "constants were folded\n",
                      ep.providerKey.c_str(), tag, prevUs,
                      (long long)prevD, mean_us, (long long)D, dispatchUs,
                      work / prevWork);
        L.best = 0.0;
        L.err = "this provider folded the operands at compile time: the "
                "timed runs measure dispatch plus a reduction of a "
                "precomputed result rather than the matrix multiply, so "
                "the timings do not scale with the problem size and mean "
                "nothing";
        L.errStatus = ResultStatus::Error;
        L.folded = true;
        break;
      }
      prevUs = mean_us;
      prevD = D;

      if (rate > L.best * kImproveFactor)
      {
        strikes = 0;
        L.best = rate;
        L.bestDim = D;
      }
      else
      {
        if (rate > L.best)
        {
          L.best = rate;
          L.bestDim = D;
        }
        if (++strikes >= kMaxStrikes)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: no further gain past %lld^3\n",
                      ep.providerKey.c_str(), tag, (long long)L.bestDim);
          break;
        }
      }

      // A rung that measured slower than the ceiling ends the ladder here.
      //
      // The gate at the top of the loop asks the same question of the *next*
      // size, but it has to answer it by extrapolating from the rate of the
      // size before -- and an extrapolation cannot see a cliff, which is
      // exactly what it is being asked to look for.  A provider that falls
      // off one reads as fast right up to the rung that collapses: Core ML
      // runs fp16 at 6.1 TFLOPS at 4096 and 0.34 at 8192, so the prediction
      // for 8192 came out nineteen times short of the truth.
      //
      // This one is not a prediction.  The rung has been measured, it took
      // longer than a whole iteration is allowed to take, and the next size
      // is eight times the work -- so there is nothing above this worth the
      // wait, whatever the rate did.
      if (per_iter_us > kMaxIterUs)
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 measured %.1f s per iteration, "
                    "stopping\n",
                    ep.providerKey.c_str(), tag,
                    (long long)D, per_iter_us / 1.0e6);
        endedOnWork = true;
        break;
      }

      // A rung that compiled past the budget ends the ladder after it is
      // measured.  The first rung is allowed to exceed the budget once -- its
      // time seeds the prediction above; truncating it would discard a valid
      // peak.
      if (D != minDim && createUs > kOnnxMaxCreateUs)
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 create %.1f s > %.1f s, stopping\n",
                    ep.providerKey.c_str(), tag,
                    (long long)D, createUs / 1.0e6, kOnnxMaxCreateUs / 1.0e6);
        break;
      }
      prevPrevCreateUs = prevCreateUs;
      prevCreateUs = createUs;
    }

    // Whole-ladder backstop for the foldable shape: real work grows with the
    // cube of the size -- 64x across a 1024-to-8192 ladder -- so anything
    // close to flat means nothing was computed.  The tolerance has to clear
    // what a legitimately improving rate does to the time: TensorRT's int8
    // improves 9.4x between 1024 and 16384, so its time grows 433x where the
    // work grew 4096x, and a factor of 8 once threw away a correct 124 TOPS
    // reading.  A rate improving 64x across one ladder has never been
    // observed.
    double expectedGrowth = 1.0;
    for (int64_t d = L.firstDim; d > 0 && d < lastDim; d *= 2)
      expectedGrowth *= 8.0; // each doubling is 8x work
    if (foldable && L.best > 0.0 && firstUs > 0.0 && lastDim > L.firstDim &&
        lastUs < firstUs * expectedGrowth / 64.0)
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %.1f us at %lld vs %.1f us at %lld -- "
                  "work does not scale, constants were folded\n",
                  ep.providerKey.c_str(), tag, firstUs,
                  (long long)L.firstDim, lastUs, (long long)lastDim);
      L.best = 0.0;
      L.err = "this runtime folded the operands at load time: it accepted "
              "the request to disable constant folding and ignored it, "
              "which ONNX Runtime did before about 1.18, so the timings "
              "do not scale with the problem size and mean nothing";
      L.errStatus = ResultStatus::Error;
      L.folded = true;
    }

    // A foldable ladder that ended after one rung has nothing to compare
    // that rung against.  When it ended because the next size would not fit
    // or the rung itself took seconds, real work demonstrably happened; when
    // a compile-time gate ended it, the gate is the fold's own signature --
    // a compiler evaluating 2 GFLOP with reference code takes a minute --
    // and the one timing is indistinguishable from dispatch.  QNN published
    // 12 TFLOPS fp32, 12 TFLOPS fp16 and 12 TOPS int8 from one rung each,
    // all the same 179 us, before this rule existed.
    if (foldable && L.best > 0.0 && rungs == 1 && !endedOnWork)
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: one rung, ended on a compile-time gate; "
                  "cannot rule out folding\n",
                  ep.providerKey.c_str(), tag);
      L.best = 0.0;
      L.err = "only one size could be measured before the provider's "
              "compile time ran out, so nothing confirms the timing "
              "scaled with the work rather than measuring dispatch";
      L.errStatus = ResultStatus::Error;
      L.folded = true;
    }
    if (L.best <= 0.0)
      L.bestDim = 0;
    return L;
  };

  // One row: every scheme the probe kept, each down its shapes, the fastest
  // reported.  `layers` > 1 is a chained row under `label`, built from the
  // variant `v` whose probe result it shares.
  auto runVariant = [&](const Variant &v, int layers, const std::string &label) {
    const bool isInt = isIntVariant(v);
    const bool chained = layers > 1;

    logger::EmitOptions o;
    const std::string intro =
        chained ? "Experimental: " + std::to_string(layers) +
                      " distinct square multiplies chained in one dispatch, "
                      "each layer's product feeding the next, swept over "
                      "layer widths and reported at its best.  "
                : std::string("Peak over a doubling sweep of square sizes.  ");
    o.description = intro + v.note;
    if (isInt)
      o.unit = "ops";

    // The probe already learned, at 32^3, whether this provider can run the
    // variant at all, which quantization scheme fuses, and which graph shape
    // keeps the multiply live without changing its arithmetic.  The ladder
    // reproduces exactly that.
    const auto &cache = onnxProbeGemmCache(rt, ep);
    auto it = cache.find(v.label);
    if (it == cache.end())
    {
      test.skip(label, ResultStatus::Unsupported,
                "no probe result for " + std::string(v.label), o);
      return;
    }
    const OnnxProbeResult &pr = it->second;
    if (!pr.ok)
    {
      test.skip(label, onnxFailureStatus(pr.reason), pr.reason, o);
      return;
    }

    std::vector<Scheme> schemes = schemesOf(pr);
    if (chained)
    {
      // A chain runs the scheme its single-multiply row settled on.
      auto s = schemeOf.find(v.label);
      const size_t keep = (s != schemeOf.end() && s->second < schemes.size())
                              ? s->second
                              : 0;
      schemes = {schemes[keep]};
    }

    OnnxGemmTail tail = OnnxGemmTail::Rows;
    Ladder best;
    size_t bestScheme = 0;
    bool measured = false;
    Ladder firstFail;
    bool haveFail = false;
    std::vector<std::pair<size_t, double>> schemePeaks;
    for (size_t si = 0; si < schemes.size(); si++)
    {
      if (clpeak::cancelRequested())
        break;
      const Scheme &sc = schemes[si];

      // The probe left one or more viable shapes, result-scaled first.  Try
      // them in order: the first that is not caught folding is the
      // measurement, and a fold drops to the next (a live shape a vendor
      // compiler cannot evaluate at build time).  A non-folding provider
      // never leaves the first.  A chain of constants is one long constant
      // expression, so a chain starts on a live shape; so does any row on a
      // provider that has already folded one.
      std::vector<OnnxLiveShape> shapes = sc.shapes;
      if ((chained || providerFolds) && shapes.size() > 1 &&
          shapes.front() == OnnxLiveShape::ResultScaled)
      {
        if (providerFolds && !chained)
          CLPEAK_VLOG("onnx-gemm[%s/%s]: this provider folded an earlier "
                      "row's resident product; starting at %s\n",
                      ep.providerKey.c_str(), label.c_str(),
                      shapeNameFor(shapes[1]));
        shapes.erase(shapes.begin());
      }
      if (shapes.empty() ||
          (chained && shapes.front() == OnnxLiveShape::ResultScaled))
      {
        if (!haveFail)
        {
          firstFail.err = "no shape that keeps the multiply live builds at "
                          "this row's width on this provider, and a chain of "
                          "constants would be evaluated at build time";
          firstFail.errStatus = ResultStatus::Unsupported;
          haveFail = true;
        }
        continue;
      }

      Ladder L;
      for (size_t shi = 0; shi < shapes.size(); shi++)
      {
        L = ladder(v, label, pr, sc, shapes[shi], layers,
                   chained ? kChainMinDim : kMinDim, tail);
        if (L.folded)
          providerFolds = true;
        // Caught folding and another shape remains: drop to it (a live shape
        // the compiler cannot evaluate at build time) rather than reporting
        // an error.
        if (L.folded && shi + 1 < shapes.size())
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %s folded, retrying %s\n",
                      ep.providerKey.c_str(), label.c_str(),
                      shapeNameFor(shapes[shi]), shapeNameFor(shapes[shi + 1]));
          continue;
        }
        break; // settled: this shape produced the ladder (measurement or error)
      }

      if (schemes.size() > 1)
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %s: %s\n", ep.providerKey.c_str(),
                    label.c_str(), sc.name,
                    L.best > 0.0 ? (opsText(L.best) + " at " +
                                    std::to_string(L.bestDim) + "^3").c_str()
                                 : L.err.c_str());
      if (L.best > 0.0)
      {
        schemePeaks.push_back({si, L.best});
        if (!measured || L.best > best.best)
        {
          best = L;
          bestScheme = si;
          measured = true;
        }
      }
      else if (!haveFail)
      {
        firstFail = L;
        haveFail = true;
      }
    }

    if (!measured)
    {
      if (firstFail.folded && !chained)
        onnxNoteGemmFolded(ep, v.label);
      test.skip(label, onnxFailureStatus(firstFail.err, firstFail.errStatus),
                firstFail.err.empty() ? "no supported datatype" : firstFail.err,
                o);
      return;
    }
    if (!chained)
      schemeOf[v.label] = bestScheme;

    const Scheme &sc = schemes[bestScheme];
    o.description =
        chained ? intro + "Fastest at " + std::to_string(best.bestDim) +
                      "-wide layers.  " + v.note
                : "Peak over a doubling sweep of square sizes; fastest at " +
                      std::to_string(best.bestDim) + " cubed.  " + v.note;
    if (!sc.ranAs.empty())
      o.description += "  Ran as " + sc.ranAs + " (" + sc.name + ").";
    for (const auto &peak : schemePeaks)
      if (peak.first != bestScheme)
        o.description += "  It also runs " + std::string(schemes[peak.first].name) +
                         ", which peaked at " + opsText(peak.second) +
                         "; this is the faster of the two.";
    if (sc.castedActs)
      o.description += "  The provider converts the activations first, a "
                       "full pass inside this figure.";
    if (!pr.ranWider.empty())
      o.description += "  This provider has no " + std::string(v.label) +
                       " matmul kernel and ran it in " + pr.ranWider +
                       ", so this is that width rather than " + v.label + ".";
    if (pr.reduceInFloat)
      o.description += "  The product is cast to fp32 before the reduction; "
                       "the multiply is unaffected.";
    o.description += shapeNote(v, best.shape);
    if (best.rank4From > 0)
      o.description += "  From " + std::to_string(best.rank4From) +
                       " the product was reduced through a 4-D view of itself, "
                       "because this provider refused the 2-D reduction there; "
                       "the values reduced and the work are the same.";
    if (best.offDeviceBelow > 0)
      o.description += "  Sizes below " + std::to_string(best.firstDim) +
                       " cubed were sent to another compute unit by the "
                       "provider's runtime and are not in this figure.";
    if (best.offDeviceAbove > 0)
      o.description += "  From " + std::to_string(best.offDeviceAbove) +
                       " cubed the provider's runtime sent the work to "
                       "another compute unit, which ended the sweep.";
    test.emit(label, (float)best.best, o);

    // What the provider's own profiler says the time went to, for the log.
    // One more build, never one that was timed (a profiled session is not
    // the session measured), and only when someone asked for the detail.
    const std::string profilePath =
        clpeak::verboseEnabled() ? onnxNativeProfilePath(ep) : std::string();
    if (!profilePath.empty() && !clpeak::cancelRequested())
    {
      int64_t pd = 0;
      for (const auto &c : best.creates)
        if (c.first <= best.bestDim && c.second <= kNativeProfileCreateCapUs)
          pd = c.first;
      if (pd > 0)
      {
        const OnnxGemmTail pt = (best.rank4From > 0 && pd >= best.rank4From)
                                    ? OnnxGemmTail::Rank4
                                    : OnnxGemmTail::Rows;
        GemmSetup g = makeSetup(rt, ep, v, pd, /*profile=*/false, sc.actDtype,
                                pr.reduceInFloat, sc.wgtDtype, best.shape,
                                /*verifyPlacement=*/true, pt, layers,
                                profilePath);
        if (g.session && timeRuns(rt, g, 1 + warmupCount) > 0.0)
          timeRuns(rt, g, 3);
        else
          CLPEAK_VLOG("onnx-gemm[%s/%s]: profiled build at %lld failed: %s\n",
                      ep.providerKey.c_str(), label.c_str(), (long long)pd,
                      g.error.c_str());
        destroySetup(rt, g);
        onnxLogNativeProfile(profilePath, ep.providerKey + "/" + label + " " +
                                              std::to_string(pd) + "^3" +
                                              (chained ? " x" + std::to_string(layers) : ""));
      }
      else
        CLPEAK_VLOG("onnx-gemm[%s/%s]: no measured size compiled within %.0f s; "
                    "not profiled\n",
                    ep.providerKey.c_str(), label.c_str(),
                    kNativeProfileCreateCapUs / 1.0e6);
    }
  };

  for (size_t i = 0; i < kFpVariantCount; i++)
  {
    if (clpeak::cancelRequested()) break;
    runVariant(kFpVariants[i], 1, kFpVariants[i].label);
  }
  for (size_t i = 0; i < kIntVariantCount; i++)
  {
    if (clpeak::cancelRequested()) break;
    runVariant(kIntVariants[i], 1, kIntVariants[i].label);
  }
  // The chained rows, after every single-multiply row they are read against.
  for (size_t i = 0; i < kFpVariantCount; i++)
  {
    if (clpeak::cancelRequested()) break;
    if (std::string(kFpVariants[i].label) == "fp16")
      runVariant(kFpVariants[i], kChainLayers,
                 std::string(kFpVariants[i].label) + "_chain");
  }
  for (size_t i = 0; i < kIntVariantCount; i++)
  {
    if (clpeak::cancelRequested()) break;
    runVariant(kIntVariants[i], kChainLayers,
               std::string(kIntVariants[i].label) + "_chain");
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
