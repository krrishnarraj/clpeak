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
// onnx_probe.cpp); the result-scaled form survives where no live operand
// builds, and there the fold check below stands guard over it.
//
// Every row but NVFP4 is a chain: sixteen distinct square layers in one
// graph, each one's product the next one's activations.  One multiply per
// dispatch measured what surrounds the multiply as much as the multiply.  On
// QNN's HTP, whose profiler counts cycles per operation, the pass that keeps
// a single int8 8192-cube live and the reduction that brings its row back
// took 48% of the run -- the multiply itself at 26 TOPS inside a row that
// read 13 -- while sixteen layers per dispatch read 29 TOPS with each layer
// at 35.  The Neural Engine gained 25% the same way, ONNX Runtime's CPU 13%,
// the Adreno 6%.  A chain is also how a model runs.
//
// One test, `onnx_gemm`: the same chained model on whichever formats the
// provider accepts.  The int8 QDQ reading is measured in ops rather than
// flops and carries that unit itself.  int8 is the dtype most NPUs are
// actually built for, so an NPU whose only measured reading is the int8 one
// is the expected shape, not a gap.

#include <onnx/onnx_peak.h>
#include "gemm_setup.h"
#include "onnx_model.h"
#include "onnx_probe.h"
#include "onnx_session.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace onnxgemm;

namespace
{

  // The ladder doubles the layer width from 1024 until the rate plateaus
  // (kOnnxPlateauGain), and the peak is reported along with the width that
  // produced it.
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

  // Layers per chain.  Sixteen spreads the pass that keeps the graph live and
  // the final reduction thin -- 15% of a 4096-wide int8 chain on QNN's HTP,
  // against 48% of one multiply -- while a 4096-wide fp16 chain still fits
  // the protobuf ceiling and compiles in under a minute there.
  constexpr int kChainLayers = 16;

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

  // Every layer's weights together, capped at a quarter of physical memory.
  // A fixed ceiling here would be a crash on a phone and a needless limit on
  // a workstation; see clpeak::memoryBudget.
  //
  // And capped again by protobuf.  An ONNX model is a protobuf message, whose
  // serialized size cannot exceed 2 GiB, and the weights live inside it as
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

  // A verbose run asks the provider's own profiler (onnxNativeProfilePath) to
  // watch one more build of the row's best size.  Where that size took longer
  // than this to compile, the largest measured size that did not is watched
  // instead: how the time splits between operations barely moves from one
  // size to the next, and the top rung can cost minutes.
  constexpr double kNativeProfileCreateCapUs = 90.0e6;

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
  // Only the live forms need a sentence: they cost a pass over the first
  // layer's activations that is inside the figure, and a reader dividing rows
  // should know it.
  const char *shapeNote(const Variant &v, OnnxLiveShape shape)
  {
    switch (shape)
    {
    case OnnxLiveShape::ResultScaled:
      return "";
    case OnnxLiveShape::OperandScaled:
      return v.qdq
                 ? "  The activations are scaled at run time and quantized on "
                   "device ahead of the first multiply, so no compiler can fold "
                   "the multiplies away; that pass is inside the figure."
                 : "  The activations are scaled at run time ahead of the first "
                   "multiply, so no compiler can fold the multiplies away; that "
                   "pass is inside the figure.";
    case OnnxLiveShape::Add0:
      return "  The activations take a runtime zero ahead of the first "
             "multiply, so no compiler can fold the multiplies away; that pass "
             "is inside the figure.";
    case OnnxLiveShape::QdqAdd0:
      return "  The activations take a runtime zero as a quantized op ahead of "
             "the first multiply, so no compiler can fold the multiplies away; "
             "that pass is inside the figure.";
    }
    return "";
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
       "per row: sixteen distinct square multiplies chained in one dispatch, "
       "each layer's product feeding the next, swept over layer widths and "
       "reported at its best.  A chain is how a model runs, and it spreads "
       "what surrounds a multiply -- the dispatch, the pass that keeps the "
       "graph from being computed at build time, the reduction that brings "
       "one row back -- over sixteen of them.  The same model runs on every "
       "provider, and one that cannot run it entirely on its own device "
       "reports unsupported rather than quietly measuring the host.",
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

  // What one ladder -- one variant in one shape -- found.
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

  // One doubling ladder.  `view` is the row's and sticks: once a provider
  // refuses the 2-D reduction at some size it gets the rank-4 view for every
  // size after, and so does every later graph on that provider
  // (onnxPrefersRank4Reduce).
  auto ladder = [&](const Variant &v, const OnnxProbeResult &pr,
                    OnnxLiveShape shape, int layers,
                    OnnxReduceView &view) -> Ladder {
    Ladder L;
    L.shape = shape;
    const bool foldable = (shape == OnnxLiveShape::ResultScaled);
    const char *tag = v.label;

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

    for (int64_t D = kMinDim; D <= kMaxDim; D *= 2)
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
      // The compile cap, checked before paying for the build
      // (onnxPredictCreateUs).  The first rung is always attempted -- its time
      // seeds the prediction.
      if (D > kMinDim && prevCreateUs > 0.0)
      {
        const double predictedUs =
            onnxPredictCreateUs(prevCreateUs, prevPrevCreateUs);
        if (predictedUs > kOnnxMaxCreateUs)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 predicted create %.1f s "
                      "(prev %.1f s) > %.1f s, stopping\n",
                      ep.providerKey.c_str(), tag, (long long)D,
                      predictedUs / 1.0e6, prevCreateUs / 1.0e6,
                      kOnnxMaxCreateUs / 1.0e6);
          break;
        }
      }

      auto build = [&](OnnxReduceView rv, double &createUs) {
        auto createStart = std::chrono::steady_clock::now();
        GemmSetup g = makeSetup(rt, ep, v, D, /*profile=*/false, pr.actDtype,
                                pr.reduceInFloat, pr.wgtDtype, shape,
                                /*verifyPlacement=*/true, rv, layers);
        createUs = std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - createStart)
                       .count();
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 x%d session create %.1f s%s\n",
                    ep.providerKey.c_str(), tag, (long long)D, layers,
                    createUs / 1.0e6,
                    rv == OnnxReduceView::Rank4 ? " (rank-4 reduction)" : "");
        return g;
      };
      double createUs = 0.0;
      GemmSetup g = build(view, createUs);

      // A provider can take the multiply and refuse the reduction behind it
      // at a real size -- the QNN Adreno backend did at every size from 1024,
      // leaving the multiply in a partition fed by constants alone, which it
      // then rejected as "Zero tensor size!".  The same values reduced
      // through a rank-4 view is the one other spelling worth a compile.  Not
      // after running out of memory, which says the size is too big however
      // it is spelled.
      if (!g.session && !g.offDevice && view == OnnxReduceView::Rows &&
          onnxFailureStatus(g.error) == ResultStatus::Unsupported &&
          !onnxReasonIsOutOfMemory(g.error) && !clpeak::cancelRequested())
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 refused (%s); retrying the "
                    "reduction through a rank-4 view\n",
                    ep.providerKey.c_str(), tag, (long long)D,
                    g.error.c_str());
        double retryUs = 0.0;
        GemmSetup r4 = build(OnnxReduceView::Rank4, retryUs);
        if (r4.session)
        {
          // Every later size builds once, in this view, so this build's time
          // is the one the compile gate should extrapolate from.
          view = OnnxReduceView::Rank4;
          onnxNoteRank4Reduce(rt, ep);
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
      if (view == OnnxReduceView::Rank4 && L.rank4From == 0)
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
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 x%d -> %.3f\n",
                  ep.providerKey.c_str(), tag, (long long)D, layers, rate);

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
      const double dispatchUs = (pr.probeUs > 0.0) ? pr.probeUs : 0.0;
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

      // Climb to the plateau (kOnnxPlateauGain, one grace size).
      if (rate > L.best * kOnnxPlateauGain)
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
        if (++strikes >= kOnnxPlateauStrikes)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: plateau past %lld^3\n",
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

      // A rung that compiled past the cap anyway is kept, and ends the
      // ladder.  The first rung is allowed past it once -- its time seeds
      // the prediction above; truncating it would discard a valid peak.
      if (D != kMinDim && createUs > kOnnxMaxCreateUs)
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

  auto runVariant = [&](const Variant &v) {
    const bool isInt = isIntVariant(v);
    // NVFP4 is the one row measured as a single multiply: chaining it would
    // mean requantizing every layer's product to NVFP4 with block scales
    // taken from the data at run time, which this test does not build.
    const int layers = v.nvfp4 ? 1 : kChainLayers;
    const bool chained = layers > 1;

    logger::EmitOptions o;
    const std::string sweep =
        chained ? "Peak over a doubling sweep of layer widths, sixteen layers "
                  "chained per dispatch."
                : "Peak over a doubling sweep of square sizes, one multiply "
                  "per dispatch.";
    o.description = sweep + "  " + v.note;
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
      test.skip(v.label, ResultStatus::Unsupported,
                "no probe result for " + std::string(v.label), o);
      return;
    }
    const OnnxProbeResult &pr = it->second;
    if (!pr.ok)
    {
      test.skip(v.label, onnxFailureStatus(pr.reason), pr.reason, o);
      return;
    }

    // The probe left one or more viable shapes, result-scaled first.  A
    // single multiply takes them in that order: result-scaled is the fastest
    // shape wherever it does not fold, and the fold check drops to the next
    // where it does.  A chain takes its live shapes first: result-scaled
    // would be sixteen constant multiplies for a folding compiler to work
    // through before the check could catch it -- minutes on QNN -- while the
    // live pass is spread across the sixteen.  Result-scaled stays in the
    // list as the last resort, for the rows no live shape builds.
    std::vector<OnnxLiveShape> shapes = pr.shapes;
    if (chained)
      std::stable_partition(shapes.begin(), shapes.end(),
                            [](OnnxLiveShape s) {
                              return s != OnnxLiveShape::ResultScaled;
                            });
    else if (providerFolds && shapes.size() > 1 &&
             shapes.front() == OnnxLiveShape::ResultScaled)
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: this provider folded an earlier row's "
                  "resident product; starting at %s\n",
                  ep.providerKey.c_str(), v.label, shapeNameFor(shapes[1]));
      shapes.erase(shapes.begin());
    }

    OnnxReduceView view = onnxPrefersRank4Reduce(rt, ep)
                              ? OnnxReduceView::Rank4
                              : OnnxReduceView::Rows;
    Ladder L;
    for (size_t shi = 0; shi < shapes.size(); shi++)
    {
      L = ladder(v, pr, shapes[shi], layers, view);
      if (L.folded)
        providerFolds = true;
      // Caught folding and another shape remains: drop to it (a live shape
      // the compiler cannot evaluate at build time) rather than reporting an
      // error.
      if (L.folded && shi + 1 < shapes.size())
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %s folded, retrying %s\n",
                    ep.providerKey.c_str(), v.label,
                    shapeNameFor(shapes[shi]), shapeNameFor(shapes[shi + 1]));
        continue;
      }
      break; // settled: this shape produced the row (measurement or error)
    }

    if (L.best <= 0.0)
    {
      if (L.folded)
        onnxNoteGemmFolded(ep, v.label);
      test.skip(v.label, onnxFailureStatus(L.err, L.errStatus),
                L.err.empty() ? "no supported datatype" : L.err, o);
      return;
    }

    o.description =
        chained ? "Peak over a doubling sweep of layer widths, sixteen layers "
                  "chained per dispatch; fastest at " +
                      std::to_string(L.bestDim) + "-wide layers.  " + v.note
                : "Peak over a doubling sweep of square sizes; fastest at " +
                      std::to_string(L.bestDim) + " cubed.  " + v.note +
                      "  A single multiply per dispatch: chaining NVFP4 would "
                      "requantize every layer's product with block scales "
                      "taken from the data at run time, which this test does "
                      "not build.";
    if (!pr.ranAs.empty())
      o.description += "  Ran as " + pr.ranAs + " (" + pr.schemeName + ").";
    if (pr.castedActs)
      o.description += "  The provider converts the activations first, a "
                       "full pass inside this figure.";
    if (!pr.ranWider.empty())
      o.description += "  This provider has no " + std::string(v.label) +
                       " matmul kernel and ran it in " + pr.ranWider +
                       ", so this is that width rather than " + v.label + ".";
    if (pr.reduceInFloat)
      o.description += "  The product is cast to fp32 before the reduction; "
                       "the multiplies are unaffected.";
    o.description += shapeNote(v, L.shape);
    if (L.rank4From > 0)
      o.description += "  From " + std::to_string(L.rank4From) +
                       " the product was reduced through a 4-D view of itself, "
                       "because this provider refused the 2-D reduction there; "
                       "the values reduced and the work are the same.";
    if (L.offDeviceBelow > 0)
      o.description += "  Sizes below " + std::to_string(L.firstDim) +
                       " were sent to another compute unit by the provider's "
                       "runtime and are not in this figure.";
    if (L.offDeviceAbove > 0)
      o.description += "  From " + std::to_string(L.offDeviceAbove) +
                       " the provider's runtime sent the work to another "
                       "compute unit, which ended the sweep.";
    test.emit(v.label, (float)L.best, o);

    // What the provider's own profiler says the time went to, for the log.
    // One more build, never one that was timed (a profiled session is not
    // the session measured), and only when someone asked for the detail.
    const std::string profilePath =
        clpeak::verboseEnabled() ? onnxNativeProfilePath(ep) : std::string();
    if (!profilePath.empty() && !clpeak::cancelRequested())
    {
      int64_t pd = 0;
      for (const auto &c : L.creates)
        if (c.first <= L.bestDim && c.second <= kNativeProfileCreateCapUs)
          pd = c.first;
      if (pd > 0)
      {
        const OnnxReduceView pv = (L.rank4From > 0 && pd >= L.rank4From)
                                      ? OnnxReduceView::Rank4
                                      : OnnxReduceView::Rows;
        GemmSetup g = makeSetup(rt, ep, v, pd, /*profile=*/false, pr.actDtype,
                                pr.reduceInFloat, pr.wgtDtype, L.shape,
                                /*verifyPlacement=*/true, pv, layers,
                                profilePath);
        if (g.session && timeRuns(rt, g, 1 + warmupCount) > 0.0)
          timeRuns(rt, g, 3);
        else
          CLPEAK_VLOG("onnx-gemm[%s/%s]: profiled build at %lld failed: %s\n",
                      ep.providerKey.c_str(), v.label, (long long)pd,
                      g.error.c_str());
        destroySetup(rt, g);
        onnxLogNativeProfile(profilePath, ep.providerKey + "/" + v.label + " " +
                                              std::to_string(pd) + "^3 x" +
                                              std::to_string(layers));
      }
      else
        CLPEAK_VLOG("onnx-gemm[%s/%s]: no measured size compiled within %.0f s; "
                    "not profiled\n",
                    ep.providerKey.c_str(), v.label,
                    kNativeProfileCreateCapUs / 1.0e6);
    }
  };

  for (size_t i = 0; i < kFpVariantCount; i++)
  {
    if (clpeak::cancelRequested()) break;
    runVariant(kFpVariants[i]);
  }
  for (size_t i = 0; i < kIntVariantCount; i++)
  {
    if (clpeak::cancelRequested()) break;
    runVariant(kIntVariants[i]);
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
