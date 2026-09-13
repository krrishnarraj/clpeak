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
#include <cstring>
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
                 ? "  The activations are scaled by a runtime value and "
                   "quantized on the device before the multiply, so it cannot "
                   "be evaluated at compile time; that pass is inside this "
                   "figure."
                 : "  The activations are scaled by a runtime value before the "
                   "multiply, so it cannot be evaluated at compile time; that "
                   "one elementwise pass is inside this figure.";
    case OnnxLiveShape::Add0:
      return "  The activations add a runtime zero on the way in, so the "
             "multiply cannot be evaluated at compile time; that pass is "
             "inside this figure.";
    case OnnxLiveShape::QdqAdd0:
      return "  The activations add a runtime zero as a quantized op on the "
             "way in, so the multiply cannot be evaluated at compile time; "
             "that pass is inside this figure.";
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
       "Matrix-multiply speed through ONNX Runtime on this execution "
       "provider, using a single-operation model with constant weights.  "
       "The identical model runs on every provider, so rows from different "
       "providers are directly comparable -- and the gap against a vendor's "
       "advertised TOPS is real, not an artifact of different test code.  "
       "Providers that cannot run an operation entirely on their device "
       "report it as unsupported instead of quietly measuring the CPU.  "
       "Each reading is a different input format.",
       TestShape::Heterogeneous, "data type"});

  // runAll also clears per EP before dispatching; this entry clear keeps
  // direct runGemm callers (tests, future entry points) from inheriting a
  // stale record from an earlier run in the same process.
  onnxClearGemmFolded(ep);

  const bool fp32As16 = onnxEpRunsFp32AsFp16(ep);

  auto runVariant = [&](const Variant &v) {
    const bool isInt = isIntVariant(v);

    logger::EmitOptions o;
    o.description = std::string("Peak over a doubling sweep of square sizes.  ") + v.note;
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
      test.skip(v.label, ResultStatus::Unsupported, pr.reason, o);
      return;
    }
    // The probe left one or more viable shapes, result-scaled first.  Try
    // them in order: the first that is not caught folding is the measurement,
    // and a fold drops to the next (a live shape a vendor compiler cannot
    // evaluate at build time).  A non-folding provider never leaves the first.
    for (size_t shapeIdx = 0; shapeIdx < pr.shapes.size(); shapeIdx++)
    {
    const OnnxLiveShape shape = pr.shapes[shapeIdx];
    const bool foldable = (shape == OnnxLiveShape::ResultScaled);

    double best = 0.0;
    int64_t bestDim = 0;
    std::string firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;
    // Set by the folding guards below; drives the paired suppression in
    // runNumericError (see onnx_probe.h).
    bool folded = false;
    int rungs = 0;
    // Why the ladder ended, for the single-rung verdict below.
    bool endedOnWork = false; // memory or measured time: real work happened

    // First and last timings with their sizes, to confirm the work actually
    // happened (see the folding check after the loop).
    double firstUs = 0.0, lastUs = 0.0;
    int64_t firstDim = 0, lastDim = 0;
    double lastRate = 0.0;
    double prevCreateUs = 0.0;
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
      const uint64_t weightBytes = operandBytes(v, D, shape);
      if (weightBytes > maxWeightBytes())
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 needs %llu MB of operands, "
                    "stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)D, (unsigned long long)(weightBytes >> 20));
        endedOnWork = true;
        break;
      }
      if (lastRate > 0.0)
      {
        const double predictedUs =
            2.0 * (double)D * (double)D * (double)D / lastRate;
        if (predictedUs > kMaxIterUs)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 would take ~%.1f s per "
                      "iteration, stopping\n",
                      ep.providerKey.c_str(),
                      v.label, (long long)D, predictedUs / 1.0e6);
          endedOnWork = true;
          break;
        }
      }
      // Predicted compilation gate: skip the next rung before paying its
      // build when the previous rung already predicts worse than the budget
      // (compilation scales roughly 4x memory per 2x dim).  The first rung
      // is always attempted -- its time seeds this prediction.
      if (D > kMinDim && prevCreateUs > 0.0 &&
          prevCreateUs * 4.0 > kOnnxMaxCreateUs)
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 predicted create %.1f s (prev %.1f s *4) > %.1f s, stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)D, prevCreateUs * 4.0 / 1.0e6,
                    prevCreateUs / 1.0e6, kOnnxMaxCreateUs / 1.0e6);
        break;
      }

      auto createStart = std::chrono::steady_clock::now();
      GemmSetup g = makeSetup(rt, ep, v, D, /*profile=*/false, pr.actDtype,
                              pr.reduceInFloat, pr.wgtDtype, shape);
      auto createEnd = std::chrono::steady_clock::now();
      double createUs = std::chrono::duration<double, std::micro>(
                            createEnd - createStart).count();
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 session create %.1f s\n",
                  ep.providerKey.c_str(), v.label,
                  (long long)D, createUs / 1.0e6);

      if (!g.session)
      {
        if (firstErr.empty())
          firstErr = g.error;
        // Larger sizes need strictly more of everything, so nothing above
        // this one can succeed either.
        break;
      }

      double per_iter_us = -1.0;
      if (timeRuns(rt, g, 1 + warmupCount) > 0.0) // compile + warmup
        per_iter_us = timeRuns(rt, g, 1);         // calibration probe
      if (per_iter_us <= 0.0)
      {
        if (firstErr.empty())
        {
          firstErr = g.error.empty() ? "run failed" : g.error;
          errStatus = ResultStatus::Error;
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
      if (mean_us <= 0.0 && firstErr.empty())
      {
        firstErr = g.error.empty() ? "run failed" : g.error;
        errStatus = ResultStatus::Error;
      }
      destroySetup(rt, g);
      if (mean_us <= 0.0)
        break;

      rungs++;
      if (firstUs == 0.0)
      {
        firstUs = mean_us;
        firstDim = D;
      }
      lastUs = mean_us;
      lastDim = D;

      const double ops = 2.0 * (double)D * (double)D * (double)D;
      const double rate = ops * 1.0e6 / mean_us;
      // Rate per FLOP-count is what the ladder is searching on; the raw rate
      // in ops/us drives the time prediction for the next rung.
      lastRate = ops / mean_us;
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 -> %.3f\n", ep.providerKey.c_str(),
                  v.label, (long long)D, rate);

      // Per-doubling fold detector, for the foldable shape only.  A folded
      // graph leaves dispatch plus a reduction over D elements, so its time
      // barely moves while the work grows 8x: QNN read ~170 us at both 1024
      // and 2048.  The test is on the time, not the rate -- a provider with
      // an expensive dispatch shows a rate that jumps several-fold across
      // its first doubling on perfectly real work, because the first rung
      // was mostly dispatch, and a rate threshold mistook that for folding.
      // Real work at least doubles the time of the previous rung whatever
      // the dispatch cost; a fold does not.  The live shapes cannot fold, so
      // their ladders are never second-guessed.
      if (foldable && prevUs > 0.0 && D == prevD * 2 && mean_us < prevUs * 2.0)
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %.1f us at %lld vs %.1f us at %lld "
                    "(%.2fx for 8x the work) -- work does not scale, "
                    "constants were folded\n",
                    ep.providerKey.c_str(), v.label, prevUs,
                    (long long)prevD, mean_us, (long long)D, mean_us / prevUs);
        best = 0.0;
        firstErr = "this provider folded the operands at compile time: the "
                   "timed runs measure dispatch plus a reduction of a "
                   "precomputed result rather than the matrix multiply, so "
                   "the timings do not scale with the problem size and mean "
                   "nothing";
        errStatus = ResultStatus::Error;
        folded = true;
        break;
      }
      prevUs = mean_us;
      prevD = D;

      if (rate > best * kImproveFactor)
      {
        strikes = 0;
        best = rate;
        bestDim = D;
      }
      else
      {
        if (rate > best)
        {
          best = rate;
          bestDim = D;
        }
        if (++strikes >= kMaxStrikes)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: no further gain past %lld^3\n",
                      ep.providerKey.c_str(), v.label, (long long)bestDim);
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
                    ep.providerKey.c_str(), v.label,
                    (long long)D, per_iter_us / 1.0e6);
        endedOnWork = true;
        break;
      }

      // Session creation (graph compilation) can dominate on ahead-of-time
      // providers.  Two guards: absolute and factor.  The first rung is
      // allowed to exceed the absolute once -- its time seeds the factor
      // gate; truncating it would discard a valid peak.  Factor catches a
      // cliff before the next model is even built.
      bool createCliff = (prevCreateUs > 0.0 &&
                          createUs > prevCreateUs * kOnnxCreateGrowthFactor);
      if (createCliff)
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 create grew %.1fx (%.1f s -> %.1f s) > %.1fx, stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)D, createUs / prevCreateUs,
                    prevCreateUs / 1.0e6, createUs / 1.0e6,
                    kOnnxCreateGrowthFactor);
        prevCreateUs = createUs;
        break;
      }
      if (D != kMinDim && createUs > kOnnxMaxCreateUs)
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 create %.1f s > %.1f s, stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)D, createUs / 1.0e6, kOnnxMaxCreateUs / 1.0e6);
        prevCreateUs = createUs;
        break;
      }
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
    for (int64_t d = firstDim; d > 0 && d < lastDim; d *= 2)
      expectedGrowth *= 8.0; // each doubling is 8x work
    if (foldable && best > 0.0 && firstUs > 0.0 && lastDim > firstDim &&
        lastUs < firstUs * expectedGrowth / 64.0)
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %.1f us at %lld vs %.1f us at %lld -- "
                  "work does not scale, constants were folded\n",
                  ep.providerKey.c_str(), v.label, firstUs,
                  (long long)firstDim, lastUs, (long long)lastDim);
      best = 0.0;
      firstErr = "this runtime folded the operands at load time: it accepted "
                 "the request to disable constant folding and ignored it, "
                 "which ONNX Runtime did before about 1.18, so the timings "
                 "do not scale with the problem size and mean nothing";
      errStatus = ResultStatus::Error;
      folded = true;
    }

    // A foldable ladder that ended after one rung has nothing to compare
    // that rung against.  When it ended because the next size would not fit
    // or the rung itself took seconds, real work demonstrably happened; when
    // a compile-time gate ended it, the gate is the fold's own signature --
    // a compiler evaluating 2 GFLOP with reference code takes a minute --
    // and the one timing is indistinguishable from dispatch.  QNN published
    // 12 TFLOPS fp32, 12 TFLOPS fp16 and 12 TOPS int8 from one rung each,
    // all the same 179 us, before this rule existed.
    if (foldable && best > 0.0 && rungs == 1 && !endedOnWork)
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: one rung, ended on a compile-time gate; "
                  "cannot rule out folding\n",
                  ep.providerKey.c_str(), v.label);
      best = 0.0;
      firstErr = "only one size could be measured before the provider's "
                 "compile time ran out, so nothing confirms the timing "
                 "scaled with the work rather than measuring dispatch";
      errStatus = ResultStatus::Error;
      folded = true;
    }

    // Caught folding and another shape remains: drop to it (a live shape the
    // compiler cannot evaluate at build time) rather than reporting an error.
    if (folded && shapeIdx + 1 < pr.shapes.size())
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %s folded, retrying %s\n",
                  ep.providerKey.c_str(), v.label, shapeNameFor(shape),
                  shapeNameFor(pr.shapes[shapeIdx + 1]));
      continue;
    }

    if (best > 0.0)
    {
      o.description = "Peak over a doubling sweep of square sizes; fastest at " + std::to_string(bestDim) + " cubed.  " + v.note;
      if (!pr.ranAs.empty())
        o.description += "  The provider ran the multiply as " + pr.ranAs +
                         " with " + pr.schemeName +
                         ", confirming it really ran in " + v.label + ".";
      if (pr.castedActs)
        o.description += "  This provider does not take the activations in the "
                         "width they were given, so it converts them first: "
                         "that is a full pass over them on every run and it is "
                         "inside this figure.";
      if (pr.reduceInFloat)
        o.description += "  This provider has no reduction for the datatype, "
                         "so the product is cast to fp32 before being reduced; "
                         "the multiply itself is unaffected, but the cast is a "
                         "full pass over the result and costs a few percent.";
      o.description += shapeNote(v, shape);
      if (fp32As16 && v.dtype == ONNX_DT_FLOAT && !v.qdq)
        o.description += "  This provider runs fp32 graphs at 16-bit precision "
                         "by default, which is how an fp32 model reaches its "
                         "hardware at all; the fp32 numeric-error row shows "
                         "the cost.";
      test.emit(v.label, (float)best, o);
    }
    else
    {
      if (folded)
        onnxNoteGemmFolded(ep, v.label);
      test.skip(v.label, errStatus,
                firstErr.empty() ? "no supported datatype" : firstErr, o);
    }
    break; // settled: this shape produced the row (measurement or error)
    }      // shape-retry loop
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
