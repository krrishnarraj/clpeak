#ifdef ENABLE_COREML

// coreml-gemm: matrix-multiply peak through Core ML, on whichever of the
// Neural Engine, the GPU and the CPU this device row names.
//
// The shape is the ONNX backend's (src/onnx/gemm.cpp): both operands are
// model constants and the result is reduced to one row, so nothing large
// crosses the host boundary per run; a runtime scalar keeps the graph from
// being a constant expression.  Sizes double from 1024 until the rate stops
// improving, and the peak is reported with the size that produced it.
//
// One test, `coreml_gemm`: the same single-operation model in every format
// Core ML can store a weight in.  Core ML's arithmetic is fp16 or fp32 and
// nothing else, so most narrow rows report flops -- the weights are unpacked
// into a float multiply -- and only the quantized-activation row (int8_qdq)
// reports ops, because on a Neural Engine with integer multipliers (A17 Pro,
// M4 and later) that is the one that runs as integer arithmetic.
//
// The compute plan is the guard.  Core ML never refuses a model for lack of
// a Neural Engine kernel; it moves the operation to another compute unit and
// says nothing.  Every session here reads its plan, and a row whose matmul
// landed anywhere but the device named reports unsupported with the
// operations that moved, rather than a CPU number under an accelerator's
// name -- which is exactly what happens to blockwise int4 on an M1.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kMinDim = 1024;
constexpr int64_t kMaxDim = 32768;

// A size counts as an improvement only if it beats the best so far by this
// much; two failures in a row end the search.
constexpr double kImproveFactor = 1.03;
constexpr int kMaxStrikes = 2;

// One iteration, predicted from the previous rung's rate, must not exceed
// this; a rung that measures slower than it ends the ladder outright.
constexpr double kMaxIterUs = 2.0e6;

// Per-size budget for the timed phase.
constexpr unsigned int kSizeBudgetUs = 2000000;

// The size whose whole time is the cost of asking: 64^3 is 0.5 MFLOP.
constexpr int64_t kFloorDim = 64;

// A rung has to do at least this much work, as a share of the submission
// cost, before it counts as having computed anything (see the ONNX ladder,
// where the value was calibrated across seven providers).
constexpr double kFoldWorkFloor = 0.15;

// Both operands together, capped at a quarter of physical memory: a fixed
// ceiling would be a crash on a phone and a needless limit on a Mac Studio.
uint64_t maxOperandBytes() { return clpeak::memoryBudget(3ull << 30); }

struct Variant
{
  CoremlWeight w;
  const char *note;
};

const Variant kVariants[] = {
    {CoremlWeight::Fp16,
     "16-bit inputs, the Neural Engine's native width and the currency of "
     "every shipping Core ML model."},
    {CoremlWeight::Fp32,
     "Full 32-bit precision.  The Neural Engine has no fp32 path, so on that "
     "row Core ML sends the multiply elsewhere and the plan says where."},
    {CoremlWeight::Bf16,
     "bfloat16: in Core ML's type list but accepted by none of its "
     "operations, so this row records what the compiler says to it."},
    {CoremlWeight::Int8Channel,
     "8-bit integer weights with one scale per output column, unpacked into a "
     "16-bit multiply -- Core ML's affine weight quantization.  The "
     "arithmetic is unchanged, so what the narrow weights buy is traffic."},
    {CoremlWeight::Int4Block,
     "4-bit integer weights, one scale per block of 32 along the reduction "
     "axis, unpacked into a 16-bit multiply.  The blocked form needs macOS 15 "
     "/ iOS 18, and a Neural Engine older than the A17 Pro / M4 generation "
     "may take it only with the activations resident, as they are here -- "
     "the transformer-block rows, whose activations are live, say whether it "
     "runs the way a model would use it."},
    {CoremlWeight::Int4Lut,
     "4-bit palettized weights: every value is an index into a 16-entry "
     "table, the compression Apple's Neural Engine was built to decode.  "
     "The same 16-level grid as the blocked row, so the two differ only in "
     "how the levels are found."},
    {CoremlWeight::Fp8Block,
     "8-bit float (E4M3) weights with block scales.  The type exists in Core "
     "ML's format since macOS 26; whether any compute unit takes it is what "
     "this row asks."},
    {CoremlWeight::Int8Qdq,
     "8-bit weights and 8-bit activations: the activations are quantized on "
     "device and the result quantized back, the pattern Core ML fuses into an "
     "integer multiply on Neural Engines that have one (A17 Pro, M4 and "
     "later).  Measured in ops; a reading no faster than the fp16 row is a "
     "Neural Engine doing this in 16-bit."},
};

uint64_t operandBytes(CoremlWeight w, int64_t D)
{
  const int act = coremlActDtype(w);
  return coremlElemBytes(act, D * D) + coremlWeightBytes(w, D, D);
}

} // namespace

int CoreMLPeak::runGemm(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_gemm", "Core ML matmul peak", "flops", Category::Unknown,
       "Matrix-multiply rate through Core ML on this compute unit, one weight "
       "format per row, swept over square sizes and reported at its best.  "
       "The same model runs on the Neural Engine, the GPU and the CPU, and "
       "the compute plan proves which one actually did the multiply: a row "
       "Core ML would have moved to another unit reports unsupported instead.",
       TestShape::Heterogeneous, "weight format"});

  // The cost of asking, measured once: a 64^3 multiply whose arithmetic is
  // nothing, so its time is submission and the reduction.  Every fold check
  // below is against this.
  double floorUs = 0.0;
  {
    std::string err;
    auto s = CoremlSession::create(dev, coremlResidentMatMulModel(spec, kFloorDim, kFloorDim, kFloorDim,
                                                                  CoremlWeight::Fp16, true), err);
    if (s && coremlBindScalar(*s, "s", CML_FP16, err))
    {
      auto m = coremlMeasure(*s, warmupCount, 200000, false, 0, 50);
      if (m.meanUs > 0.0)
        floorUs = m.meanUs;
    }
    CLPEAK_VLOG("coreml-gemm[%s]: submission floor %.1f us\n", dev.displayName.c_str(), floorUs);
  }

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;
    const char *label = coremlWeightLabel(v.w);
    const bool isInt = (v.w == CoremlWeight::Int8Qdq);
    const int act = coremlActDtype(v.w);
    const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;

    logger::EmitOptions o;
    o.description = std::string("Peak over a doubling sweep of square sizes.  ") + v.note;
    if (isInt)
      o.unit = "ops";

    if (coremlSpecForWeight(v.w) > spec)
    {
      test.skip(label, ResultStatus::Unsupported,
                "needs " + coremlOsForSpec(coremlSpecForWeight(v.w)) +
                    " (model specification " + std::to_string(coremlSpecForWeight(v.w)) +
                    "); this OS accepts " + std::to_string(spec),
                o);
      continue;
    }

    // Result-scaled first (the cheapest live shape), operand-scaled if the
    // compiler is caught folding it.  W8A8 quantizes a live operand and has
    // only the second.
    std::vector<bool> shapes;
    if (isInt)
      shapes = {false};
    else
      shapes = {true, false};

    for (size_t si = 0; si < shapes.size(); si++)
    {
      const bool resultScaled = shapes[si];

      double best = 0.0;
      int64_t bestDim = 0;
      std::string firstErr;
      ResultStatus errStatus = ResultStatus::Unsupported;
      bool folded = false;
      int rungs = 0;
      bool endedOnWork = false;
      std::string offDeviceNote, glueNote;
      double firstUs = 0.0, lastUs = 0.0;
      int64_t firstDim = 0, lastDim = 0;
      double lastRate = 0.0, prevCreateUs = 0.0, prevUs = 0.0;
      int64_t prevD = 0;
      int strikes = 0;

      for (int64_t D = kMinDim; D <= kMaxDim; D *= 2)
      {
        if (clpeak::cancelRequested())
          break;

        const uint64_t bytes = operandBytes(v.w, D);
        if (bytes > maxOperandBytes())
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 needs %llu MB of operands, stopping\n",
                      dev.displayName.c_str(), label, (long long)D,
                      (unsigned long long)(bytes >> 20));
          endedOnWork = true;
          break;
        }
        if (lastRate > 0.0)
        {
          const double predictedUs = 2.0 * (double)D * (double)D * (double)D / lastRate;
          if (predictedUs > kMaxIterUs)
          {
            CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 would take ~%.1f s per iteration, stopping\n",
                        dev.displayName.c_str(), label, (long long)D, predictedUs / 1.0e6);
            endedOnWork = true;
            break;
          }
        }
        if (D > kMinDim && prevCreateUs > 0.0 && prevCreateUs * 4.0 > kCoremlMaxCreateUs)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 predicted create %.1f s > %.1f s, stopping\n",
                      dev.displayName.c_str(), label, (long long)D,
                      prevCreateUs * 4.0 / 1.0e6, kCoremlMaxCreateUs / 1.0e6);
          break;
        }

        std::string err;
        auto s = CoremlSession::create(dev, coremlResidentMatMulModel(spec, D, D, D, v.w, resultScaled), err);
        if (!s)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 create failed: %s\n",
                      dev.displayName.c_str(), label, (long long)D, err.c_str());
          if (firstErr.empty())
            firstErr = err;
          // Larger sizes need strictly more of everything.
          break;
        }
        const double createUs = coremlCreateUs(*s);
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 create %.2f s (compile %.2f, load %.2f, plan %.2f)\n",
                    dev.displayName.c_str(), label, (long long)D, createUs / 1.0e6,
                    s->compileUs / 1.0e6, s->loadUs / 1.0e6, s->planUs / 1.0e6);

        if (!s->onDevice())
        {
          const std::string why = coremlOffDeviceReason(dev, *s);
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 %s\n", dev.displayName.c_str(), label,
                      (long long)D, why.c_str());
          if (rungs == 0)
          {
            if (firstErr.empty())
              firstErr = why;
            // Too small for the planner is not too big for the unit; the
            // ladder climbs on.  A shape the unit cannot run at all will not
            // become runnable by growing.
            if (s->offDeviceCapable())
            {
              s.reset();
              continue;
            }
          }
          else
            offDeviceNote = "  Larger sizes were declined by the " +
                            std::string(coremlKindName(dev.kind)) + ".";
          break;
        }

        if (!coremlBindScalar(*s, "s", ioDtype, err))
        {
          if (firstErr.empty())
          {
            firstErr = err;
            errStatus = ResultStatus::Error;
          }
          break;
        }

        auto m = coremlMeasure(*s, warmupCount, kSizeBudgetUs, forceIters, specifiedIters);
        if (m.meanUs > 0.0)
          glueNote = coremlGlueNote(*s);
        s.reset();   // the temp files go before the next size is written
        if (m.meanUs <= 0.0)
        {
          if (firstErr.empty())
          {
            firstErr = m.error;
            errStatus = m.status;
          }
          break;
        }

        rungs++;
        if (firstUs == 0.0)
        {
          firstUs = m.meanUs;
          firstDim = D;
        }
        lastUs = m.meanUs;
        lastDim = D;

        const double ops = 2.0 * (double)D * (double)D * (double)D;
        const double rate = ops * 1.0e6 / m.meanUs;
        lastRate = ops / m.meanUs;
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 -> %.3f (%.1f us, %u iters)\n",
                    dev.displayName.c_str(), label, (long long)D, rate, m.meanUs, m.iters);

        // Fold detector for the foldable shape: work is time less the
        // submission floor, and eight times the arithmetic has to cost at
        // least twice the time however much the rate improves.
        const double work = m.meanUs - floorUs;
        const double prevWork = prevUs - floorUs;
        const bool computedNothing = resultScaled && floorUs > 0.0 && work < kFoldWorkFloor * floorUs;
        const bool workFlat = resultScaled && prevUs > 0.0 && D == prevD * 2 && prevWork > 0.0 &&
                              work < prevWork * 2.0;
        if (computedNothing || workFlat)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 took %.1f us against a %.1f us floor "
                      "(previous %.1f us) -- constants were folded\n",
                      dev.displayName.c_str(), label, (long long)D, m.meanUs, floorUs, prevUs);
          best = 0.0;
          firstErr = "Core ML folded the operands at compile time: the timed runs measure "
                     "dispatch plus a reduction of a precomputed result rather than the "
                     "matrix multiply, so the timings do not scale with the problem size";
          errStatus = ResultStatus::Error;
          folded = true;
          break;
        }
        prevUs = m.meanUs;
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
            CLPEAK_VLOG("coreml-gemm[%s/%s]: no further gain past %lld^3\n",
                        dev.displayName.c_str(), label, (long long)bestDim);
            break;
          }
        }

        if (m.probeUs > kMaxIterUs)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld^3 measured %.1f s per iteration, stopping\n",
                      dev.displayName.c_str(), label, (long long)D, m.probeUs / 1.0e6);
          endedOnWork = true;
          break;
        }

        // Compilation gates: a cliff between rungs, or an absolute ceiling
        // (the first rung may exceed it once; its time seeds the growth gate).
        if (prevCreateUs > 0.0 && createUs > kCoremlCreateGrowthFloor &&
            createUs > prevCreateUs * kCoremlCreateGrowthFactor)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: create grew %.1fx at %lld^3, stopping\n",
                      dev.displayName.c_str(), label, createUs / prevCreateUs, (long long)D);
          prevCreateUs = createUs;
          break;
        }
        if (D != kMinDim && createUs > kCoremlMaxCreateUs)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: create %.1f s > %.1f s at %lld^3, stopping\n",
                      dev.displayName.c_str(), label, createUs / 1.0e6,
                      kCoremlMaxCreateUs / 1.0e6, (long long)D);
          prevCreateUs = createUs;
          break;
        }
        prevCreateUs = createUs;
      }

      // Whole-ladder backstop for the foldable shape: real work grows with
      // the cube of the size, so anything near flat computed nothing.
      double expectedGrowth = 1.0;
      for (int64_t d = firstDim; d > 0 && d < lastDim; d *= 2)
        expectedGrowth *= 8.0;
      if (resultScaled && best > 0.0 && firstUs > 0.0 && lastDim > firstDim &&
          lastUs < firstUs * expectedGrowth / 64.0)
      {
        best = 0.0;
        firstErr = "Core ML folded the operands at compile time: the timings do not scale "
                   "with the problem size and mean nothing";
        errStatus = ResultStatus::Error;
        folded = true;
      }
      if (resultScaled && best > 0.0 && rungs == 1 && !endedOnWork)
      {
        best = 0.0;
        firstErr = "only one size could be measured before the compile budget ran out, so "
                   "nothing confirms the timing scaled with the work rather than measuring "
                   "dispatch";
        errStatus = ResultStatus::Error;
        folded = true;
      }

      if (folded && si + 1 < shapes.size())
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: result-scaled folded, retrying operand-scaled\n",
                    dev.displayName.c_str(), label);
        continue;
      }

      if (best > 0.0)
      {
        o.description = "Peak over a doubling sweep of square sizes; fastest at " +
                        std::to_string(bestDim) + " cubed.  " + v.note + offDeviceNote + glueNote;
        if (!resultScaled && !isInt)
          o.description += "  The activations are scaled at run time so the compiler "
                           "cannot fold the multiply away; that pass is inside the figure.";
        test.emit(label, (float)best, o);
      }
      else
        test.skip(label, errStatus, firstErr.empty() ? "no size could be measured" : firstErr, o);
      break;
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
