#ifdef ENABLE_COREML

// coreml-gemm: matrix-multiply peak through Core ML, on whichever of the
// Neural Engine, the GPU and the CPU this device row names.
//
// The shape is the ONNX backend's (src/onnx/gemm.cpp): sixteen distinct
// square layers chained in one prediction, each layer's product the next
// one's activations, swept over layer width.  The weights are model
// constants and the last product is reduced to one row, so nothing large
// crosses the host boundary per run, and the chain starts from a 64-wide seed
// that takes a runtime scalar and one more multiply, not counted, widens into
// the first layer's activations -- so every multiply has a live operand and
// no compiler can evaluate the chain at build time.  Widths double from 1024
// until the rate stops improving, and the peak is reported with the width
// that produced it.
//
// One multiply per prediction measured what surrounds the multiply as much
// as the multiply.  On an M1 Pro's Neural Engine a single 2048-cube read 8.3
// TFLOPS fp16 and sixteen chained 2048-wide layers read 10.5 -- int8_weight
// 8.4 against 11.0, int4_lut 8.4 against 11.4, int8_qdq 9.6 against 11.5
// TOPS -- the figure the ONNX backend's CoreML provider reads for its own
// chain (10.5).  A chain is also how a model runs, with live activations,
// and two readings the single multiply made were its constant activations'
// doing.  Blockwise int4, which they let through, is unsupported on that
// Neural Engine here, as it is in the transformer block.  And the CPU read
// fp16 at 5.0-5.2 TFLOPS with both operands constant, against 4.2 for the
// same multiply with a live input and 4.5-5.0 for the chain over three runs:
// its float rows read up to a tenth lower than before, and the chain is
// still the faster form.
//
// One test, `coreml_gemm`: the same chain in every format Core ML can store a
// weight in.  Core ML's arithmetic is fp16 or fp32 and nothing else, so most
// narrow rows report flops -- the weights are unpacked into a float multiply
// -- and only the quantized-activation row (int8_qdq) reports ops, because on
// a Neural Engine with integer multipliers (A17 Pro, M4 and later) that is
// the one that runs as integer arithmetic.
//
// The compute plan is the guard.  Core ML never refuses a model for lack of
// a Neural Engine kernel; it moves the operation to another compute unit and
// says nothing.  Every session here reads its plan, and a row whose matmuls
// landed anywhere but the device named reports unsupported with the
// operations that moved, rather than a CPU number under an accelerator's
// name.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

#include <algorithm>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kMinDim = 1024;
constexpr int64_t kMaxDim = 32768;

// Layers per chain, and the width of the seed the runtime scalar scales: the
// ONNX backend's (src/onnx/gemm.cpp, gemm_setup.h), so the rows divide.
constexpr int kChainLayers = 16;
constexpr int64_t kSeedWidth = 64;

// A size counts as an improvement only if it beats the best so far by this
// much; two failures in a row end the search.
constexpr double kImproveFactor = 1.03;
constexpr int kMaxStrikes = 2;

// One multiply, predicted from the previous rung's rate, must not take
// longer than this; a rung that measures slower ends the ladder outright.
// One multiply, not one prediction: bounding the sixteen together stops a
// slow unit at a quarter of the width it would reach alone, as the ONNX
// ladder found on a CPU provider.
constexpr double kMaxIterUs = 2.0e6;

// Per-size budget for the timed phase.
constexpr unsigned int kSizeBudgetUs = 2000000;

// Everything the chain holds, capped at a quarter of physical memory: a
// fixed ceiling would be a crash on a phone and a needless limit on a Mac
// Studio.
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
     "may decline it with live activations, which every layer of the chain "
     "has -- as the transformer-block rows do."},
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
     "device and every product quantized back, the pattern Core ML fuses into "
     "an integer multiply on Neural Engines that have one (A17 Pro, M4 and "
     "later).  Measured in ops; a reading no faster than the fp16 row is a "
     "Neural Engine doing this in 16-bit."},
};

// What the chain holds: the seed and the weights that widen it, then every
// layer's weights.
uint64_t operandBytes(CoremlWeight w, int64_t D)
{
  const int64_t sw = std::min(kSeedWidth, D);
  return coremlElemBytes(coremlActDtype(w), D * sw) + coremlWeightBytes(w, sw, D) +
         (uint64_t)kChainLayers * coremlWeightBytes(w, D, D);
}

} // namespace

int CoreMLPeak::runGemm(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_gemm", "Core ML matmul peak", "flops", Category::Unknown,
       "Matrix-multiply rate through Core ML on this compute unit, one weight "
       "format per row: sixteen distinct square multiplies chained in one "
       "prediction, each layer's product feeding the next, swept over layer "
       "widths and reported at its best.  The same model runs on the Neural "
       "Engine, the GPU and the CPU, and the compute plan proves which one "
       "actually did the multiplies: a row Core ML would have moved to another "
       "unit reports unsupported instead.",
       TestShape::Heterogeneous, "weight format"});

  const std::string sweep = "Peak over a doubling sweep of layer widths, sixteen layers chained "
                            "per prediction";
  const std::string seedNote =
      "  The chain starts from a " + std::to_string(kSeedWidth) +
      "-wide seed that takes a runtime value, and one more multiply, not counted, "
      "widens it into the first layer's activations -- so no compiler can fold the "
      "multiplies away, and the seed's pass and that multiply are inside the figure.";

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;
    const char *label = coremlWeightLabel(v.w);
    const bool isInt = (v.w == CoremlWeight::Int8Qdq);
    const int act = coremlActDtype(v.w);
    const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;

    logger::EmitOptions o;
    o.description = sweep + ".  " + v.note;
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

    double best = 0.0;
    int64_t bestDim = 0;
    std::string firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;
    int rungs = 0;
    std::string offDeviceNote, glueNote;
    double lastRate = 0.0, prevCreateUs = 0.0, prevPrevCreateUs = 0.0;
    int strikes = 0;

    for (int64_t D = kMinDim; D <= kMaxDim; D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      const uint64_t bytes = operandBytes(v.w, D);
      if (bytes > maxOperandBytes())
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers need %llu MB of operands, stopping\n",
                    dev.displayName.c_str(), label, (long long)D,
                    (unsigned long long)(bytes >> 20));
        break;
      }
      const double layerOps = 2.0 * (double)D * (double)D * (double)D;
      if (lastRate > 0.0 && layerOps / lastRate > kMaxIterUs)
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers would take ~%.1f s per multiply, "
                    "stopping\n",
                    dev.displayName.c_str(), label, (long long)D, layerOps / lastRate / 1.0e6);
        break;
      }
      // The compile cap, checked before paying for the build; the first size
      // always builds, since its time seeds the prediction.
      if (D > kMinDim && prevCreateUs > 0.0)
      {
        const double predictedUs = coremlPredictCreateUs(prevCreateUs, prevPrevCreateUs, strikes > 0);
        if (predictedUs > kCoremlMaxChainCreateUs)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers predicted create %.1f s (prev %.1f s%s) "
                      "> %.1f s, stopping\n",
                      dev.displayName.c_str(), label, (long long)D, predictedUs / 1.0e6,
                      prevCreateUs / 1.0e6, strikes > 0 ? ", after a size that did not gain" : "",
                      kCoremlMaxChainCreateUs / 1.0e6);
          break;
        }
      }

      std::string err;
      auto s = CoremlSession::create(dev, coremlMatMulChainModel(spec, D, kChainLayers, kSeedWidth, v.w),
                                     err);
      if (!s)
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide create failed: %s\n", dev.displayName.c_str(),
                    label, (long long)D, err.c_str());
        if (firstErr.empty())
          firstErr = err;
        // Larger sizes need strictly more of everything.
        break;
      }
      const double createUs = coremlCreateUs(*s);
      CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide x%d create %.2f s (compile %.2f, load %.2f, plan %.2f)\n",
                  dev.displayName.c_str(), label, (long long)D, kChainLayers, createUs / 1.0e6,
                  s->compileUs / 1.0e6, s->loadUs / 1.0e6, s->planUs / 1.0e6);

      if (!s->onDevice())
      {
        const std::string why = coremlOffDeviceReason(dev, *s);
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide %s\n", dev.displayName.c_str(), label,
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
            prevPrevCreateUs = prevCreateUs;
            prevCreateUs = createUs;
            continue;
          }
        }
        else
          offDeviceNote = "  Wider layers were declined by the " +
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
      const double ops = layerOps * kChainLayers;
      const double rate = ops * 1.0e6 / m.meanUs;
      lastRate = ops / m.meanUs;
      CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide x%d -> %.3f (%.1f us, %u iters)\n",
                  dev.displayName.c_str(), label, (long long)D, kChainLayers, rate, m.meanUs,
                  m.iters);

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
          CLPEAK_VLOG("coreml-gemm[%s/%s]: no further gain past %lld-wide layers\n",
                      dev.displayName.c_str(), label, (long long)bestDim);
          break;
        }
      }

      // Measured, not predicted: an extrapolation cannot see a cliff, and the
      // next size is eight times the work.
      if (m.probeUs / kChainLayers > kMaxIterUs)
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers measured %.1f s per multiply, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, m.probeUs / kChainLayers / 1.0e6);
        break;
      }

      // A size that compiled past the cap anyway is kept, and ends the
      // ladder.  The first may exceed it once: its time seeds the prediction.
      if (D != kMinDim && createUs > kCoremlMaxChainCreateUs)
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: create %.1f s > %.1f s at %lld-wide layers, stopping\n",
                    dev.displayName.c_str(), label, createUs / 1.0e6,
                    kCoremlMaxChainCreateUs / 1.0e6, (long long)D);
        break;
      }
      prevPrevCreateUs = prevCreateUs;
      prevCreateUs = createUs;
    }

    if (best > 0.0)
    {
      o.description = sweep + "; fastest at " + std::to_string(bestDim) + "-wide layers.  " + v.note +
                      seedNote + offDeviceNote + glueNote;
      test.emit(label, (float)best, o);
    }
    else
      test.skip(label, errStatus, firstErr.empty() ? "no size could be measured" : firstErr, o);
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
