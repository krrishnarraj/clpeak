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
// Each width is timed with the weights stored both ways, [in, out] and
// [out, in], until the readings settle which one this unit runs faster
// (CoremlLayoutRace), and the row takes the faster and names it.  On an M1
// Pro every row's race settles at the first width or the second but the
// GPU's int4_weight, whose layouts stay 7-18% apart all the way up, so the
// race costs about one more model per row; and on the Neural Engine, which
// runs the two at one rate, the climb goes on in the [out, in] layout its
// compiler takes as it is: 2048-wide layers compile in about a second there
// instead of fifteen.
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
#include <cstdio>
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

// On a GPU one prediction is bounded too, at --max-time-gpu (gpuRunCapUs),
// and each layout's first rung, which has no rate to be predicted from, is
// predicted from a chain this wide timed once beforehand -- the ONNX ladder's
// rule, for its reason (src/onnx/gemm.cpp).  The scout is no rung: the row
// neither publishes it nor counts it toward the plateau.
constexpr int64_t kScoutDim = 512;

// Per-size budget for the timed phase.
constexpr unsigned int kSizeBudgetUs = 2000000;

// Everything a rung holds at once (rungBytes), capped at a quarter of
// physical memory -- a fixed ceiling would be a crash on a phone and a
// needless limit on a Mac Studio -- and at 8 GB, which still admits an
// 8192-wide fp16 chain, for a GPU that runs one inside --max-time-gpu.
uint64_t maxRungBytes() { return clpeak::memoryBudget(8ull << 30); }

struct Variant
{
  CoremlWeight w;
  const char *note;
};

const Variant kVariants[] = {
    {CoremlWeight::Fp16, "16-bit weights and arithmetic."},
    {CoremlWeight::Fp32, "Full 32-bit precision."},
    {CoremlWeight::Bf16, "bfloat16 weights and arithmetic."},
    {CoremlWeight::Int8Channel,
     "8-bit weights, one scale per output column, widened to 16 bits for the multiply."},
    {CoremlWeight::Int4Block, "4-bit weights in blocks of 32, widened to 16 bits for the multiply."},
    {CoremlWeight::Int4Lut, "4-bit indices into a 16-entry table, widened to 16 bits for the multiply."},
    {CoremlWeight::Fp8Block,
     "8-bit float (E4M3) weights in blocks of 32, widened to 16 bits for the multiply."},
    {CoremlWeight::Int8Qdq,
     "Full-integer int8: 8-bit activations and weights; only A17 Pro / M4 and later Neural Engines "
     "multiply them as integers."},
};

// A row's word on the layout of its weights (CoremlLayoutRace): the order
// the peak was measured in, and why there was no race when there was none.
std::string layoutNote(bool transposed, int otherRungs, bool otherWrong)
{
  std::string s = ", weights stored " + std::string(coremlLayoutName(transposed));
  if (otherWrong)
    s += "; " + std::string(coremlLayoutName(!transposed)) + " answered wrong";
  else if (otherRungs == 0)
    s += ", the only order that ran";
  return s;
}

// The chain's constants -- the seed and the weights that widen it, then
// every layer's weights -- held as coremlHeldBytes says, and every layer's
// output.
uint64_t rungBytes(CoremlWeight w, int64_t D)
{
  const int64_t sw = std::min(kSeedWidth, D);
  const int act = coremlActDtype(w);
  const uint64_t constants = coremlElemBytes(act, D * sw) + coremlWeightBytes(w, sw, D) +
                             (uint64_t)kChainLayers * coremlWeightBytes(w, D, D);
  return coremlHeldBytes(constants, (uint64_t)(kChainLayers + 1) * coremlElemBytes(act, D * D));
}

} // namespace

int CoreMLPeak::runGemm(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  // The longest one prediction may be predicted to take: on a GPU,
  // --max-time-gpu (gpuRunCapUs); elsewhere 0, unbounded.
  const double runCapUs = gpuRunCapUs(dev.deviceType, cfg);
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_gemm", "Core ML matmul peak", "flops", Category::Unknown,
       "Matrix-multiply rate on this compute unit, one weight format per row: "
       "sixteen chained square matmuls in one prediction, swept over layer width "
       "and reported at its best in the faster of the two weight layouts.  A row "
       "Core ML would move to another unit reports unsupported.",
       TestShape::Heterogeneous, "weight format"});

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;
    const char *label = coremlWeightLabel(v.w);
    const bool isInt = (v.w == CoremlWeight::Int8Qdq);
    const int act = coremlActDtype(v.w);
    const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;

    logger::EmitOptions o;
    o.description = v.note;
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

    // A layout whose answer is wrong leaves the race (wrongAnswer,
    // numeric_error.cpp), and a format with no layout that answers right has
    // no rate worth publishing.
    std::string wrongWhy[2];
    for (int t = 0; t < 2; t++)
      wrongWhy[t] = wrongAnswer(dev, v.w, t);
    if (!wrongWhy[0].empty() && !wrongWhy[1].empty())
    {
      const int read = answerCheck(dev, v.w).read;
      CLPEAK_VLOG("coreml-gemm[%s/%s]: %s\n", dev.displayName.c_str(), label, wrongWhy[read == 1].c_str());
      test.skip(label, ResultStatus::Error, wrongWhy[read == 1], o);
      continue;
    }

    double best = 0.0;
    int64_t bestDim = 0;
    bool bestTransposed = false;
    std::string firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;
    int strikes = 0;
    bool wrong = false;

    // The two weight layouts climb together, one race per row
    // (CoremlLayoutRace): every width times each layout the race still
    // runs, each with its own caps and its own compile history, and the
    // width's rate is the faster one's.
    CoremlLayoutRace race;
    for (int t = 0; t < 2; t++)
      if (!wrongWhy[t].empty())
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %s\n", dev.displayName.c_str(), label, wrongWhy[t].c_str());
        race.drop(t);
      }
    struct Lane
    {
      int rungs = 0;
      double lastRate = 0.0, prevCreateUs = 0.0, prevPrevCreateUs = 0.0;
      std::string offDeviceNote, glueNote;
    } lanes[2];

    // On a GPU each layout's first rung is predicted from a kScoutDim chain;
    // a scout that cannot be built, is sent to another unit or cannot run
    // leaves it unpredicted.
    for (int t = 0; t < 2 && runCapUs > 0.0; t++)
    {
      if (!race.runs(t) || clpeak::cancelRequested())
        continue;
      std::string err;
      auto s = CoremlSession::create(
          dev, coremlMatMulChainModel(spec, kScoutDim, kChainLayers, kSeedWidth, v.w, t), err);
      double us = -1.0;
      if (s && !s->onDevice())
        err = coremlOffDeviceReason(dev, *s);
      else if (s && coremlBindScalar(*s, "s", ioDtype, err) && s->timeRuns(1, err) > 0.0)   // compile + warmup
        us = s->timeRuns(1, err);
      s.reset();
      if (us > 0.0)
      {
        lanes[t].lastRate = 2.0 * (double)kScoutDim * (double)kScoutDim * (double)kScoutDim * kChainLayers / us;
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide x%d %s scout %.1f ms\n", dev.displayName.c_str(), label,
                    (long long)kScoutDim, kChainLayers, coremlLayoutName(t), us / 1.0e3);
      }
      else
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide %s scout failed, first rung unpredicted: %s\n",
                    dev.displayName.c_str(), label, (long long)kScoutDim, coremlLayoutName(t),
                    err.empty() ? "no reason given" : err.c_str());
    }

    for (int64_t D = kMinDim; D <= kMaxDim && !race.done(); D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      const uint64_t bytes = rungBytes(v.w, D), budget = maxRungBytes();
      if (bytes > budget)
      {
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers need %llu MB of a %llu MB budget, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, (unsigned long long)(bytes >> 20),
                    (unsigned long long)(budget >> 20));
        break;
      }
      const double layerOps = 2.0 * (double)D * (double)D * (double)D;
      double rate[2] = {0.0, 0.0}, createUs[2] = {0.0, 0.0};

      for (int t = 0; t < 2 && !wrong; t++)
      {
        if (!race.runs(t))
          continue;
        Lane &ln = lanes[t];
        const char *layout = coremlLayoutName(t);
        const double runUs = ln.lastRate > 0.0 ? layerOps * kChainLayers / ln.lastRate : 0.0;
        const bool runTooLong = runCapUs > 0.0 && runUs > runCapUs;
        if (runTooLong || runUs / kChainLayers > kMaxIterUs)
        {
          if (runTooLong)
            CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers %s would keep the GPU busy ~%.2f s a "
                        "prediction, past --max-time-gpu (%.2f s), stopping\n",
                        dev.displayName.c_str(), label, (long long)D, layout, runUs / 1.0e6,
                        runCapUs / 1.0e6);
          else
            CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers %s would take ~%.1f s per multiply, "
                        "stopping\n",
                        dev.displayName.c_str(), label, (long long)D, layout,
                        runUs / kChainLayers / 1.0e6);
          // Before any rung the prediction is the scout's, and it is all the
          // row has to say.
          if (ln.rungs == 0 && firstErr.empty())
          {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.1f", runUs / 1.0e6);
            firstErr = "at the rate a " + std::to_string(kScoutDim) + "-wide chain ran, a " +
                       std::to_string(D) + "-wide one would take about " + buf + " s a prediction, " +
                       (runTooLong ? "longer than --max-time-gpu lets one run hold a GPU -- a driver "
                                     "may reset a GPU held longer -- "
                                   : "longer than one multiply may take, ") +
                       "so no size was measured";
            errStatus = ResultStatus::Error;
          }
          race.drop(t);
          continue;
        }
        // The compile cap, checked before paying for the build; the first
        // size always builds, since its time seeds the prediction.
        if (D > kMinDim && ln.prevCreateUs > 0.0)
        {
          const double predictedUs =
              coremlPredictCreateUs(ln.prevCreateUs, ln.prevPrevCreateUs, strikes > 0);
          if (predictedUs > kCoremlMaxChainCreateUs)
          {
            CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers %s predicted create %.1f s (prev "
                        "%.1f s%s) > %.1f s, stopping\n",
                        dev.displayName.c_str(), label, (long long)D, layout, predictedUs / 1.0e6,
                        ln.prevCreateUs / 1.0e6, strikes > 0 ? ", after a size that did not gain" : "",
                        kCoremlMaxChainCreateUs / 1.0e6);
            race.drop(t);
            continue;
          }
        }

        std::string err;
        auto s = CoremlSession::create(
            dev, coremlMatMulChainModel(spec, D, kChainLayers, kSeedWidth, v.w, t), err);
        if (!s)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide %s create failed: %s\n",
                      dev.displayName.c_str(), label, (long long)D, layout, err.c_str());
          if (firstErr.empty())
            firstErr = err;
          // Larger sizes need strictly more of everything.
          race.drop(t);
          continue;
        }
        const double cu = coremlCreateUs(*s);
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide x%d %s create %.2f s (compile %.2f, load %.2f, "
                    "plan %.2f)\n",
                    dev.displayName.c_str(), label, (long long)D, kChainLayers, layout, cu / 1.0e6,
                    s->compileUs / 1.0e6, s->loadUs / 1.0e6, s->planUs / 1.0e6);

        if (!s->onDevice())
        {
          const std::string why = coremlOffDeviceReason(dev, *s);
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide %s %s\n", dev.displayName.c_str(), label,
                      (long long)D, layout, why.c_str());
          if (ln.rungs == 0)
          {
            if (firstErr.empty())
              firstErr = why;
            // Too small for the planner is not too big for the unit; the
            // ladder climbs on.  A shape the unit cannot run at all will not
            // become runnable by growing.
            if (s->offDeviceCapable())
            {
              s.reset();
              ln.prevPrevCreateUs = ln.prevCreateUs;
              ln.prevCreateUs = cu;
              continue;
            }
          }
          else
            ln.offDeviceNote = "; the " + std::string(coremlKindName(dev.kind)) + " declined wider layers";
          race.drop(t);
          continue;
        }

        if (!coremlBindScalar(*s, "s", ioDtype, err))
        {
          if (firstErr.empty())
          {
            firstErr = err;
            errStatus = ResultStatus::Error;
          }
          race.drop(t);
          continue;
        }

        auto m = coremlMeasure(*s, warmupCount, kSizeBudgetUs, forceIters, specifiedIters);
        std::string bad;
        if (m.meanUs > 0.0)
        {
          ln.glueNote = coremlGlueNote(*s);
          bad = coremlNonFiniteReason(*s, "out", ioDtype,
                                      "at " + std::to_string(D) + "-wide layers, weights stored " + layout);
        }
        s.reset();   // the temp files go before the next model is written
        if (m.meanUs <= 0.0)
        {
          // Logged whatever the row says: above a measured width it
          // publishes the widths below, and this failure would leave no trace.
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide %s run failed: %s\n", dev.displayName.c_str(), label,
                      (long long)D, layout, m.error.c_str());
          if (firstErr.empty())
          {
            firstErr = m.error;
            errStatus = m.status;
          }
          race.drop(t);
          continue;
        }
        // A wrong answer withholds the whole row, not just the rungs from
        // here up: a rung that came back finite, or the other layout, is no
        // alibi for the unit.
        if (!bad.empty())
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %s\n", dev.displayName.c_str(), label, bad.c_str());
          firstErr = bad;
          errStatus = ResultStatus::Error;
          wrong = true;
          break;
        }

        ln.rungs++;
        rate[t] = layerOps * kChainLayers * 1.0e6 / m.meanUs;
        createUs[t] = cu;
        ln.lastRate = layerOps * kChainLayers / m.meanUs;
        CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide x%d %s -> %.3f (%.1f us, %u iters)\n",
                    dev.displayName.c_str(), label, (long long)D, kChainLayers, layout, rate[t],
                    m.meanUs, m.iters);

        // Measured, not predicted: an extrapolation cannot see a cliff, and
        // the next size is eight times the work.
        if (m.probeUs / kChainLayers > kMaxIterUs)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers %s measured %.1f s per multiply, "
                      "stopping\n",
                      dev.displayName.c_str(), label, (long long)D, layout,
                      m.probeUs / kChainLayers / 1.0e6);
          race.drop(t);
        }
        // The same for a GPU's prediction (gpuRunCapUs): this one held the
        // device longer than --max-time-gpu and survived it, and the next
        // would hold it eight times as long.
        else if (runCapUs > 0.0 && m.probeUs > runCapUs)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers %s measured %.2f s a prediction, past "
                      "--max-time-gpu (%.2f s), stopping\n",
                      dev.displayName.c_str(), label, (long long)D, layout, m.probeUs / 1.0e6,
                      runCapUs / 1.0e6);
          race.drop(t);
        }
        // A size that compiled past the cap anyway is kept, and ends this
        // layout's climb.  The first may exceed it once: its time seeds the
        // prediction.
        else if (D != kMinDim && cu > kCoremlMaxChainCreateUs)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: %lld-wide layers %s create %.1f s > %.1f s, stopping\n",
                      dev.displayName.c_str(), label, (long long)D, layout, cu / 1.0e6,
                      kCoremlMaxChainCreateUs / 1.0e6);
          race.drop(t);
        }
        ln.prevPrevCreateUs = ln.prevCreateUs;
        ln.prevCreateUs = cu;
      }
      if (wrong)
      {
        best = 0.0;
        break;
      }

      if (rate[0] > 0.0 && rate[1] > 0.0)
      {
        race.settle(rate, createUs);
        if (!race.runs(0) || !race.runs(1))
          CLPEAK_VLOG("coreml-gemm[%s/%s]: layouts settled at %lld-wide layers: %s goes on\n",
                      dev.displayName.c_str(), label, (long long)D,
                      race.runs(1) ? coremlLayoutName(true) : coremlLayoutName(false));
      }
      const bool t = rate[1] > rate[0];
      if (rate[t] <= 0.0)
        continue;   // neither layout measured this width: the planner's size decision

      if (rate[t] > best * kImproveFactor)
      {
        strikes = 0;
        best = rate[t];
        bestDim = D;
        bestTransposed = t;
      }
      else
      {
        if (rate[t] > best)
        {
          best = rate[t];
          bestDim = D;
          bestTransposed = t;
        }
        if (++strikes >= kMaxStrikes)
        {
          CLPEAK_VLOG("coreml-gemm[%s/%s]: no further gain past %lld-wide layers\n",
                      dev.displayName.c_str(), label, (long long)bestDim);
          break;
        }
      }
    }

    if (best > 0.0)
    {
      const Lane &ln = lanes[bestTransposed];
      o.description = std::string(v.note) + "  Fastest at " + std::to_string(bestDim) + "-wide layers" +
                      layoutNote(bestTransposed, lanes[!bestTransposed].rungs,
                                 !wrongWhy[!bestTransposed].empty()) +
                      ln.offDeviceNote + ln.glueNote + ".";
      test.emit(label, (float)best, o);
    }
    else
      test.skip(label, errStatus, firstErr.empty() ? "no size could be measured" : firstErr, o);
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
