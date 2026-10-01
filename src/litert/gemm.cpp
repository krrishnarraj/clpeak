#ifdef ENABLE_LITERT

// litert-gemm: matrix-multiply peak through LiteRT, on whichever of the NPU,
// the GPU and the CPU this device row names.
//
// The shape is the ONNX and Core ML backends' (src/onnx/gemm.cpp): sixteen
// distinct square FULLY_CONNECTED layers chained in one dispatch, each one's
// product the next one's input, swept over layer width.  The weights are
// model constants and the last product is reduced to one row, so nothing
// large crosses the host boundary per run, and the chain starts from a
// 64-wide seed that takes a runtime scalar and one more FULLY_CONNECTED, not
// counted, widens into the first layer's input -- so every multiply has a
// live operand and the scaling pass is a sliver of the work.  Widths double
// from 1024 until the rate stops improving, and the peak is reported with
// the width that produced it.
//
// One multiply per dispatch measured what surrounds the multiply as much as
// the multiply.  On QNN's HTP, through the ONNX backend, a single int8
// 8192-cube read 13 TOPS where the chain reads 35, and the Neural Engine
// gained 25% through Core ML -- the vendor runtimes this backend reaches on
// Android are where the difference lives.  On an M1 Pro, XNNPACK read 4-13%
// more in a back-to-back pair, inside its run-to-run spread, and the Metal
// accelerator up to 8% more in fp32 and the same elsewhere, where a single
// multiply already ran near the GPU's peak.
//
// One test, `litert_gemm`: the same chain in every format a LiteRT model
// ships in.  Each accelerator runs a format as it can and the
// row says which kernel that was: XNNPACK names its GEMM by the types it
// packed ("Fully Connected (NC, QD8, F32, QB4W)" is 4-bit weights against
// dynamically quantized int8 activations), the GPU accelerator names the
// shader ("convolution1x1(conv_wave_matrix)" is a simdgroup-matrix kernel).
//
// LiteRT's own answer is the guard.  The runtime keeps the CPU as a fallback
// for any operation an accelerator declines and says nothing about it in
// the result; LiteRtCompiledModelIsFullyAccelerated says whether that
// happened, and a row it happened to reports unsupported rather than a CPU
// number under the accelerator's name.

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
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
// One multiply, not one dispatch: bounding the sixteen together stops a slow
// accelerator at a quarter of the width it would reach alone, as the ONNX
// ladder found on a CPU provider.
constexpr double kMaxIterUs = 2.0e6;

// Per-size budget for the timed phase.
constexpr unsigned int kSizeBudgetUs = 2000000;

// The size whose whole time is the cost of asking: 64^3 is 0.5 MFLOP.
constexpr int64_t kFloorDim = 64;

// A rung has to do at least this much work, as a share of the submission
// cost, before it counts as having computed anything.
constexpr double kFoldWorkFloor = 0.15;

// Everything a rung holds at once, capped at a quarter of physical memory
// (a fixed ceiling would be a crash on a phone and a needless limit on a
// workstation): the model's own copy of the seed and every layer's weights
// in their stored types, the accelerator's packed copy of the weights, and a
// layer's input and output.
uint64_t maxRungBytes() { return clpeak::memoryBudget(3ull << 30); }

uint64_t rungBytes(const LitertPlan &p, int64_t D)
{
  const int64_t sw = std::min(kSeedWidth, D);
  const uint64_t w = litertWeightBytes(p, D, sw) + (uint64_t)kChainLayers * litertWeightBytes(p, D, D);
  return litertElemBytes(litertConstantType(p), D * sw) + 2 * w + 2 * litertElemBytes(p.act, D * D);
}

struct Variant
{
  LitertFormat f;
  const char *note;
};

const Variant kVariants[] = {
    {LitertFormat::Fp32,
     "Full 32-bit precision as a control; the GPU is asked for its fp32 policy, "
     "since it otherwise computes an fp32 graph in half."},
    {LitertFormat::Fp16,
     "16-bit storage and arithmetic: half-typed tensors and XNNPACK's fp16 GEMM "
     "on the CPU, the fp16 policy over half-stored weights on the GPU."},
    {LitertFormat::Fp16Acc32,
     "The GPU's fp16 policy with the matmul accumulated in fp32; the accuracy "
     "row says what that buys over plain fp16."},
    {LitertFormat::Bf16,
     "bfloat16 tensors, which the .tflite schema allows; whether any kernel "
     "takes them is the row."},
    {LitertFormat::Int8Qdq,
     "8-bit weights, activations and result -- TFLite's full-integer "
     "quantization, what headline TOPS figures are quoted for."},
    {LitertFormat::Int16x8,
     "16-bit activations over 8-bit weights, TFLite's higher-accuracy integer "
     "scheme; the CPU has only the reference kernel for it, an NPU may have a "
     "real one."},
    {LitertFormat::Int8Weight,
     "8-bit per-row weights against float activations (dynamic-range "
     "quantization): int8 arithmetic on XNNPACK, unpacked to half and "
     "multiplied in float on the GPU."},
    {LitertFormat::Int4Weight,
     "4-bit blockwise weights (32 per scale) against float activations, what "
     "an on-device language model ships as; XNNPACK runs it as int8 "
     "arithmetic, the GPU unpacks to half."},
    {LitertFormat::Fp8Weight,
     "8-bit float (E4M3) weights with a scale per row; which accelerator has a "
     "kernel for them is the row."},
};

// The kernel that did the multiplies, from a profiled run: the one most of
// them ran as.  The seed's 64-deep widening can be given another (the Metal
// accelerator runs it as conv_generic where the layers run conv_wave_matrix).
std::string matmulKernel(const std::vector<std::string> &ops)
{
  std::map<std::string, int> count;
  std::string best;
  for (const std::string &o : ops)
  {
    std::string l = o;
    std::transform(l.begin(), l.end(), l.begin(), ::tolower);
    if (l.find("fully") != std::string::npos || l.find("conv") != std::string::npos ||
        l.find("matmul") != std::string::npos || l.find("gemm") != std::string::npos)
      if (++count[o] > (best.empty() ? 0 : count[best]))
        best = o;
  }
  return best;
}

} // namespace

int LitertPeak::runGemm(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"litert_gemm", "LiteRT matmul peak", "flops", Category::Unknown,
       "Matrix-multiply rate through LiteRT on this accelerator, one model "
       "format per row: sixteen distinct square multiplies chained in one "
       "dispatch, swept over layer widths and reported at its best.  Each row "
       "names the kernel that ran; a format handed back to the CPU reports "
       "unsupported.",
       TestShape::Heterogeneous, "model format"});

  // The cost of asking, measured once: a 64^3 multiply whose arithmetic is
  // nothing, so its time is submission and the reduction.
  double floorUs = 0.0;
  {
    const LitertPlan fp = litertPlanFor(LitertFormat::Fp32, dev.accel);
    std::string err;
    auto s = LitertSession::create(rt, dev, litertMatMulModel(fp, kFloorDim, kFloorDim, kFloorDim),
                                   litertConfigFor(fp), err);
    if (s && s->onDevice() && litertBindScalar(*s, fp, err))
    {
      auto m = litertMeasure(*s, warmupCount, 200000, false, 0, 50);
      if (m.meanUs > 0.0)
        floorUs = m.meanUs;
    }
    CLPEAK_VLOG("litert-gemm[%s]: submission floor %.1f us\n", dev.displayName.c_str(), floorUs);
  }

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;
    const char *label = litertFormatLabel(v.f);
    const LitertPlan plan = litertPlanFor(v.f, dev.accel);

    logger::EmitOptions o;
    o.description = v.note;
    if (plan.integerOps)
      o.unit = "ops";

    if (!plan.applies)
    {
      test.skip(label, ResultStatus::Unsupported, plan.whyNot, o);
      continue;
    }
    // A kernel whose answer is wrong has no rate worth publishing.
    if (const std::string wrong = wrongAnswer(rt, dev, v.f); !wrong.empty())
    {
      test.skip(label, ResultStatus::Error, wrong, o);
      continue;
    }

    double best = 0.0;
    int64_t bestDim = 0;
    std::string firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;
    std::string kernel;
    int rungs = 0;
    double firstUs = 0.0, lastUs = 0.0;
    int64_t firstDim = 0, lastDim = 0;
    double lastRate = 0.0, prevCreateUs = 0.0, prevPrevCreateUs = 0.0, prevUs = 0.0;
    int64_t prevD = 0;
    int strikes = 0;

    for (int64_t D = kMinDim; D <= kMaxDim; D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      const uint64_t bytes = rungBytes(plan, D);
      if (bytes > maxRungBytes())
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide layers need %llu MB, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, (unsigned long long)(bytes >> 20));
        break;
      }
      const double layerOps = 2.0 * (double)D * (double)D * (double)D;
      if (lastRate > 0.0 && layerOps / lastRate > kMaxIterUs)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide layers would take ~%.1f s per multiply, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, layerOps / lastRate / 1.0e6);
        break;
      }
      // The compile cap, checked before paying for the build; the first size
      // always builds, since its time seeds the prediction.
      if (D > kMinDim && prevCreateUs > 0.0)
      {
        const double predictedUs = litertPredictCreateUs(prevCreateUs, prevPrevCreateUs, strikes > 0);
        if (predictedUs > kLitertMaxChainCreateUs)
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide layers predicted create %.1f s (prev %.1f s%s) > "
                      "%.1f s, stopping\n",
                      dev.displayName.c_str(), label, (long long)D, predictedUs / 1.0e6,
                      prevCreateUs / 1.0e6, strikes > 0 ? ", after a size that did not gain" : "",
                      kLitertMaxChainCreateUs / 1.0e6);
          break;
        }
      }

      // The first rung is also profiled once, in a session of its own: the
      // kernel name is the row's evidence of what ran, and profiling costs
      // enough on the GPU (every kernel waited on) that it never touches a
      // timed session.
      if (rungs == 0 && kernel.empty())
      {
        std::string err;
        auto ps = LitertSession::create(rt, dev, litertMatMulChainModel(plan, D, kChainLayers, kSeedWidth),
                                        litertConfigFor(plan, true), err);
        if (ps && ps->onDevice() && litertBindScalar(*ps, plan, err))
        {
          std::string perr;
          kernel = matmulKernel(ps->profileOps(perr));
          CLPEAK_VLOG("litert-gemm[%s/%s]: kernel '%s'\n", dev.displayName.c_str(), label, kernel.c_str());
        }
      }

      std::string err;
      auto s = LitertSession::create(rt, dev, litertMatMulChainModel(plan, D, kChainLayers, kSeedWidth),
                                     litertConfigFor(plan), err);
      if (!s)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide create failed: %s\n", dev.displayName.c_str(), label,
                    (long long)D, err.c_str());
        if (firstErr.empty())
          firstErr = err;
        break;   // larger sizes need strictly more of everything
      }
      const double createUs = s->createUs;
      CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide x%d create %.3f s\n", dev.displayName.c_str(), label,
                  (long long)D, kChainLayers, createUs / 1.0e6);
      if (!s->onDevice())
      {
        const std::string why = s->offDevice();
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s\n", dev.displayName.c_str(), label, (long long)D,
                    why.c_str());
        if (firstErr.empty())
          firstErr = why;
        break;
      }
      if (!litertBindScalar(*s, plan, err))
      {
        if (firstErr.empty())
        {
          firstErr = err;
          errStatus = ResultStatus::Error;
        }
        break;
      }

      auto m = litertMeasure(*s, warmupCount, kSizeBudgetUs, forceIters, specifiedIters);
      const std::string wrong =
          (m.meanUs > 0.0) ? litertNonFiniteReason(*s, plan.act, "at " + std::to_string(D) + "-wide layers")
                           : std::string();
      s.reset();   // the model's memory goes before the next size is built
      if (m.meanUs <= 0.0)
      {
        if (firstErr.empty())
        {
          firstErr = m.error;
          errStatus = m.status;
        }
        break;
      }
      // A wrong answer withholds the whole row, not just the rungs from here
      // up: a rung that came back finite is no alibi for the accelerator.
      if (!wrong.empty())
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %s\n", dev.displayName.c_str(), label, wrong.c_str());
        best = 0.0;
        firstErr = wrong;
        errStatus = ResultStatus::Error;
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

      const double ops = layerOps * kChainLayers;
      const double rate = ops * 1.0e6 / m.meanUs;
      lastRate = ops / m.meanUs;
      CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide x%d -> %.3f (%.1f us, %u iters)\n",
                  dev.displayName.c_str(), label, (long long)D, kChainLayers, rate, m.meanUs, m.iters);

      // Fold detector: work is time less the submission floor, and eight
      // times the arithmetic has to cost at least twice the time however
      // much the rate improves.  The runtime scalar makes folding
      // impossible for LiteRT's own runtime; a vendor compiler that hoisted
      // the scalar out of the float chain and folded the constant product
      // behind it would land here.
      const double work = m.meanUs - floorUs;
      const double prevWork = prevUs - floorUs;
      const bool computedNothing = floorUs > 0.0 && work < kFoldWorkFloor * floorUs;
      const bool workFlat = prevUs > 0.0 && D == prevD * 2 && prevWork > 0.0 && work < prevWork * 2.0;
      if (computedNothing || workFlat)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide took %.1f us against a %.1f us floor (previous %.1f "
                    "us) -- the multiplies were folded away\n",
                    dev.displayName.c_str(), label, (long long)D, m.meanUs, floorUs, prevUs);
        best = 0.0;
        firstErr = "the compiler folded the multiplies away: the timings do not scale with the "
                   "problem size, so they measure dispatch and a reduction rather than the "
                   "matrix multiplies";
        errStatus = ResultStatus::Error;
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
          CLPEAK_VLOG("litert-gemm[%s/%s]: no further gain past %lld-wide layers\n",
                      dev.displayName.c_str(), label, (long long)bestDim);
          break;
        }
      }

      // Measured, not predicted: an extrapolation cannot see a cliff, and the
      // next size is eight times the work.
      if (m.probeUs / kChainLayers > kMaxIterUs)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide layers measured %.1f s per multiply, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, m.probeUs / kChainLayers / 1.0e6);
        break;
      }

      // A size that compiled past the cap anyway is kept, and ends the
      // ladder.  The first may exceed it once: its time seeds the prediction.
      if (D != kMinDim && createUs > kLitertMaxChainCreateUs)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: create %.1f s > %.1f s at %lld-wide layers, stopping\n",
                    dev.displayName.c_str(), label, createUs / 1.0e6, kLitertMaxChainCreateUs / 1.0e6,
                    (long long)D);
        break;
      }
      prevPrevCreateUs = prevCreateUs;
      prevCreateUs = createUs;
    }

    // Whole-ladder backstop: real work grows with the cube of the size, so
    // anything near flat computed nothing.
    double expectedGrowth = 1.0;
    for (int64_t d = firstDim; d > 0 && d < lastDim; d *= 2)
      expectedGrowth *= 8.0;
    if (best > 0.0 && firstUs > 0.0 && lastDim > firstDim && lastUs < firstUs * expectedGrowth / 64.0)
    {
      best = 0.0;
      firstErr = "the compiler folded the multiplies away: the timings do not scale with the "
                 "problem size and mean nothing";
      errStatus = ResultStatus::Error;
    }

    if (best > 0.0)
    {
      o.description = std::string(v.note) + "  Sixteen layers chained per dispatch from a " +
                      std::to_string(kSeedWidth) + "-wide seed, whose widening multiply is not "
                      "counted; fastest at " + std::to_string(bestDim) + "-wide layers";
      if (!kernel.empty())
      {
        o.description += ", as `" + kernel + "`";
        // A row that says "ops" for float arithmetic needs the words; the
        // kernel's name, not the passes around it, says which it was.
        if (plan.integerOps && litertKernelIsFloatForInteger(kernel, dev.accel))
          o.description += litertFloatKernelNote();
      }
      o.description += ".";
      test.emit(label, (float)best, o);
    }
    else
      test.skip(label, errStatus, firstErr.empty() ? "no size could be measured" : firstErr, o);
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
