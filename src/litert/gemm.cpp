#ifdef ENABLE_LITERT

// litert-gemm: matrix-multiply peak through LiteRT, on whichever of the NPU,
// the GPU and the CPU this device row names.
//
// The shape is the ONNX and Core ML backends' (src/onnx/gemm.cpp): both
// operands are model constants and the result is reduced to one row, so
// nothing large crosses the host boundary per run; a runtime scalar scales
// the activations so the graph is never a constant expression.  Sizes
// double from 1024 until the rate stops improving, and the peak is reported
// with the size that produced it.
//
// One test, `litert_gemm`: the same FULLY_CONNECTED in every format a
// LiteRT model ships in.  Each accelerator runs a format as it can and the
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
// cost, before it counts as having computed anything.
constexpr double kFoldWorkFloor = 0.15;

// Everything a rung holds at once, capped at a quarter of physical memory
// (a fixed ceiling would be a crash on a phone and a needless limit on a
// workstation): the model's own copy of A and W in their stored types, the
// scaled activations, the result, and the accelerator's packed copy of the
// weights.
uint64_t maxRungBytes() { return clpeak::memoryBudget(3ull << 30); }

uint64_t rungBytes(const LitertPlan &p, int64_t D)
{
  const uint64_t a = litertElemBytes(litertConstantType(p), D * D);
  const uint64_t w = litertWeightBytes(p, D, D);
  return 2 * a + 2 * w + a;
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

// The kernel that did the multiply, from a profiled run.
std::string matmulKernel(const std::vector<std::string> &ops)
{
  for (const std::string &o : ops)
  {
    std::string l = o;
    std::transform(l.begin(), l.end(), l.begin(), ::tolower);
    if (l.find("fully") != std::string::npos || l.find("conv") != std::string::npos ||
        l.find("matmul") != std::string::npos || l.find("gemm") != std::string::npos)
      return o;
  }
  return std::string();
}

} // namespace

int LitertPeak::runGemm(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"litert_gemm", "LiteRT matmul peak", "flops", Category::Unknown,
       "Matrix-multiply rate through LiteRT on this accelerator, one model "
       "format per row, swept over square sizes and reported at its best.  Each "
       "row names the kernel that ran; a format handed back to the CPU reports "
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
    double lastRate = 0.0, prevCreateUs = 0.0, prevUs = 0.0;
    int64_t prevD = 0;
    int strikes = 0;

    for (int64_t D = kMinDim; D <= kMaxDim; D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      const uint64_t bytes = rungBytes(plan, D);
      if (bytes > maxRungBytes())
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 needs %llu MB, stopping\n", dev.displayName.c_str(),
                    label, (long long)D, (unsigned long long)(bytes >> 20));
        break;
      }
      if (lastRate > 0.0)
      {
        const double predictedUs = 2.0 * (double)D * (double)D * (double)D / lastRate;
        if (predictedUs > kMaxIterUs)
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 would take ~%.1f s per iteration, stopping\n",
                      dev.displayName.c_str(), label, (long long)D, predictedUs / 1.0e6);
          break;
        }
      }
      if (D > kMinDim && prevCreateUs > 0.0 && prevCreateUs * 4.0 > kLitertMaxCreateUs)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 predicted create %.1f s > %.1f s, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, prevCreateUs * 4.0 / 1.0e6,
                    kLitertMaxCreateUs / 1.0e6);
        break;
      }

      // The first rung is also profiled once, in a session of its own: the
      // kernel name is the row's evidence of what ran, and profiling costs
      // enough on the GPU (every kernel waited on) that it never touches a
      // timed session.
      if (rungs == 0 && kernel.empty())
      {
        std::string err;
        auto ps = LitertSession::create(rt, dev, litertMatMulModel(plan, D, D, D),
                                        litertConfigFor(plan, true), err);
        if (ps && ps->onDevice() && litertBindScalar(*ps, plan, err))
        {
          std::string perr;
          kernel = matmulKernel(ps->profileOps(perr));
          CLPEAK_VLOG("litert-gemm[%s/%s]: kernel '%s'\n", dev.displayName.c_str(), label, kernel.c_str());
        }
      }

      std::string err;
      auto s = LitertSession::create(rt, dev, litertMatMulModel(plan, D, D, D), litertConfigFor(plan), err);
      if (!s)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 create failed: %s\n", dev.displayName.c_str(), label,
                    (long long)D, err.c_str());
        if (firstErr.empty())
          firstErr = err;
        break;   // larger sizes need strictly more of everything
      }
      const double createUs = s->createUs;
      CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 create %.3f s\n", dev.displayName.c_str(), label,
                  (long long)D, createUs / 1.0e6);
      if (!s->onDevice())
      {
        const std::string why = s->offDevice();
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 %s\n", dev.displayName.c_str(), label, (long long)D,
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
      CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 -> %.3f (%.1f us, %u iters)\n", dev.displayName.c_str(),
                  label, (long long)D, rate, m.meanUs, m.iters);

      // Fold detector: work is time less the submission floor, and eight
      // times the arithmetic has to cost at least twice the time however
      // much the rate improves.  The runtime scalar makes folding
      // impossible for LiteRT's own runtime; a vendor compiler that
      // rewrote (A*s)*W as s*(A*W) and folded A*W would land here.
      const double work = m.meanUs - floorUs;
      const double prevWork = prevUs - floorUs;
      const bool computedNothing = floorUs > 0.0 && work < kFoldWorkFloor * floorUs;
      const bool workFlat = prevUs > 0.0 && D == prevD * 2 && prevWork > 0.0 && work < prevWork * 2.0;
      if (computedNothing || workFlat)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 took %.1f us against a %.1f us floor (previous %.1f "
                    "us) -- the multiply was folded away\n",
                    dev.displayName.c_str(), label, (long long)D, m.meanUs, floorUs, prevUs);
        best = 0.0;
        firstErr = "the compiler folded the multiply away: the timings do not scale with the "
                   "problem size, so they measure dispatch and a reduction rather than the "
                   "matrix multiply";
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
          CLPEAK_VLOG("litert-gemm[%s/%s]: no further gain past %lld^3\n", dev.displayName.c_str(),
                      label, (long long)bestDim);
          break;
        }
      }

      if (m.probeUs > kMaxIterUs)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld^3 measured %.1f s per iteration, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, m.probeUs / 1.0e6);
        break;
      }

      // Compilation gates: a cliff between rungs, or an absolute ceiling
      // (the first rung may exceed it once; its time seeds the growth gate).
      if (prevCreateUs > 0.0 && createUs > kLitertCreateGrowthFloor &&
          createUs > prevCreateUs * kLitertCreateGrowthFactor)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: create grew %.1fx at %lld^3, stopping\n",
                    dev.displayName.c_str(), label, createUs / prevCreateUs, (long long)D);
        break;
      }
      if (D != kMinDim && createUs > kLitertMaxCreateUs)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: create %.1f s > %.1f s at %lld^3, stopping\n",
                    dev.displayName.c_str(), label, createUs / 1.0e6, kLitertMaxCreateUs / 1.0e6,
                    (long long)D);
        break;
      }
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
      firstErr = "the compiler folded the multiply away: the timings do not scale with the "
                 "problem size and mean nothing";
      errStatus = ResultStatus::Error;
    }

    if (best > 0.0)
    {
      o.description = std::string(v.note) + "  Fastest at " + std::to_string(bestDim) + " cubed";
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
