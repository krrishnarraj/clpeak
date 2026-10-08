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
// A weight-only row whose kernel multiplies int8 that way counts ops, not
// flops (litertRateUnit).
//
// The int8 row races two spellings of its layers (clpeak::MultiFormRace):
// FULLY_CONNECTED, and a 1x1 CONV_2D over a grid of as many positions as
// the layer is wide -- the same arithmetic in the operator NPU compilers
// were first built for, and on some of them the faster int8 path.  Through
// the ONNX backend (src/onnx/gemm_setup.cpp, kIntVariants) QNN's HTP ran
// the convolutions at 38 TOPS against 35, ONNX Runtime 1.24.4's built-in
// QNN at 36 against 5, and TensorRT at 142 against 135, while fp16 lost or
// tied on every provider tried -- so only int8 races.  On a GPU the int8
// rows (int8_qdq, and int8_weight, which the kernels were written for) also
// race the accelerator's 8-bit FULLY_CONNECTED and convolution kernels,
// which it uses only when allowed: allowed, a Pixel 7a's Mali ran the
// int8_qdq chain at 1.24 TOPS against 702 GOPS, as int8 multiplies
// (`convolution_int8(conv_generic)`) rather than a float kernel between
// quantize and dequantize passes, and the Metal accelerator has none.  So
// int8_qdq is four forms there.  The row reports the fastest and names it.
//
// LiteRT's own answer is the guard.  The runtime keeps the CPU as a fallback
// for any operation an accelerator declines and says nothing about it in
// the result; LiteRtCompiledModelIsFullyAccelerated says whether that
// happened, and a row it happened to reports unsupported rather than a CPU
// number under the accelerator's name.

#include <common/form_race.h>
#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// On a GPU one run is bounded too, at --max-time-gpu (gpuRunCapUs), and each
// operator's first rung, which has no rate to be predicted from, is predicted
// from a chain this wide timed once beforehand -- the ONNX ladder's rule, for
// its reason (src/onnx/gemm.cpp).  The scout is no rung: the row neither
// publishes it nor counts it toward the plateau.
constexpr int64_t kScoutDim = 512;

// Per-size budget for the timed phase.
constexpr unsigned int kSizeBudgetUs = 2000000;

// The size whose whole time is the cost of asking: 64^3 is 0.5 MFLOP.
constexpr int64_t kFloorDim = 64;

// A rung has to do at least this much work, as a share of the submission
// cost, before it counts as having computed anything.
constexpr double kFoldWorkFloor = 0.15;

// Everything a rung holds at once (litertHeldBytes: the seed and every
// layer's weights, and every layer's output), capped at a quarter of
// physical memory (a fixed ceiling would be a crash on a phone and a
// needless limit on a workstation) and at 8 GB, which still admits the
// 8192-wide integer rungs XNNPACK and the Metal accelerator peak at.
uint64_t maxRungBytes() { return clpeak::memoryBudget(8ull << 30); }

uint64_t rungBytes(const LitertPlan &p, int64_t D)
{
  const int64_t sw = std::min(kSeedWidth, D);
  const uint64_t constants = litertElemBytes(litertConstantType(p), D * sw) + litertWeightBytes(p, D, sw) +
                             (uint64_t)kChainLayers * litertWeightBytes(p, D, D);
  return litertHeldBytes(constants, (uint64_t)(kChainLayers + 1) * litertElemBytes(p.act, D * D));
}

struct Variant
{
  LitertFormat f;
  const char *note;
};

const Variant kVariants[] = {
    {LitertFormat::Fp32, "Full 32-bit precision."},
    {LitertFormat::Fp16, "16-bit weights and arithmetic."},
    {LitertFormat::Fp16Acc32, "The GPU's fp16 policy with fp32 accumulation."},
    {LitertFormat::Bf16, "bfloat16 weights and arithmetic."},
    {LitertFormat::Int8Qdq, "Full-integer int8: 8-bit activations and weights."},
    {LitertFormat::Int16x8, "Full-integer int16x8: 16-bit activations over 8-bit weights."},
    {LitertFormat::Int8Weight, "8-bit weights, one scale per output column, against float activations."},
    {LitertFormat::Int4Weight, "4-bit weights in blocks of 32 against float activations."},
    {LitertFormat::Fp8Weight,
     "8-bit float (E4M3) weights, one scale per output column, against float activations."},
};

// The forms a format's chain is raced in on this accelerator (LitertForm):
// int8's layers as FULLY_CONNECTED and as 1x1 CONV_2Ds over a grid of as
// many positions as the layer is wide -- the same multiply-accumulates in
// the operator NPU compilers were first built for -- and, where the GPU has
// 8-bit kernels for the format, each with them disallowed and allowed.  The
// first listed carries on through a tie.
std::vector<LitertForm> formsFor(LitertFormat f, const LitertPlan &plan)
{
  const bool ops = f == LitertFormat::Int8Qdq;
  std::vector<LitertForm> forms;
  for (int conv = 0; conv <= (ops ? 1 : 0); conv++)
    for (int k = 0; k <= (plan.gpuInt8KernelChoice ? 1 : 0); k++)
      forms.push_back(LitertForm{conv == 1, k == 1, false});
  return forms;
}

// One form in a row's words.
std::string formName(const LitertForm &form, bool kernelsRaced)
{
  std::string s = form.conv1x1 ? "1x1 convolutions" : "FULLY_CONNECTED operators";
  if (kernelsRaced)
    s += form.gpuInt8Kernels ? " with the GPU's 8-bit kernels" : " without the GPU's 8-bit kernels";
  return s;
}

// One form's climb up the ladder: its own caps, compile history, fold
// check, kernel and peak.
struct Lane
{
  std::string kernel;
  int rungs = 0, strikes = 0;
  double best = 0.0;
  int64_t bestDim = 0;
  double lastRate = 0.0, prevCreateUs = 0.0, prevPrevCreateUs = 0.0, prevUs = 0.0;
  int64_t prevD = 0;
  double firstUs = 0.0, lastUs = 0.0;
  int64_t firstDim = 0, lastDim = 0;
  std::string err;   // why it stopped, the first reason
  ResultStatus errStatus = ResultStatus::Unsupported;

  void fail(const std::string &why, ResultStatus status = ResultStatus::Unsupported)
  {
    if (err.empty())
    {
      err = why;
      errStatus = status;
    }
  }
  // Caught computing nothing: whatever this form measured is void.
  void folded(const std::string &why)
  {
    best = 0.0;
    err = why;
    errStatus = ResultStatus::Error;
  }
};

std::string foldReason(const std::string &form, const char *tail)
{
  return (form.empty() ? std::string() : "written as " + form + ", ") +
         "the compiler folded the multiplies away: " + tail;
}

// A raced row's winner (clpeak::MultiFormRace), in the words that follow
// "Fastest at N-wide layers": how its layers were written where that was
// raced, and whether the GPU's 8-bit kernels were allowed where that was.
std::string winnerWords(const std::vector<LitertForm> &forms, bool kernelsRaced, size_t win)
{
  std::string s;
  if (forms.front().conv1x1 != forms.back().conv1x1)
    s += forms[win].conv1x1 ? " as 1x1 convolutions" : " as FULLY_CONNECTED";
  if (kernelsRaced)
    s += forms[win].gpuInt8Kernels ? " with the GPU's 8-bit kernels" : " without the GPU's 8-bit kernels";
  return s;
}

} // namespace

int LitertPeak::runGemm(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg)
{
  // The longest one run may be predicted to take: on a GPU, --max-time-gpu
  // (gpuRunCapUs); elsewhere 0, unbounded.
  const double runCapUs = gpuRunCapUs(dev.deviceType, cfg);

  auto test = currentDeviceScope->beginTest(
      {"litert_gemm", "LiteRT matmul peak", "flops", Category::Unknown,
       "Matrix-multiply rate on this accelerator, one model format per row: "
       "sixteen chained square matmuls in one dispatch, swept over layer width "
       "and reported at its best.  Each row names the kernel that ran, and a "
       "format LiteRT hands back to the CPU reports unsupported.",
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
    // The forms this row races (formsFor), each in the inputs and outputs
    // it answers right with (resolveIo): one form for most formats.
    std::vector<LitertForm> forms = formsFor(v.f, plan);
    const size_t nf = forms.size();
    const bool raced = nf > 1;
    for (LitertForm &form : forms)
      form = resolveIo(rt, dev, v.f, form);
    clpeak::MultiFormRace race(nf);

    // A form whose answer is wrong has no rate worth publishing, and leaves
    // the race rather than withholding the row: a broken convolution kernel
    // cannot take a FULLY_CONNECTED row with it.
    std::vector<std::string> wrongForm(nf);
    size_t wrongCount = 0;
    for (size_t f = 0; f < nf; f++)
    {
      wrongForm[f] = wrongAnswer(rt, dev, v.f, forms[f]);
      if (!wrongForm[f].empty())
      {
        race.drop(f);
        wrongCount++;
      }
    }
    if (wrongCount == nf)
    {
      test.skip(label, ResultStatus::Error, wrongForm[0], o);
      continue;
    }

    std::vector<Lane> lanes(nf);
    std::string wrongRow;   // a NaN from any form withholds the row
    // A form in the log's words, and in a reason's, which names it only
    // where the row has more than one.
    auto what = [&](size_t f) { return formName(forms[f], plan.gpuInt8KernelChoice); };
    auto whose = [&](size_t f) { return raced ? what(f) : std::string(); };

    // On a GPU each form's first rung is predicted from a kScoutDim chain; a
    // scout that cannot be built or run leaves it unpredicted.
    for (size_t f = 0; f < nf && runCapUs > 0.0; f++)
    {
      if (!race.runs(f) || clpeak::cancelRequested())
        continue;
      const LitertPlan fp = litertFormPlan(plan, forms[f]);
      std::string err;
      auto s = LitertSession::create(rt, dev,
                                     litertMatMulChainModel(fp, kScoutDim, kChainLayers, kSeedWidth, forms[f].conv1x1),
                                     litertConfigFor(fp, forms[f]), err);
      double us = -1.0;
      if (s && !s->onDevice())
        err = s->offDevice();
      else if (s && litertBindScalar(*s, fp, err) && s->timeRuns(1, err) > 0.0)   // compile + warmup
        us = s->timeRuns(1, err);
      if (us > 0.0)
      {
        lanes[f].lastRate = 2.0 * (double)kScoutDim * (double)kScoutDim * (double)kScoutDim * kChainLayers / us;
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide x%d %s scout %.1f ms\n", dev.displayName.c_str(), label,
                    (long long)kScoutDim, kChainLayers, what(f).c_str(), us / 1.0e3);
      }
      else
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s scout failed, first rung unpredicted: %s\n",
                    dev.displayName.c_str(), label, (long long)kScoutDim, what(f).c_str(), err.c_str());
    }

    for (int64_t D = kMinDim; D <= kMaxDim && !race.done() && wrongRow.empty(); D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      const uint64_t bytes = rungBytes(plan, D), budget = maxRungBytes();
      if (bytes > budget)
      {
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide layers need %llu MB of a %llu MB budget, stopping\n",
                    dev.displayName.c_str(), label, (long long)D, (unsigned long long)(bytes >> 20),
                    (unsigned long long)(budget >> 20));
        break;
      }
      const double layerOps = 2.0 * (double)D * (double)D * (double)D;
      std::vector<double> rate(nf, 0.0), createUs(nf, 0.0);

      for (size_t f = 0; f < nf && wrongRow.empty(); f++)
      {
        if (!race.runs(f))
          continue;
        Lane &ln = lanes[f];
        const LitertPlan fp = litertFormPlan(plan, forms[f]);
        const std::string form = what(f);
        const double runUs = ln.lastRate > 0.0 ? layerOps * kChainLayers / ln.lastRate : 0.0;
        const bool runTooLong = runCapUs > 0.0 && runUs > runCapUs;
        if (runTooLong || runUs / kChainLayers > kMaxIterUs)
        {
          if (runTooLong)
            CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s would keep the GPU busy ~%.2f s a run, past "
                        "--max-time-gpu (%.2f s), stopping\n",
                        dev.displayName.c_str(), label, (long long)D, form.c_str(), runUs / 1.0e6,
                        runCapUs / 1.0e6);
          else
            CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s would take ~%.1f s per multiply, stopping\n",
                        dev.displayName.c_str(), label, (long long)D, form.c_str(), runUs / kChainLayers / 1.0e6);
          // Before any rung the prediction is the scout's, and it is all the
          // row has to say.
          if (ln.rungs == 0)
          {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.1f", runUs / 1.0e6);
            ln.fail("at the rate a " + std::to_string(kScoutDim) + "-wide chain ran, a " +
                        std::to_string(D) + "-wide one would take about " + buf + " s a run, " +
                        (runTooLong ? "longer than --max-time-gpu lets one run hold a GPU -- a driver may "
                                      "reset a GPU held longer -- "
                                    : "longer than one multiply may take, ") +
                        "so no size was measured",
                    ResultStatus::Error);
          }
          race.drop(f);
          continue;
        }
        // The compile cap, checked before paying for the build; the first
        // size always builds, since its time seeds the prediction.
        if (D > kMinDim && ln.prevCreateUs > 0.0)
        {
          const double predictedUs =
              litertPredictCreateUs(ln.prevCreateUs, ln.prevPrevCreateUs, ln.strikes > 0);
          if (predictedUs > kLitertMaxChainCreateUs)
          {
            CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s predicted create %.1f s (prev %.1f s%s) > "
                        "%.1f s, stopping\n",
                        dev.displayName.c_str(), label, (long long)D, form.c_str(), predictedUs / 1.0e6,
                        ln.prevCreateUs / 1.0e6, ln.strikes > 0 ? ", after a size that did not gain" : "",
                        kLitertMaxChainCreateUs / 1.0e6);
            race.drop(f);
            continue;
          }
        }

        // A form's first rung is also profiled once, in a session of its
        // own: the kernel name is the row's evidence of what ran, and
        // profiling costs enough on the GPU (every kernel waited on) that it
        // never touches a timed session.
        if (ln.rungs == 0 && ln.kernel.empty())
        {
          std::string err;
          auto ps = LitertSession::create(rt, dev,
                                          litertMatMulChainModel(fp, D, kChainLayers, kSeedWidth, forms[f].conv1x1),
                                          litertConfigFor(fp, forms[f], true), err);
          if (ps && ps->onDevice() && litertBindScalar(*ps, fp, err))
          {
            std::string perr;
            ln.kernel = litertMatMulKernel(ps->profileOps(perr));
            CLPEAK_VLOG("litert-gemm[%s/%s]: %s kernel '%s'\n", dev.displayName.c_str(), label, form.c_str(),
                        ln.kernel.c_str());
          }
        }

        std::string err;
        auto s = LitertSession::create(rt, dev,
                                       litertMatMulChainModel(fp, D, kChainLayers, kSeedWidth, forms[f].conv1x1),
                                       litertConfigFor(fp, forms[f]), err);
        if (!s)
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s create failed: %s\n", dev.displayName.c_str(), label,
                      (long long)D, form.c_str(), err.c_str());
          ln.fail(err);
          race.drop(f);   // larger sizes need strictly more of everything
          continue;
        }
        const double cu = s->createUs;
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide x%d %s create %.3f s\n", dev.displayName.c_str(), label,
                    (long long)D, kChainLayers, form.c_str(), cu / 1.0e6);
        if (!s->onDevice())
        {
          const std::string why = s->offDevice();
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s %s\n", dev.displayName.c_str(), label, (long long)D,
                      form.c_str(), why.c_str());
          ln.fail(why);
          race.drop(f);
          continue;
        }
        if (!litertBindScalar(*s, fp, err))
        {
          ln.fail(err, ResultStatus::Error);
          race.drop(f);
          continue;
        }

        auto m = litertMeasure(*s, warmupCount, kSizeBudgetUs, forceIters, specifiedIters);
        // What the race weighs a form's build by: until it can run, the
        // first inference included -- constant sharing moves the GPU's
        // weight conversion out of creation and into that inference.
        const double readyUs = cu + s->firstRunUs;
        const std::string wrong =
            (m.meanUs > 0.0)
                ? litertNonFiniteReason(*s, litertIoType(fp),
                                        "at " + std::to_string(D) + "-wide layers" +
                                            (raced ? " written as " + form : std::string()))
                : std::string();
        s.reset();   // the model's memory goes before the next one is built
        if (m.meanUs <= 0.0)
        {
          // Logged whatever the row says: above a measured rung it publishes
          // the rungs below, and this failure would leave no trace.
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s run failed: %s\n", dev.displayName.c_str(), label,
                      (long long)D, form.c_str(), m.error.c_str());
          ln.fail(m.error, m.status);
          race.drop(f);
          continue;
        }
        // A wrong answer withholds the whole row, not just the rungs from
        // here up: a rung that came back finite, or another form, is no
        // alibi for the accelerator.
        if (!wrong.empty())
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %s\n", dev.displayName.c_str(), label, wrong.c_str());
          wrongRow = wrong;
          break;
        }

        ln.rungs++;
        if (ln.firstUs == 0.0)
        {
          ln.firstUs = m.meanUs;
          ln.firstDim = D;
        }
        ln.lastUs = m.meanUs;
        ln.lastDim = D;
        const double r = layerOps * kChainLayers * 1.0e6 / m.meanUs;
        ln.lastRate = layerOps * kChainLayers / m.meanUs;
        CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide x%d %s -> %.3f (%.1f us, %u iters)\n",
                    dev.displayName.c_str(), label, (long long)D, kChainLayers, form.c_str(), r, m.meanUs, m.iters);

        // Fold detector: work is time less the submission floor, and eight
        // times the arithmetic has to cost at least twice the time however
        // much the rate improves.  The runtime scalar makes folding
        // impossible for LiteRT's own runtime; a vendor compiler that
        // hoisted the scalar out of the float chain and folded the constant
        // product behind it would land here.  It voids what this form
        // measured, and the others race on.
        const double work = m.meanUs - floorUs;
        const double prevWork = ln.prevUs - floorUs;
        const bool computedNothing = floorUs > 0.0 && work < kFoldWorkFloor * floorUs;
        const bool workFlat = ln.prevUs > 0.0 && D == ln.prevD * 2 && prevWork > 0.0 && work < prevWork * 2.0;
        if (computedNothing || workFlat)
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s took %.1f us against a %.1f us floor (previous "
                      "%.1f us) -- the multiplies were folded away\n",
                      dev.displayName.c_str(), label, (long long)D, form.c_str(), m.meanUs, floorUs, ln.prevUs);
          ln.folded(foldReason(whose(f), "the timings do not scale with the problem size, so they "
                                     "measure dispatch and a reduction rather than the matrix "
                                     "multiplies"));
          race.drop(f);
          continue;
        }
        ln.prevUs = m.meanUs;
        ln.prevD = D;
        rate[f] = r;
        createUs[f] = readyUs;

        if (r > ln.best * kImproveFactor)
        {
          ln.strikes = 0;
          ln.best = r;
          ln.bestDim = D;
        }
        else
        {
          if (r > ln.best)
          {
            ln.best = r;
            ln.bestDim = D;
          }
          if (++ln.strikes >= kMaxStrikes)
          {
            CLPEAK_VLOG("litert-gemm[%s/%s]: %s: no further gain past %lld-wide layers\n",
                        dev.displayName.c_str(), label, form.c_str(), (long long)ln.bestDim);
            race.drop(f);
          }
        }

        // Measured, not predicted: an extrapolation cannot see a cliff, and
        // the next size is eight times the work.
        if (race.runs(f) && m.probeUs / kChainLayers > kMaxIterUs)
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s measured %.1f s per multiply, stopping\n",
                      dev.displayName.c_str(), label, (long long)D, form.c_str(),
                      m.probeUs / kChainLayers / 1.0e6);
          race.drop(f);
        }
        else if (race.runs(f) && runCapUs > 0.0 && m.probeUs > runCapUs)
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %lld-wide %s measured %.2f s a run, past --max-time-gpu (%.2f s), "
                      "stopping\n",
                      dev.displayName.c_str(), label, (long long)D, form.c_str(), m.probeUs / 1.0e6,
                      runCapUs / 1.0e6);
          race.drop(f);
        }
        // A size that compiled past the cap anyway is kept, and ends this
        // form's climb.  The first may exceed it once: its time seeds the
        // prediction.
        else if (race.runs(f) && D != kMinDim && cu > kLitertMaxChainCreateUs)
        {
          CLPEAK_VLOG("litert-gemm[%s/%s]: %s create %.1f s > %.1f s at %lld-wide layers, stopping\n",
                      dev.displayName.c_str(), label, form.c_str(), cu / 1.0e6, kLitertMaxChainCreateUs / 1.0e6,
                      (long long)D);
          race.drop(f);
        }
        ln.prevPrevCreateUs = ln.prevCreateUs;
        ln.prevCreateUs = cu;
      }

      size_t timed = 0;
      for (size_t f = 0; f < nf; f++)
        timed += rate[f] > 0.0;
      if (timed >= 2)
      {
        size_t before = 0, after = 0;
        for (size_t f = 0; f < nf; f++)
          before += race.runs(f);
        race.settle(rate, createUs);
        for (size_t f = 0; f < nf; f++)
          after += race.runs(f);
        if (after < before)
          for (size_t f = 0; f < nf; f++)
            if (race.runs(f) && after == 1)
              CLPEAK_VLOG("litert-gemm[%s/%s]: forms settled at %lld-wide layers: %s go on\n",
                          dev.displayName.c_str(), label, (long long)D, what(f).c_str());
      }
    }

    // Whole-ladder backstop, per form: real work grows with the cube of the
    // size, so anything near flat computed nothing.
    for (size_t f = 0; f < nf; f++)
    {
      Lane &ln = lanes[f];
      double expectedGrowth = 1.0;
      for (int64_t d = ln.firstDim; d > 0 && d < ln.lastDim; d *= 2)
        expectedGrowth *= 8.0;
      if (ln.best > 0.0 && ln.firstUs > 0.0 && ln.lastDim > ln.firstDim &&
          ln.lastUs < ln.firstUs * expectedGrowth / 64.0)
        ln.folded(foldReason(whose(f), "the timings do not scale with the problem size and mean nothing"));
    }

    size_t w = 0;
    for (size_t f = 1; f < nf; f++)
      if (lanes[f].best > lanes[w].best)
        w = f;
    const Lane &win = lanes[w];
    if (wrongRow.empty() && win.best > 0.0)
    {
      // In the unit litertRateUnit gives the format and the winner's kernel:
      // a weight-only one's in ops where it multiplied int8.
      o.unit = std::strcmp(litertRateUnit(plan, win.kernel, dev.accel), "ops") == 0 ? "ops" : "";
      const std::string winner = raced ? winnerWords(forms, plan.gpuInt8KernelChoice, w) : std::string();
      o.description += "  Fastest at " + std::to_string(win.bestDim) + "-wide layers" + winner;
      if (!win.kernel.empty())
        o.description += std::string(winner.empty() ? ", as `" : ", run as `") + win.kernel + "`" +
                         litertKernelNote(plan, win.kernel, dev.accel);
      // What its own int8 or int16 inputs and outputs did is the accuracy row's to say.
      if (forms[w].floatIo)
        o.description += ", with float inputs and outputs";
      o.description += ".";
      test.emit(label, (float)win.best, o);
    }
    else
    {
      // The row's reason: a wrong answer first, then an error (a caught
      // fold, a failed run) over a capability, and the first form's over the
      // others'.
      std::string why = wrongRow;
      for (size_t f = 0; f < nf && why.empty(); f++)
        why = wrongForm[f];
      ResultStatus status = ResultStatus::Error;
      if (why.empty())
      {
        const Lane *l = nullptr;
        for (const Lane &ln : lanes)
          if (!ln.err.empty() && (!l || (ln.errStatus == ResultStatus::Error && l->errStatus != ResultStatus::Error)))
            l = &ln;
        if (l)
        {
          why = l->err;
          status = l->errStatus;
        }
      }
      test.skip(label, status, why.empty() ? "no size could be measured" : why, o);
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
