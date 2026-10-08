#ifdef ENABLE_LITERT

// litert-conv: 2-D convolution peak through LiteRT.
//
// Mobile accelerators were built for convolution before they were asked to
// do anything else -- every NPU's headline TOPS figure is an int8
// convolution -- and the gap between this and litert-gemm is an
// architectural number in its own right.  Three shapes at a fixed 256
// channels -- a 3x3 at stride 2, a 1x1 (arithmetically a matmul per pixel)
// and a depthwise 3x3 (a fraction of the arithmetic per byte loaded) -- each
// swept over feature-map size until the rate stops improving, in fp32, fp16
// and full-integer int8.  The recipe is the ONNX and Core ML backends'
// (src/onnx/conv.cpp, whose kShapes says why the 3x3's stride is 2: at 1,
// the GPU accelerator runs it as `convolution_winograd_3x3`, about a quarter
// of the multiplies counted), so the ladders divide row for row; the int8 rows
// are this backend's own, because TFLite is where int8 convolution ships.

#include <common/form_race.h>
#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kChannels = 256;
constexpr int64_t kMinSpatial = 32;
constexpr int64_t kMaxSpatial = 4096;
constexpr double kImproveFactor = 1.03;
constexpr int kMaxStrikes = 2;
constexpr double kMaxIterUs = 2.0e6;
constexpr unsigned int kSizeBudgetUs = 2000000;

// A GPU's 8-bit kernels race only on feature maps large enough to be work,
// not dispatch: a size settles a row's race once every form timed there took
// this many times as long a pass as at its first size.  The accelerator
// uses those kernels for large layers only, and on an M1 Pro both forms of
// every int8 row tied at 64 by 64, where a pass is the cost of asking.
constexpr double kRaceWorkFactor = 8.0;

// Everything a rung holds at once (litertHeldBytes), capped at a quarter of
// physical memory and at 2 GB, which still admits the 1024-square int8 maps
// XNNPACK is fastest at.
uint64_t maxRungBytes() { return clpeak::memoryBudget(2ull << 30); }

struct Format
{
  LitertFormat f;
  const char *label;
  const char *note;
};

struct Shape
{
  int64_t kernel;
  int64_t stride;
  bool depthwise;
  const char *label;
  const char *note;
};

// A row's first sentence is its shape's words and then its format's.
const Format kFormats[] = {
    {LitertFormat::Fp32, "fp32", "full 32-bit precision"},
    {LitertFormat::Fp16, "fp16", "16-bit weights and arithmetic"},
    {LitertFormat::Int8Qdq, "int8", "full-integer int8 with per-channel weights"},
};

const Shape kShapes[] = {
    {3, 2, false, "conv3x3s2", "Stride-2 3x3"},
    {1, 1, false, "conv1x1", "1x1 (a matmul per pixel)"},
    {3, 1, true, "depthwise3x3", "Depthwise 3x3 (each channel on its own)"},
};

double convFlops(const Shape &v, int64_t spatial)
{
  const double inPerGroup = v.depthwise ? 1.0 : (double)kChannels;
  const double out = (double)(spatial / v.stride);   // the output's side
  return 2.0 * (double)kChannels * out * out * inPerGroup * (double)(v.kernel * v.kernel);
}

// One form's climb over feature-map sizes.
struct ConvLane
{
  double best = 0.0;
  int64_t bestSp = 0;
  double firstUs = 0.0;   // the mean pass at its first size: what asking costs
  std::string firstErr, kernel;
  ResultStatus errStatus = ResultStatus::Unsupported;
  double lastRate = 0.0, prevCreateUs = 0.0;
  int strikes = 0, rungs = 0;

  void fail(const std::string &why, ResultStatus status = ResultStatus::Unsupported)
  {
    if (firstErr.empty())
    {
      firstErr = why;
      errStatus = status;
    }
  }
};

} // namespace

int LitertPeak::runConv(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg)
{
  // The longest one pass may be predicted to take: on a GPU, --max-time-gpu
  // (gpuRunCapUs); elsewhere 0, unbounded.
  const double runCapUs = gpuRunCapUs(dev.deviceType, cfg);

  auto test = currentDeviceScope->beginTest(
      {"litert_conv", "LiteRT convolution peak", "flops", Category::Compute,
       "2-D convolution rate on this accelerator at 256 channels -- a stride-2 "
       "3x3, a 1x1 and a depthwise 3x3 -- each swept over feature-map size and "
       "reported at its best.  Set against the matmul rows, it shows how well "
       "this accelerator handles convolution.",
       TestShape::Heterogeneous, "format and shape"});

  for (const Format &fmt : kFormats)
  {
    const LitertPlan plan = litertPlanFor(fmt.f, dev.accel);
    // The forms a row races: on a GPU with 8-bit kernels for the format,
    // with them disallowed and allowed (gemm.cpp has why).  Each is checked
    // once, as the same product written as a 1x1 convolution -- the conv1x1
    // row's own graph, and for the 3x3 and the depthwise the nearest one
    // there is -- in the inputs and outputs it answers right with.  A form
    // whose answer is wrong leaves every shape's race.
    std::vector<LitertForm> forms;
    std::vector<std::string> wrong;
    if (plan.applies)
      for (int k = 0; k <= (plan.gpuInt8KernelChoice ? 1 : 0); k++)
      {
        const LitertForm form = resolveIo(rt, dev, fmt.f, LitertForm{true, k == 1, false});
        forms.push_back(LitertForm{false, form.gpuInt8Kernels, form.floatIo});
        wrong.push_back(wrongAnswer(rt, dev, fmt.f, form));
      }
    const size_t nf = forms.size();
    const bool raced = nf > 1;
    size_t wrongCount = 0;
    for (const std::string &w : wrong)
      wrongCount += !w.empty();

    for (const Shape &v : kShapes)
    {
      if (clpeak::cancelRequested())
        break;
      const std::string row = std::string(fmt.label) + "_" + v.label;
      logger::EmitOptions o;
      o.description = std::string(v.note) + ", " + fmt.note + ".";
      if (plan.integerOps)
        o.unit = "ops";
      if (!plan.applies)
      {
        test.skip(row, ResultStatus::Unsupported, plan.whyNot, o);
        continue;
      }
      if (wrongCount == nf)
      {
        test.skip(row, ResultStatus::Error, wrong[0], o);
        continue;
      }

      clpeak::MultiFormRace race(nf);
      for (size_t f = 0; f < nf; f++)
        if (!wrong[f].empty())
          race.drop(f);
      std::vector<ConvLane> lanes(nf);
      auto what = [&](size_t f) {
        return raced ? std::string(forms[f].gpuInt8Kernels ? " with" : " without") + " the GPU's 8-bit kernels"
                     : std::string();
      };

      for (int64_t sp = kMinSpatial; sp <= kMaxSpatial && !race.done(); sp *= 2)
      {
        if (clpeak::cancelRequested())
          break;
        // The feature map and the filter are constants (litertHeldBytes);
        // the scaled copy of the map and the result, a quarter of it at
        // stride 2, are activations.
        const uint64_t elemBytes = litertElemBytes(litertConstantType(plan), 1);
        const uint64_t pixels = (uint64_t)kChannels * (uint64_t)sp * (uint64_t)sp;
        const uint64_t filter = litertElemBytes(plan.weight, kChannels * (v.depthwise ? 1 : kChannels) *
                                                                 v.kernel * v.kernel);
        const uint64_t act = litertElemBytes(plan.act, 1);
        const uint64_t bytes = litertHeldBytes(pixels * elemBytes + filter,
                                               pixels * act + pixels * act / (uint64_t)(v.stride * v.stride));
        const uint64_t budget = maxRungBytes();
        if (bytes > budget)
        {
          CLPEAK_VLOG("litert-conv[%s/%s]: %lld needs %llu MB of a %llu MB budget, stopping\n",
                      dev.displayName.c_str(), row.c_str(), (long long)sp, (unsigned long long)(bytes >> 20),
                      (unsigned long long)(budget >> 20));
          break;
        }
        std::vector<double> rate(nf, 0.0), createUsAt(nf, 0.0);
        std::vector<bool> workBound(nf, false);

        for (size_t f = 0; f < nf; f++)
        {
          if (!race.runs(f))
            continue;
          ConvLane &ln = lanes[f];
          const LitertPlan fp = litertFormPlan(plan, forms[f]);
          const std::string form = what(f);
          if (ln.lastRate > 0.0 && convFlops(v, sp) / ln.lastRate > kMaxIterUs)
          {
            CLPEAK_VLOG("litert-conv[%s/%s]: %lld%s would take too long, stopping\n", dev.displayName.c_str(),
                        row.c_str(), (long long)sp, form.c_str());
            race.drop(f);
            continue;
          }
          // On a GPU a pass is also held to what one run may keep the device
          // busy (gpuRunCapUs): a driver resets a GPU held too long.
          if (ln.lastRate > 0.0 && runCapUs > 0.0 && convFlops(v, sp) / ln.lastRate > runCapUs)
          {
            CLPEAK_VLOG("litert-conv[%s/%s]: %lld%s would keep the GPU busy ~%.2f s a pass, past "
                        "--max-time-gpu (%.2f s), stopping\n",
                        dev.displayName.c_str(), row.c_str(), (long long)sp, form.c_str(),
                        convFlops(v, sp) / ln.lastRate / 1.0e6, runCapUs / 1.0e6);
            race.drop(f);
            continue;
          }
          if (sp > kMinSpatial && ln.prevCreateUs > 0.0 && ln.prevCreateUs * 4.0 > kLitertMaxCreateUs)
          {
            race.drop(f);
            continue;
          }

          if (ln.rungs == 0 && ln.kernel.empty())
          {
            std::string err;
            auto ps = LitertSession::create(rt, dev,
                                            litertConvModel(fp, kChannels, sp, v.kernel, v.stride, v.depthwise),
                                            litertConfigFor(fp, forms[f], true), err);
            if (ps && ps->onDevice() && litertBindScalar(*ps, fp, err))
            {
              std::string perr;
              ln.kernel = litertMatMulKernel(ps->profileOps(perr));
            }
          }

          std::string err;
          auto s = LitertSession::create(rt, dev, litertConvModel(fp, kChannels, sp, v.kernel, v.stride, v.depthwise),
                                         litertConfigFor(fp, forms[f]), err);
          if (!s)
          {
            CLPEAK_VLOG("litert-conv[%s/%s]: %lld%s create failed: %s\n", dev.displayName.c_str(), row.c_str(),
                        (long long)sp, form.c_str(), err.c_str());
            ln.fail(err);
            race.drop(f);
            continue;
          }
          const double createUs = s->createUs;
          if (!s->onDevice())
          {
            ln.fail(s->offDevice());
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
          const double readyUs = createUs + s->firstRunUs;   // the race's build cost, as in gemm.cpp
          const std::string nonFinite =
              (m.meanUs > 0.0) ? litertNonFiniteReason(*s, litertIoType(fp),
                                                       "on a " + std::to_string(sp) + "x" + std::to_string(sp) +
                                                           " feature map" + form)
                               : std::string();
          s.reset();
          if (m.meanUs <= 0.0)
          {
            // Logged whatever the row says: above a measured size it
            // publishes the sizes below, and this failure would leave no
            // trace.
            CLPEAK_VLOG("litert-conv[%s/%s]: %lld%s run failed: %s\n", dev.displayName.c_str(), row.c_str(),
                        (long long)sp, form.c_str(), m.error.c_str());
            ln.fail(m.error, m.status);
            race.drop(f);
            continue;
          }
          // A wrong answer withholds the whole row, as in gemm.cpp.
          if (!nonFinite.empty())
          {
            CLPEAK_VLOG("litert-conv[%s/%s]: %s\n", dev.displayName.c_str(), row.c_str(), nonFinite.c_str());
            for (ConvLane &l : lanes)
            {
              l.best = 0.0;
              l.firstErr = nonFinite;
              l.errStatus = ResultStatus::Error;
            }
            for (size_t g = 0; g < nf; g++)
              race.drop(g);
            break;
          }
          ln.rungs++;
          if (ln.firstUs == 0.0)
            ln.firstUs = m.meanUs;
          workBound[f] = m.meanUs >= kRaceWorkFactor * ln.firstUs;
          const double flops = convFlops(v, sp);
          const double r = flops * 1.0e6 / m.meanUs;
          ln.lastRate = flops / m.meanUs;
          CLPEAK_VLOG("litert-conv[%s/%s]: %lld%s -> %.3f (%.1f us)\n", dev.displayName.c_str(), row.c_str(),
                      (long long)sp, form.c_str(), r, m.meanUs);
          rate[f] = r;
          createUsAt[f] = readyUs;

          if (r > ln.best * kImproveFactor)
          {
            ln.strikes = 0;
            ln.best = r;
            ln.bestSp = sp;
          }
          else
          {
            if (r > ln.best)
            {
              ln.best = r;
              ln.bestSp = sp;
            }
            if (++ln.strikes >= kMaxStrikes)
              race.drop(f);
          }
          if (race.runs(f) && m.probeUs > kMaxIterUs)
            race.drop(f);
          else if (race.runs(f) && runCapUs > 0.0 && m.probeUs > runCapUs)
          {
            CLPEAK_VLOG("litert-conv[%s/%s]: %lld%s measured %.2f s a pass, past --max-time-gpu (%.2f s), "
                        "stopping\n",
                        dev.displayName.c_str(), row.c_str(), (long long)sp, form.c_str(), m.probeUs / 1.0e6,
                        runCapUs / 1.0e6);
            race.drop(f);
          }
          else if (race.runs(f) && ln.prevCreateUs > 0.0 && createUs > kLitertCreateGrowthFloor &&
                   createUs > ln.prevCreateUs * kLitertCreateGrowthFactor)
            race.drop(f);
          else if (race.runs(f) && sp != kMinSpatial && createUs > kLitertMaxCreateUs)
            race.drop(f);
          ln.prevCreateUs = createUs;
        }

        if (rate.size() > 1 && std::count_if(rate.begin(), rate.end(), [](double r) { return r > 0.0; }) >= 2)
        {
          bool work = true;
          for (size_t f = 0; f < nf; f++)
            if (rate[f] > 0.0)
              work = work && workBound[f];
          if (work)
            race.settle(rate, createUsAt);
        }
      }

      size_t w = 0;
      for (size_t f = 1; f < nf; f++)
        if (lanes[f].best > lanes[w].best)
          w = f;
      const ConvLane &win = lanes[w];
      if (win.best > 0.0)
      {
        // In the unit litertRateUnit gives the format and the kernel that ran
        // it; a raced row names its winner (`what`).
        o.unit = std::strcmp(litertRateUnit(plan, win.kernel, dev.accel), "ops") == 0 ? "ops" : "";
        o.description += "  Fastest at a " + std::to_string(win.bestSp) + "x" + std::to_string(win.bestSp) +
                         " feature map" + what(w);
        if (!win.kernel.empty())
          o.description += ", as `" + win.kernel + "`" + litertKernelNote(plan, win.kernel, dev.accel);
        if (forms[w].floatIo)
          o.description += ", with float inputs and outputs";
        o.description += ".";
        test.emit(row, (float)win.best, o);
      }
      else
      {
        const ConvLane *l = nullptr;
        for (const ConvLane &ln : lanes)
          if (!ln.firstErr.empty() && (!l || (ln.errStatus == ResultStatus::Error && l->errStatus != ResultStatus::Error)))
            l = &ln;
        std::string why = l ? l->firstErr : std::string();
        ResultStatus status = l ? l->errStatus : ResultStatus::Unsupported;
        for (size_t f = 0; f < nf && why.empty(); f++)
          if (!wrong[f].empty())
          {
            why = wrong[f];
            status = ResultStatus::Error;
          }
        test.skip(row, status, why.empty() ? "no size could be measured" : why, o);
      }
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
