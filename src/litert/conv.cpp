#ifdef ENABLE_LITERT

// litert-conv: 2-D convolution peak through LiteRT.
//
// Mobile accelerators were built for convolution before they were asked to
// do anything else -- every NPU's headline TOPS figure is an int8
// convolution -- and the gap between this and litert-gemm is an
// architectural number in its own right.  Three shapes at a fixed 256
// channels -- a 3x3, a 1x1 (arithmetically a matmul per pixel) and a
// depthwise 3x3 (a fraction of the arithmetic per byte loaded) -- each swept
// over feature-map size until the rate stops improving, in fp32, fp16 and
// full-integer int8.  The recipe is the ONNX and Core ML backends'
// (src/onnx/conv.cpp), so the ladders divide row for row; the int8 rows are
// this backend's own, because TFLite is where int8 convolution ships.

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <algorithm>
#include <cctype>
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

uint64_t maxTensorBytes() { return clpeak::memoryBudget(1ull << 30); }

struct Format
{
  LitertFormat f;
  const char *label;
  const char *note;
};

struct Shape
{
  int64_t kernel;
  bool depthwise;
  const char *label;
  const char *note;
};

const Format kFormats[] = {
    {LitertFormat::Fp32, "fp32", "In fp32, the GPU under its fp32 policy."},
    {LitertFormat::Fp16, "fp16",
     "In fp16: half-typed on the CPU, the GPU's fp16 policy over half-stored "
     "weights."},
    {LitertFormat::Int8Qdq, "int8",
     "Full-integer int8 with per-channel weights, the operation every mobile "
     "NPU is rated on."},
};

const Shape kShapes[] = {
    {3, false, "conv3x3",
     "A 3x3 convolution over 256 channels, the shape vision networks are built "
     "from, counted as direct multiplies (a Winograd kernel does fewer and can "
     "read above the matmul peak)."},
    {1, false, "conv1x1",
     "A 1x1 convolution, arithmetically a matmul at every pixel, so it should "
     "land near the matmul rows."},
    {3, true, "depthwise3x3",
     "A depthwise 3x3, each channel on its own: far less arithmetic per byte "
     "loaded, where hardware built around dense arrays collapses."},
};

double convFlops(const Shape &v, int64_t spatial)
{
  const double inPerGroup = v.depthwise ? 1.0 : (double)kChannels;
  return 2.0 * (double)kChannels * (double)spatial * (double)spatial * inPerGroup *
         (double)(v.kernel * v.kernel);
}

std::string convKernel(const std::vector<std::string> &ops)
{
  for (const std::string &o : ops)
  {
    std::string l = o;
    std::transform(l.begin(), l.end(), l.begin(), ::tolower);
    if (l.find("conv") != std::string::npos)
      return o;
  }
  return std::string();
}

} // namespace

int LitertPeak::runConv(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"litert_conv", "LiteRT convolution peak", "flops", Category::Compute,
       "2-D convolution rate through LiteRT on this accelerator: a 3x3, a 1x1 "
       "and a depthwise 3x3 at 256 channels, in fp32, fp16 and full-integer "
       "int8, each swept over feature-map size.  Against the matmul rows it says "
       "whether the accelerator was built for convolution, as mobile NPUs were.",
       TestShape::Heterogeneous, "format and shape"});

  for (const Format &fmt : kFormats)
  {
    const LitertPlan plan = litertPlanFor(fmt.f, dev.accel);
    // The format's matmul answer, checked once: a kernel that gets the
    // format wrong there gets no convolution rate either.
    const std::string wrong = plan.applies ? wrongAnswer(rt, dev, fmt.f) : std::string();
    for (const Shape &v : kShapes)
    {
      if (clpeak::cancelRequested())
        break;
      const std::string row = std::string(fmt.label) + "_" + v.label;
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  " + fmt.note;
      if (plan.integerOps)
        o.unit = "ops";
      if (!plan.applies)
      {
        test.skip(row, ResultStatus::Unsupported, plan.whyNot, o);
        continue;
      }
      if (!wrong.empty())
      {
        test.skip(row, ResultStatus::Error, wrong, o);
        continue;
      }

      const uint64_t elemBytes = litertElemBytes(litertConstantType(plan), 1);
      double best = 0.0;
      int64_t bestSp = 0;
      std::string firstErr, kernel;
      ResultStatus errStatus = ResultStatus::Unsupported;
      double lastRate = 0.0, prevCreateUs = 0.0;
      int strikes = 0;
      int rungs = 0;

      for (int64_t sp = kMinSpatial; sp <= kMaxSpatial; sp *= 2)
      {
        if (clpeak::cancelRequested())
          break;
        // The feature map, held three times (the constant as stored, the
        // scaled copy, the result).
        const uint64_t bytes = 3ull * (uint64_t)kChannels * (uint64_t)sp * (uint64_t)sp * elemBytes;
        if (bytes > maxTensorBytes())
        {
          CLPEAK_VLOG("litert-conv[%s/%s]: %lld needs %llu MB, stopping\n", dev.displayName.c_str(),
                      row.c_str(), (long long)sp, (unsigned long long)(bytes >> 20));
          break;
        }
        if (lastRate > 0.0 && convFlops(v, sp) / lastRate > kMaxIterUs)
        {
          CLPEAK_VLOG("litert-conv[%s/%s]: %lld would take too long, stopping\n", dev.displayName.c_str(),
                      row.c_str(), (long long)sp);
          break;
        }
        if (sp > kMinSpatial && prevCreateUs > 0.0 && prevCreateUs * 4.0 > kLitertMaxCreateUs)
          break;

        if (rungs == 0 && kernel.empty())
        {
          std::string err;
          auto ps = LitertSession::create(rt, dev, litertConvModel(plan, kChannels, sp, v.kernel, v.depthwise),
                                          litertConfigFor(plan, true), err);
          if (ps && ps->onDevice() && litertBindScalar(*ps, plan, err))
          {
            std::string perr;
            kernel = convKernel(ps->profileOps(perr));
          }
        }

        std::string err;
        auto s = LitertSession::create(rt, dev, litertConvModel(plan, kChannels, sp, v.kernel, v.depthwise),
                                       litertConfigFor(plan), err);
        if (!s)
        {
          CLPEAK_VLOG("litert-conv[%s/%s]: %lld create failed: %s\n", dev.displayName.c_str(), row.c_str(),
                      (long long)sp, err.c_str());
          if (firstErr.empty())
            firstErr = err;
          break;
        }
        const double createUs = s->createUs;
        if (!s->onDevice())
        {
          if (firstErr.empty())
            firstErr = s->offDevice();
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
        s.reset();
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
        const double flops = convFlops(v, sp);
        const double rate = flops * 1.0e6 / m.meanUs;
        lastRate = flops / m.meanUs;
        CLPEAK_VLOG("litert-conv[%s/%s]: %lld -> %.3f (%.1f us)\n", dev.displayName.c_str(), row.c_str(),
                    (long long)sp, rate, m.meanUs);

        if (rate > best * kImproveFactor)
        {
          strikes = 0;
          best = rate;
          bestSp = sp;
        }
        else
        {
          if (rate > best)
          {
            best = rate;
            bestSp = sp;
          }
          if (++strikes >= kMaxStrikes)
            break;
        }
        if (m.probeUs > kMaxIterUs)
          break;
        if (prevCreateUs > 0.0 && createUs > kLitertCreateGrowthFloor &&
            createUs > prevCreateUs * kLitertCreateGrowthFactor)
          break;
        if (sp != kMinSpatial && createUs > kLitertMaxCreateUs)
          break;
        prevCreateUs = createUs;
      }

      if (best > 0.0)
      {
        o.description = std::string(v.note) + "  " + fmt.note + "  Fastest at a " + std::to_string(bestSp) +
                        " by " + std::to_string(bestSp) + " feature map";
        if (!kernel.empty())
        {
          o.description += ", as `" + kernel + "`";
          if (plan.integerOps && litertKernelIsFloatForInteger(kernel, dev.accel))
            o.description += litertFloatKernelNote();
        }
        o.description += ".";
        test.emit(row, (float)best, o);
      }
      else
        test.skip(row, errStatus, firstErr.empty() ? "no size could be measured" : firstErr, o);
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
