#ifdef ENABLE_COREML

// coreml-conv: 2-D convolution peak through Core ML.
//
// Accelerators were built for convolution before they were asked to do
// anything else, and the gap between this and coreml-gemm is an
// architectural number in its own right: the Neural Engine convolves faster
// than it multiplies.  Three shapes at a fixed 256 channels -- a 3x3, a 1x1
// (arithmetically a matmul per pixel) and a depthwise 3x3 (a fraction of the
// arithmetic per byte loaded) -- each swept over feature-map size until the
// rate stops improving, in fp16 and fp32.  The recipe is the ONNX backend's
// (src/onnx/conv.cpp), so the two ladders divide row for row.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

#include <algorithm>
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

struct DType
{
  int dtype;
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

const DType kDTypes[] = {
    {CML_FP16, "fp16", "16-bit floats, the native currency of the Neural Engine's convolution engine."},
    {CML_FP32, "fp32",
     "FP32 in and out.  The Neural Engine has no fp32 path, so on that device "
     "this row reads whichever unit Core ML sent it to, or reports so."},
};

const Shape kShapes[] = {
    {3, false, "conv3x3",
     "A 3x3 convolution over 256 channels, the shape most vision networks are "
     "built from and the one accelerators were designed around."},
    {1, false, "conv1x1",
     "Arithmetically a matrix multiply at every pixel, so it should land near "
     "the matmul rows; where it does not, the two shapes reach different "
     "machinery."},
    {3, true, "depthwise3x3",
     "The 3x3 shape with each channel kept separate, so far less arithmetic "
     "per value loaded.  Hardware built around dense arrays collapses here, "
     "which is why mobile networks run slower than their FLOP counts."},
};

double convFlops(const Shape &v, int64_t spatial)
{
  const double inPerGroup = v.depthwise ? 1.0 : (double)kChannels;
  return 2.0 * (double)kChannels * (double)spatial * (double)spatial * inPerGroup *
         (double)(v.kernel * v.kernel);
}

} // namespace

int CoreMLPeak::runConv(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_conv", "Core ML convolution peak", "flops", Category::Compute,
       "2-D convolution rate through Core ML on this compute unit, three "
       "shapes at 256 channels, each swept over feature-map size and reported "
       "at its best.  Against the matmul rows this says whether the unit was "
       "built for convolution -- the Neural Engine was -- and the depthwise "
       "row says what it does when the arithmetic per byte collapses.",
       TestShape::Heterogeneous, "data type and shape"});

  for (const DType &dt : kDTypes)
  {
    for (const Shape &v : kShapes)
    {
      if (clpeak::cancelRequested())
        break;
      const std::string row = std::string(dt.label) + "_" + v.label;
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  " + dt.note;

      const int64_t group = v.depthwise ? kChannels : 1;
      const uint64_t elemBytes = coremlElemBytes(dt.dtype, 1);

      double best = 0.0;
      int64_t bestSp = 0;
      std::string firstErr;
      ResultStatus errStatus = ResultStatus::Unsupported;
      double lastRate = 0.0, prevCreateUs = 0.0;
      int strikes = 0;
      std::string offDeviceNote, glueNote;
      int rungs = 0;

      for (int64_t sp = kMinSpatial; sp <= kMaxSpatial; sp *= 2)
      {
        if (clpeak::cancelRequested())
          break;
        // The feature map, held twice (the constant and the scaled copy).
        const uint64_t bytes = 2ull * (uint64_t)kChannels * (uint64_t)sp * (uint64_t)sp * elemBytes;
        if (bytes > maxTensorBytes())
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld needs %llu MB, stopping\n", dev.displayName.c_str(),
                      row.c_str(), (long long)sp, (unsigned long long)(bytes >> 20));
          break;
        }
        if (lastRate > 0.0 && convFlops(v, sp) / lastRate > kMaxIterUs)
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld would take too long, stopping\n",
                      dev.displayName.c_str(), row.c_str(), (long long)sp);
          break;
        }
        if (sp > kMinSpatial && prevCreateUs > 0.0 && prevCreateUs * 4.0 > kCoremlMaxCreateUs)
          break;

        std::string err;
        auto s = CoremlSession::create(dev, coremlConvModel(spec, kChannels, sp, v.kernel, group, dt.dtype), err);
        if (!s)
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld create failed: %s\n", dev.displayName.c_str(),
                      row.c_str(), (long long)sp, err.c_str());
          if (firstErr.empty())
            firstErr = err;
          break;
        }
        const double createUs = coremlCreateUs(*s);
        CLPEAK_VLOG("coreml-conv[%s/%s]: %lld create %.2f s\n", dev.displayName.c_str(), row.c_str(),
                    (long long)sp, createUs / 1.0e6);
        if (!s->onDevice())
        {
          const std::string why = coremlOffDeviceReason(dev, *s);
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld %s\n", dev.displayName.c_str(), row.c_str(),
                      (long long)sp, why.c_str());
          if (rungs == 0)
          {
            if (firstErr.empty())
              firstErr = why;
            // Too small for the planner is not too big for the unit: a 32x32
            // map stays on the CPU where a 256x256 one goes to the Neural
            // Engine, so the ladder climbs on.  A shape the unit cannot run
            // at all will not become runnable by growing.
            if (s->offDeviceCapable())
            {
              s.reset();
              continue;
            }
          }
          else
            offDeviceNote = "  Larger feature maps were declined by the " +
                            std::string(coremlKindName(dev.kind)) + ".";
          break;
        }
        if (!coremlBindScalar(*s, "s", dt.dtype, err))
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
        CLPEAK_VLOG("coreml-conv[%s/%s]: %lld -> %.3f (%.1f us)\n", dev.displayName.c_str(), row.c_str(),
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
        if (prevCreateUs > 0.0 && createUs > kCoremlCreateGrowthFloor &&
            createUs > prevCreateUs * kCoremlCreateGrowthFactor)
        {
          prevCreateUs = createUs;
          break;
        }
        if (sp != kMinSpatial && createUs > kCoremlMaxCreateUs)
        {
          prevCreateUs = createUs;
          break;
        }
        prevCreateUs = createUs;
      }

      if (best > 0.0)
      {
        o.description = std::string(v.note) + "  " + dt.note + "  Fastest at a " +
                        std::to_string(bestSp) + " by " + std::to_string(bestSp) + " feature map." +
                        offDeviceNote + glueNote;
        test.emit(row, (float)best, o);
      }
      else
        test.skip(row, errStatus, firstErr.empty() ? "no size could be measured" : firstErr, o);
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
