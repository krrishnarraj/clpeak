#ifdef ENABLE_COREML

// coreml-conv: 2-D convolution peak through Core ML.
//
// Accelerators were built for convolution before they were asked to do
// anything else, and the gap between this and coreml-gemm is an
// architectural number in its own right.  Three shapes at a fixed 256
// channels -- a 3x3 at stride 2, a 1x1 (arithmetically a matmul per pixel)
// and a depthwise 3x3 (a fraction of the arithmetic per byte loaded) -- each
// swept over feature-map size until the rate stops improving, in fp16 and
// fp32.  The recipe is the ONNX backend's (src/onnx/conv.cpp, whose kShapes
// says why the 3x3's stride is 2: at 1, Core ML's GPU ran it with fewer
// multiplies than counted and read 6.0 TFLOPS on an M1 Pro whose GPU peaks
// at 5.3), so the two ladders divide row for row.

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

// Everything a rung holds at once (coremlHeldBytes), capped at a quarter of
// physical memory and at 3 GB, which still admits the 1024-square fp16 map
// the GPU's depthwise row is fastest at.
uint64_t maxRungBytes() { return clpeak::memoryBudget(3ull << 30); }

struct DType
{
  int dtype;
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

// A row's description is its shape's note, then its type's: "Stride-2 3x3,
// 16-bit weights and arithmetic."
const DType kDTypes[] = {
    {CML_FP16, "fp16", "16-bit weights and arithmetic."},
    {CML_FP32, "fp32", "full 32-bit precision."},
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

} // namespace

int CoreMLPeak::runConv(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  // The longest one prediction may be predicted to take: on a GPU,
  // --max-time-gpu (gpuRunCapUs); elsewhere 0, unbounded.
  const double runCapUs = gpuRunCapUs(dev.deviceType, cfg);
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_conv", "Core ML convolution peak", "flops", Category::Compute,
       "2-D convolution rate on this compute unit at 256 channels -- a stride-2 "
       "3x3, a 1x1 and a depthwise 3x3 -- each swept over feature-map size and "
       "reported at its best.  Set against the matmul rows, it shows how well "
       "this compute unit handles convolution.",
       TestShape::Heterogeneous, "data type and shape"});

  for (const DType &dt : kDTypes)
  {
    // A unit whose answer for this precision, written as a 1x1 convolution,
    // is wrong has no convolution rate worth publishing (wrongAnswer,
    // numeric_error.cpp): the conv1x1 row's operator, and the nearest there
    // is to the other shapes'.
    const std::string wrongConv =
        clpeak::cancelRequested()
            ? std::string()
            : wrongAnswer(dev, dt.dtype == CML_FP32 ? CoremlWeight::Fp32 : CoremlWeight::Fp16, false,
                          /*conv1x1=*/true);
    for (const Shape &v : kShapes)
    {
      if (clpeak::cancelRequested())
        break;
      const std::string row = std::string(dt.label) + "_" + v.label;
      logger::EmitOptions o;
      o.description = std::string(v.note) + ", " + dt.note;
      if (!wrongConv.empty())
      {
        CLPEAK_VLOG("coreml-conv[%s/%s]: %s\n", dev.displayName.c_str(), row.c_str(), wrongConv.c_str());
        test.skip(row, ResultStatus::Error, wrongConv, o);
        continue;
      }

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
        // The feature map and the filter are constants (coremlHeldBytes);
        // the scaled copy of the map and the result, a quarter of it at
        // stride 2, are activations.
        const uint64_t map = (uint64_t)kChannels * (uint64_t)sp * (uint64_t)sp * elemBytes;
        const uint64_t filter = (uint64_t)kChannels * (uint64_t)(kChannels / group) *
                                (uint64_t)(v.kernel * v.kernel) * elemBytes;
        const uint64_t bytes = coremlHeldBytes(map + filter, map + map / (uint64_t)(v.stride * v.stride));
        const uint64_t budget = maxRungBytes();
        if (bytes > budget)
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld needs %llu MB of a %llu MB budget, stopping\n",
                      dev.displayName.c_str(), row.c_str(), (long long)sp, (unsigned long long)(bytes >> 20),
                      (unsigned long long)(budget >> 20));
          break;
        }
        if (lastRate > 0.0 && convFlops(v, sp) / lastRate > kMaxIterUs)
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld would take too long, stopping\n",
                      dev.displayName.c_str(), row.c_str(), (long long)sp);
          break;
        }
        // On a GPU a prediction is also held to what one run may keep the
        // device busy (gpuRunCapUs): a driver resets a GPU held too long.
        if (lastRate > 0.0 && runCapUs > 0.0 && convFlops(v, sp) / lastRate > runCapUs)
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld would keep the GPU busy ~%.2f s a prediction, past "
                      "--max-time-gpu (%.2f s), stopping\n",
                      dev.displayName.c_str(), row.c_str(), (long long)sp, convFlops(v, sp) / lastRate / 1.0e6,
                      runCapUs / 1.0e6);
          break;
        }
        if (sp > kMinSpatial && prevCreateUs > 0.0 && prevCreateUs * 4.0 > kCoremlMaxCreateUs)
          break;

        std::string err;
        auto s = CoremlSession::create(
            dev, coremlConvModel(spec, kChannels, sp, v.kernel, v.stride, group, dt.dtype), err);
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
            offDeviceNote = "; the " + std::string(coremlKindName(dev.kind)) + " declined larger maps";
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
        std::string wrong;
        if (m.meanUs > 0.0)
        {
          glueNote = coremlGlueNote(*s);
          wrong = coremlNonFiniteReason(*s, "out", dt.dtype,
                                        "on a " + std::to_string(sp) + "x" + std::to_string(sp) +
                                            " feature map");
        }
        s.reset();
        if (m.meanUs <= 0.0)
        {
          // Logged whatever the row says: above a measured size it publishes
          // the sizes below, and this failure would leave no trace.
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld run failed: %s\n", dev.displayName.c_str(), row.c_str(),
                      (long long)sp, m.error.c_str());
          if (firstErr.empty())
          {
            firstErr = m.error;
            errStatus = m.status;
          }
          break;
        }
        // A wrong answer withholds the whole row, as in gemm.cpp.
        if (!wrong.empty())
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %s\n", dev.displayName.c_str(), row.c_str(), wrong.c_str());
          best = 0.0;
          firstErr = wrong;
          errStatus = ResultStatus::Error;
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
        if (runCapUs > 0.0 && m.probeUs > runCapUs)
        {
          CLPEAK_VLOG("coreml-conv[%s/%s]: %lld measured %.2f s a prediction, past --max-time-gpu (%.2f s), "
                      "stopping\n",
                      dev.displayName.c_str(), row.c_str(), (long long)sp, m.probeUs / 1.0e6, runCapUs / 1.0e6);
          break;
        }
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
        o.description += "  Fastest at a " + std::to_string(bestSp) + "x" + std::to_string(bestSp) +
                         " feature map" + offDeviceNote + glueNote + ".";
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
