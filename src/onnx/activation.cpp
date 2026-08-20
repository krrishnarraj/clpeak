#ifdef ENABLE_ONNX

// onnx-activation: how fast the operations *between* the matrix multiplies
// run.
//
// A transformer layer is mostly matmul by arithmetic and mostly other things
// by op count -- softmax, normalisation, the gate in a feed-forward.  None of
// them does meaningful arithmetic: they read a tensor and write one back, so
// their ceiling is memory bandwidth, and their rate is reported here as the
// bandwidth they achieve.  Compare it with onnx-tensor-bw: an accelerator
// that streams weights at hundreds of gigabytes a second but normalises at a
// fraction of that will spend a surprising share of a layer doing the cheap
// part, and these are also the operations most likely to be handed back to
// the CPU by a provider that does not implement them.
//
// Each rate is measured against a reference graph that reads and reduces the
// same constant with no operation applied.  Subtracting it leaves the
// operation's own cost rather than the cost of the scaffolding around it.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace
{

// A transformer-shaped tensor: rows of model-width vectors.  The row count
// doubles until the rate stops improving, so the reported figure is the
// operation's best and does not depend on a size chosen today.
constexpr int64_t kCols    = 4096;
constexpr int64_t kMinRows = 1024;            // 8 MB
static uint64_t maxTensorBytes() { return clpeak::memoryBudget(1ull << 30); }

constexpr double kImproveFactor = 1.03;
constexpr int    kMaxStrikes    = 2;
constexpr unsigned int kSizeBudgetUs = 1000000;

struct Variant
{
  OnnxActivation act;
  const char    *label;
  const char    *note;
};

const Variant kVariants[] = {
  {OnnxActivation::Silu, "silu",
   "x times sigmoid(x) -- the gate in the feed-forward network of most "
   "current language models."},
  {OnnxActivation::Softmax, "softmax",
   "Softmax across the row, the operation at the heart of attention.  It "
   "needs two passes over the data and a maximum before it can divide, which "
   "makes it far harder for fixed-function hardware than its cost suggests."},
  {OnnxActivation::LayerNorm, "layernorm",
   "Layer normalisation: mean and variance across each row, then rescale.  "
   "Every transformer layer does this at least twice."},
};

struct Run
{
  double us = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Run measure(const OrtRuntime &rt, const onnx_ep_info_t &ep,
            OnnxActivation act, int64_t rows,
            unsigned int warmup, bool forceIters, unsigned int forced)
{
  Run r;

  OrtSession *session = nullptr;
  {
    std::string xRaw((size_t)rows * kCols * 2, '\0');
    {
      uint16_t *h = reinterpret_cast<uint16_t *>(&xRaw[0]);
      uint32_t s = 0x9e3779b9u;
      for (int64_t i = 0; i < rows * kCols; i++)
      {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        h[i] = floatToHalf((float)(s >> 8) / 16777216.0f - 0.5f);
      }
    }
    std::string model = onnxResidentActivationModel(rows, kCols, act, xRaw);
    xRaw.clear(); xRaw.shrink_to_fit();

    auto ses = onnxCreateSession(rt, ep, model, /*keepConstantsUnfolded=*/true);
    model.clear(); model.shrink_to_fit();
    if (!ses.session)
    {
      r.error  = ses.error;
      r.status = ResultStatus::Unsupported;
      return r;
    }
    session = ses.session;
  }

  uint16_t sVal = floatToHalf(1.0009765625f);
  std::vector<uint16_t> outBuf((size_t)kCols, 0);

  OrtMemoryInfo *mi = nullptr;
  OrtValue *inVal = nullptr, *outVal = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  const int64_t outShape[1] = {kCols};
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, &sVal, sizeof(sVal), nullptr, 0,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, outBuf.data(), outBuf.size() * 2, outShape, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &outVal);
  if (mi) rt.api->ReleaseMemoryInfo(mi);

  auto run = [&](unsigned int n) -> double {
    static const char *ins[]  = {"S"};
    static const char *outs[] = {"Y"};
    auto a = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n; i++)
    {
      OrtStatus *rs = rt.api->Run(session, nullptr, ins,
                                  (const OrtValue *const *)&inVal, 1,
                                  outs, 1, &outVal);
      if (rs)
      {
        r.error = onnxStatusText(rt, rs);
        return -1.0;
      }
    }
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - a).count() / (double)n;
  };

  if (!st && run(1 + warmup) > 0.0)
  {
    double probe = run(3);
    if (probe > 0.0)
    {
      unsigned int iters = pickIters(probe, kSizeBudgetUs,
                                     forceIters ? forced : 0);
      r.us = run(iters);
    }
  }
  if (st)
    r.error = onnxStatusText(rt, st);
  if (r.us <= 0.0 && r.status == ResultStatus::Ok)
    r.status = ResultStatus::Error;

  if (inVal)  rt.api->ReleaseValue(inVal);
  if (outVal) rt.api->ReleaseValue(outVal);
  rt.api->ReleaseSession(session);
  return r;
}

} // namespace

int OnnxPeak::runActivation(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                            benchmark_config_t &cfg)
{
  OnnxDeadline deadline(kOnnxTestBudgetSec);
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx-activation", "ONNX activation throughput", "gbps",
       Category::Bandwidth,
       "How fast this provider runs the operations between the matrix "
       "multiplies -- normalisation, softmax, the feed-forward gate.  They do "
       "almost no arithmetic, so their limit is how fast data moves, and the "
       "figure here is the bandwidth each one achieves.  Held against the "
       "resident-tensor rows, it shows how much of a layer goes on the cheap "
       "parts: hardware built for matrix multiplication often runs these at a "
       "small fraction of its streaming speed, and they are also the "
       "operations a provider is most likely to hand back to the CPU."});

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;

    double       best = 0.0;
    int          strikes = 0;
    std::string  firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;

    for (int64_t rows = kMinRows;; rows *= 2)
    {
      if (clpeak::cancelRequested())
        break;
      if (deadline.expired())
      {
        CLPEAK_VLOG("onnx-activation[%s/%s]: out of time, stopping at %lld "
                    "rows\n", ep.providerKey.c_str(), v.label,
                    (long long)rows);
        break;
      }
      const uint64_t bytes = (uint64_t)rows * kCols * 2ull;
      if (bytes > maxTensorBytes())
        break;

      // The reference: same tensor, same read and reduction, no operation.
      Run floor = measure(rt, ep, OnnxActivation::None, rows, warmupCount,
                          forceIters, specifiedIters);
      Run full  = measure(rt, ep, v.act, rows, warmupCount, forceIters,
                          specifiedIters);
      if (full.us <= 0.0)
      {
        if (firstErr.empty())
        {
          firstErr  = full.error.empty() ? "run failed" : full.error;
          errStatus = full.status;
        }
        break;
      }

      const double netUs = full.us - ((floor.us > 0.0) ? floor.us : 0.0);
      if (netUs <= 0.0)
      {
        // Too cheap to separate from reading the tensor at this size; a
        // bigger one will separate them.
        CLPEAK_VLOG("onnx-activation[%s/%s]: %lld rows lost in the noise\n",
                    ep.providerKey.c_str(), v.label, (long long)rows);
        continue;
      }

      // One pass in, one pass out.
      const double gbps = 2.0 * (double)bytes / (netUs * 1.0e-6) / 1.0e9;
      CLPEAK_VLOG("onnx-activation[%s/%s]: %lld rows -> %.1f GB/s (%.0f us, "
                  "floor %.0f us)\n", ep.providerKey.c_str(), v.label,
                  (long long)rows, gbps, full.us, floor.us);

      if (gbps > best * kImproveFactor)
      {
        strikes = 0;
        best = gbps;
      }
      else
      {
        if (gbps > best) best = gbps;
        if (++strikes >= kMaxStrikes)
          break;
      }
    }

    logger::EmitOptions o;
    o.description = v.note;
    if (best > 0.0)
      test.emit(v.label, (float)best, o);
    else
      test.skip(v.label, errStatus,
                firstErr.empty() ? "unsupported" : firstErr, o.description);
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
