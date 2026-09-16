#ifdef ENABLE_LITERT

// litert-transfer-bw: what it costs to get a tensor to this accelerator and
// a result back.
//
// Every other test here keeps its tensors resident, because a model input
// has to be presented to the accelerator in memory it can read -- a Metal or
// OpenCL buffer, an NPU's DMA region -- and that presentation would land in
// every throughput figure.  This test measures it head-on.  What an
// application pays per inference is what is timed: the host copy into
// LiteRT's tensor buffer (a lock, a memcpy, an unlock that uploads), the
// run, and for the round trip the copy back out.  On unified-memory devices
// the rows read close to memory speed, which is itself the answer.
//
// The graphs are the ONNX backend's (src/onnx/transfer.cpp): everything
// arrives and one element comes back, so that on an accelerator with no real
// transfer the row does not become its reduction rate; the round trip
// returns the whole tensor squared.

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kMinElems = 8ll << 20;    // 16 MB of fp16
uint64_t maxTensorBytes() { return clpeak::memoryBudget(128ull << 20); }
constexpr unsigned int kSizeBudgetUs = 300000;

struct Run
{
  double us = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Run measure(const LitertRuntime &rt, const litert_device_info_t &dev, const LitertPlan &plan,
            LitertTransfer dir, int64_t elems, unsigned warmup, bool forceIters, unsigned forced)
{
  Run r;
  std::string err;
  auto s = LitertSession::create(rt, dev, litertTransferModel(plan, dir, elems), litertConfigFor(plan), err);
  if (!s)
  {
    r.error = err;
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (!s->onDevice())
  {
    r.error = s->offDevice();
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (s->createUs > kLitertMaxCreateUs)
  {
    r.error = "model creation took " + std::to_string((long long)(s->createUs / 1.0e6)) +
              " s, exceeds " + std::to_string((long long)(kLitertMaxCreateUs / 1.0e6)) +
              " s compilation budget";
    r.status = ResultStatus::Unsupported;
    return r;
  }

  const std::string half = litertScalarBytes(plan.act, 0.5f);
  std::vector<uint8_t> in((size_t)elems * half.size());
  for (int64_t i = 0; i < elems; i++)
    std::memcpy(&in[(size_t)i * half.size()], half.data(), half.size());
  std::vector<uint8_t> out;

  const bool roundTrip = (dir == LitertTransfer::RoundTrip);
  auto once = [&](unsigned n) -> double {
    const auto t0 = std::chrono::steady_clock::now();
    for (unsigned i = 0; i < n; i++)
    {
      if (!s->writeInput(0, in.data(), in.size(), err) || !s->run(err))
        return -1.0;
      // The result back in full, or the one element that proves the run
      // finished: either way the trip is timed to completion.
      if (roundTrip ? !s->outputBytes(0, out, err) : !s->sync(err))
        return -1.0;
    }
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
  };

  if (once(1 + warmup) < 0.0)
  {
    r.error = err;
    r.status = ResultStatus::Error;
    return r;
  }
  const double probe = once(1);
  if (probe <= 0.0)
  {
    r.error = err;
    r.status = ResultStatus::Error;
    return r;
  }
  const unsigned iters = pickIters(probe, kSizeBudgetUs, forceIters ? forced : 0, kLitertMaxIters);
  r.us = (iters > 1) ? once(iters) : probe;
  if (r.us <= 0.0)
  {
    r.error = err;
    r.status = ResultStatus::Error;
  }
  return r;
}

} // namespace

int LitertPeak::runTransferBandwidth(const LitertRuntime &rt, const litert_device_info_t &dev,
                                     benchmark_config_t &cfg)
{
  (void)cfg;
  const LitertPlan plan = litertPlanFor(LitertFormat::Fp16, dev.accel);
  const uint64_t elem = litertElemBytes(plan.act, 1);

  auto test = currentDeviceScope->beginTest(
      {"litert_transfer_bw", "LiteRT host transfer bandwidth", "bps", Category::Bandwidth,
       "How fast a tensor reaches this accelerator and comes back -- the copy "
       "into LiteRT's tensor buffer, the run, and for the round trip the copy "
       "out -- which every input and output of a real model pays.  Swept over "
       "sizes; the trip out is reported at the largest, the round trip at the "
       "smallest, and the trip back is their difference.",
       TestShape::Heterogeneous, "direction"});

  const char *h2dNote = "Host to accelerator: a tensor handed to LiteRT and one element read "
                        "back, at the largest size measured.";
  const char *rtNote = "Round trip: the tensor in, squared, and the whole result back, at "
                       "the smallest size.";
  const char *d2hNote = "Accelerator to host: the round trip less the trip out at the same size.";

  double h2dBest = 0.0, h2dSmallUs = 0.0;
  std::string h2dErr;
  ResultStatus h2dStatus = ResultStatus::Unsupported;
  int64_t smallElems = 0;
  for (int64_t elems = kMinElems;; elems *= 2)
  {
    if (clpeak::cancelRequested())
      break;
    const uint64_t bytes = (uint64_t)elems * elem;
    if (bytes > maxTensorBytes())
      break;
    Run r = measure(rt, dev, plan, LitertTransfer::ToDevice, elems, warmupCount, forceIters, specifiedIters);
    if (r.us <= 0.0)
    {
      if (h2dErr.empty())
      {
        h2dErr = r.error;
        h2dStatus = r.status;
      }
      break;
    }
    const double bps = (double)bytes / (r.us * 1.0e-6);
    CLPEAK_VLOG("litert-transfer[%s]: h2d %llu MB -> %.2f GB/s (%.1f us)\n", dev.displayName.c_str(),
                (unsigned long long)(bytes >> 20), bps / 1.0e9, r.us);
    if (smallElems == 0)
    {
      smallElems = elems;
      h2dSmallUs = r.us;
    }
    if (bps > h2dBest * 1.03)
      h2dBest = bps;
    else
      break;   // the link saturated
  }
  if (h2dBest > 0.0)
    test.emit("h2d", (float)h2dBest, h2dNote);
  else
    test.skip("h2d", h2dStatus, h2dErr.empty() ? "no size could be measured" : h2dErr, h2dNote);

  if (smallElems > 0)
  {
    Run r = measure(rt, dev, plan, LitertTransfer::RoundTrip, smallElems, warmupCount, forceIters, specifiedIters);
    const uint64_t bytes = (uint64_t)smallElems * elem;
    if (r.us > 0.0)
    {
      const double bps = 2.0 * (double)bytes / (r.us * 1.0e-6);
      CLPEAK_VLOG("litert-transfer[%s]: roundtrip %llu MB -> %.2f GB/s (%.1f us)\n", dev.displayName.c_str(),
                  (unsigned long long)(bytes >> 20), bps / 1.0e9, r.us);
      test.emit("roundtrip", (float)bps, rtNote);
      const double backUs = r.us - h2dSmallUs;
      if (backUs > 0.05 * r.us)
        test.emit("d2h", (float)((double)bytes / (backUs * 1.0e-6)), d2hNote);
      else
        test.skip("d2h", ResultStatus::Error,
                  "the round trip took no longer than the trip out, so the return copy cannot "
                  "be separated from it -- on a unified-memory device the result is read in "
                  "place",
                  d2hNote);
    }
    else
    {
      test.skip("roundtrip", r.status, r.error.empty() ? "run failed" : r.error, rtNote);
      test.skip("d2h", r.status, "the round trip could not be measured", d2hNote);
    }
  }
  else
  {
    test.skip("roundtrip", h2dStatus, "no size could be measured", rtNote);
    test.skip("d2h", h2dStatus, "no size could be measured", d2hNote);
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
