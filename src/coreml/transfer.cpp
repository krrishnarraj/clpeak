#ifdef ENABLE_COREML

// coreml-transfer-bw: what it costs to get a tensor to this compute unit
// and a result back.
//
// Every other test here keeps its tensors resident, because a model input
// is copied into whatever memory the compute unit reads from -- on the
// Neural Engine an IOSurface the driver owns -- and that copy would land in
// every throughput figure.  This test measures the handover head-on.  All
// three units share the host's memory, so what the rows show is the
// framework's own cost of presenting a tensor to each, which is real: an
// MLMultiArray reaches the Neural Engine through a copy, not a pointer.
//
// It cannot be done with a trivial graph, the way the ONNX backend does it
// (src/onnx/transfer.cpp): Core ML's planner keeps trivial work on the CPU
// whatever unit was asked for, so a graph that only slices its input would
// measure the CPU under every device name.  Instead the same fp16 matmul --
// heavy enough to be placed -- is built three ways that differ only in what
// crosses the boundary, and the transfers are the differences between them:
//
//   resident   x constant, one row of the result     nothing crosses
//   to-device  x a model input, one row of the result x crosses, out
//   round trip x a model input, whole result out      x out and y back
//
// The matmul's own time cancels in every subtraction.  The kept row is a
// slice, not a reduction: a reduce over 8192 rows is a pass the Neural
// Engine runs at ~20 GB/s, which does not cancel against the round trip's
// graph (it has none) and outweighed the copy back on macOS 27.  Sizes
// double from 16 MB; the trip out is reported at the largest size (a
// runtime may pass small tensors by pointer and copy only big ones), the
// round trip and the trip back at the smallest.  The M1 Pro's Neural Engine
// declines a live [32768, 1024] input, so its ladder ends at 32 MB.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

#include <cstring>
#include <string>
#include <vector>

namespace
{

// x[rows, kK] * W[kK, kN]: 2 KB per row in and out.
constexpr int64_t kK = 1024;
constexpr int64_t kN = 1024;
constexpr int64_t kMinRows = 8192;   // 16 MB of fp16
uint64_t maxTensorBytes() { return clpeak::memoryBudget(128ull << 20); }
constexpr unsigned int kSizeBudgetUs = 500000;

// A resident graph whose time is less than this share of the input-fed one
// computed nothing: its constants were folded, and the difference would be
// the whole multiply rather than the transfer.
constexpr double kFoldShare = 0.25;

struct Run
{
  double us = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Run measure(const coreml_device_info_t &dev, int spec, CoremlTransfer dir, int64_t rows,
            bool resultScaled, unsigned warmup, bool forceIters, unsigned forced)
{
  Run r;
  std::string err;
  auto s = CoremlSession::create(dev, coremlTransferModel(spec, dir, rows, kK, kN, resultScaled), err);
  if (!s)
  {
    r.error = err;
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (!s->onDevice())
  {
    r.error = coremlOffDeviceReason(dev, *s);
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (dir == CoremlTransfer::Resident)
  {
    if (!coremlBindScalar(*s, "s", CML_FP16, err))
    {
      r.error = err;
      r.status = ResultStatus::Error;
      return r;
    }
  }
  else
  {
    void *x = s->bindInput("x", CML_FP16, {rows, kK}, (size_t)rows * kK * 2, err);
    if (!x)
    {
      r.error = err;
      r.status = ResultStatus::Error;
      return r;
    }
    // Pseudo-random bit patterns: a copy of predictable data could be
    // compressed on the way, and the row is about bytes moved.
    uint16_t *p = static_cast<uint16_t *>(x);
    uint32_t seed = 0x9e3779b9u;
    for (int64_t i = 0; i < rows * kK; i++)
    {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      p[i] = coremlFloatToHalf((float)(seed >> 8) / 16777216.0f - 0.5f);
    }
  }
  auto m = coremlMeasure(*s, warmup, kSizeBudgetUs, forceIters, forced);
  if (m.meanUs <= 0.0)
  {
    r.error = m.error;
    r.status = m.status;
    return r;
  }
  r.us = m.meanUs;
  return r;
}

} // namespace

int CoreMLPeak::runTransferBandwidth(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_transfer_bw", "Core ML input transfer bandwidth", "bps", Category::Bandwidth,
       "How fast a tensor handed to Core ML reaches this compute unit and how "
       "fast a result comes back -- the copy every other test here keeps its "
       "tensors resident to avoid.  Measured as the difference between one "
       "matrix multiply built with its input resident, with it handed in, and "
       "with its result handed back, so the arithmetic cancels.  All three "
       "units share the host's memory: this is the framework's cost of "
       "presenting a tensor, which for the Neural Engine is a real copy.",
       TestShape::Heterogeneous, "direction"});

  // The sizes are named in the rows because they differ: the trip out is
  // reported at the largest size and the round trip at the smallest, so a
  // reader sees the round trip outrun the trip out (47.6 against 14.7 GB/s
  // on an M1 Pro's Neural Engine) and should not take it as one subtraction
  // from the other.
  auto mb = [](int64_t rows, int64_t width) {
    return std::to_string(((uint64_t)rows * (uint64_t)width * 2) >> 20) + " MB";
  };
  std::string h2dNote =
      "Host to device: the input-fed multiply less the resident one, so what is "
      "left is the trip out.  Reported at the largest size measured";
  std::string d2hNote =
      "Device to host: the multiply that returns its whole result less the one that "
      "reduces it on the device.  What is left is the return journey alone, reported "
      "at the largest size measured";
  std::string rtNote =
      "Both transfers of an offloaded operation, in and out, less the arithmetic "
      "between them -- the bar any offloaded work has to clear.  Measured at the "
      "smallest size, where the trip out is reported at the largest, so the two rows "
      "are not one subtraction apart";

  double lastH2dBps = 0.0;
  double firstResidentUs = 0.0, firstToDeviceUs = 0.0;
  double lastResidentUs = 0.0, lastToDeviceUs = 0.0;
  int64_t firstRows = 0, lastRows = 0;
  std::string firstErr;
  ResultStatus errStatus = ResultStatus::Unsupported;

  for (int64_t rows = kMinRows;; rows *= 2)
  {
    if (clpeak::cancelRequested())
      break;
    const uint64_t bytes = (uint64_t)rows * kK * 2;
    if (bytes > maxTensorBytes())
      break;

    Run res = measure(dev, spec, CoremlTransfer::Resident, rows, true, warmupCount, forceIters, specifiedIters);
    Run in = measure(dev, spec, CoremlTransfer::ToDevice, rows, true, warmupCount, forceIters, specifiedIters);
    if (res.us > 0.0 && in.us > 0.0 && res.us < kFoldShare * in.us)
    {
      // The reduced resident form was folded; the live-operand one cannot be.
      CLPEAK_VLOG("coreml-transfer[%s]: resident %.0f us against %.0f us input-fed -- folded, "
                  "retrying the operand-scaled form\n",
                  dev.displayName.c_str(), res.us, in.us);
      res = measure(dev, spec, CoremlTransfer::Resident, rows, false, warmupCount, forceIters, specifiedIters);
    }
    if (res.us <= 0.0 || in.us <= 0.0)
    {
      const Run &bad = res.us <= 0.0 ? res : in;
      CLPEAK_VLOG("coreml-transfer[%s/h2d]: %lld MB: %s form failed: %s\n",
                  dev.displayName.c_str(), (long long)(bytes >> 20),
                  res.us <= 0.0 ? "resident" : "input-fed",
                  bad.error.empty() ? "run failed" : bad.error.c_str());
      if (firstErr.empty())
      {
        firstErr = bad.error.empty() ? "run failed" : bad.error;
        errStatus = bad.status;
      }
      break;
    }
    const double tripUs = in.us - res.us;
    CLPEAK_VLOG("coreml-transfer[%s/h2d]: %lld MB: input-fed %.0f us, resident %.0f us\n",
                dev.displayName.c_str(), (long long)(bytes >> 20), in.us, res.us);
    if (firstRows == 0)
    {
      firstRows = rows;
      firstResidentUs = res.us;
      firstToDeviceUs = in.us;
    }
    lastRows = rows;
    lastResidentUs = res.us;
    lastToDeviceUs = in.us;
    if (tripUs > 0.0)
      lastH2dBps = (double)bytes / (tripUs * 1.0e-6);
  }

  if (lastRows > 0)
    h2dNote += " (" + mb(lastRows, kK) + ")";
  h2dNote += ".";
  if (firstRows > 0)
    rtNote += " (" + mb(firstRows, kK) + " in, " + mb(firstRows, kN) + " out)";
  rtNote += ".";
  if (lastRows > 0)
    d2hNote += " (" + mb(lastRows, kN) + ")";
  d2hNote += ".";

  if (lastH2dBps > 0.0)
    test.emit("h2d", (float)lastH2dBps, h2dNote.c_str());
  else if (firstRows > 0)
    test.skip("h2d", ResultStatus::Error,
              "handing the input over cost no more than holding it resident, so no copy "
              "could be measured",
              h2dNote);
  else
    test.skip("h2d", errStatus, firstErr.empty() ? "unsupported" : firstErr, h2dNote);

  if (firstRows > 0 && !clpeak::cancelRequested())
  {
    Run rt = measure(dev, spec, CoremlTransfer::RoundTrip, firstRows, true, warmupCount, forceIters, specifiedIters);
    const uint64_t inBytes = (uint64_t)firstRows * kK * 2;
    const uint64_t outBytes = (uint64_t)firstRows * kN * 2;
    CLPEAK_VLOG("coreml-transfer[%s/roundtrip]: %.0f us (input-fed %.0f, resident %.0f)\n",
                dev.displayName.c_str(), rt.us, firstToDeviceUs, firstResidentUs);
    if (rt.us <= 0.0)
    {
      test.skip("roundtrip", rt.status, rt.error.empty() ? "run failed" : rt.error, rtNote);
      test.skip("d2h", ResultStatus::Error, "the round trip it is derived from could not be measured", d2hNote);
    }
    else
    {
      if (rt.us > firstResidentUs)
        test.emit("roundtrip", (float)((double)(inBytes + outBytes) / ((rt.us - firstResidentUs) * 1.0e-6)), rtNote.c_str());
      else
        test.skip("roundtrip", ResultStatus::Error,
                  "moving the tensors cost no more than keeping them resident", rtNote);

      // Does returning the result copy anything?  Core ML can hand an
      // output back in the buffer the unit wrote -- the Neural Engine's
      // 16 MB result came back in 15 us, a terabyte a second of nothing --
      // so the trip back is measured at the largest size too and has to
      // have grown with it, and has to be resolvable beside the trip out
      // at that size: the difference of two nearly equal times is noise
      // around zero, and 3 us against 51 us once passed the growth test and
      // published 659 GB/s for a return that costs nothing.  A twentieth of
      // the trip out keeps the GPU's real copy (1.9 ms back against 4.7 ms
      // out for 128 MB) and drops the Neural Engine's nothing (-12 us
      // against 2 ms).  Where it did grow, the largest size is reported,
      // as for the trip out.
      constexpr double kMinReturnShare = 0.05;
      double d2hFirst = rt.us - firstToDeviceUs;
      double d2hBps = d2hFirst > 0.0 ? (double)outBytes / (d2hFirst * 1.0e-6) : 0.0;
      bool copies = d2hFirst >= kMinReturnShare * (firstToDeviceUs - firstResidentUs);
      if (lastRows > firstRows && !clpeak::cancelRequested())
      {
        Run rtLast = measure(dev, spec, CoremlTransfer::RoundTrip, lastRows, true, warmupCount, forceIters,
                             specifiedIters);
        const double d2hLast = rtLast.us > 0.0 ? rtLast.us - lastToDeviceUs : -1.0;
        CLPEAK_VLOG("coreml-transfer[%s/d2h]: %.0f us at %lld MB, %.0f us at %lld MB\n",
                    dev.displayName.c_str(), d2hFirst, (long long)(outBytes >> 20), d2hLast,
                    (long long)(((uint64_t)lastRows * kN * 2) >> 20));
        copies = d2hLast > 0.0 && d2hFirst > 0.0 && d2hLast > d2hFirst * 2.0 &&
                 d2hLast >= kMinReturnShare * (lastToDeviceUs - lastResidentUs);
        if (copies)
          d2hBps = (double)((uint64_t)lastRows * kN * 2) / (d2hLast * 1.0e-6);
      }
      if (copies && d2hBps > 0.0)
        test.emit("d2h", (float)d2hBps, d2hNote.c_str());
      else if (!copies)
        test.skip("d2h", ResultStatus::Unsupported,
                  "the result is handed back without a copy -- the time did not grow with its size",
                  d2hNote);
      else
        test.skip("d2h", ResultStatus::Error,
                  "returning the result cost no more than keeping it on device", d2hNote);
    }
  }
  else
  {
    test.skip("roundtrip", errStatus, firstErr.empty() ? "no usable size" : firstErr, rtNote);
    test.skip("d2h", errStatus, firstErr.empty() ? "no usable size" : firstErr, d2hNote);
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
