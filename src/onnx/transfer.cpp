#ifdef ENABLE_ONNX

// onnx-transfer-bw: what it costs to get data to the accelerator and back.
//
// Every other test in this backend is built to keep tensors resident,
// because otherwise a discrete GPU is measured through its PCIe bus rather
// than its arithmetic -- a mistake that cost this backend two rounds of
// wrong numbers before it was caught.  This test measures that cost head-on
// instead of designing around it.
//
// It matters because vendors never quote it and it decides whether
// offloading is worth doing at all: an accelerator that computes ten times
// faster than the host is no help if reaching it costs more than the work
// saved.  On unified-memory devices there is no transfer to speak of and
// these rows read close to memory speed, which is itself the answer.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <common/console_mute.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr int64_t  kMinElems       = 8ll << 20;    // 16 MB of fp16
// Transfer rate saturates early -- a link is a fixed number of bytes per
// second, not something that improves with problem size -- so unlike the
// compute sweeps this ceiling is a safety bound rather than a measurement
// limit.  The improvement rule normally stops well before it.
constexpr uint64_t kMaxTensorBytes = 128ull << 20;
constexpr double   kImproveFactor  = 1.03;
constexpr int      kMaxStrikes     = 2;
constexpr unsigned int kSizeBudgetUs = 300000;

struct Run
{
  double us = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Run measure(const OrtRuntime &rt, const onnx_ep_info_t &ep,
            OnnxTransfer dir, int64_t elems,
            unsigned int warmup, bool forceIters, unsigned int forced)
{
  Run r;
  const bool bigIn  = (dir != OnnxTransfer::FromDevice);
  const bool bigOut = (dir == OnnxTransfer::FromDevice ||
                       dir == OnnxTransfer::RoundTrip);

  OrtSession *session = nullptr;
  {
    std::string model = onnxTransferModel(dir, elems, std::string());

    auto ses = onnxCreateSession(rt, ep, model);
    model.clear(); model.shrink_to_fit();
    if (!ses.session)
    {
      r.error  = ses.error;
      r.status = ResultStatus::Unsupported;
      return r;
    }
    session = ses.session;
  }

  std::vector<uint16_t> inBuf((size_t)(bigIn ? elems : 1), floatToHalf(0.5f));
  std::vector<uint16_t> outBuf((size_t)(bigOut ? elems : 1), 0);

  OrtMemoryInfo *mi = nullptr;
  OrtValue *inVal = nullptr, *outVal = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  const int64_t big[1] = {elems};
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, inBuf.data(), inBuf.size() * 2, bigIn ? big : nullptr,
        bigIn ? 1 : 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, outBuf.data(), outBuf.size() * 2, bigOut ? big : nullptr,
        bigOut ? 1 : 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &outVal);
  if (mi) rt.api->ReleaseMemoryInfo(mi);

  const char *inName  = (dir == OnnxTransfer::FromDevice) ? "S" : "X";
  auto run = [&](unsigned int n) -> double {
    const char *ins[]  = {inName};
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

  // Some providers compile lazily on the first run and complain to the
  // console when they cannot -- the ANE refuses this graph shape out loud.
  // The status still tells us what happened.
  double warmed = 0.0;
  {
    clpeak::ScopedConsoleMute mute;
    warmed = run(1 + warmup);
  }
  if (!st && warmed > 0.0)
  {
    double probe = run(3);
    if (probe > 0.0)
      r.us = run(pickIters(probe, kSizeBudgetUs, forceIters ? forced : 0));
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

int OnnxPeak::runTransferBandwidth(const OrtRuntime &rt,
                                   const onnx_ep_info_t &ep,
                                   benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx-transfer-bw", "ONNX host transfer bandwidth", "gbps",
       Category::Bandwidth,
       "How fast data reaches this provider and comes back.  Every other test "
       "here keeps its tensors on the device on purpose, because otherwise a "
       "discrete accelerator gets measured through the cable rather than its "
       "arithmetic.  This measures that cable.  It decides whether offloading "
       "is worth doing at all -- an accelerator ten times faster than the host "
       "is no help if reaching it costs more than the work saved -- and "
       "vendors never quote it.  Devices that share memory with the CPU have "
       "no real transfer and read close to memory speed here."});

  // ---- Trip out: swept, since the rate varies with size until it saturates
  double  bestGbps = 0.0, bestUs = 0.0;
  int64_t bestElems = 0;
  // The smallest rung is kept separately: the round trip is measured there
  // and the two have to be compared at one size.
  double  firstUs = 0.0;
  int64_t firstElems = 0;
  int     strikes = 0;
  std::string firstErr;
  ResultStatus errStatus = ResultStatus::Unsupported;

  for (int64_t elems = kMinElems;; elems *= 2)
  {
    if (clpeak::cancelRequested())
      break;
    if ((uint64_t)elems * 2ull > kMaxTensorBytes)
      break;

    Run r = measure(rt, ep, OnnxTransfer::ToDevice, elems, warmupCount,
                    forceIters, specifiedIters);
    if (r.us <= 0.0)
    {
      if (firstErr.empty())
      {
        firstErr  = r.error.empty() ? "run failed" : r.error;
        errStatus = r.status;
      }
      break;
    }

    if (firstElems == 0) { firstUs = r.us; firstElems = elems; }

    const double gbps = (double)elems * 2.0 / (r.us * 1.0e-6) / 1.0e9;
    CLPEAK_VLOG("onnx-transfer[%s/h2d]: %lld MB -> %.1f GB/s\n",
                ep.providerKey.c_str(), (long long)((elems * 2) >> 20), gbps);

    if (gbps > bestGbps * kImproveFactor)
    {
      strikes = 0;
      bestGbps = gbps; bestUs = r.us; bestElems = elems;
    }
    else
    {
      if (gbps > bestGbps) { bestGbps = gbps; bestUs = r.us; bestElems = elems; }
      if (++strikes >= kMaxStrikes)
        break;
    }
  }

  if (bestGbps > 0.0)
    test.emit("h2d", (float)bestGbps,
              "Host to device: the tensor is handed over and a single value "
              "comes back, so almost all of the time is the trip out.");
  else
    test.skip("h2d", errStatus, firstErr.empty() ? "unsupported" : firstErr,
              "Host to device: the tensor is handed over and a single value "
              "comes back.");

  // ---- Round trip: one measurement, at the smallest rung.
  //
  // Not swept, and deliberately at the small end.  A link saturates rather
  // than improving with size, so a sweep buys nothing -- and the compilation
  // this graph provokes does not scale gently: the ANE compiles a 16 MB
  // elementwise graph in seconds and a 64 MB one in more than ten minutes.
  double roundUs = 0.0;
  if (firstElems > 0 && !clpeak::cancelRequested())
  {
    Run r = measure(rt, ep, OnnxTransfer::RoundTrip, firstElems, warmupCount,
                    forceIters, specifiedIters);
    if (r.us > 0.0)
    {
      roundUs = r.us;
      const double gbps = 2.0 * (double)firstElems * 2.0 / (r.us * 1.0e-6) / 1.0e9;
      CLPEAK_VLOG("onnx-transfer[%s/roundtrip]: %lld MB -> %.1f GB/s\n",
                  ep.providerKey.c_str(),
                  (long long)((firstElems * 2) >> 20), gbps);
      test.emit("roundtrip", (float)gbps,
                "The full cost of offloading: sending a tensor, applying one "
                "trivial operation to it, and getting the result back.  This "
                "is the bar any offloaded work has to clear -- a calculation "
                "the host finishes faster than this number moves the data is "
                "not worth sending away.  It includes one pass over the data "
                "on the device, so on hardware that is slow at simple "
                "elementwise work (compare the activation rows) that pass, "
                "rather than the transfer, is what this measures.");
    }
    else
    {
      test.skip("roundtrip", r.status,
                r.error.empty() ? "run failed" : r.error,
                "Both directions, which is what an offloaded operation "
                "actually pays.");
    }
  }
  else
  {
    test.skip("roundtrip", ResultStatus::Error, "no usable size",
              "Both directions, which is what an offloaded operation pays.");
  }

  // The return journey is deliberately not reported.
  //
  // Isolating it needs a third graph -- everything the round trip does except
  // shipping the result back -- because subtracting the trip out would leave
  // the elementwise pass in the answer, and that pass is not always cheap:
  // the ANE applies a pointwise operation at about a seventh of the rate it
  // reads memory (see onnx-activation), which made a naive difference read
  // four times too slow.  That third graph turned out to cost more than it is
  // worth: Core ML compiles ahead of time and took over twenty minutes on it.
  //
  // Two honest rows beat three with one wrong.  Where the return trip matters,
  // the round trip minus the provider's own elementwise rate from
  // onnx-activation gives it, on the same hardware, for free.

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
