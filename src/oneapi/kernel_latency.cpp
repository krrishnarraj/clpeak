#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>
#include <chrono>

class kernel_latency_noop;

// Host-observed empty-kernel roundtrip: submit + wait per iter.  Matches the
// ROCm "roundtrip" metric.  We don't try to break out the pure-dispatch
// component because SYCL doesn't expose it consistently across L0/OpenCL.

int OneapiPeak::runKernelLatency(OneapiDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = forceIters ? specifiedIters
                                  : (cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000);

  auto test = currentDeviceScope->beginTest(
    {"kernel_launch_latency", "Kernel launch latency", "us"});

  bool submitFailed = false;

  auto launchNoop = [&]() {
    dev.stream.submit([&](sycl::handler &h) {
      h.parallel_for<kernel_latency_noop>(sycl::range<1>(1), [=](sycl::id<1>) {});
    });
  };

  try
  {
    for (unsigned int w = 0; w < warmupCount; w++)
      launchNoop();
    dev.stream.wait_and_throw();
  }
  catch (const sycl::exception &e)
  {
    CLPEAK_VLOG("SYCL kernel-latency warmup failed: %s\n", e.what());
    submitFailed = true;
  }

  double totalRoundtripUs = 0.0;
  if (!submitFailed)
  {
    for (unsigned int i = 0; i < iters; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      try { launchNoop(); dev.stream.wait_and_throw(); }
      catch (const sycl::exception &e)
      {
        CLPEAK_VLOG("SYCL kernel-latency submit failed: %s\n", e.what());
        submitFailed = true;
        break;
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      totalRoundtripUs += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    }
  }

  // SYCL doesn't expose a dispatch-only timer comparable to e.g. Metal's
  // GPUStartTime / GPUEndTime fields, so we report only the roundtrip and
  // skip the dispatch metric — consistent with how the ROCm backend reports.
  test.skip("dispatch", ResultStatus::Unsupported,
            "Not measurable via SYCL runtime");
  if (submitFailed)
    test.skip("roundtrip", ResultStatus::Error, "SYCL submit/wait failed");
  else
    test.emit("roundtrip", (float)(totalRoundtripUs / iters));

  return 0;
}

#endif // ENABLE_ONEAPI
