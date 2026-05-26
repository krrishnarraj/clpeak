#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>
#include <chrono>

// H2D / D2H peak via queue.memcpy with pinned (USM host) staging buffer.
// Equivalent to hipMemcpyAsync between pinned host and device memory in the
// ROCm backend.

int OneapiPeak::runTransferBandwidth(OneapiDevice &dev, benchmark_config_t &cfg)
{
  const uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ull << 27);
  unsigned int forced = forceIters ? specifiedIters : 0;

  auto test = currentDeviceScope->beginTest(
    {"transfer_bandwidth", "Transfer bandwidth", "gbps"});

  void *dBuf    = sycl::malloc_device(bytes, dev.stream);
  void *hPinned = sycl::malloc_host(bytes, dev.stream);
  if (!dBuf || !hPinned)
  {
    test.skip("h2d_pinned", ResultStatus::Error, "Failed to allocate buffers");
    test.skip("d2h_pinned", ResultStatus::Error, "Failed to allocate buffers");
    if (dBuf)    sycl::free(dBuf, dev.stream);
    if (hPinned) sycl::free(hPinned, dev.stream);
    return -1;
  }

  populate((float *)hPinned, bytes / sizeof(float));

  auto timeXfer = [&](bool h2d) -> float
  {
    auto runBatch = [&](unsigned int n) -> float {
      try
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (unsigned i = 0; i < n; i++)
        {
          dev.stream.memcpy(h2d ? dBuf : hPinned,
                            h2d ? hPinned : dBuf,
                            bytes);
        }
        dev.stream.wait_and_throw();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return (float)((double)ns / 1000.0);
      }
      catch (const sycl::exception &)
      {
        return -1.0f;
      }
    };

    try
    {
      for (unsigned w = 0; w < warmupCount; w++)
        dev.stream.memcpy(h2d ? dBuf : hPinned, h2d ? hPinned : dBuf, bytes);
      dev.stream.wait_and_throw();
    }
    catch (const sycl::exception &) { return -1.0f; }

    float probeUs = runBatch(1);
    if (probeUs <= 0.0f) return -1.0f;
    unsigned int iters = pickIters((double)probeUs, cfg.targetTimeUs, forced);
    float totalUs = runBatch(iters);
    return totalUs > 0.0f ? totalUs / iters : -1.0f;
  };

  float usH2D = timeXfer(true);
  if (usH2D > 0.0f) test.emit("h2d_pinned", (float)bytes / usH2D / 1e3f);
  else              test.skip("h2d_pinned", ResultStatus::Error, "transfer failed");

  float usD2H = timeXfer(false);
  if (usD2H > 0.0f) test.emit("d2h_pinned", (float)bytes / usD2H / 1e3f);
  else              test.skip("d2h_pinned", ResultStatus::Error, "transfer failed");

  sycl::free(hPinned, dev.stream);
  sycl::free(dBuf,    dev.stream);
  return 0;
}

#endif // ENABLE_ONEAPI
