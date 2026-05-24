#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runTransferBandwidth(RocmDevice &dev, benchmark_config_t &cfg)
{
  const uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ull << 27);
  unsigned int forced = forceIters ? specifiedIters : 0;

  auto test = currentDeviceScope->beginTest(
    {"transfer_bandwidth", "Transfer bandwidth", "gbps"});

  void *dBuf = nullptr;
  if (hipMalloc(&dBuf, bytes) != hipSuccess)
  {
    test.skip("h2d_pinned", ResultStatus::Error, "Failed to allocate device buffer");
    test.skip("d2h_pinned", ResultStatus::Error, "Failed to allocate device buffer");
    return -1;
  }
  void *hPinned = nullptr;
  if (hipHostMalloc(&hPinned, bytes) != hipSuccess)
  {
    hipFree(dBuf);
    test.skip("h2d_pinned", ResultStatus::Error, "Failed to allocate pinned host buffer");
    test.skip("d2h_pinned", ResultStatus::Error, "Failed to allocate pinned host buffer");
    return -1;
  }

  populate((float *)hPinned, bytes / sizeof(float));

  auto timeXfer = [&](bool h2d) -> float
  {
    hipEvent_t s = nullptr, e = nullptr;
    hipEventCreate(&s);
    hipEventCreate(&e);

    auto runBatch = [&](unsigned int n) -> float {
      hipEventRecord(s, dev.stream);
      hipError_t status = hipSuccess;
      for (unsigned i = 0; i < n; i++)
      {
        status = hipMemcpyAsync(h2d ? dBuf : hPinned,
                                h2d ? hPinned : dBuf,
                                bytes,
                                h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost,
                                dev.stream);
        if (status != hipSuccess)
          break;
      }
      if (status != hipSuccess)
        return -1.0f;
      hipEventRecord(e, dev.stream);
      if (hipEventSynchronize(e) != hipSuccess)
        return -1.0f;
      float ms = 0.0f;
      hipEventElapsedTime(&ms, s, e);
      return ms * 1000.0f;
    };

    for (unsigned w = 0; w < warmupCount; w++)
    {
      hipMemcpyAsync(h2d ? dBuf : hPinned,
                     h2d ? hPinned : dBuf,
                     bytes,
                     h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost,
                     dev.stream);
      hipStreamSynchronize(dev.stream);
    }

    float probeUs = runBatch(1);
    if (probeUs <= 0.0f)
    {
      hipEventDestroy(s);
      hipEventDestroy(e);
      return -1.0f;
    }
    unsigned int iters = pickIters((double)probeUs, cfg.targetTimeUs, forced);
    float totalUs = runBatch(iters);

    hipEventDestroy(s);
    hipEventDestroy(e);
    return totalUs > 0.0f ? totalUs / iters : -1.0f;
  };

  float usH2D = timeXfer(true);
  if (usH2D > 0.0f)
    test.emit("h2d_pinned", (float)bytes / usH2D / 1e3f);
  else
    test.skip("h2d_pinned", ResultStatus::Error, "transfer failed");

  float usD2H = timeXfer(false);
  if (usD2H > 0.0f)
    test.emit("d2h_pinned", (float)bytes / usD2H / 1e3f);
  else
    test.skip("d2h_pinned", ResultStatus::Error, "transfer failed");

  hipHostFree(hPinned);
  hipFree(dBuf);
  return 0;
}

#endif // ENABLE_ROCM
