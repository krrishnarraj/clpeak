#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runTransferBandwidth(RocmDevice &dev, benchmark_config_t &cfg)
{
  const uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ull << 27);
  unsigned int forced = forceIters ? specifiedIters : 0;

  auto test = currentDeviceScope->beginTest(
    {"transfer_bandwidth", "Transfer bandwidth", "gbps", Category::Unknown,
     "How fast data crosses between the computer's own memory and the card's, "
     "over the PCIe link.  That link is far narrower than either side's own "
     "memory, which is what makes moving data to the GPU worth avoiding.  Both "
     "readings use pinned host memory, the fast path.",
     TestShape::Heterogeneous, "direction"});

  const char *h2dNote = "Host to device: sending data up to the card.";
  const char *d2hNote = "Device to host: reading results back down.  Often a "
                        "little slower than the other direction.";

  void *dBuf = nullptr;
  if (hipMalloc(&dBuf, bytes) != hipSuccess)
  {
    test.skip("h2d_pinned", ResultStatus::Error, "Failed to allocate device buffer", h2dNote);
    test.skip("d2h_pinned", ResultStatus::Error, "Failed to allocate device buffer", d2hNote);
    return -1;
  }
  void *hPinned = nullptr;
  if (hipHostMalloc(&hPinned, bytes) != hipSuccess)
  {
    (void)hipFree(dBuf);
    test.skip("h2d_pinned", ResultStatus::Error, "Failed to allocate pinned host buffer", h2dNote);
    test.skip("d2h_pinned", ResultStatus::Error, "Failed to allocate pinned host buffer", d2hNote);
    return -1;
  }

  populate((float *)hPinned, bytes / sizeof(float));

  auto timeXfer = [&](bool h2d) -> float
  {
    hipEvent_t s = nullptr, e = nullptr;
    (void)hipEventCreate(&s);
    (void)hipEventCreate(&e);

    auto runBatch = [&](unsigned int n) -> float {
      (void)hipEventRecord(s, dev.stream);
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
      (void)hipEventRecord(e, dev.stream);
      if (hipEventSynchronize(e) != hipSuccess)
        return -1.0f;
      float ms = 0.0f;
      (void)hipEventElapsedTime(&ms, s, e);
      return ms * 1000.0f;
    };

    for (unsigned w = 0; w < warmupCount; w++)
    {
      (void)hipMemcpyAsync(h2d ? dBuf : hPinned,
                           h2d ? hPinned : dBuf,
                           bytes,
                           h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost,
                           dev.stream);
      (void)hipStreamSynchronize(dev.stream);
    }

    float probeUs = runBatch(1);
    if (probeUs <= 0.0f)
    {
      (void)hipEventDestroy(s);
      (void)hipEventDestroy(e);
      return -1.0f;
    }
    unsigned int iters = pickIters((double)probeUs, cfg.targetTimeUs, forced);
    float totalUs = runBatch(iters);

    (void)hipEventDestroy(s);
    (void)hipEventDestroy(e);
    return totalUs > 0.0f ? totalUs / iters : -1.0f;
  };

  float usH2D = timeXfer(true);
  if (usH2D > 0.0f)
    test.emit("h2d_pinned", (float)bytes / usH2D / 1e3f, h2dNote);
  else
    test.skip("h2d_pinned", ResultStatus::Error, "transfer failed", h2dNote);

  float usD2H = timeXfer(false);
  if (usD2H > 0.0f)
    test.emit("d2h_pinned", (float)bytes / usD2H / 1e3f, d2hNote);
  else
    test.skip("d2h_pinned", ResultStatus::Error, "transfer failed", d2hNote);

  (void)hipHostFree(hPinned);
  (void)hipFree(dBuf);
  return 0;
}

#endif // ENABLE_ROCM
