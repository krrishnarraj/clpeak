#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>
#include <chrono>

int CudaPeak::runTransferBandwidth(CudaDevice &dev, benchmark_config_t &cfg)
{
  const uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ull << 27);
  unsigned int forced = forceIters ? specifiedIters : 0;

  auto test = currentDeviceScope->beginTest(
    {"transfer_bandwidth", "Transfer bandwidth", "gbps"});

  CUdeviceptr dBuf = 0;
  if (cuMemAlloc(&dBuf, bytes) != CUDA_SUCCESS)
  {
    test.skip("h2d_pinned", ResultStatus::Error, "Failed to allocate device buffer");
    test.skip("d2h_pinned", ResultStatus::Error, "Failed to allocate device buffer");
    return -1;
  }
  void *hPinned = nullptr;
  if (cuMemAllocHost(&hPinned, bytes) != CUDA_SUCCESS)
  {
    cuMemFree(dBuf);
    test.skip("h2d_pinned", ResultStatus::Error, "Failed to allocate pinned host buffer");
    test.skip("d2h_pinned", ResultStatus::Error, "Failed to allocate pinned host buffer");
    return -1;
  }

  // Fill host buffer with pseudo-random data to defeat hardware memory
  // compression on both H2D and D2H paths.
  populate((float *)hPinned, bytes / sizeof(float));

  auto timeXfer = [&](bool h2d) -> float
  {
    CUevent s, e;
    cuEventCreate(&s, CU_EVENT_DEFAULT);
    cuEventCreate(&e, CU_EVENT_DEFAULT);

    auto runBatch = [&](unsigned int n) -> float {
      cuEventRecord(s, dev.stream);
      for (unsigned i = 0; i < n; i++)
      {
        if (h2d)
          cuMemcpyHtoDAsync(dBuf, hPinned, bytes, dev.stream);
        else
          cuMemcpyDtoHAsync(hPinned, dBuf, bytes, dev.stream);
      }
      cuEventRecord(e, dev.stream);
      cuEventSynchronize(e);
      float ms = 0;
      cuEventElapsedTime(&ms, s, e);
      return ms * 1000.0f; // total us
    };

    // Phase 1: untimed warmup.
    for (unsigned w = 0; w < warmupCount; w++)
    {
      if (h2d)
        cuMemcpyHtoDAsync(dBuf, hPinned, bytes, dev.stream);
      else
        cuMemcpyDtoHAsync(hPinned, dBuf, bytes, dev.stream);
      cuStreamSynchronize(dev.stream);
    }

    // Phase 2: timed probe -> per-iter time -> calibrated iters.
    unsigned int probeIters = 1;
    float probeUs = runBatch(probeIters);
    double per_iter_us = (double)probeUs / (double)probeIters;
    unsigned int iters = pickIters(per_iter_us, cfg.targetTimeUs, forced);

    // Phase 3: real timed run.
    float totalUs = runBatch(iters);

    cuEventDestroy(s);
    cuEventDestroy(e);
    return totalUs / iters; // microseconds per transfer
  };

  float usH2D = timeXfer(true);
  float gbpsH2D = (float)bytes / usH2D / 1e3f;
  test.emit("h2d_pinned", gbpsH2D);

  float usD2H = timeXfer(false);
  float gbpsD2H = (float)bytes / usD2H / 1e3f;
  test.emit("d2h_pinned", gbpsD2H);

  cuMemFreeHost(hPinned);
  cuMemFree(dBuf);
  return 0;
}

// ---------------------------------------------------------------------------
// Kernel launch latency (CUDA)
// ---------------------------------------------------------------------------


#endif // ENABLE_CUDA
