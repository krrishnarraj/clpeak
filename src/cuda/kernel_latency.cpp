#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

int CudaPeak::runKernelLatency(CudaDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = forceIters ? specifiedIters
                                  : (cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000);

  auto test = currentDeviceScope->beginTest(
    {"kernel_launch_latency", "Kernel launch latency", "us"});

  CUfunction fn;
  if (!dev.getKernel(cuda_kernels::kernel_latency_src,
                     cuda_kernels::kernel_latency_name,
                     "kernel_latency_noop", fn))
  {
    test.skip("dispatch", ResultStatus::Error, "Kernel compile failed");
    test.skip("roundtrip", ResultStatus::Error, "Kernel compile failed");
    return -1;
  }

  void *args[1] = {nullptr};

  // CUDA's driver API has no primitive equivalent to OpenCL's QUEUED -> START
  // profiling info or VK_EXT_calibrated_timestamps -- cuEventRecord captures
  // GPU stream-time, with no portable way to project a host timestamp into
  // the same domain.  So we report only the round-trip metric here, leaving
  // dispatch latency as "not measurable" rather than a misleading estimate.

  bool submitFailed = false;

  // Warmup
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    CUresult lr = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, dev.stream, args, nullptr);
    CUresult sr = cuStreamSynchronize(dev.stream);
    if (lr != CUDA_SUCCESS || sr != CUDA_SUCCESS) { submitFailed = true; break; }
  }

  double totalRoundtripUs = 0;
  if (!submitFailed)
  {
    for (unsigned int i = 0; i < iters; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      CUresult lr = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, dev.stream, args, nullptr);
      CUresult sr = cuStreamSynchronize(dev.stream);
      auto t1 = std::chrono::high_resolution_clock::now();
      if (lr != CUDA_SUCCESS || sr != CUDA_SUCCESS) { submitFailed = true; break; }
      totalRoundtripUs += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    }
  }

  test.skip("dispatch", ResultStatus::Unsupported,
            "Not measurable via CUDA driver API");
  if (submitFailed)
  {
    test.skip("roundtrip", ResultStatus::Error,
              "cuLaunchKernel/cuStreamSynchronize failed");
  }
  else
  {
    float roundtripUs = (float)(totalRoundtripUs / iters);
    test.emit("roundtrip", roundtripUs);
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Local memory bandwidth (CUDA -- __shared__ memory)
// ---------------------------------------------------------------------------


#endif // ENABLE_CUDA
