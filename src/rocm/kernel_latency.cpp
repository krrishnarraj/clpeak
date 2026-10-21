#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <chrono>

int RocmPeak::runKernelLatency(RocmDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = forceIters ? specifiedIters
                                  : (cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000);

  auto test = currentDeviceScope->beginTest(
    {"kernel_launch_latency", "Kernel launch latency", "s", Category::Unknown,
     "Time to launch an empty kernel, the fixed cost every dispatch pays.",
     TestShape::Heterogeneous});

  // dispatch is always skipped here: HIP exposes no clock the host and GPU
  // share, so the one-way time cannot be taken.
  const char *dispatchNote = "Host submit to kernel start, one way.";
  const char *roundtripNote = "Host submit to completion seen by the host.";

  RocmKernel k = dev.getKernel(rocm_kernels::kernel_latency, "kernel_latency_noop");
  if (!k)
  {
    test.skip("dispatch", k.status, k.reason, dispatchNote);
    test.skip("roundtrip", k.status, k.reason, roundtripNote);
    return -1;
  }
  hipFunction_t fn = k.fn;

  void *args[1] = {nullptr};
  bool submitFailed = false;

  for (unsigned int w = 0; w < warmupCount; w++)
  {
    hipError_t lr = hipModuleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, dev.stream, args, nullptr);
    hipError_t sr = hipStreamSynchronize(dev.stream);
    if (lr != hipSuccess || sr != hipSuccess) { submitFailed = true; break; }
  }

  double totalRoundtripUs = 0.0;
  if (!submitFailed)
  {
    for (unsigned int i = 0; i < iters; i++)
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      hipError_t lr = hipModuleLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, dev.stream, args, nullptr);
      hipError_t sr = hipStreamSynchronize(dev.stream);
      auto t1 = std::chrono::high_resolution_clock::now();
      if (lr != hipSuccess || sr != hipSuccess) { submitFailed = true; break; }
      totalRoundtripUs += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    }
  }

  test.skip("dispatch", ResultStatus::Unsupported,
            "Not measurable via HIP runtime/module API", dispatchNote);
  if (submitFailed)
  {
    test.skip("roundtrip", ResultStatus::Error,
              "hipModuleLaunchKernel/hipStreamSynchronize failed", roundtripNote);
  }
  else
  {
    test.emit("roundtrip", (float)(totalRoundtripUs / iters * 1e-6), roundtripNote);
  }

  return 0;
}

#endif // ENABLE_ROCM
