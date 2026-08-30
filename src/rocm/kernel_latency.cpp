#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <chrono>

int RocmPeak::runKernelLatency(RocmDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = forceIters ? specifiedIters
                                  : (cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000);

  auto test = currentDeviceScope->beginTest(
    {"kernel_launch_latency", "Kernel launch latency", "us", Category::Unknown,
     "The overhead of asking the GPU to do anything at all, measured with a "
     "kernel that does no work.  It is what small, frequent GPU jobs pay "
     "before any of their own work begins.",
     TestShape::Heterogeneous});

  const char *dispatchNote = "One way only: from the moment the host submits the "
                             "work to the moment the GPU starts running it.  HIP "
                             "exposes no clock the host and GPU share, so this is "
                             "reported on the other backends but not here.";
  const char *roundtripNote = "The full round trip -- submit, run, and hear back "
                              "that it finished.  This is what the host waits for "
                              "if it has nothing else to get on with.";

  hipFunction_t fn;
  if (!dev.getKernel(rocm_kernels::kernel_latency,
                     "kernel_latency_noop", fn))
  {
    test.skip("dispatch", ResultStatus::Error, "Kernel compile failed", dispatchNote);
    test.skip("roundtrip", ResultStatus::Error, "Kernel compile failed", roundtripNote);
    return -1;
  }

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
    test.emit("roundtrip", (float)(totalRoundtripUs / iters), roundtripNote);
  }

  return 0;
}

#endif // ENABLE_ROCM
