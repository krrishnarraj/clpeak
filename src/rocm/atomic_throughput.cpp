#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runAtomicThroughput(RocmDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"atomic_throughput", "Atomic throughput", "gops"});

  const uint32_t blockSize = 256;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint32_t numBlocks = (uint32_t)(globalThreads / blockSize);

  {
    void *buf = nullptr;
    if (hipMalloc(&buf, globalThreads * sizeof(int)) == hipSuccess)
    {
      hipMemset(buf, 0, globalThreads * sizeof(int));
      hipFunction_t fn;
      if (dev.getKernel(rocm_kernels::atomic_throughput_src,
                        rocm_kernels::atomic_throughput_name,
                        "atomic_throughput_global", fn))
      {
        void *args[1] = {&buf};
        float us = runKernel(dev, fn, numBlocks, blockSize, args,
                             cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        if (us > 0.0f)
        {
          float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
          test.emit("int_global", gops);
        }
        else
        {
          test.skip("int_global", ResultStatus::Error, "kernel launch failed");
        }
      }
      else
      {
        test.skip("int_global", ResultStatus::Error, "Kernel compile failed");
      }
      hipFree(buf);
    }
    else
    {
      test.skip("int_global", ResultStatus::Error, "Buffer alloc failed");
    }
  }

  {
    void *buf = nullptr;
    if (hipMalloc(&buf, (uint64_t)numBlocks * sizeof(int)) == hipSuccess)
    {
      hipFunction_t fn;
      if (dev.getKernel(rocm_kernels::atomic_throughput_src,
                        rocm_kernels::atomic_throughput_name,
                        "atomic_throughput_local", fn))
      {
        void *args[1] = {&buf};
        float us = runKernel(dev, fn, numBlocks, blockSize, args,
                             cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        if (us > 0.0f)
        {
          float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
          test.emit("int_local", gops);
        }
        else
        {
          test.skip("int_local", ResultStatus::Error, "kernel launch failed");
        }
      }
      else
      {
        test.skip("int_local", ResultStatus::Error, "Kernel compile failed");
      }
      hipFree(buf);
    }
    else
    {
      test.skip("int_local", ResultStatus::Error, "Buffer alloc failed");
    }
  }

  return 0;
}

#endif // ENABLE_ROCM
