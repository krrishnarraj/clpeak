#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

int CudaPeak::runAtomicThroughput(CudaDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"atomic_throughput", "Atomic throughput", "gops"});

  const uint32_t blockSize = 256;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numSMs);
  uint32_t numBlocks = (uint32_t)(globalThreads / blockSize);

  // Global: per-thread counter (128 MB).
  {
    CUdeviceptr buf = 0;
    if (cuMemAlloc(&buf, globalThreads * sizeof(int)) == CUDA_SUCCESS)
    {
      cuMemsetD32(buf, 0, globalThreads);
      CUfunction fn;
      if (dev.getKernel(cuda_kernels::atomic_throughput_src,
                        cuda_kernels::atomic_throughput_name,
                        "atomic_throughput_global", fn))
      {
        void *args[1] = {&buf};
        float us = runKernel(dev, fn, numBlocks, blockSize, args,
                             cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
        test.emit("int_global", gops);
      }
      else
      {
        test.skip("int_global", ResultStatus::Error, "Kernel compile failed");
      }
      cuMemFree(buf);
    }
    else
    {
      test.skip("int_global", ResultStatus::Error, "Buffer alloc failed");
    }
  }

  // Local: one counter per block.
  {
    CUdeviceptr buf = 0;
    if (cuMemAlloc(&buf, (uint64_t)numBlocks * sizeof(int)) == CUDA_SUCCESS)
    {
      CUfunction fn;
      if (dev.getKernel(cuda_kernels::atomic_throughput_src,
                        cuda_kernels::atomic_throughput_name,
                        "atomic_throughput_local", fn))
      {
        void *args[1] = {&buf};
        float us = runKernel(dev, fn, numBlocks, blockSize, args,
                             cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
        test.emit("int_local", gops);
      }
      else
      {
        test.skip("int_local", ResultStatus::Error, "Kernel compile failed");
      }
      cuMemFree(buf);
    }
    else
    {
      test.skip("int_local", ResultStatus::Error, "Buffer alloc failed");
    }
  }

  return 0;
}

// Free-function enumeration used by --list-devices and the Android JNI surface.
// Uses the static driver API directly — no CudaPeak instance required.

#endif // ENABLE_CUDA
