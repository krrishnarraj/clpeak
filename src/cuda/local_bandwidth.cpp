#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

int CudaPeak::runLocalBandwidth(CudaDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"local_memory_bandwidth", "Local memory bandwidth", "gbps",
     Category::Unknown,
     "How many bytes per second the GPU moves through shared memory -- the "
     "small on-chip scratchpad a block of threads passes data through, which "
     "never goes out to the card's main memory."});

  const uint32_t blockSize = 256;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numSMs);
  uint32_t numBlocks = (uint32_t)(globalThreads / blockSize);

  CUdeviceptr outBuf = 0;
  if (cuMemAlloc(&outBuf, globalThreads * sizeof(float)) != CUDA_SUCCESS)
  {
    test.skip("float", ResultStatus::Error, "Buffer alloc failed", cudaWidthNote(1));
    test.skip("float2", ResultStatus::Error, "Buffer alloc failed", cudaWidthNote(2));
    test.skip("float4", ResultStatus::Error, "Buffer alloc failed", cudaWidthNote(4));
    return -1;
  }

  struct V
  {
    const char *label;
    const char *kname;
    uint32_t width;
  };
  const V vs[] = {
      {"float  ", "local_bandwidth_v1", 1},
      {"float2 ", "local_bandwidth_v2", 2},
      {"float4 ", "local_bandwidth_v4", 4},
  };
  for (const auto &v : vs)
  {
    CUfunction fn;
    if (!dev.getKernel(cuda_kernels::local_bandwidth, v.kname, fn))
    {
      std::string key(v.label);
      while (!key.empty() && key.back() == ' ')
        key.pop_back();
      test.skip(key, ResultStatus::Error, "Kernel compile failed",
                cudaWidthNote(v.width));
      continue;
    }
    void *args[1] = {&outBuf};
    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.width * sizeof(float) * globalThreads;
    float gbps = (float)bytes / us / 1e3f;
    std::string key(v.label);
    while (!key.empty() && key.back() == ' ')
      key.pop_back();
    test.emit(key, gbps, cudaWidthNote(v.width));
  }

  cuMemFree(outBuf);
  return 0;
}

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (CUDA -- cudaTextureObject_t via driver API)
// ---------------------------------------------------------------------------


#endif // ENABLE_CUDA
