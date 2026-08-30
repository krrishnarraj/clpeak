#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runLocalBandwidth(RocmDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"local_memory_bandwidth", "Local memory bandwidth", "gbps",
     Category::Unknown,
     "How many bytes per second the GPU moves through shared local memory -- "
     "the small on-chip scratchpad a block of threads passes data through, "
     "which never goes out to the card's main memory.",
     TestShape::Homogeneous, "vector width"});

  const uint32_t blockSize = 256;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint32_t numBlocks = (uint32_t)(globalThreads / blockSize);

  void *outBuf = nullptr;
  if (hipMalloc(&outBuf, globalThreads * sizeof(float)) != hipSuccess)
  {
    test.skip("float", ResultStatus::Error, "Buffer alloc failed", rocmWidthNote(1));
    test.skip("float2", ResultStatus::Error, "Buffer alloc failed", rocmWidthNote(2));
    test.skip("float4", ResultStatus::Error, "Buffer alloc failed", rocmWidthNote(4));
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
    std::string key(v.label);
    while (!key.empty() && key.back() == ' ')
      key.pop_back();

    hipFunction_t fn;
    if (!dev.getKernel(rocm_kernels::local_bandwidth, v.kname, fn))
    {
      test.skip(key, ResultStatus::Error, "Kernel compile failed",
                rocmWidthNote(v.width));
      continue;
    }
    void *args[1] = {&outBuf};
    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      test.skip(key, ResultStatus::Error, "kernel launch failed",
                rocmWidthNote(v.width));
      continue;
    }
    uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.width * sizeof(float) * globalThreads;
    float gbps = (float)bytes / us / 1e3f;
    test.emit(key, gbps, rocmWidthNote(v.width));
  }

  (void)hipFree(outBuf);
  return 0;
}

#endif // ENABLE_ROCM
