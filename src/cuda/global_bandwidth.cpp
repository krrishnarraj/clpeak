#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

int CudaPeak::runGlobalBandwidth(CudaDevice &dev, benchmark_config_t &cfg)
{
  const uint32_t blockSize = 256;

  uint64_t maxItems = dev.info.totalGlobalMem / sizeof(float) / 4; // input+output, plus margin
  uint64_t numItems = (maxItems / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);
  if (numItems > cfg.globalBWMaxSize / sizeof(float))
    numItems = (cfg.globalBWMaxSize / sizeof(float) / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);

  uint32_t numBlocks = (uint32_t)(numItems / FETCH_PER_WI / blockSize);
  if (numBlocks == 0)
    numBlocks = 1;

  // Opened before the sizing diagnostic below, so that line lands under
  // this test's header rather than under the previous test's readings.
  auto test = currentDeviceScope->beginTest(
    {"global_memory_bandwidth", "Global memory bandwidth", "bps",
     Category::Unknown,
     "How many bytes per second the GPU can stream out of its own memory, "
     "reading a buffer far too large to cache.  Each reading fetches a "
     "different number of values per instruction, since wider fetches usually "
     "pull more through before the memory system saturates.",
     TestShape::Homogeneous, "vector width"});

  // The one number that decides whether this test measured memory or cache:
  // the timed phase re-reads the same buffer, so a working set that fits behind
  // the last-level cache reports the cache.  benchmark_config_t::forDevice sizes
  // globalBWMaxSize to clear it; print both so an implausible reading can be
  // checked against them without a rebuild.
  CLPEAK_VLOG("global_memory_bandwidth: working set %llu MB, device cache %llu MB\n",
              (unsigned long long)(numItems * sizeof(float) >> 20),
              (unsigned long long)(dev.info.l2CacheSize >> 20));

  CUdeviceptr inBuf = 0, outBuf = 0;
  if (cuMemAlloc(&inBuf, numItems * sizeof(float)) != CUDA_SUCCESS ||
      cuMemAlloc(&outBuf, numItems * sizeof(float)) != CUDA_SUCCESS)
  {
    const char *labels[] = {"float", "float2", "float4"};
    const uint32_t widths[] = {1, 2, 4};
    for (int i = 0; i < 3; i++)
      test.skip(labels[i], ResultStatus::Error, "Failed to allocate buffers",
                cudaWidthNote(widths[i]));
    if (inBuf)
      cuMemFree(inBuf);
    return -1;
  }
  // Fill input with pseudo-random data to defeat hardware memory compression.
  float *hInput = new float[numItems];
  populate(hInput, numItems);
  cuMemcpyHtoD(inBuf, hInput, numItems * sizeof(float));
  delete[] hInput;

  // Each variant takes input typed as floatN* so element index strides by
  // V floats per iteration -- matching the OpenCL backend.  numBlocks
  // shrinks accordingly so the total bytes touched remain the same.
  struct Variant
  {
    const char *label;
    const char *kernelName;
    uint32_t width;
  };
  static const Variant variants[] = {
      {"float   ", "global_bandwidth_v1", 1},
      {"float2  ", "global_bandwidth_v2", 2},
      {"float4  ", "global_bandwidth_v4", 4},
  };

  for (const auto &v : variants)
  {
    std::string key(v.label);
    while (!key.empty() && key.back() == ' ')
      key.pop_back();

    CUfunction fn;
    if (!dev.getKernel(cuda_kernels::global_bandwidth,
                       v.kernelName, fn))
    {
      test.skip(key, ResultStatus::Error, "Kernel compile failed",
                cudaWidthNote(v.width));
      continue;
    }

    uint64_t blocks = numItems / FETCH_PER_WI / v.width / blockSize;
    if (blocks == 0)
      blocks = 1;
    uint32_t blocksU = (uint32_t)blocks;

    void *args[2] = {&inBuf, &outBuf};
    float us = runKernel(dev, fn, blocksU, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    double bytes = (double)blocksU * blockSize * FETCH_PER_WI * v.width * sizeof(float);
    float gbps = (float)(bytes / us * 1e6);
    test.emit(key, gbps, cudaWidthNote(v.width));
  }

  cuMemFree(inBuf);
  cuMemFree(outBuf);
  return 0;
}

// ---------------------------------------------------------------------------
// Host<->device transfer bandwidth (CUDA)
// ---------------------------------------------------------------------------


#endif // ENABLE_CUDA
