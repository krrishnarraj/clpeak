#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runGlobalBandwidth(RocmDevice &dev, benchmark_config_t &cfg)
{
  const uint32_t blockSize = 256;

  uint64_t maxItems = dev.info.totalGlobalMem / sizeof(float) / 4;
  uint64_t numItems = (maxItems / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);
  if (numItems > cfg.globalBWMaxSize / sizeof(float))
    numItems = (cfg.globalBWMaxSize / sizeof(float) / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);

  uint32_t numBlocks = (uint32_t)(numItems / FETCH_PER_WI / blockSize);
  if (numBlocks == 0)
    numBlocks = 1;

  auto test = currentDeviceScope->beginTest(
    {"global_memory_bandwidth", "Global memory bandwidth", "gbps"});

  void *inBuf = nullptr;
  void *outBuf = nullptr;
  if (hipMalloc(&inBuf, numItems * sizeof(float)) != hipSuccess ||
      hipMalloc(&outBuf, numItems * sizeof(float)) != hipSuccess)
  {
    const char *labels[] = {"float", "float2", "float4"};
    for (int i = 0; i < 3; i++)
      test.skip(labels[i], ResultStatus::Error, "Failed to allocate buffers");
    if (inBuf)
      (void)hipFree(inBuf);
    if (outBuf)
      (void)hipFree(outBuf);
    return -1;
  }

  float *hInput = new float[numItems];
  populate(hInput, numItems);
  hipError_t copyStatus = hipMemcpy(inBuf, hInput, numItems * sizeof(float),
                                    hipMemcpyHostToDevice);
  delete[] hInput;
  if (copyStatus != hipSuccess)
  {
    const char *labels[] = {"float", "float2", "float4"};
    for (int i = 0; i < 3; i++)
      test.skip(labels[i], ResultStatus::Error, "Failed to upload input buffer");
    (void)hipFree(inBuf);
    (void)hipFree(outBuf);
    return -1;
  }

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

    hipFunction_t fn;
    if (!dev.getKernel(rocm_kernels::global_bandwidth_src,
                       rocm_kernels::global_bandwidth_name,
                       v.kernelName, fn))
    {
      test.skip(key, ResultStatus::Error, "Kernel compile failed");
      continue;
    }

    uint64_t blocks = numItems / FETCH_PER_WI / v.width / blockSize;
    if (blocks == 0)
      blocks = 1;
    uint32_t blocksU = (uint32_t)blocks;

    void *args[2] = {&inBuf, &outBuf};
    float us = runKernel(dev, fn, blocksU, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      test.skip(key, ResultStatus::Error, "kernel launch failed");
      continue;
    }
    double bytes = (double)blocksU * blockSize * FETCH_PER_WI * v.width * sizeof(float);
    float gbps = (float)(bytes / us / 1e3);
    test.emit(key, gbps);
  }

  (void)hipFree(inBuf);
  (void)hipFree(outBuf);
  return 0;
}

#endif // ENABLE_ROCM
