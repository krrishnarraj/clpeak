#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <vector>

// ---------------------------------------------------------------------------
// Shared compute-peak driver.  Mirrors vkPeak::runComputeKernel in spirit:
// allocate a single device-local output buffer, dispatch each variant of
// the same kernel against it with NVRTC-compiled kernels.
// ---------------------------------------------------------------------------

int CudaPeak::runComputeKernel(CudaDevice &dev, benchmark_config_t &cfg,
                               const cuda_compute_desc_t &d)
{
  auto test = currentDeviceScope->beginTest({d.resultTag, d.title, d.unit});

  if (d.skip)
  {
    if (d.variants && d.numVariants > 0)
    {
      for (uint32_t i = 0; i < d.numVariants; i++)
        test.skip(d.variants[i].label, ResultStatus::Unsupported,
                  d.skipMsg ? d.skipMsg : "Skipped");
    }
    else
    {
      test.skip(d.metricLabel, ResultStatus::Unsupported,
                d.skipMsg ? d.skipMsg : "Skipped");
    }
    return 0;
  }

  struct Variant
  {
    const char *label;
    const char *kernelName;
    const char *src;
    const char *srcName;
  };
  std::vector<Variant> variants;
  if (d.variants && d.numVariants > 0)
    for (uint32_t i = 0; i < d.numVariants; i++)
      variants.push_back({d.variants[i].label, d.variants[i].kernelName,
                          d.variants[i].src, d.variants[i].srcName});
  else
    variants.push_back({d.metricLabel, d.kernelName, d.src, d.srcName});

  // Scale to numSMs so high-SM parts (H100, B200, …) don't get under-saturated;
  // floor at 32M preserves behavior on small dev cards.  Clamp by VRAM below.
  const uint32_t blockSize = d.blockSize ? d.blockSize : 256;
  const uint32_t outPerBlock = d.outElemsPerBlock ? d.outElemsPerBlock : blockSize;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numSMs);
  uint64_t bytesPerBlock = (uint64_t)outPerBlock * d.elemSize;
  uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock; // cap at 1/4 VRAM
  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  uint32_t numBlocks = (uint32_t)pickBlocks;
  uint64_t bufferBytes = (uint64_t)numBlocks * bytesPerBlock;

  CUdeviceptr outputBuf = 0;
  if (cuMemAlloc(&outputBuf, bufferBytes) != CUDA_SUCCESS)
  {
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  std::vector<const char *> nvrtcOpts;
  for (uint32_t i = 0; i < d.numExtraNvrtcOpts; i++)
    nvrtcOpts.push_back(d.extraNvrtcOpts[i]);

  for (const auto &v : variants)
  {
    CUfunction fn;
    if (!dev.getKernel(v.src, v.srcName, v.kernelName, fn, nvrtcOpts))
    {
      test.skip(v.label, ResultStatus::Error, "compile/load failed");
      continue;
    }

    void *args[2];
    args[0] = &outputBuf;
    args[1] = const_cast<void *>(d.scalarArg);

    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
    double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
    float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us / divider);

    test.emit(v.label, value);
  }

  cuMemFree(outputBuf);
  return 0;
}

#endif // ENABLE_CUDA
