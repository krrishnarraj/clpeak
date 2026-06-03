#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <vector>

int RocmPeak::runComputeKernel(RocmDevice &dev, benchmark_config_t &cfg,
                               const rocm_compute_desc_t &d)
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

  const uint32_t blockSize = d.blockSize ? d.blockSize : 256;
  const uint32_t outPerBlock = d.outElemsPerBlock ? d.outElemsPerBlock : blockSize;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint64_t bytesPerBlock = (uint64_t)outPerBlock * d.elemSize;
  uint64_t maxBlocks = bytesPerBlock ? (dev.info.totalGlobalMem / 4 / bytesPerBlock) : 0;
  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  if (pickBlocks == 0)
    pickBlocks = 1;
  uint32_t numBlocks = (uint32_t)pickBlocks;
  uint64_t bufferBytes = (uint64_t)numBlocks * bytesPerBlock;

  void *outputBuf = nullptr;
  if (hipMalloc(&outputBuf, bufferBytes) != hipSuccess)
  {
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  std::vector<const char *> hiprtcOpts;
  for (uint32_t i = 0; i < d.numExtraHiprtcOpts; i++)
    hiprtcOpts.push_back(d.extraHiprtcOpts[i]);

  for (const auto &v : variants)
  {
    hipFunction_t fn;
    if (!dev.getKernel(v.src, v.srcName, v.kernelName, fn, hiprtcOpts, d.quietCompile))
    {
      test.skip(v.label, ResultStatus::Error, "compile/load failed");
      continue;
    }

    void *args[2];
    args[0] = &outputBuf;
    args[1] = const_cast<void *>(d.scalarArg);

    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "kernel launch failed");
      continue;
    }

    uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
    double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
    float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us / divider);

    test.emit(v.label, value);
  }

  (void)hipFree(outputBuf);
  return 0;
}

#endif // ENABLE_ROCM
