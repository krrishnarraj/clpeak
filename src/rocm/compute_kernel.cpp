#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <string>
#include <vector>

int RocmPeak::runComputeKernel(RocmDevice &dev, benchmark_config_t &cfg,
                               const rocm_compute_desc_t &d)
{
  auto test = currentDeviceScope->beginTest(
    {d.resultTag, d.title, d.unit, Category::Unknown,
     d.description ? d.description : ""});

  struct Variant
  {
    const char *label;
    const char *kernelName;
    const rocm_kernels::Blob *blob;
    const char *description;
  };
  std::vector<Variant> variants;
  if (d.variants && d.numVariants > 0)
    for (uint32_t i = 0; i < d.numVariants; i++)
      variants.push_back({d.variants[i].label, d.variants[i].kernelName,
                          d.variants[i].blob, d.variants[i].description});
  else
    // Single-variant tests have one reading whose name restates the title, so
    // the test description covers it.
    variants.push_back({d.metricLabel, d.kernelName, d.blob, nullptr});

  auto note = [](const char *text) { return text ? std::string(text) : std::string(); };

  if (d.skip)
  {
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Unsupported,
                d.skipMsg ? d.skipMsg : "Skipped", note(v.description));
    return 0;
  }

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
      test.skip(v.label, ResultStatus::Error, "Failed to allocate output buffer",
                note(v.description));
    return -1;
  }

  for (const auto &v : variants)
  {
    hipFunction_t fn;
    if (!dev.getKernel(*v.blob, v.kernelName, fn))
    {
      test.skip(v.label, ResultStatus::Error, "compile/load failed",
                note(v.description));
      continue;
    }

    void *args[2];
    args[0] = &outputBuf;
    args[1] = const_cast<void *>(d.scalarArg);

    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "kernel launch failed",
                note(v.description));
      continue;
    }

    uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
    double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
    float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us / divider);

    test.emit(v.label, value, {false, note(v.description)});
  }

  (void)hipFree(outputBuf);
  return 0;
}

#endif // ENABLE_ROCM
