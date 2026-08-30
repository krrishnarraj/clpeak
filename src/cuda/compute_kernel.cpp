#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Shared compute-peak driver.  Mirrors vkPeak::runComputeKernel in spirit:
// allocate a single device-local output buffer, dispatch each variant of
// the same kernel against it with NVRTC-compiled kernels.
// ---------------------------------------------------------------------------

int CudaPeak::runComputeKernel(CudaDevice &dev, benchmark_config_t &cfg,
                               const cuda_compute_desc_t &d)
{
  auto test = currentDeviceScope->beginTest(
    {d.resultTag, d.title, d.unit, Category::Unknown,
     d.description ? d.description : "",
     d.shape, d.axis ? d.axis : ""});

  struct Variant
  {
    const char *label;
    const char *kernelName;
    const cuda_kernels::Blob *blob;
    const char *description;
  };
  std::vector<Variant> variants;
  if (d.variants && d.numVariants > 0)
    for (uint32_t i = 0; i < d.numVariants; i++)
      variants.push_back({d.variants[i].label, d.variants[i].kernelName,
                          d.variants[i].blob, d.variants[i].description});
  else
    // Single-variant tests: one reading, documented by d.metricDescription.
    // The tensor-core tests use this path once per data type, all into the
    // same test.
    variants.push_back({d.metricLabel, d.kernelName, d.blob,
                        d.metricDescription});

  auto note = [](const char *text) { return text ? std::string(text) : std::string(); };

  // Unit override for the single-variant path: an integer member of an
  // otherwise floating-point family carries its own.
  auto emitOpts = [&](const char *description) {
    logger::EmitOptions o;
    o.description = note(description);
    if (d.metricUnit) o.unit = d.metricUnit;
    return o;
  };

  if (d.skip)
  {
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Unsupported,
                d.skipMsg ? d.skipMsg : "Skipped", note(v.description));
    return 0;
  }

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
      test.skip(v.label, ResultStatus::Error, "Failed to allocate output buffer",
                note(v.description));
    return -1;
  }

  for (const auto &v : variants)
  {
    CUfunction fn;
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
    uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
    double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
    float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us / divider);

    test.emit(v.label, value, emitOpts(v.description));
  }

  cuMemFree(outputBuf);
  return 0;
}

#endif // ENABLE_CUDA
