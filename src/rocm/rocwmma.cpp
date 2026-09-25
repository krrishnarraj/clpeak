#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runRocwmma(RocmDevice &dev, benchmark_config_t &cfg)
{
  // One test for all data types -- integer reading carries its own unit.
  auto test = currentDeviceScope->beginTest(
    {"rocwmma", "rocWMMA matrix multiply",
     "flops", Category::Unknown,
     "Matrix-core speed reached through AMD's rocWMMA library rather than the "
     "raw instructions.  Compare each reading with the WMMA or MFMA row for "
     "the same format to see what the library layer costs.",
     TestShape::Heterogeneous, "data type"});

#ifndef CLPEAK_ROCM_HAS_ROCWMMA
  {
    logger::EmitOptions o; o.description = "16-bit inputs with a 32-bit running total, 16x16x16 tile.";
    test.skip("fp16", ResultStatus::Unsupported, "rocWMMA headers not found at configure time", o);
    logger::EmitOptions oi; oi.description = "8-bit whole numbers with a 32-bit running total, 16x16x32 tile."; oi.unit = "ops";
    test.skip("int8", ResultStatus::Unsupported, "rocWMMA headers not found at configure time", oi);
  }
  return 0;
#else
  // Which archs rocWMMA supports is whatever the installed version's headers
  // accepted at build time (probed in EmbedRocmKernels.cmake), so the kernels'
  // bundles are the list -- a GPU without a slice is one rocWMMA lacks.
  // Helper to run one rocWMMA variant
  auto runOne = [&](const char *metric, const char *note, const char *kernelName,
                    const rocm_kernels::Blob &blob, uint32_t K, size_t elemBytes,
                    logger::EmitOptions opts) {
    const uint32_t waveSize = dev.info.warpSize > 0 ? (uint32_t)dev.info.warpSize : 64;
    const uint32_t blockSize = waveSize;
    uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
    constexpr uint32_t M = 16;
    constexpr uint32_t N = 16;
    constexpr uint32_t Iters = 256;
    uint64_t wantBlocks = globalThreads / blockSize;
    uint64_t bytesPerBlock = (uint64_t)M * N * sizeof(float);
    uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock;
    uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
    if (pickBlocks == 0)
      pickBlocks = 1;
    uint32_t numBlocks = (uint32_t)pickBlocks;
    const uint64_t outElems = (uint64_t)numBlocks * M * N;
    const uint64_t outBytes = outElems * elemBytes;

    void *outBuf = nullptr;
    if (hipMalloc(&outBuf, outBytes) != hipSuccess)
    {
      test.skip(metric, ResultStatus::Error, "Failed to allocate output buffer", opts);
      return;
    }

    RocmKernel k = dev.getKernel(blob, kernelName,
                                 "rocWMMA does not support this GPU architecture");
    if (!k)
    {
      (void)hipFree(outBuf);
      test.skip(metric, k.status, k.reason, opts);
      return;
    }

    void *args[1] = {&outBuf};
    float us = runKernel(dev, k.fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      (void)hipFree(outBuf);
      test.skip(metric, ResultStatus::Error, "kernel launch failed", opts);
      return;
    }

    const double ops = (double)numBlocks * (double)M * (double)N *
                       (double)K * 2.0 * (double)Iters;
    float value = (float)(ops * 1.0e6 / us);
    test.emit(metric, value, opts);

    (void)hipFree(outBuf);
  };

  {
    logger::EmitOptions o; o.description = "16-bit inputs with a 32-bit running total, 16x16x16 tile.";
    runOne("fp16", "fp16", "rocwmma_fp16", rocm_kernels::rocwmma_fp16, 16u, sizeof(float), o);
  }
  {
    logger::EmitOptions o; o.description = "8-bit whole numbers with a 32-bit running total, 16x16x32 tile."; o.unit = "ops";
    runOne("int8", "int8", "rocwmma_int8", rocm_kernels::rocwmma_int8, 32u, sizeof(int), o);
  }
  return 0;
#endif
}

#endif // ENABLE_ROCM
