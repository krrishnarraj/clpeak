#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runRocwmma(RocmDevice &dev, benchmark_config_t &cfg, Category category)
{
  const bool isInt = category == Category::IntCompute;
  auto test = currentDeviceScope->beginTest(
    {isInt ? "rocwmma-int" : "rocwmma-fp",
     isInt ? "rocWMMA int8xint8+int32 16x16x16"
           : "rocWMMA fp16xfp16+fp32 16x16x16",
     isInt ? "tops" : "tflops"});

  const char *metric = isInt ? "rocwmma_int8" : "rocwmma_fp16";

#ifndef CLPEAK_ROCM_HAS_ROCWMMA
  test.skip(metric, ResultStatus::Unsupported, "rocWMMA headers not found at configure time");
  return 0;
#else
  if (!dev.info.rocwmmaSupported)
  {
    test.skip(metric, ResultStatus::Unsupported, "rocWMMA does not support this GPU architecture");
    return 0;
  }

  const uint32_t waveSize = dev.info.warpSize > 0 ? (uint32_t)dev.info.warpSize : 64;
  const uint32_t blockSize = waveSize;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);

  constexpr uint32_t M = 16;
  constexpr uint32_t N = 16;
  constexpr uint32_t K = 16;
  constexpr uint32_t Iters = 256;
  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t bytesPerBlock = (uint64_t)M * N * sizeof(float);
  uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  if (pickBlocks == 0)
    pickBlocks = 1;
  uint32_t numBlocks = (uint32_t)pickBlocks;
  const uint64_t outElems = (uint64_t)numBlocks * M * N;
  const uint64_t outBytes = outElems * (isInt ? sizeof(int) : sizeof(float));

  void *outBuf = nullptr;
  if (hipMalloc(&outBuf, outBytes) != hipSuccess)
  {
    test.skip(metric, ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  hipFunction_t fn;
  const char *opts[] = {"--std=c++17"};
  if (!dev.getKernel(isInt ? rocm_kernels::rocwmma_int8_src : rocm_kernels::rocwmma_fp16_src,
                     isInt ? rocm_kernels::rocwmma_int8_name : rocm_kernels::rocwmma_fp16_name,
                     isInt ? "rocwmma_int8" : "rocwmma_fp16", fn,
                     std::vector<const char *>(opts, opts + 1)))
  {
    (void)hipFree(outBuf);
    test.skip(metric, ResultStatus::Error, "Kernel compile failed");
    return 0;
  }

  void *args[1] = {&outBuf};
  float us = runKernel(dev, fn, numBlocks, blockSize, args,
                       cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
  {
    (void)hipFree(outBuf);
    test.skip(metric, ResultStatus::Error, "kernel launch failed");
    return 0;
  }

  const double ops = (double)numBlocks * (double)M * (double)N *
                     (double)K * 2.0 * (double)Iters;
  float value = (float)(ops * 1.0e6 / us / 1.0e12);
  test.emit(metric, value);

  (void)hipFree(outBuf);
  return 0;
#endif
}

#endif // ENABLE_ROCM
