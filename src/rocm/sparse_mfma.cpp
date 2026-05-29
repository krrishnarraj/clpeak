#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <string>

// 2:4 structured-sparse MFMA peak microbenchmarks (SMFMAC). Sibling of mfma.cpp
// for the datasheet's "with Structured Sparsity" columns: a sparse instruction
// consumes a 2:4-compressed A operand and does the work of a dense tile twice
// the K depth in the same time, so the *effective* throughput is ~2x the dense
// MFMA (MI300X: fp16/bf16 2.61 PFLOPS, int8 5.22 POPS, fp8 5.22 PFLOPS).
//
// Same capability model as mfma.cpp: gate on the CDNA (gfx9xx) family, then let
// the HIPRTC compile of __builtin_amdgcn_smfmac_* settle exact per-datatype
// support so future CDNA parts are picked up with no list edits. FLOPs are
// counted with the DENSE-equivalent K (the work the instruction actually does),
// which is what yields the sparse 2x figure. SMFMAC_ITERS * SMFMAC_ACC must
// match the #defines in smfmac_*.hip.
namespace {

constexpr uint64_t kIters = 512;
constexpr uint64_t kAcc = 8;
constexpr uint64_t kPerWave = kIters * kAcc;

struct SparseEntry
{
  const char *metric;
  const char *title;
  const char *unit;     // "tflops" or "tops"
  const char *src;
  const char *srcName;
  const char *kernelName;
  uint32_t M, N, Kdense; // dense-equivalent K for op accounting
  bool isInt;
};

std::string archBaseOf(const std::string &a)
{
  return a.substr(0, a.find(':'));
}

// SMFMAC is CDNA3+. The family check (gfx9xx) excludes RDNA cleanly; a CDNA
// part without a given sparse datatype fails the HIPRTC compile and is reported
// Unsupported rather than emitting a fabricated number.
bool isCdnaFamily(const std::string &base)
{
  return base.compare(0, 4, "gfx9") == 0;
}

} // namespace

int RocmPeak::runSparseMfma(RocmDevice &dev, benchmark_config_t &cfg, Category category)
{
  const bool isInt = category == Category::IntCompute;

  static const SparseEntry fpEntries[] = {
    {"smfmac_fp16", "Sparse MFMA fp16 2:4 16x16x32 (TFLOPS)", "tflops",
     rocm_kernels::smfmac_fp16_src, rocm_kernels::smfmac_fp16_name, "smfmac_fp16",
     16, 16, 32, false},
    {"smfmac_bf16", "Sparse MFMA bf16 2:4 16x16x32 (TFLOPS)", "tflops",
     rocm_kernels::smfmac_bf16_src, rocm_kernels::smfmac_bf16_name, "smfmac_bf16",
     16, 16, 32, false},
    {"smfmac_fp8", "Sparse MFMA fp8 2:4 16x16x64 (TFLOPS)", "tflops",
     rocm_kernels::smfmac_fp8_src, rocm_kernels::smfmac_fp8_name, "smfmac_fp8",
     16, 16, 64, false},
  };
  static const SparseEntry intEntries[] = {
    {"smfmac_int8", "Sparse MFMA int8 2:4 16x16x64 (TOPS)", "tops",
     rocm_kernels::smfmac_int8_src, rocm_kernels::smfmac_int8_name, "smfmac_int8",
     16, 16, 64, true},
  };

  const SparseEntry *entries = isInt ? intEntries : fpEntries;
  const size_t numEntries = isInt ? (sizeof(intEntries) / sizeof(intEntries[0]))
                                  : (sizeof(fpEntries) / sizeof(fpEntries[0]));

  const bool cdna = isCdnaFamily(archBaseOf(dev.info.archName));

  const uint32_t waveSize = dev.info.warpSize > 0 ? (uint32_t)dev.info.warpSize : 64;
  const uint32_t blockSize = waveSize;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint64_t wantBlocks = globalThreads / blockSize;

  for (size_t e = 0; e < numEntries; e++)
  {
    const SparseEntry &se = entries[e];
    auto test = currentDeviceScope->beginTest({se.metric, se.title, se.unit});

    if (!cdna)
    {
      test.skip(se.metric, ResultStatus::Unsupported,
                "Sparse MFMA is a CDNA (gfx9xx) matrix-core feature; absent on this architecture");
      continue;
    }

    hipFunction_t fn;
    if (!dev.getKernel(se.src, se.srcName, se.kernelName, fn))
    {
      test.skip(se.metric, ResultStatus::Unsupported,
                "Sparse MFMA instruction for this datatype not available on this GPU");
      continue;
    }

    const uint32_t elemSize = se.isInt ? (uint32_t)sizeof(int) : (uint32_t)sizeof(float);
    uint64_t bytesPerBlock = (uint64_t)blockSize * elemSize;
    uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock;
    uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
    if (pickBlocks == 0)
      pickBlocks = 1;
    uint32_t numBlocks = (uint32_t)pickBlocks;

    void *outBuf = nullptr;
    if (hipMalloc(&outBuf, (uint64_t)numBlocks * bytesPerBlock) != hipSuccess)
    {
      test.skip(se.metric, ResultStatus::Error, "Failed to allocate output buffer");
      continue;
    }

    void *args[1] = {&outBuf};
    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      (void)hipFree(outBuf);
      test.skip(se.metric, ResultStatus::Error, "kernel launch failed");
      continue;
    }

    // Dense-equivalent work: each sparse op produces a full 2*M*N*Kdense result.
    const double ops = (double)numBlocks * (double)kPerWave *
                       2.0 * (double)se.M * (double)se.N * (double)se.Kdense;
    float value = (float)(ops * 1.0e6 / us / 1.0e12);
    test.emit(se.metric, value);

    (void)hipFree(outBuf);
  }

  return 0;
}

#endif // ENABLE_ROCM
