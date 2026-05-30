#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <string>

// Raw MFMA (matrix-core) peak microbenchmarks. Unlike the vector-ALU FMA
// compute tests (single/half/double/mp/bf16), these drive the matrix cores
// directly via __builtin_amdgcn_mfma_* and target the datasheet's PFLOPS/POPS
// numbers (FP16/BF16 1.3 PFLOPS, INT8 2.6 POPS, FP8 2.61 PFLOPS on MI300X).
//
// Capability detection is intentionally NOT a per-arch equality/threshold list:
//   * A numeric "gfx >= N" compare is unsafe -- the gfx space is partitioned by
//     family, not capability. gfx9xx (CDNA / data-center) has MFMA; gfx10/11/12
//     (RDNA / consumer) have WMMA instead and no MFMA, yet sort numerically
//     ABOVE gfx9xx. So a threshold would wrongly claim MFMA on RDNA.
//   * Exact arch lists silently exclude future CDNA parts.
// Instead we gate on the one stable invariant -- MFMA lives on the CDNA family
// (gfx9xx) -- and let the HIPRTC compile decide the exact per-datatype support.
// clang hard-gates each mfma builtin per sub-target, so a future CDNA arch is
// picked up automatically and an unsupported datatype fails to compile and is
// reported Unsupported. The kernels' op count (MFMA_ITERS * MFMA_ACC) must match
// the #defines in mfma_*.hip.
namespace {

constexpr uint64_t kMfmaIters = 512;
constexpr uint64_t kMfmaAcc = 8;
constexpr uint64_t kMfmaPerWave = kMfmaIters * kMfmaAcc;

struct MfmaEntry
{
  const char *metric;
  const char *title;
  const char *unit;     // "tflops" or "tops"
  const char *src;
  const char *srcName;
  const char *kernelName;
  uint32_t M, N, K;
  bool isInt;           // output buffer element type
};

// Strip the gcnArchName feature suffix (e.g. "gfx942:sramecc+:xnack-").
std::string archBaseOf(const std::string &a)
{
  return a.substr(0, a.find(':'));
}

// MFMA exists only on the CDNA / data-center line (gfx9xx). RDNA (gfx10/11/12)
// has WMMA, not MFMA. This family check is the only hardcoded assumption; the
// precise per-datatype capability is settled by the HIPRTC compile below.
bool isCdnaFamily(const std::string &base)
{
  return base.compare(0, 4, "gfx9") == 0;
}

} // namespace

int RocmPeak::runMfma(RocmDevice &dev, benchmark_config_t &cfg, Category category)
{
  const bool isInt = category == Category::IntCompute;

  static const MfmaEntry fpEntries[] = {
    {"mfma_fp16", "MFMA fp16xfp16+fp32 16x16x16", "tflops",
     rocm_kernels::mfma_fp16_src, rocm_kernels::mfma_fp16_name, "mfma_fp16",
     16, 16, 16, false},
    {"mfma_bf16", "MFMA bf16xbf16+fp32 16x16x16", "tflops",
     rocm_kernels::mfma_bf16_src, rocm_kernels::mfma_bf16_name, "mfma_bf16",
     16, 16, 16, false},
    {"mfma_fp8", "MFMA fp8xfp8+fp32 16x16x32", "tflops",
     rocm_kernels::mfma_fp8_src, rocm_kernels::mfma_fp8_name, "mfma_fp8",
     16, 16, 32, false},
    {"mfma_mxfp4", "MFMA mxfp4(e2m1)+fp32 16x16x128", "tflops",
     rocm_kernels::mfma_mxfp4_src, rocm_kernels::mfma_mxfp4_name, "mfma_mxfp4",
     16, 16, 128, false},
  };
  static const MfmaEntry intEntries[] = {
    {"mfma_int8", "MFMA int8xint8+int32 16x16x32", "tops",
     rocm_kernels::mfma_int8_src, rocm_kernels::mfma_int8_name, "mfma_int8",
     16, 16, 32, true},
  };

  const MfmaEntry *entries = isInt ? intEntries : fpEntries;
  const size_t numEntries = isInt ? (sizeof(intEntries) / sizeof(intEntries[0]))
                                  : (sizeof(fpEntries) / sizeof(fpEntries[0]));

  const bool cdna = isCdnaFamily(archBaseOf(dev.info.archName));

  const uint32_t waveSize = dev.info.warpSize > 0 ? (uint32_t)dev.info.warpSize : 64;
  const uint32_t blockSize = waveSize;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint64_t wantBlocks = globalThreads / blockSize;

  for (size_t e = 0; e < numEntries; e++)
  {
    const MfmaEntry &me = entries[e];
    auto test = currentDeviceScope->beginTest({me.metric, me.title, me.unit});

    if (!cdna)
    {
      test.skip(me.metric, ResultStatus::Unsupported,
                "MFMA is a CDNA (gfx9xx) matrix-core feature; absent on this architecture");
      continue;
    }

    // Compiler-driven capability check: if this GPU lacks the instruction the
    // HIPRTC compile of the builtin fails. Report that as Unsupported rather
    // than a hard error, since for MFMA it means the datatype isn't available.
    hipFunction_t fn;
    // quiet: a compile failure here just means the datatype isn't supported on
    // this arch; suppress the HIPRTC log so it doesn't break result formatting.
    if (!dev.getKernel(me.src, me.srcName, me.kernelName, fn, {}, /*quiet=*/true))
    {
      test.skip(me.metric, ResultStatus::Unsupported,
                "MFMA instruction for this datatype not available on this GPU");
      continue;
    }

    const uint32_t elemSize = me.isInt ? (uint32_t)sizeof(int) : (uint32_t)sizeof(float);
    uint64_t bytesPerBlock = (uint64_t)blockSize * elemSize;
    uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock;
    uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
    if (pickBlocks == 0)
      pickBlocks = 1;
    uint32_t numBlocks = (uint32_t)pickBlocks;

    void *outBuf = nullptr;
    if (hipMalloc(&outBuf, (uint64_t)numBlocks * bytesPerBlock) != hipSuccess)
    {
      test.skip(me.metric, ResultStatus::Error, "Failed to allocate output buffer");
      continue;
    }

    void *args[1] = {&outBuf};
    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      (void)hipFree(outBuf);
      test.skip(me.metric, ResultStatus::Error, "kernel launch failed");
      continue;
    }

    // Each wave issues kMfmaPerWave MFMA ops, each doing 2*M*N*K flops/ops.
    const double ops = (double)numBlocks * (double)kMfmaPerWave *
                       2.0 * (double)me.M * (double)me.N * (double)me.K;
    float value = (float)(ops * 1.0e6 / us / 1.0e12);
    test.emit(me.metric, value);

    (void)hipFree(outBuf);
  }

  return 0;
}

#endif // ENABLE_ROCM
