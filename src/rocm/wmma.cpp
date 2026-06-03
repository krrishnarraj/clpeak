#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <string>

// Raw WMMA (matrix-core) peak microbenchmarks for RDNA. Counterpart to mfma.cpp
// (CDNA): RDNA3 (gfx11) and RDNA4 (gfx12) have WMMA, not MFMA, and drive their
// matrix units via __builtin_amdgcn_wmma_*. This is the native, library-free
// path to the datasheet matrix peaks (e.g. on RX 9070 XT / gfx1201: 195 TFLOPS
// fp16, 389 TOPS int8) -- the analogue of the Vulkan cooperative-matrix tests.
//
// Distinct from rocwmma.cpp (the rocWMMA *library* path, which needs headers at
// configure time) and from mfma.cpp (CDNA-only). Capability is gated on the
// RDNA-WMMA family (gfx11/gfx12); the exact per-datatype support is then settled
// by the HIPRTC compile -- a missing builtin fails quietly and is reported
// Unsupported, never a fabricated number.
namespace {

constexpr uint64_t kWmmaIters = 512;
constexpr uint64_t kWmmaAcc = 8;
constexpr uint64_t kWmmaPerWave = kWmmaIters * kWmmaAcc;

struct WmmaEntry
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

std::string archBaseOf(const std::string &a)
{
  return a.substr(0, a.find(':'));
}

// WMMA lives on RDNA3+ (gfx11) and RDNA4 (gfx12). RDNA1/2 (gfx10) and CDNA
// (gfx9) are excluded; CDNA uses MFMA (mfma.cpp). The precise per-datatype
// capability is settled by the HIPRTC compile.
bool isRdnaWmmaFamily(const std::string &base)
{
  return base.compare(0, 5, "gfx11") == 0 || base.compare(0, 5, "gfx12") == 0;
}

} // namespace

int RocmPeak::runWmma(RocmDevice &dev, benchmark_config_t &cfg, Category category)
{
  const bool isInt = category == Category::IntCompute;

  static const WmmaEntry fpEntries[] = {
    {"wmma_fp16", "WMMA fp16xfp16+fp32 16x16x16", "tflops",
     rocm_kernels::wmma_fp16_src, rocm_kernels::wmma_fp16_name, "wmma_fp16",
     16, 16, 16, false},
  };
  static const WmmaEntry intEntries[] = {
    {"wmma_int8", "WMMA int8xint8+int32 16x16x16", "tops",
     rocm_kernels::wmma_int8_src, rocm_kernels::wmma_int8_name, "wmma_int8",
     16, 16, 16, true},
  };

  const WmmaEntry *entries = isInt ? intEntries : fpEntries;
  const size_t numEntries = isInt ? (sizeof(intEntries) / sizeof(intEntries[0]))
                                  : (sizeof(fpEntries) / sizeof(fpEntries[0]));

  const bool rdna = isRdnaWmmaFamily(archBaseOf(dev.info.archName));

  // WMMA builtins are wave32 (_w32). HIP defaults to wave32 on RDNA; if a part
  // is compiled wave64 the _w32 builtin layout would not match, so restrict to
  // wave32 rather than emit a misleading number.
  const uint32_t waveSize = dev.info.warpSize > 0 ? (uint32_t)dev.info.warpSize : 32;
  const bool wave32 = waveSize == 32;
  const uint32_t blockSize = waveSize;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint64_t wantBlocks = globalThreads / blockSize;

  for (size_t e = 0; e < numEntries; e++)
  {
    const WmmaEntry &me = entries[e];
    auto test = currentDeviceScope->beginTest({me.metric, me.title, me.unit});

    if (!rdna)
    {
      test.skip(me.metric, ResultStatus::Unsupported,
                "WMMA is an RDNA3+/RDNA4 (gfx11/gfx12) matrix-core feature; "
                "absent on this architecture");
      continue;
    }
    if (!wave32)
    {
      test.skip(me.metric, ResultStatus::Unsupported,
                "WMMA peak measured only in wave32 mode on this build");
      continue;
    }

    // Compiler-driven capability check: a missing builtin for this datatype
    // fails the HIPRTC compile. Suppress the log (quiet) and report Unsupported.
    hipFunction_t fn;
    if (!dev.getKernel(me.src, me.srcName, me.kernelName, fn, {}, /*quiet=*/true))
    {
      test.skip(me.metric, ResultStatus::Unsupported,
                "WMMA instruction for this datatype not available on this GPU");
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

    // Each wave issues kWmmaPerWave WMMA ops, each doing 2*M*N*K flops/ops.
    const double ops = (double)numBlocks * (double)kWmmaPerWave *
                       2.0 * (double)me.M * (double)me.N * (double)me.K;
    float value = (float)(ops * 1.0e6 / us / 1.0e12);
    test.emit(me.metric, value);

    (void)hipFree(outBuf);
  }

  return 0;
}

#endif // ENABLE_ROCM
