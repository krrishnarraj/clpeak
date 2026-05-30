#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>

#include <sycl/sycl.hpp>
#if defined(CLPEAK_ONEAPI_HAS_BF16) || __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif

namespace clpeak_oneapi {
uint32_t pickComputeBlocks(const oneapi_device_info_t &info,
                           uint32_t blockSize, uint32_t outElemsPerBlock,
                           uint32_t elemSize);
float    computeGflops(uint64_t totalThreads, uint32_t workPerWI, float meanUs,
                       double unitDivider);
}

// --------------------------------------------------------------------------
// MAD macro shape matches the ROCm/CUDA/OpenCL backends: 16 fused mul-adds
// per MAD_16.  The alternating read/write (x depends on y, then y on x)
// builds a dependency chain the compiler cannot hoist or vectorize away.
// One MAD_16 = 16 fma = 32 flops per lane.
//
// Total ops/WI is width-invariant: for vector width W we run baseIters/W
// outer iterations, each doing 32*W flops, so total = baseIters*32 flops/WI.
// SP/HP: baseIters=128 -> 4096 (COMPUTE_FP_WORK_PER_WI).
// DP:    baseIters=16  -> 512  (COMPUTE_DP_WORK_PER_WI).
// --------------------------------------------------------------------------
#define MAD_4(x, y)  x = sycl::fma(y, x, y); y = sycl::fma(x, y, x); \
                     x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

// Per-family kernel-name tags (SYCL needs a unique type per parallel_for).
namespace { struct SpTag; struct HpTag; struct DpTag; }
template <typename Tag, typename T, int W> class compute_fp_vec_kernel;

// One vector-width variant of an FP compute test.  Builds sycl::vec<T,W>
// with distinct per-lane seeds (so the compiler can't collapse the vector to
// a scalar broadcast), runs the FMA dependency chain, reduces lanes into the
// output, times via runKernel, and emits the metric.
template <typename Tag, typename T, int W>
static void runFpWidth(OneapiPeak &peak, OneapiDevice &dev,
                       logger::TestScope &test, const char *label,
                       T *out, uint64_t totalThreads, uint32_t blockSize,
                       int baseIters, double scalarA, uint32_t workPerWI,
                       unsigned int targetTimeUs, unsigned int forced)
{
  using VecT = sycl::vec<T, W>;
  int iters = baseIters / W;
  if (iters < 1) iters = 1;
  const T A = (T)scalarA;

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_fp_vec_kernel<Tag, T, W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          VecT x, y;
          #pragma unroll
          for (int k = 0; k < W; k++)
          {
            x[k] = (T)((double)A + (double)k);
            y[k] = (T)((double)it.get_local_id(0) + (double)k);
          }
          #pragma unroll 1
          for (int i = 0; i < iters; i++) { MAD_16(x, y) }
          VecT r = x + y;
          T acc = (T)0;
          #pragma unroll
          for (int k = 0; k < W; k++) acc += r[k];
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  float us = peak.runKernel(dev, submit, targetTimeUs, forced);
  if (us <= 0.0f) test.skip(label, ResultStatus::Error, "kernel launch failed");
  else            test.emit(label, clpeak_oneapi::computeGflops(totalThreads, workPerWI, us, 1e9));
}

// Drive the {1,2,4,8,16} sweep for one FP family.
template <typename Tag, typename T>
static void runFpSweep(OneapiPeak &peak, OneapiDevice &dev,
                       logger::TestScope &test, const char *baseLabel,
                       T *out, uint64_t totalThreads, uint32_t blockSize,
                       int baseIters, double scalarA, uint32_t workPerWI,
                       unsigned int targetTimeUs, unsigned int forced)
{
  const std::string b(baseLabel);
  runFpWidth<Tag, T, 1 >(peak, dev, test, b.c_str(),          out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 2 >(peak, dev, test, (b + "2").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 4 >(peak, dev, test, (b + "4").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 8 >(peak, dev, test, (b + "8").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 16>(peak, dev, test, (b + "16").c_str(), out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
}

// --------------------------------------------------------------------------
// Single precision — float / float2 / float4 / float8 / float16
// --------------------------------------------------------------------------
int OneapiPeak::runComputeSP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"single_precision_compute", "Single-precision compute", "gflops"});

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  float *out = sycl::malloc_device<float>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"float", "float2", "float4", "float8", "float16"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  runFpSweep<SpTag, float>(*this, dev, test, "float", out, totalThreads, blockSize,
                           /*baseIters=*/128, /*A=*/1.3, COMPUTE_FP_WORK_PER_WI,
                           cfg.targetTimeUs, forceIters ? specifiedIters : 0);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Half precision — half / half2 / half4 / half8 / half16
// --------------------------------------------------------------------------
int OneapiPeak::runComputeHP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"half_precision_compute", "Half-precision compute", "gflops"});

  if (!dev.info.fp16Supported)
  {
    test.skipAll({"half", "half2", "half4", "half8", "half16"},
                 ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(sycl::half));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  sycl::half *out = sycl::malloc_device<sycl::half>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"half", "half2", "half4", "half8", "half16"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  runFpSweep<HpTag, sycl::half>(*this, dev, test, "half", out, totalThreads, blockSize,
                                /*baseIters=*/128, /*A=*/1.3, COMPUTE_FP_WORK_PER_WI,
                                cfg.targetTimeUs, forceIters ? specifiedIters : 0);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Double precision — double / double2 / double4 / double8 / double16
// workPerWI = 512 (COMPUTE_DP_WORK_PER_WI), baseIters = 16.
// --------------------------------------------------------------------------
int OneapiPeak::runComputeDP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"double_precision_compute", "Double-precision compute", "gflops"});

  if (!dev.info.fp64Supported)
  {
    test.skipAll({"double", "double2", "double4", "double8", "double16"},
                 ResultStatus::Unsupported, "fp64 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(double));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  double *out = sycl::malloc_device<double>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"double", "double2", "double4", "double8", "double16"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  runFpSweep<DpTag, double>(*this, dev, test, "double", out, totalThreads, blockSize,
                            /*baseIters=*/16, /*A=*/1.3, COMPUTE_DP_WORK_PER_WI,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Mixed precision (fp16 multiply -> fp32 accumulate).  Mirrors compute_mp.hip:
// round-trip through half to force the lower-precision multiply, accumulate
// in float.  Scalar only (the round-trip is inherently per-element).  4096 ops/WI.
// --------------------------------------------------------------------------
class compute_mp_kernel;

int OneapiPeak::runComputeMP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"mixed_precision_compute", "Mixed-precision compute fp16xfp16+fp32", "gflops"});

  if (!dev.info.fp16Supported)
  {
    test.skip("mp", ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  float *out = sycl::malloc_device<float>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("mp", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const float A = 1.3f;

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_mp_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float x = (float)(sycl::half)A;
          float y = (float)(sycl::half)(float)it.get_local_id(0);
          #pragma unroll 1
          for (int i = 0; i < 128; i++) {
            MAD_16(x, y)
            x = (float)(sycl::half)x;
            y = (float)(sycl::half)y;
          }
          out[it.get_global_id(0)] = x + y;
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
    test.skip("mp", ResultStatus::Error, "kernel launch failed");
  else
    test.emit("mp", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9));

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// BF16 compute (bf16xbf16 -> fp32 accumulate).  Gated by aspect probe;
// emulated on iGPUs without native bf16, hardware on Arc/PVC/Battlemage.
// Scalar only.  16 outer iters * MAD_128 (256 ops) = 4096 ops/WI.
// --------------------------------------------------------------------------
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
class compute_bf16_kernel;

int OneapiPeak::runComputeBF16(OneapiDevice &dev, benchmark_config_t &cfg)
{
  using bfloat16 = sycl::ext::oneapi::bfloat16;

  auto test = currentDeviceScope->beginTest(
    {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops"});

  if (!dev.info.bf16Supported)
  {
    test.skip("bf16", ResultStatus::Unsupported, "bf16 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  float *out = sycl::malloc_device<float>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("bf16", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const float A = 1.3f;

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_bf16_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float x = (float)bfloat16(A);
          float y = (float)bfloat16((float)it.get_local_id(0));
          #pragma unroll 1
          for (int i = 0; i < 16; i++) {
            // MAD_128 = 8 * MAD_16 = 256 ops
            MAD_16(x, y) MAD_16(x, y) MAD_16(x, y) MAD_16(x, y)
            MAD_16(x, y) MAD_16(x, y) MAD_16(x, y) MAD_16(x, y)
            x = (float)bfloat16(x);
            y = (float)bfloat16(y);
          }
          out[it.get_global_id(0)] = x + y;
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
    test.skip("bf16", ResultStatus::Error, "kernel launch failed");
  else
    test.emit("bf16", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9));

  sycl::free(out, dev.stream);
  return 0;
}
#else
int OneapiPeak::runComputeBF16(OneapiDevice &, benchmark_config_t &)
{
  auto test = currentDeviceScope->beginTest(
    {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops"});
  test.skip("bf16", ResultStatus::Unsupported,
            "SYCL bfloat16 header not available in this oneAPI toolchain");
  return 0;
}
#endif

#endif // ENABLE_ONEAPI
