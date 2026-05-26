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
// per MAD_16, 128 outer iterations.  Total per-WI = 128*16*2 = 4096 fp ops
// (= COMPUTE_FP_WORK_PER_WI).  Compiler-level fma() emits hardware FMA.
// --------------------------------------------------------------------------
#define MAD_4(x, y)  x = sycl::fma(y, x, y); y = sycl::fma(x, y, x); \
                     x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

// --------------------------------------------------------------------------
// Single precision
// --------------------------------------------------------------------------
class compute_sp_kernel;

int OneapiPeak::runComputeSP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"single_precision_compute", "Single-precision compute", "gflops"});

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
  uint64_t bufElems = totalThreads;

  float *out = sycl::malloc_device<float>(bufElems, dev.stream);
  if (!out)
  {
    test.skip("float", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  const float A = 1.3f;
  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_sp_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float x = A;
          float y = (float)it.get_local_id(0);
          #pragma unroll 1
          for (int i = 0; i < 128; i++) { MAD_16(x, y) }
          out[it.get_global_id(0)] = y;
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
    test.skip("float", ResultStatus::Error, "kernel launch failed");
  else
    test.emit("float", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9));

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Half precision — half and half2 (via sycl::vec<half,2>)
// --------------------------------------------------------------------------
class compute_hp_kernel;
class compute_hp2_kernel;

int OneapiPeak::runComputeHP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"half_precision_compute", "Half-precision compute", "gflops"});

  if (!dev.info.fp16Supported)
  {
    test.skip("half",  ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");
    test.skip("half2", ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  float *out = sycl::malloc_device<float>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("half",  ResultStatus::Error, "Failed to allocate output buffer");
    test.skip("half2", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const float A = 1.3f;

  // ---- half (scalar) — 64 outer iters * 2 MAD_16 sequences = 4096 ops/WI
  {
    auto submit = [&](sycl::queue &q) -> sycl::event {
      return q.submit([&](sycl::handler &h) {
        h.parallel_for<compute_hp_kernel>(
          sycl::nd_range<1>(totalThreads, blockSize),
          [=](sycl::nd_item<1> it) {
            sycl::half x0 = (sycl::half)A;
            sycl::half y0 = (sycl::half)(float)it.get_local_id(0);
            sycl::half x1 = (sycl::half)(A + 1.0f);
            sycl::half y1 = (sycl::half)((float)it.get_local_id(0) + 7.0f);
            #pragma unroll 1
            for (int i = 0; i < 64; i++) { MAD_16(x0, y0) MAD_16(x1, y1) }
            out[it.get_global_id(0)] = (float)(x0 + y0 + x1 + y1);
          });
      });
    };
    float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f) test.skip("half", ResultStatus::Error, "kernel launch failed");
    else            test.emit("half", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9));
  }

  // ---- half2 (vec<half,2>) — 64 iters * MAD2_16 (32 fp ops/lane) = 4096 ops/WI
  {
    using half2 = sycl::vec<sycl::half, 2>;
    auto submit = [&](sycl::queue &q) -> sycl::event {
      return q.submit([&](sycl::handler &h) {
        h.parallel_for<compute_hp2_kernel>(
          sycl::nd_range<1>(totalThreads, blockSize),
          [=](sycl::nd_item<1> it) {
            half2 x{(sycl::half)A, (sycl::half)A};
            half2 y{(sycl::half)(float)it.get_local_id(0),
                    (sycl::half)(float)it.get_local_id(0)};
            #pragma unroll 1
            for (int i = 0; i < 64; i++) {
              // 16 fma calls per iter on a 2-lane vector = 32 fp ops/WI/iter
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
              x = sycl::fma(y, x, y); y = sycl::fma(x, y, x);
            }
            half2 r = x + y;
            out[it.get_global_id(0)] = (float)(r[0] + r[1]);
          });
      });
    };
    float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f) test.skip("half2", ResultStatus::Error, "kernel launch failed");
    else            test.emit("half2", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9));
  }

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Double precision — workPerWI = 512 (matches COMPUTE_DP_WORK_PER_WI).
// 16 outer iters * MAD_16 (32 ops) = 512 ops/WI.
// --------------------------------------------------------------------------
class compute_dp_kernel;

int OneapiPeak::runComputeDP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"double_precision_compute", "Double-precision compute", "gflops"});

  if (!dev.info.fp64Supported)
  {
    test.skip("double", ResultStatus::Unsupported, "fp64 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(double));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  double *out = sycl::malloc_device<double>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("double", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const double A = 1.3;

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_dp_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          double x = A;
          double y = (double)it.get_local_id(0);
          #pragma unroll 1
          for (int i = 0; i < 16; i++) { MAD_16(x, y) }
          out[it.get_global_id(0)] = y;
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
    test.skip("double", ResultStatus::Error, "kernel launch failed");
  else
    test.emit("double", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_DP_WORK_PER_WI, us, 1e9));

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Mixed precision (fp16 multiply -> fp32 accumulate).  Mirrors compute_mp.hip:
// round-trip through half to force the lower-precision multiply, accumulate
// in float.  4096 ops/WI.
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
// 16 outer iters * MAD_128 (256 ops) = 4096 ops/WI.
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
