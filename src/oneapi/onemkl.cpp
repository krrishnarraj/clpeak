#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>

#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#define CLPEAK_ONEMKL_HAS_BF16 1
#endif

#ifdef CLPEAK_ONEAPI_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

// oneMKL GEMM peak — analog of rocBLAS / cuBLAS / MPSGraph.  FP category
// reports tflops for FP32, FP64, FP16, BF16 (each gated by device aspect);
// INT category reports tops for INT8 (s8 x u8 -> s32 via gemm_bias).  This
// mirrors the datatype coverage of the joint_matrix microbenchmark so every
// SDK-supported GEMM dtype is measured by both the raw and the library path.
//
// Sizing matches the ROCm rocBLAS benchmark: D = round_up_256(2048 + 128*CUs),
// clamped to [2048, 16384], halved while 3*D*D*8 > totalGlobalMem/4.

static uint32_t pickOnemklGemmDim(const oneapi_device_info_t &info)
{
  uint32_t cus = (uint32_t)(info.numCUs > 0 ? info.numCUs : 16);
  uint64_t D = 2048 + (uint64_t)cus * 128;
  D = (D + 255) & ~uint64_t(255);
  if (D < 2048)  D = 2048;
  if (D > 16384) D = 16384;
  uint64_t budget = info.totalGlobalMem ? info.totalGlobalMem / 4 : ((uint64_t)4 << 30);
  while (D > 1024 && 3ULL * D * D * 8 > budget)
    D /= 2;
  return (uint32_t)D;
}

int OneapiPeak::runOnemkl(OneapiDevice &dev, benchmark_config_t &, Category category)
{
  const bool fpPhase = (category != Category::IntCompute);

  auto test = fpPhase
    ? currentDeviceScope->beginTest({"onemkl-fp", "oneMKL GEMM peak", "tflops"})
    : currentDeviceScope->beginTest({"onemkl-int", "oneMKL GEMM peak", "tops"});

#ifndef CLPEAK_ONEAPI_HAS_ONEMKL
  if (fpPhase)
  {
    test.skip("fp32", ResultStatus::Unsupported, "oneMKL not found at configure time");
    test.skip("fp64", ResultStatus::Unsupported, "oneMKL not found at configure time");
    test.skip("fp16", ResultStatus::Unsupported, "oneMKL not found at configure time");
    test.skip("bf16", ResultStatus::Unsupported, "oneMKL not found at configure time");
  }
  else
  {
    test.skip("int8", ResultStatus::Unsupported, "oneMKL not found at configure time");
  }
  return 0;
#else
  namespace mkl = oneapi::mkl;
  const std::int64_t D = pickOnemklGemmDim(dev.info);

  // fp64 throughput on most GPUs is a small fraction of fp32 (often 1/16..1/64+,
  // measured ~1/16 on Arc-class parts).  A full-size fp64 GEMM can therefore run
  // for many seconds in a SINGLE call and trip the GPU watchdog, surfacing as
  // CL_OUT_OF_RESOURCES.  Shrink the fp64 tile so one call stays short;
  // pickIters() then runs proportionally more iterations to still fill the 5 s
  // budget, so the measured peak is unaffected (a 3584^3 GEMM still saturates).
  const std::int64_t fp64Dim = std::max<std::int64_t>(1024, D / 4);

  // Run ONE GEMM dtype in full isolation: its own context + queue + buffers,
  // all torn down before the next dtype.  Two reasons:
  //  1. A faulting GEMM (e.g. fp64 returning a *sticky* CL_OUT_OF_RESOURCES on
  //     some drivers) corrupts only this disposable context, so the shared
  //     dev.stream stays healthy for every later benchmark.
  //  2. Per-dtype isolation means each dtype reports its own pass/fail instead
  //     of one bad dtype poisoning all the rest — a precise signal for the
  //     driver team about exactly which GEMM dtype faults.
  // gemmFn(q, dA, dB, dC, dCo, dim) issues one square dim*dim*dim GEMM (dCo is
  // the int8 bias buffer, unused by the FP paths).  `dim` lets fp64 use a
  // smaller tile than the other dtypes.  Buffers are sized at fp64 width and
  // reinterpreted per dtype.
  auto measure = [&](const char *label, std::int64_t dim, auto gemmFn) {
    const size_t cells = (size_t)dim * (size_t)dim;
    const double flops = 2.0 * (double)dim * (double)dim * (double)dim;
    sycl::queue q = [&]() -> sycl::queue {
      try
      {
        return sycl::queue(sycl::context(dev.dev), dev.dev,
                           sycl::property::queue::in_order{});
      }
      catch (const std::exception &e)
      {
        CLPEAK_VLOG("oneMKL %s: private context create failed (%s); shared queue\n",
                    label, e.what());
        return dev.stream;
      }
    }();

    void *dA  = sycl::malloc_device(cells * sizeof(double), q);
    void *dB  = sycl::malloc_device(cells * sizeof(double), q);
    void *dC  = sycl::malloc_device(cells * sizeof(double), q);
    void *dCo = sycl::malloc_device(sizeof(std::int32_t), q);  // int8 bias
    auto freeAll = [&]() {
      if (dA)  { try { sycl::free(dA,  q); } catch (...) {} }
      if (dB)  { try { sycl::free(dB,  q); } catch (...) {} }
      if (dC)  { try { sycl::free(dC,  q); } catch (...) {} }
      if (dCo) { try { sycl::free(dCo, q); } catch (...) {} }
    };
    if (!dA || !dB || !dC || !dCo)
    {
      test.skip(label, ResultStatus::Error, "Failed to allocate GEMM buffers");
      freeAll();
      return;
    }
    try { q.memset(dA,  0x3f, cells * sizeof(double)).wait(); } catch (...) {}
    try { q.memset(dB,  0x3f, cells * sizeof(double)).wait(); } catch (...) {}
    try { q.memset(dC,  0,    cells * sizeof(double)).wait(); } catch (...) {}
    try { q.memset(dCo, 0,    sizeof(std::int32_t)).wait();   } catch (...) {}

    auto runBatch = [&](unsigned int n) -> double {
      try
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < n; i++)
          gemmFn(q, dA, dB, dC, dCo, dim);
        q.wait_and_throw();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return (double)ns / 1000.0 / (double)n;
      }
      catch (const std::exception &e)
      {
        // Contained to this dtype's private context (see above).
        CLPEAK_VLOG("oneMKL %s failed: %s\n", label, e.what());
        return -1.0;
      }
    };

    const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
    double probeUs = runBatch(warm);
    if (probeUs <= 0.0)
      test.skip(label, ResultStatus::Error, "timing probe failed");
    else
    {
      unsigned int iters = pickIters(probeUs, 5000000u, forceIters ? specifiedIters : 0);
      double meanUs = runBatch(iters);
      if (meanUs <= 0.0)
        test.skip(label, ResultStatus::Error, "oneMKL GEMM failed");
      else
        test.emit(label, (float)(flops * 1.0e6 / meanUs / 1.0e12));
    }
    freeAll();
    // q and its private context are destroyed here.
  };

  if (fpPhase)
  {
    measure("fp32", D, [&](sycl::queue &q, void *dA, void *dB, void *dC, void *,
                           std::int64_t n) {
      mkl::blas::row_major::gemm(
        q, mkl::transpose::nontrans, mkl::transpose::nontrans,
        n, n, n, 1.0f,
        (const float *)dA, n, (const float *)dB, n, 0.0f, (float *)dC, n);
    });

    if (dev.info.fp64Supported)
      measure("fp64", fp64Dim, [&](sycl::queue &q, void *dA, void *dB, void *dC, void *,
                                   std::int64_t n) {
        mkl::blas::row_major::gemm(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          n, n, n, 1.0,
          (const double *)dA, n, (const double *)dB, n, 0.0, (double *)dC, n);
      });
    else
      test.skip("fp64", ResultStatus::Unsupported, "fp64 not supported by this oneAPI device");

    if (dev.info.fp16Supported)
      measure("fp16", D, [&](sycl::queue &q, void *dA, void *dB, void *dC, void *,
                             std::int64_t n) {
        mkl::blas::row_major::gemm(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          n, n, n, sycl::half(1.0f),
          (const sycl::half *)dA, n, (const sycl::half *)dB, n,
          sycl::half(0.0f), (sycl::half *)dC, n);
      });
    else
      test.skip("fp16", ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");

    // BF16: bf16 inputs, fp32 output + accumulate (HPA) -- the dtype combo the
    // Intel XMX bf16 GEMM peak is quoted against, matching joint_matrix.cpp.
#if defined(CLPEAK_ONEMKL_HAS_BF16)
    if (dev.info.bf16Supported)
      measure("bf16", D, [&](sycl::queue &q, void *dA, void *dB, void *dC, void *,
                             std::int64_t n) {
        using bfloat16 = sycl::ext::oneapi::bfloat16;
        mkl::blas::row_major::gemm(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          n, n, n, 1.0f,
          (const bfloat16 *)dA, n, (const bfloat16 *)dB, n, 0.0f, (float *)dC, n);
      });
    else
      test.skip("bf16", ResultStatus::Unsupported, "bf16 not supported by this oneAPI device");
#else
    test.skip("bf16", ResultStatus::Unsupported,
              "SYCL bfloat16 header not available in this oneAPI toolchain");
#endif
  }
  else
  {
    // INT8: s8 x u8 -> s32 via gemm_bias (the oneMKL integer GEMM entrypoint).
    // Zero offsets / zero bias -- we only measure throughput, so the numeric
    // result is irrelevant.  Reported in tops.  Gate on XMX (Arc/PVC/Battlemage
    // carry the int8 matmul path); on devices without it gemm_bias throws and
    // measure() reports a clean skip.
    if (!dev.info.xmxSupported)
      test.skip("int8", ResultStatus::Unsupported,
                "int8 GEMM requires Intel XMX (Arc/PVC/Battlemage)");
    else
      measure("int8", D, [&](sycl::queue &q, void *dA, void *dB, void *dC, void *dCo,
                             std::int64_t n) {
        // gemm_bias with offset::fix reads a single int32 bias from `co`.
        mkl::blas::row_major::gemm_bias(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          mkl::offset::fix,
          n, n, n, 1.0f,
          (const std::int8_t *)dA, n, (std::int8_t)0,
          (const std::uint8_t *)dB, n, (std::uint8_t)0,
          0.0f,
          (std::int32_t *)dC, n, (const std::int32_t *)dCo);
      });
  }

  return 0;
#endif
}

#endif // ENABLE_ONEAPI
