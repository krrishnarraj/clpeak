#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>
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
  const uint32_t D = pickOnemklGemmDim(dev.info);
  const std::int64_t M = D, N = D, K = D;
  const double flops = 2.0 * (double)M * (double)N * (double)K;

  const size_t cells = (size_t)D * (size_t)D;

  // Run ONE GEMM dtype in full isolation: its own context + queue + buffers,
  // all torn down before the next dtype.  Two reasons:
  //  1. A faulting GEMM (e.g. fp64 returning a *sticky* CL_OUT_OF_RESOURCES on
  //     some drivers) corrupts only this disposable context, so the shared
  //     dev.stream stays healthy for every later benchmark.
  //  2. Per-dtype isolation means each dtype reports its own pass/fail instead
  //     of one bad dtype poisoning all the rest — a precise signal for the
  //     driver team about exactly which GEMM dtype faults.
  // gemmFn(q, dA, dB, dC, dCo) issues one GEMM (dCo is the int8 bias buffer,
  // unused by the FP paths).  Buffers are sized at fp64 width and reinterpreted.
  auto measure = [&](const char *label, auto gemmFn) {
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
          gemmFn(q, dA, dB, dC, dCo);
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
    measure("fp32", [&](sycl::queue &q, void *dA, void *dB, void *dC, void *) {
      mkl::blas::row_major::gemm(
        q, mkl::transpose::nontrans, mkl::transpose::nontrans,
        M, N, K, 1.0f,
        (const float *)dA, K, (const float *)dB, N, 0.0f, (float *)dC, N);
    });

    if (dev.info.fp64Supported)
      measure("fp64", [&](sycl::queue &q, void *dA, void *dB, void *dC, void *) {
        mkl::blas::row_major::gemm(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          M, N, K, 1.0,
          (const double *)dA, K, (const double *)dB, N, 0.0, (double *)dC, N);
      });
    else
      test.skip("fp64", ResultStatus::Unsupported, "fp64 not supported by this oneAPI device");

    if (dev.info.fp16Supported)
      measure("fp16", [&](sycl::queue &q, void *dA, void *dB, void *dC, void *) {
        mkl::blas::row_major::gemm(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          M, N, K, sycl::half(1.0f),
          (const sycl::half *)dA, K, (const sycl::half *)dB, N,
          sycl::half(0.0f), (sycl::half *)dC, N);
      });
    else
      test.skip("fp16", ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");

    // BF16: bf16 inputs, fp32 output + accumulate (HPA) -- the dtype combo the
    // Intel XMX bf16 GEMM peak is quoted against, matching joint_matrix.cpp.
#if defined(CLPEAK_ONEMKL_HAS_BF16)
    if (dev.info.bf16Supported)
      measure("bf16", [&](sycl::queue &q, void *dA, void *dB, void *dC, void *) {
        using bfloat16 = sycl::ext::oneapi::bfloat16;
        mkl::blas::row_major::gemm(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          M, N, K, 1.0f,
          (const bfloat16 *)dA, K, (const bfloat16 *)dB, N, 0.0f, (float *)dC, N);
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
      measure("int8", [&](sycl::queue &q, void *dA, void *dB, void *dC, void *dCo) {
        // gemm_bias with offset::fix reads a single int32 bias from `co`.
        mkl::blas::row_major::gemm_bias(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans,
          mkl::offset::fix,
          M, N, K, 1.0f,
          (const std::int8_t *)dA, K, (std::int8_t)0,
          (const std::uint8_t *)dB, N, (std::uint8_t)0,
          0.0f,
          (std::int32_t *)dC, N, (const std::int32_t *)dCo);
      });
  }

  return 0;
#endif
}

#endif // ENABLE_ONEAPI
