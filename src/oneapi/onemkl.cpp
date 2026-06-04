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

  // Allocate enough memory for the widest type (fp64) and reuse for each
  // precision via reinterpret.  Same trick the rocBLAS benchmark uses.
  // Skip every label for the active phase with one error message.
  auto skipPhase = [&](ResultStatus status, const char *msg) {
    if (fpPhase)
    {
      test.skip("fp32", status, msg);
      test.skip("fp64", status, msg);
      test.skip("fp16", status, msg);
      test.skip("bf16", status, msg);
    }
    else
    {
      test.skip("int8", status, msg);
    }
  };

  const size_t cells = (size_t)D * (size_t)D;
  void *dA = sycl::malloc_device(cells * sizeof(double), dev.stream);
  void *dB = sycl::malloc_device(cells * sizeof(double), dev.stream);
  void *dC = sycl::malloc_device(cells * sizeof(double), dev.stream);
  if (!dA || !dB || !dC)
  {
    skipPhase(ResultStatus::Error, "Failed to allocate GEMM buffers");
    if (dA) sycl::free(dA, dev.stream);
    if (dB) sycl::free(dB, dev.stream);
    if (dC) sycl::free(dC, dev.stream);
    return -1;
  }
  try { dev.stream.memset(dA, 0x3f, cells * sizeof(double)).wait(); } catch (...) {}
  try { dev.stream.memset(dB, 0x3f, cells * sizeof(double)).wait(); } catch (...) {}
  try { dev.stream.memset(dC, 0,    cells * sizeof(double)).wait(); } catch (...) {}

  auto timeBatch = [&](const char *label, auto launchFn) {
    auto runBatch = [&](unsigned int n) -> double {
      try
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < n; i++)
          launchFn();
        dev.stream.wait_and_throw();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return (double)ns / 1000.0 / (double)n;
      }
      catch (const sycl::exception &e)
      {
        CLPEAK_VLOG("oneMKL %s failed: %s\n", label, e.what());
        // Recover the shared queue: a failed GEMM can wedge the in-order queue
        // and cascade "kernel launch failed" into every later benchmark.
        dev.resetQueue();
        return -1.0;
      }
      catch (const std::exception &e)
      {
        CLPEAK_VLOG("oneMKL %s failed: %s\n", label, e.what());
        dev.resetQueue();
        return -1.0;
      }
    };

    const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
    double probeUs = runBatch(warm);
    if (probeUs <= 0.0)
    {
      test.skip(label, ResultStatus::Error, "timing probe failed");
      return;
    }
    unsigned int iters = pickIters(probeUs, 5000000u, forceIters ? specifiedIters : 0);
    double meanUs = runBatch(iters);
    if (meanUs <= 0.0)
    {
      test.skip(label, ResultStatus::Error, "oneMKL GEMM failed");
      return;
    }
    test.emit(label, (float)(flops * 1.0e6 / meanUs / 1.0e12));
  };

  if (fpPhase)
  {
    // FP32
    timeBatch("fp32", [&]() {
      mkl::blas::row_major::gemm(
        dev.stream, mkl::transpose::nontrans, mkl::transpose::nontrans,
        M, N, K, 1.0f,
        (const float *)dA, K,
        (const float *)dB, N,
        0.0f, (float *)dC, N);
    });

    // FP64
    if (dev.info.fp64Supported)
    {
      timeBatch("fp64", [&]() {
        mkl::blas::row_major::gemm(
          dev.stream, mkl::transpose::nontrans, mkl::transpose::nontrans,
          M, N, K, 1.0,
          (const double *)dA, K,
          (const double *)dB, N,
          0.0, (double *)dC, N);
      });
    }
    else
    {
      test.skip("fp64", ResultStatus::Unsupported, "fp64 not supported by this oneAPI device");
    }

    // FP16
    if (dev.info.fp16Supported)
    {
      timeBatch("fp16", [&]() {
        mkl::blas::row_major::gemm(
          dev.stream, mkl::transpose::nontrans, mkl::transpose::nontrans,
          M, N, K, sycl::half(1.0f),
          (const sycl::half *)dA, K,
          (const sycl::half *)dB, N,
          sycl::half(0.0f), (sycl::half *)dC, N);
      });
    }
    else
    {
      test.skip("fp16", ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");
    }

    // BF16: bf16 inputs, fp32 output + accumulate (HPA) -- the dtype combo the
    // Intel XMX bf16 GEMM peak is quoted against, matching joint_matrix.cpp.
#if defined(CLPEAK_ONEMKL_HAS_BF16)
    if (dev.info.bf16Supported)
    {
      using bfloat16 = sycl::ext::oneapi::bfloat16;
      timeBatch("bf16", [&]() {
        mkl::blas::row_major::gemm(
          dev.stream, mkl::transpose::nontrans, mkl::transpose::nontrans,
          M, N, K, 1.0f,
          (const bfloat16 *)dA, K,
          (const bfloat16 *)dB, N,
          0.0f, (float *)dC, N);
      });
    }
    else
    {
      test.skip("bf16", ResultStatus::Unsupported, "bf16 not supported by this oneAPI device");
    }
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
    // the timeBatch lambda reports a clean skip.
    if (!dev.info.xmxSupported)
    {
      test.skip("int8", ResultStatus::Unsupported,
                "int8 GEMM requires Intel XMX (Arc/PVC/Battlemage)");
    }
    else
    {
      // gemm_bias with offset::fix reads a single int32 bias from `co`.
      void *dCo = sycl::malloc_device(sizeof(std::int32_t), dev.stream);
      if (!dCo)
      {
        test.skip("int8", ResultStatus::Error, "Failed to allocate int8 bias buffer");
      }
      else
      {
        try { dev.stream.memset(dCo, 0, sizeof(std::int32_t)).wait(); } catch (...) {}
        timeBatch("int8", [&]() {
          mkl::blas::row_major::gemm_bias(
            dev.stream, mkl::transpose::nontrans, mkl::transpose::nontrans,
            mkl::offset::fix,
            M, N, K, 1.0f,
            (const std::int8_t *)dA, K, (std::int8_t)0,
            (const std::uint8_t *)dB, N, (std::uint8_t)0,
            0.0f,
            (std::int32_t *)dC, N, (const std::int32_t *)dCo);
        });
        sycl::free(dCo, dev.stream);
      }
    }
  }

  sycl::free(dA, dev.stream);
  sycl::free(dB, dev.stream);
  sycl::free(dC, dev.stream);
  return 0;
#endif
}

#endif // ENABLE_ONEAPI
