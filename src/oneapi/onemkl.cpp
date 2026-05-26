#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>
#include <chrono>

#ifdef CLPEAK_ONEAPI_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

// oneMKL GEMM peak — analog of rocBLAS / cuBLAS / MPSGraph.  Reports tflops
// for FP32, FP64, FP16 (gated by device fp16 aspect).
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

int OneapiPeak::runOnemkl(OneapiDevice &dev, benchmark_config_t &)
{
  auto test = currentDeviceScope->beginTest(
    {"onemkl-fp", "oneMKL GEMM peak", "tflops"});

#ifndef CLPEAK_ONEAPI_HAS_ONEMKL
  test.skip("fp32", ResultStatus::Unsupported, "oneMKL not found at configure time");
  test.skip("fp64", ResultStatus::Unsupported, "oneMKL not found at configure time");
  test.skip("fp16", ResultStatus::Unsupported, "oneMKL not found at configure time");
  return 0;
#else
  namespace mkl = oneapi::mkl;
  const uint32_t D = pickOnemklGemmDim(dev.info);
  const std::int64_t M = D, N = D, K = D;
  const double flops = 2.0 * (double)M * (double)N * (double)K;

  // Allocate enough memory for the widest type (fp64) and reuse for each
  // precision via reinterpret.  Same trick the rocBLAS benchmark uses.
  const size_t cells = (size_t)D * (size_t)D;
  void *dA = sycl::malloc_device(cells * sizeof(double), dev.stream);
  void *dB = sycl::malloc_device(cells * sizeof(double), dev.stream);
  void *dC = sycl::malloc_device(cells * sizeof(double), dev.stream);
  if (!dA || !dB || !dC)
  {
    test.skip("fp32", ResultStatus::Error, "Failed to allocate GEMM buffers");
    test.skip("fp64", ResultStatus::Error, "Failed to allocate GEMM buffers");
    test.skip("fp16", ResultStatus::Error, "Failed to allocate GEMM buffers");
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
        fprintf(stderr, "oneMKL %s failed: %s\n", label, e.what());
        return -1.0;
      }
      catch (const std::exception &e)
      {
        fprintf(stderr, "oneMKL %s failed: %s\n", label, e.what());
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

  sycl::free(dA, dev.stream);
  sycl::free(dB, dev.stream);
  sycl::free(dC, dev.stream);
  return 0;
#endif
}

#endif // ENABLE_ONEAPI
