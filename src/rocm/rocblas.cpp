#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <cstring>

#ifdef CLPEAK_ROCM_HAS_ROCBLAS
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>
#endif

namespace {

uint32_t pickRocblasGemmDim(const rocm_device_info_t &info)
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

#ifdef CLPEAK_ROCM_HAS_ROCBLAS
template <typename Fn>
double timeRocblas(hipStream_t stream, Fn fn, unsigned int n)
{
  hipEvent_t start = nullptr, stop = nullptr;
  (void)hipEventCreate(&start);
  (void)hipEventCreate(&stop);

  (void)hipStreamSynchronize(stream);
  (void)hipEventRecord(start, stream);
  for (unsigned int i = 0; i < n; i++)
  {
    if (fn() != rocblas_status_success)
    {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
      return -1.0;
    }
  }
  (void)hipEventRecord(stop, stream);
  if (hipEventSynchronize(stop) != hipSuccess)
  {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return -1.0;
  }

  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, stop);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return (double)ms * 1000.0 / (double)n;
}
#endif

} // namespace

int RocmPeak::runRocblas(RocmDevice &dev, benchmark_config_t &)
{
  auto test = currentDeviceScope->beginTest(
    {"rocblas-fp", "rocBLAS GEMM peak", "tflops"});

#ifndef CLPEAK_ROCM_HAS_ROCBLAS
  test.skip("fp32", ResultStatus::Unsupported, "rocBLAS not found at configure time");
  test.skip("fp64", ResultStatus::Unsupported, "rocBLAS not found at configure time");
  test.skip("fp16", ResultStatus::Unsupported, "rocBLAS not found at configure time");
  return 0;
#else
  const uint32_t D = pickRocblasGemmDim(dev.info);
  const rocblas_int M = (rocblas_int)D;
  const rocblas_int N = (rocblas_int)D;
  const rocblas_int K = (rocblas_int)D;
  const double flops = 2.0 * (double)M * (double)N * (double)K;

  const size_t aBytes = (size_t)M * K * sizeof(double);
  const size_t bBytes = (size_t)K * N * sizeof(double);
  const size_t cBytes = (size_t)M * N * sizeof(double);

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (hipMalloc(&dA, aBytes) != hipSuccess ||
      hipMalloc(&dB, bBytes) != hipSuccess ||
      hipMalloc(&dC, cBytes) != hipSuccess)
  {
    test.skip("fp32", ResultStatus::Error, "Failed to allocate GEMM buffers");
    test.skip("fp64", ResultStatus::Error, "Failed to allocate GEMM buffers");
    test.skip("fp16", ResultStatus::Error, "Failed to allocate GEMM buffers");
    if (dA) (void)hipFree(dA);
    if (dB) (void)hipFree(dB);
    if (dC) (void)hipFree(dC);
    return -1;
  }

  (void)hipMemset(dA, 0x3f, aBytes);
  (void)hipMemset(dB, 0x3f, bBytes);
  (void)hipMemset(dC, 0, cBytes);

  rocblas_handle handle = nullptr;
  if (rocblas_create_handle(&handle) != rocblas_status_success)
  {
    test.skip("fp32", ResultStatus::Error, "rocblas_create_handle failed");
    test.skip("fp64", ResultStatus::Error, "rocblas_create_handle failed");
    test.skip("fp16", ResultStatus::Error, "rocblas_create_handle failed");
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    return -1;
  }
  (void)rocblas_set_stream(handle, dev.stream);

  auto runTimed = [&](const char *label, auto gemmFn) {
    const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
    double probeUs = timeRocblas(dev.stream, gemmFn, warm);
    if (probeUs <= 0.0)
    {
      test.skip(label, ResultStatus::Error, "timing probe failed");
      return;
    }
    unsigned int iters = pickIters(probeUs, 5000000u,
                                   forceIters ? specifiedIters : 0);
    double meanUs = timeRocblas(dev.stream, gemmFn, iters);
    if (meanUs <= 0.0)
    {
      test.skip(label, ResultStatus::Error, "rocBLAS GEMM failed");
      return;
    }
    test.emit(label, (float)(flops * 1.0e6 / meanUs / 1.0e12));
  };

  const float alpha32 = 1.0f, beta32 = 0.0f;
  runTimed("fp32", [&]() {
    return rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                         M, N, K, &alpha32,
                         (const float *)dA, M,
                         (const float *)dB, K,
                         &beta32,
                         (float *)dC, M);
  });

  const double alpha64 = 1.0, beta64 = 0.0;
  runTimed("fp64", [&]() {
    return rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                         M, N, K, &alpha64,
                         (const double *)dA, M,
                         (const double *)dB, K,
                         &beta64,
                         (double *)dC, M);
  });

  if (dev.info.fp16Supported)
  {
    // Use gemm_ex with f16 in/out and f32 *accumulate* (HPA). rocblas_hgemm
    // accumulates in fp16, which does not map to the fast fp16xfp16->fp32 MFMA
    // path and tops out far below peak; the f32-compute path reaches it.
    const float alphaf = 1.0f, betaf = 0.0f;
    runTimed("fp16", [&]() {
      return rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                             M, N, K, &alphaf,
                             dA, rocblas_datatype_f16_r, M,
                             dB, rocblas_datatype_f16_r, K,
                             &betaf,
                             dC, rocblas_datatype_f16_r, M,
                             dC, rocblas_datatype_f16_r, M,
                             rocblas_datatype_f32_r,
                             rocblas_gemm_algo_standard, 0, 0);
    });
  }
  else
  {
    test.skip("fp16", ResultStatus::Unsupported, "fp16 not supported by this ROCm device");
  }

  (void)rocblas_destroy_handle(handle);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  return 0;
#endif
}

#endif // ENABLE_ROCM
