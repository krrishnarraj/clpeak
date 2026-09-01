#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <cstring>
#include <string>

#ifdef CLPEAK_ROCM_HAS_ROCBLAS
#include <rocblas/rocblas.h>
#include <hip/hip_fp16.h>
#include <common/dynlib.h>

// Optional rocBLAS loader -- rocBLAS is not part of the HIP runtime, so it is
// resolved at run time; if absent the GEMM benchmark is skipped and the rest of
// the ROCm backend still runs.  Function-pointer types come from the header via
// decltype; macros redirect the call sites so the body below is unchanged.
namespace {
struct RocblasApi
{
  void *lib = nullptr;
  decltype(&::rocblas_create_handle)  create_handle = nullptr;
  decltype(&::rocblas_destroy_handle) destroy_handle = nullptr;
  decltype(&::rocblas_set_stream)     set_stream = nullptr;
  decltype(&::rocblas_sgemm)          sgemm = nullptr;
  decltype(&::rocblas_dgemm)          dgemm = nullptr;
  decltype(&::rocblas_gemm_ex)        gemm_ex = nullptr;
  bool load();
};
RocblasApi g_rb;
bool RocblasApi::load()
{
  if (lib)
    return true;
  lib = clpeak::dynOpen({"librocblas.so", "librocblas.so.4", "librocblas.so.3",
                         "rocblas.dll"});
  if (!lib)
    return false;
  bool ok = true;
#define CLPEAK_RB_SYM(member, name)                                       \
  member = reinterpret_cast<decltype(member)>(clpeak::dynSym(lib, name)); \
  ok = ok && (member != nullptr)
  CLPEAK_RB_SYM(create_handle,  "rocblas_create_handle");
  CLPEAK_RB_SYM(destroy_handle, "rocblas_destroy_handle");
  CLPEAK_RB_SYM(set_stream,     "rocblas_set_stream");
  CLPEAK_RB_SYM(sgemm,          "rocblas_sgemm");
  CLPEAK_RB_SYM(dgemm,          "rocblas_dgemm");
  CLPEAK_RB_SYM(gemm_ex,        "rocblas_gemm_ex");
#undef CLPEAK_RB_SYM
  if (!ok)
  {
    clpeak::dynClose(lib);
    lib = nullptr;
  }
  return ok;
}
} // namespace

#define rocblas_create_handle  g_rb.create_handle
#define rocblas_destroy_handle g_rb.destroy_handle
#define rocblas_set_stream     g_rb.set_stream
#define rocblas_sgemm          g_rb.sgemm
#define rocblas_dgemm          g_rb.dgemm
// rocBLAS's header defines rocblas_gemm_ex as a function-like macro; drop it so
// our object-like redirect to the loaded pointer takes over cleanly.
#undef rocblas_gemm_ex
#define rocblas_gemm_ex        g_rb.gemm_ex
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


  // One note per dtype row, shared by every emit and skip path below.
  const char *fp32Note = "Full 32-bit precision, on the ordinary shader cores "
                         "rather than the matrix cores.";
  const char *fp64Note = "Full 64-bit precision, for scientific computing.  "
                         "Gaming cards run this far slower than the compute cards.";
  const char *fp16Note = "16-bit inputs with 32-bit totals -- the everyday "
                         "precision of AI inference, and usually the fastest row "
                         "here.";
  const char *bf16Note = "bfloat16 inputs with 32-bit totals -- 16 bits arranged "
                         "for AI work, trading digits of accuracy for the number "
                         "range of a full float.";
  const char *int8Note = "8-bit whole numbers with 32-bit totals -- the format "
                         "quantized neural networks use.";

  auto test = currentDeviceScope->beginTest(
        {"rocblas_gemm", "rocBLAS GEMM peak", "tflops", Category::Unknown,
         "Matrix-multiply speed through AMD's own tuned library, on a large "
         "square problem.  Where the matrix-core rows show what the hardware "
         "can do in principle, this shows what shipping code reaches on the "
         "operation most AI work is built from.  Each reading is a different "
         "input format.",
         TestShape::Heterogeneous, "data type"});

  auto blasOpts = [&](const char *note) {
    logger::EmitOptions o;
    if (note) o.description = note;
    return o;
  };
  auto intOpts = [&](const char *note) {
    logger::EmitOptions o;
    if (note) o.description = note;
    o.unit = "tops";
    return o;
  };

#ifndef CLPEAK_ROCM_HAS_ROCBLAS
  test.skip("fp32", ResultStatus::Unsupported, "rocBLAS not found at configure time", blasOpts(fp32Note));
  test.skip("fp64", ResultStatus::Unsupported, "rocBLAS not found at configure time", blasOpts(fp64Note));
  test.skip("fp16", ResultStatus::Unsupported, "rocBLAS not found at configure time", blasOpts(fp16Note));
  test.skip("bf16", ResultStatus::Unsupported, "rocBLAS not found at configure time", blasOpts(bf16Note));
  test.skip("int8", ResultStatus::Unsupported, "rocBLAS not found at configure time", intOpts(int8Note));
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

  auto skipPhase = [&](ResultStatus status, const char *msg) {
    test.skip("fp32", status, msg, blasOpts(fp32Note));
    test.skip("fp64", status, msg, blasOpts(fp64Note));
    test.skip("fp16", status, msg, blasOpts(fp16Note));
    test.skip("bf16", status, msg, blasOpts(bf16Note));
    test.skip("int8", status, msg, intOpts(int8Note));
  };

  // rocBLAS is an optional runtime dependency (not part of the HIP runtime).
  if (!g_rb.load())
  {
    skipPhase(ResultStatus::Unsupported, "rocBLAS library not found; GEMM skipped");
    return 0;
  }

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (hipMalloc(&dA, aBytes) != hipSuccess ||
      hipMalloc(&dB, bBytes) != hipSuccess ||
      hipMalloc(&dC, cBytes) != hipSuccess)
  {
    skipPhase(ResultStatus::Error, "Failed to allocate GEMM buffers");
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
    skipPhase(ResultStatus::Error, "rocblas_create_handle failed");
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC);
    return -1;
  }
  (void)rocblas_set_stream(handle, dev.stream);

  auto runTimed = [&](const char *label, const char *note, auto gemmFn) {
    const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
    double probeUs = timeRocblas(dev.stream, gemmFn, warm);
    if (probeUs <= 0.0)
    {
      test.skip(label, ResultStatus::Error, "timing probe failed", blasOpts(note));
      return;
    }
    unsigned int iters = pickIters(probeUs, 5000000u,
                                   forceIters ? specifiedIters : 0);
    double meanUs = timeRocblas(dev.stream, gemmFn, iters);
    if (meanUs <= 0.0)
    {
      test.skip(label, ResultStatus::Error, "rocBLAS GEMM failed", blasOpts(note));
      return;
    }
    test.emit(label, (float)(flops * 1.0e6 / meanUs / 1.0e12), blasOpts(note));
  };

  {
    const float alpha32 = 1.0f, beta32 = 0.0f;
    runTimed("fp32", fp32Note, [&]() {
      return rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                           M, N, K, &alpha32,
                           (const float *)dA, M,
                           (const float *)dB, K,
                           &beta32,
                           (float *)dC, M);
    });

    const double alpha64 = 1.0, beta64 = 0.0;
    runTimed("fp64", fp64Note, [&]() {
      return rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                           M, N, K, &alpha64,
                           (const double *)dA, M,
                           (const double *)dB, K,
                           &beta64,
                           (double *)dC, M);
    });

    // fp16/bf16 both use gemm_ex with f32 *accumulate* (HPA). The native-format
    // accumulate (rocblas_hgemm etc.) does not map to the fast 16-bit x 16-bit
    // -> fp32 MFMA path and tops out far below peak; f32-compute reaches it.
    const float alphaf = 1.0f, betaf = 0.0f;
    if (dev.info.fp16Supported)
    {
      runTimed("fp16", fp16Note, [&]() {
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
      test.skip("fp16", ResultStatus::Unsupported, "fp16 not supported by this ROCm device", blasOpts(fp16Note));
    }

    if (dev.info.bf16Supported)
    {
      runTimed("bf16", bf16Note, [&]() {
        return rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                               M, N, K, &alphaf,
                               dA, rocblas_datatype_bf16_r, M,
                               dB, rocblas_datatype_bf16_r, K,
                               &betaf,
                               dC, rocblas_datatype_bf16_r, M,
                               dC, rocblas_datatype_bf16_r, M,
                               rocblas_datatype_f32_r,
                               rocblas_gemm_algo_standard, 0, 0);
      });
    }
    else
    {
      test.skip("bf16", ResultStatus::Unsupported, "bf16 not supported by this ROCm device", blasOpts(bf16Note));
    }
  }
  {
    // int8 x int8 -> int32 via gemm_ex (i8 in, i32 out + compute). No device-info
    // flag tracks int8 GEMM support, so we attempt it and let rocBLAS decide.
    // i8 GEMM wants K a multiple of 4; the 256-aligned dim from
    // pickRocblasGemmDim satisfies it.
    const int32_t alphaI = 1, betaI = 0;
    auto int8Gemm = [&]() {
      return rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                             M, N, K, &alphaI,
                             dA, rocblas_datatype_i8_r, M,
                             dB, rocblas_datatype_i8_r, K,
                             &betaI,
                             dC, rocblas_datatype_i32_r, M,
                             dC, rocblas_datatype_i32_r, M,
                             rocblas_datatype_i32_r,
                             rocblas_gemm_algo_standard, 0, 0);
    };
    // The launch status is returned synchronously, so an unsupported type combo
    // (rocblas_status_not_implemented) is distinguishable from a runtime error
    // here -- report it as Unsupported rather than Error.
    auto int8Wrapper = [&](const char *label, const char *note, auto fn) {
      const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
      double probeUs = timeRocblas(dev.stream, fn, warm);
      if (probeUs <= 0.0)
      {
        test.skip(label, ResultStatus::Error, "timing probe failed", intOpts(note));
        return;
      }
      unsigned int iters = pickIters(probeUs, 5000000u,
                                     forceIters ? specifiedIters : 0);
      double meanUs = timeRocblas(dev.stream, fn, iters);
      if (meanUs <= 0.0)
      {
        test.skip(label, ResultStatus::Error, "rocBLAS GEMM failed", intOpts(note));
        return;
      }
      test.emit(label, (float)(flops * 1.0e6 / meanUs / 1.0e12), intOpts(note));
    };
    if (int8Gemm() != rocblas_status_success)
      test.skip("int8", ResultStatus::Unsupported,
                std::string("int8 GEMM not supported on ") + dev.info.archName, intOpts(int8Note));
    else
      int8Wrapper("int8", int8Note, int8Gemm);
  }

  (void)rocblas_destroy_handle(handle);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  return 0;
#endif
}

#endif // ENABLE_ROCM