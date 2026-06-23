#ifdef ENABLE_ROCM

// hipBLASLt-based FP8 GEMM peak benchmark.
//
// rocBLAS (rocblas.cpp) covers fp32/fp64/fp16; FP8 GEMM on MI300 is the domain
// of hipBLASLt, which carries the tuned fp8 (e4m3/e5m2 fnuz) matmul kernels.
// This mirrors the CUDA cuBLASLt runner (cuda_blas.cpp): ask the library for
// its top-K algo candidates, time a short probe on each, then run the measured
// fastest over a 5 s budget. Transfers are excluded; only the matmul window is
// timed via hipEvents. Result is the "achievable" FP8 GEMM peak that sits next
// to the raw mfma_fp8 microbench (mfma.cpp).

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <common/dynlib.h>   // must stay at file scope (defines namespace clpeak)

#ifdef CLPEAK_ROCM_HAS_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <string>
#include <vector>
#ifdef _WIN32
#include <io.h>
#define CLPEAK_DUP   _dup
#define CLPEAK_DUP2  _dup2
#define CLPEAK_CLOSE _close
#define CLPEAK_DEVNULL "NUL"
#else
#include <unistd.h>
#define CLPEAK_DUP   dup
#define CLPEAK_DUP2  dup2
#define CLPEAK_CLOSE close
#define CLPEAK_DEVNULL "/dev/null"
#endif
#endif

namespace {

// Same square-GEMM sizing as rocblas.cpp / cuda_blas.cpp: scale with the
// compute budget, clamp to [2048, 16384], 256-aligned, capped to 1/4 VRAM.
uint32_t pickGemmDim(const rocm_device_info_t &info)
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

#ifdef CLPEAK_ROCM_HAS_HIPBLASLT
// Optional hipBLASLt loader -- not part of the HIP runtime, so it is resolved at
// run time; if absent the FP8 GEMM benchmark is skipped.  Function-pointer types
// come from the header via decltype; macros redirect the call sites unchanged.
struct HipblasLtApi
{
  void *lib = nullptr;
  decltype(&::hipblasLtCreate)                       Create = nullptr;
  decltype(&::hipblasLtDestroy)                      Destroy = nullptr;
  decltype(&::hipblasLtMatmul)                       Matmul = nullptr;
  decltype(&::hipblasLtMatmulAlgoGetHeuristic)       AlgoGetHeuristic = nullptr;
  decltype(&::hipblasLtMatmulDescCreate)             DescCreate = nullptr;
  decltype(&::hipblasLtMatmulDescDestroy)            DescDestroy = nullptr;
  decltype(&::hipblasLtMatmulDescSetAttribute)       DescSetAttribute = nullptr;
  decltype(&::hipblasLtMatmulPreferenceCreate)       PrefCreate = nullptr;
  decltype(&::hipblasLtMatmulPreferenceDestroy)      PrefDestroy = nullptr;
  decltype(&::hipblasLtMatmulPreferenceSetAttribute) PrefSetAttribute = nullptr;
  decltype(&::hipblasLtMatrixLayoutCreate)           LayoutCreate = nullptr;
  decltype(&::hipblasLtMatrixLayoutDestroy)          LayoutDestroy = nullptr;
  bool load();
};
static HipblasLtApi g_hlt;
bool HipblasLtApi::load()
{
  if (lib)
    return true;
  lib = clpeak::dynOpen({"libhipblaslt.so", "libhipblaslt.so.0", "hipblaslt.dll"});
  if (!lib)
    return false;
  bool ok = true;
#define CLPEAK_HLT_SYM(member, name)                                      \
  member = reinterpret_cast<decltype(member)>(clpeak::dynSym(lib, name)); \
  ok = ok && (member != nullptr)
  CLPEAK_HLT_SYM(Create,           "hipblasLtCreate");
  CLPEAK_HLT_SYM(Destroy,          "hipblasLtDestroy");
  CLPEAK_HLT_SYM(Matmul,           "hipblasLtMatmul");
  CLPEAK_HLT_SYM(AlgoGetHeuristic, "hipblasLtMatmulAlgoGetHeuristic");
  CLPEAK_HLT_SYM(DescCreate,       "hipblasLtMatmulDescCreate");
  CLPEAK_HLT_SYM(DescDestroy,      "hipblasLtMatmulDescDestroy");
  CLPEAK_HLT_SYM(DescSetAttribute, "hipblasLtMatmulDescSetAttribute");
  CLPEAK_HLT_SYM(PrefCreate,       "hipblasLtMatmulPreferenceCreate");
  CLPEAK_HLT_SYM(PrefDestroy,      "hipblasLtMatmulPreferenceDestroy");
  CLPEAK_HLT_SYM(PrefSetAttribute, "hipblasLtMatmulPreferenceSetAttribute");
  CLPEAK_HLT_SYM(LayoutCreate,     "hipblasLtMatrixLayoutCreate");
  CLPEAK_HLT_SYM(LayoutDestroy,    "hipblasLtMatrixLayoutDestroy");
#undef CLPEAK_HLT_SYM
  if (!ok)
  {
    clpeak::dynClose(lib);
    lib = nullptr;
  }
  return ok;
}

#define hipblasLtCreate                       g_hlt.Create
#define hipblasLtDestroy                      g_hlt.Destroy
#define hipblasLtMatmul                       g_hlt.Matmul
#define hipblasLtMatmulAlgoGetHeuristic       g_hlt.AlgoGetHeuristic
#define hipblasLtMatmulDescCreate             g_hlt.DescCreate
#define hipblasLtMatmulDescDestroy            g_hlt.DescDestroy
#define hipblasLtMatmulDescSetAttribute       g_hlt.DescSetAttribute
#define hipblasLtMatmulPreferenceCreate       g_hlt.PrefCreate
#define hipblasLtMatmulPreferenceDestroy      g_hlt.PrefDestroy
#define hipblasLtMatmulPreferenceSetAttribute g_hlt.PrefSetAttribute
#define hipblasLtMatrixLayoutCreate           g_hlt.LayoutCreate
#define hipblasLtMatrixLayoutDestroy          g_hlt.LayoutDestroy

// hipBLASLt has no "is this dtype supported on this device?" capability query;
// the only arch-agnostic answer is to ask hipblasLtMatmulAlgoGetHeuristic and
// treat zero algorithms as Unsupported. But the heuristic doesn't conclude
// quietly: on parts without the hardware its Tensile/rocRoller internals print
// per-candidate diagnostics straight to the console while walking instruction
// tables (e.g. FP4 on gfx942: pages of "Warning: Latency not found" and
// rocRoller XCC FatalError lines), bypassing the HIPBLASLT_LOG_* mechanism.
// Mute stdout+stderr at the fd level for the duration of the query; the
// returned status and algo count still tell the truth. Same policy as the
// HIPRTC compile logs: under --verbose the mute is a no-op so the library
// diagnostics stay visible.
class ScopedConsoleMute
{
public:
  ScopedConsoleMute()
  {
    if (clpeak::verboseEnabled())
      return;
    (void)fflush(stdout);
    (void)fflush(stderr);
    savedOut = CLPEAK_DUP(fileno(stdout));
    savedErr = CLPEAK_DUP(fileno(stderr));
    FILE *nul = fopen(CLPEAK_DEVNULL, "w");
    if (nul)
    {
      if (savedOut >= 0) (void)CLPEAK_DUP2(fileno(nul), fileno(stdout));
      if (savedErr >= 0) (void)CLPEAK_DUP2(fileno(nul), fileno(stderr));
      (void)fclose(nul);
    }
  }
  ~ScopedConsoleMute()
  {
    (void)fflush(stdout);
    (void)fflush(stderr);
    if (savedOut >= 0) { (void)CLPEAK_DUP2(savedOut, fileno(stdout)); (void)CLPEAK_CLOSE(savedOut); }
    if (savedErr >= 0) { (void)CLPEAK_DUP2(savedErr, fileno(stderr)); (void)CLPEAK_CLOSE(savedErr); }
  }
private:
  int savedOut = -1;
  int savedErr = -1;
};

// Run `n` hipblasLtMatmul calls between an event pair; return mean us/iter.
double timeHipblasLt(hipStream_t stream, hipblasLtHandle_t lt,
                     hipblasLtMatmulDesc_t opDesc,
                     const void *alpha, const void *A, hipblasLtMatrixLayout_t Adesc,
                                        const void *B, hipblasLtMatrixLayout_t Bdesc,
                     const void *beta,  const void *C, hipblasLtMatrixLayout_t Cdesc,
                                              void *D, hipblasLtMatrixLayout_t Ddesc,
                     const hipblasLtMatmulAlgo_t *algo,
                     void *workspace, size_t workspaceSize,
                     unsigned int n)
{
  hipEvent_t start = nullptr, stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess)
  {
    if (start) (void)hipEventDestroy(start);
    if (stop)  (void)hipEventDestroy(stop);
    return -1.0;
  }

  (void)hipStreamSynchronize(stream);
  (void)hipEventRecord(start, stream);
  for (unsigned int i = 0; i < n; i++)
  {
    if (hipblasLtMatmul(lt, opDesc, alpha, A, Adesc, B, Bdesc,
                        beta, C, Cdesc, D, Ddesc,
                        algo, workspace, workspaceSize, stream) != HIPBLAS_STATUS_SUCCESS)
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
#endif // CLPEAK_ROCM_HAS_HIPBLASLT

} // namespace

int RocmPeak::runHipblasLt(RocmDevice &dev, benchmark_config_t &)
{
  auto test = currentDeviceScope->beginTest(
    {"hipblaslt-fp8", "hipBLASLt FP8 GEMM peak", "tflops"});

#ifndef CLPEAK_ROCM_HAS_HIPBLASLT
  test.skip("fp8_e4m3", ResultStatus::Unsupported, "hipBLASLt not found at configure time");
  test.skip("fp8_e5m2", ResultStatus::Unsupported, "hipBLASLt not found at configure time");
  test.skip("mxf4_e2m1", ResultStatus::Unsupported, "hipBLASLt not found at configure time");
  return 0;
#else
  // hipBLASLt is an optional runtime dependency (not part of the HIP runtime).
  if (!g_hlt.load())
  {
    test.skip("fp8_e4m3", ResultStatus::Unsupported, "hipBLASLt library not found; GEMM skipped");
    test.skip("fp8_e5m2", ResultStatus::Unsupported, "hipBLASLt library not found; GEMM skipped");
    return 0;
  }

  const uint32_t D = pickGemmDim(dev.info);
  const int64_t M = (int64_t)D, N = (int64_t)D, K = (int64_t)D;
  const double flops = 2.0 * (double)M * (double)N * (double)K;

  // fp8 inputs (1 byte), fp32 compute, bf16 output -- the dtype combo MI300's
  // fp8 GEMM peak is quoted against. Worst-case buffer is the bf16 C (2 bytes).
  const size_t abBytes = (size_t)M * K;            // 1 byte/elem fp8
  const size_t cBytes  = (size_t)M * N * sizeof(uint16_t);
  const size_t wsBytes = 256ULL * 1024 * 1024;     // 256 MB workspace

  void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dWS = nullptr;
  if (hipMalloc(&dA, abBytes) != hipSuccess ||
      hipMalloc(&dB, abBytes) != hipSuccess ||
      hipMalloc(&dC, cBytes)  != hipSuccess ||
      hipMalloc(&dWS, wsBytes) != hipSuccess)
  {
    test.skip("fp8_e4m3", ResultStatus::Error, "Failed to allocate GEMM buffers");
    test.skip("fp8_e5m2", ResultStatus::Error, "Failed to allocate GEMM buffers");
    if (dA) (void)hipFree(dA);
    if (dB) (void)hipFree(dB);
    if (dC) (void)hipFree(dC);
    if (dWS) (void)hipFree(dWS);
    return -1;
  }
  // 0x38 ~ 1.0 in e4m3 fnuz; value is irrelevant for throughput.
  (void)hipMemset(dA, 0x38, abBytes);
  (void)hipMemset(dB, 0x38, abBytes);
  (void)hipMemset(dC, 0, cBytes);

  hipblasLtHandle_t lt = nullptr;
  if (hipblasLtCreate(&lt) != HIPBLAS_STATUS_SUCCESS)
  {
    test.skip("fp8_e4m3", ResultStatus::Error, "hipblasLtCreate failed");
    test.skip("fp8_e5m2", ResultStatus::Error, "hipblasLtCreate failed");
    (void)hipFree(dA); (void)hipFree(dB); (void)hipFree(dC); (void)hipFree(dWS);
    return -1;
  }

  const float alpha = 1.0f, beta = 0.0f;

  // TN layout (op(A)=T, op(B)=N) keeps K contiguous for both operands, matching
  // the fp8 MFMA load pattern -- same choice as the cuBLASLt fp8 path.
  auto runVariant = [&](const char *label, hipDataType aType, hipDataType bType)
  {
    hipblasLtMatmulDesc_t opDesc = nullptr;
    hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

    // Stored A is K x M (col-major) for the T op; B is K x N; C is M x N.
    if (hipblasLtMatmulDescCreate(&opDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&Adesc, aType, K, M, K) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&Bdesc, bType, K, N, K) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16BF, M, N, M) != HIPBLAS_STATUS_SUCCESS)
    {
      test.skip(label, ResultStatus::Error, "descriptor create failed");
      if (opDesc) (void)hipblasLtMatmulDescDestroy(opDesc);
      if (Adesc)  (void)hipblasLtMatrixLayoutDestroy(Adesc);
      if (Bdesc)  (void)hipblasLtMatrixLayoutDestroy(Bdesc);
      if (Cdesc)  (void)hipblasLtMatrixLayoutDestroy(Cdesc);
      return;
    }

    const hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
    (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

    const int kCandidates = 8;
    hipblasLtMatmulPreference_t pref = nullptr;
    (void)hipblasLtMatmulPreferenceCreate(&pref);
    (void)hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes));

    std::vector<hipblasLtMatmulHeuristicResult_t> heurs(kCandidates);
    int returnedResults = 0;
    hipblasStatus_t hs;
    {
      ScopedConsoleMute mute;
      hs = hipblasLtMatmulAlgoGetHeuristic(
          lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
          pref, kCandidates, heurs.data(), &returnedResults);
    }
    (void)hipblasLtMatmulPreferenceDestroy(pref);

    if (hs != HIPBLAS_STATUS_SUCCESS || returnedResults == 0)
    {
      test.skip(label, ResultStatus::Unsupported,
                std::string("no fp8 GEMM algorithm for this device (") + dev.info.archName + ")");
      (void)hipblasLtMatmulDescDestroy(opDesc);
      (void)hipblasLtMatrixLayoutDestroy(Adesc);
      (void)hipblasLtMatrixLayoutDestroy(Bdesc);
      (void)hipblasLtMatrixLayoutDestroy(Cdesc);
      return;
    }

    // Measured algo selection: short probe on each candidate, keep the fastest.
    const unsigned int probeIters = 4;
    int bestIdx = -1;
    double bestProbeUs = 1e30;
    for (int i = 0; i < returnedResults; i++)
    {
      double t = timeHipblasLt(dev.stream, lt, opDesc,
          &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
          &heurs[i].algo, dWS, wsBytes, probeIters);
      if (t > 0.0 && t < bestProbeUs) { bestProbeUs = t; bestIdx = i; }
    }
    if (bestIdx < 0)
    {
      test.skip(label, ResultStatus::Error, "all candidate algos failed");
      (void)hipblasLtMatmulDescDestroy(opDesc);
      (void)hipblasLtMatrixLayoutDestroy(Adesc);
      (void)hipblasLtMatrixLayoutDestroy(Bdesc);
      (void)hipblasLtMatrixLayoutDestroy(Cdesc);
      return;
    }

    const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
    double probeUs = timeHipblasLt(dev.stream, lt, opDesc,
        &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
        &heurs[bestIdx].algo, dWS, wsBytes, warm);
    if (probeUs <= 0.0)
    {
      test.skip(label, ResultStatus::Error, "timing probe failed");
      (void)hipblasLtMatmulDescDestroy(opDesc);
      (void)hipblasLtMatrixLayoutDestroy(Adesc);
      (void)hipblasLtMatrixLayoutDestroy(Bdesc);
      (void)hipblasLtMatrixLayoutDestroy(Cdesc);
      return;
    }

    unsigned int iters = pickIters(probeUs, 5000000u, forceIters ? specifiedIters : 0);
    double meanUs = timeHipblasLt(dev.stream, lt, opDesc,
        &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
        &heurs[bestIdx].algo, dWS, wsBytes, iters);
    if (meanUs <= 0.0)
      test.skip(label, ResultStatus::Error, "hipBLASLt GEMM failed");
    else
      test.emit(label, (float)(flops * 1.0e6 / meanUs / 1.0e12));

    (void)hipblasLtMatmulDescDestroy(opDesc);
    (void)hipblasLtMatrixLayoutDestroy(Adesc);
    (void)hipblasLtMatrixLayoutDestroy(Bdesc);
    (void)hipblasLtMatrixLayoutDestroy(Cdesc);
  };

  // gfx942 fp8 is the fnuz encoding. E4M3 x E4M3 is the canonical path;
  // E5M2 x E4M3 measures a path that includes an E5M2 input (same hardware
  // rate -- any asymmetry is library algo coverage, mirroring cuda_blas.cpp).
  runVariant("fp8_e4m3", HIP_R_8F_E4M3_FNUZ, HIP_R_8F_E4M3_FNUZ);
  runVariant("fp8_e5m2", HIP_R_8F_E5M2_FNUZ, HIP_R_8F_E4M3_FNUZ);

#if defined(CLPEAK_HIPBLASLT_HAS_FP4)
  // mxfp4: block-scaled 4-bit GEMM (OCP MX, 32-element UE8M0 scales) on gfx950 /
  // MI350 -- the same dtype the mfma_mxfp4 microbench measures. Both operands are
  // HIP_R_4F_E2M1 (4 bits/elem, so they fit the fp8-sized dA/dB), output is bf16,
  // compute + scale are fp32. Neutral (==1.0) per-block scale tensors are bound;
  // their values affect numerics, not throughput.
  {
    const size_t scaleABytes = (size_t)M * K; // upper bound, 1 byte/scale
    const size_t scaleBBytes = (size_t)K * N;
    void *dSA = nullptr, *dSB = nullptr;
    if (hipMalloc(&dSA, scaleABytes) != hipSuccess ||
        hipMalloc(&dSB, scaleBBytes) != hipSuccess)
    {
      test.skip("mxf4_e2m1", ResultStatus::Error, "scale buffer alloc failed");
      if (dSA) (void)hipFree(dSA);
      if (dSB) (void)hipFree(dSB);
    }
    else
    {
      (void)hipMemset(dSA, 0x7F, scaleABytes); // 0x7F = 1.0 in UE8M0
      (void)hipMemset(dSB, 0x7F, scaleBBytes);

      hipblasLtMatmulDesc_t opDesc = nullptr;
      hipblasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
      bool ok = (hipblasLtMatmulDescCreate(&opDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS &&
                 hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_4F_E2M1, K, M, K) == HIPBLAS_STATUS_SUCCESS &&
                 hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_4F_E2M1, K, N, K) == HIPBLAS_STATUS_SUCCESS &&
                 hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16BF, M, N, M) == HIPBLAS_STATUS_SUCCESS);
      if (!ok)
      {
        test.skip("mxf4_e2m1", ResultStatus::Error, "descriptor create failed");
      }
      else
      {
        const hipblasOperation_t opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
        (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

        const hipblasLtMatmulMatrixScale_t scaleMode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dSA, sizeof(dSA));
        (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dSB, sizeof(dSB));
        (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
        (void)hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));

        const int kCandidates = 8;
        hipblasLtMatmulPreference_t pref = nullptr;
        (void)hipblasLtMatmulPreferenceCreate(&pref);
        (void)hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes));
        std::vector<hipblasLtMatmulHeuristicResult_t> heurs(kCandidates);
        int returnedResults = 0;
        hipblasStatus_t hs;
        {
          ScopedConsoleMute mute;
          hs = hipblasLtMatmulAlgoGetHeuristic(
              lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
              pref, kCandidates, heurs.data(), &returnedResults);
        }
        (void)hipblasLtMatmulPreferenceDestroy(pref);

        if (hs != HIPBLAS_STATUS_SUCCESS || returnedResults == 0)
        {
          test.skip("mxf4_e2m1", ResultStatus::Unsupported,
                    std::string("no mxfp4 GEMM algorithm for this device (") + dev.info.archName + ")");
        }
        else
        {
          const unsigned int probeIters = 4;
          int bestIdx = -1;
          double bestProbeUs = 1e30;
          for (int i = 0; i < returnedResults; i++)
          {
            double t = timeHipblasLt(dev.stream, lt, opDesc,
                &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
                &heurs[i].algo, dWS, wsBytes, probeIters);
            if (t > 0.0 && t < bestProbeUs) { bestProbeUs = t; bestIdx = i; }
          }
          if (bestIdx < 0)
          {
            test.skip("mxf4_e2m1", ResultStatus::Error, "all candidate algos failed");
          }
          else
          {
            const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
            double probeUs = timeHipblasLt(dev.stream, lt, opDesc,
                &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
                &heurs[bestIdx].algo, dWS, wsBytes, warm);
            if (probeUs <= 0.0)
            {
              test.skip("mxf4_e2m1", ResultStatus::Error, "timing probe failed");
            }
            else
            {
              unsigned int iters = pickIters(probeUs, 5000000u, forceIters ? specifiedIters : 0);
              double meanUs = timeHipblasLt(dev.stream, lt, opDesc,
                  &alpha, dA, Adesc, dB, Bdesc, &beta, dC, Cdesc, dC, Cdesc,
                  &heurs[bestIdx].algo, dWS, wsBytes, iters);
              if (meanUs <= 0.0)
                test.skip("mxf4_e2m1", ResultStatus::Error, "hipBLASLt GEMM failed");
              else
                test.emit("mxf4_e2m1", (float)(flops * 1.0e6 / meanUs / 1.0e12));
            }
          }
        }
      }
      if (opDesc) (void)hipblasLtMatmulDescDestroy(opDesc);
      if (Adesc)  (void)hipblasLtMatrixLayoutDestroy(Adesc);
      if (Bdesc)  (void)hipblasLtMatrixLayoutDestroy(Bdesc);
      if (Cdesc)  (void)hipblasLtMatrixLayoutDestroy(Cdesc);
      (void)hipFree(dSA);
      (void)hipFree(dSB);
    }
  }
#else
  test.skip("mxf4_e2m1", ResultStatus::Unsupported,
            "block-scaled FP4 GEMM API not in this hipBLASLt (needs gfx950 / recent ROCm)");
#endif // CLPEAK_HIPBLASLT_HAS_FP4

  (void)hipblasLtDestroy(lt);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dWS);
  return 0;
#endif // CLPEAK_ROCM_HAS_HIPBLASLT
}

#endif // ENABLE_ROCM
