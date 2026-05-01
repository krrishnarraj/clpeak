#ifdef ENABLE_CUDA

// cuBLASLt-based GEMM peak benchmark.
//
// Uses NVIDIA's high-level cuBLASLt matmul API rather than hand-written WMMA
// to reach close to the vendor-advertised TFLOPS / TOPS.  H2D / D2H transfers
// are excluded from timing; we measure only the kernel launch + execution
// window via cuEvents.
//
// Iteration 1 (this file): fp32 only.  Subsequent iterations add fp16, bf16,
// tf32, fp8 e4m3/e5m2, int8, int4.

#include <cuda_peak.h>
#include <cublasLt.h>
#include <chrono>
#include <sstream>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const char *cuErrStrLocal(CUresult r)
{
    const char *s = nullptr;
    cuGetErrorString(r, &s);
    return s ? s : "unknown CUDA error";
}

#define CU_TRY(call) do { CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cuErrStrLocal(_r)); \
        return -1; } } while (0)

#define LT_TRY(call) do { cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLASLt error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
        return -1; } } while (0)

// Pick a square GEMM dim that scales with the device's compute budget so
// small / large GPUs land in similar wall-clock windows.  Same shape as the
// Metal version: 2048 + numSMs*256, clamped to [2048, 16384], 256-aligned,
// then capped to 25% of VRAM (worst-case fp32 = 4 bytes / element).
uint32_t pickGemmDim(const cuda_device_info_t &info)
{
    uint32_t sms = (uint32_t)(info.numSMs > 0 ? info.numSMs : 16);
    uint64_t D = 2048 + (uint64_t)sms * 256;
    D = (D + 255) & ~uint64_t(255);
    if (D < 2048)  D = 2048;
    if (D > 16384) D = 16384;

    uint64_t budget = info.totalGlobalMem ? info.totalGlobalMem / 4 : ((uint64_t)4 << 30);
    while (D > 1024 && 3ULL * D * D * 4 > budget)
        D /= 2;
    return (uint32_t)D;
}

// Auto-tune iter count from a calibration time, targeting ~5s steady-state.
unsigned int pickIters(double per_iter_us, bool forced, unsigned int forcedVal)
{
    if (forced && forcedVal > 0) return forcedVal;
    double want = 5.0e6 / per_iter_us;
    if (want < 8.0)     want = 8.0;
    if (want > 2000.0)  want = 2000.0;
    return (unsigned int)want;
}

// One-shot timing helper: run `n` cublasLtMatmul calls between an event pair.
// Returns mean per-iter time in microseconds (or <0 on error).
double timeCublasLt(cublasLtHandle_t lt, CUstream stream,
                    cublasLtMatmulDesc_t opDesc,
                    const void *alpha, const void *A, cublasLtMatrixLayout_t Adesc,
                                       const void *B, cublasLtMatrixLayout_t Bdesc,
                    const void *beta,  const void *C, cublasLtMatrixLayout_t Cdesc,
                                             void *D, cublasLtMatrixLayout_t Ddesc,
                    const cublasLtMatmulAlgo_t *algo,
                    void *workspace, size_t workspaceSize,
                    unsigned int n)
{
    CUevent start = nullptr, stop = nullptr;
    cuEventCreate(&start, CU_EVENT_DEFAULT);
    cuEventCreate(&stop,  CU_EVENT_DEFAULT);

    cuStreamSynchronize(stream);
    cuEventRecord(start, stream);
    for (unsigned int i = 0; i < n; i++)
    {
        cublasStatus_t s = cublasLtMatmul(lt, opDesc,
                                          alpha, A, Adesc, B, Bdesc,
                                          beta,  C, Cdesc, D, Ddesc,
                                          algo, workspace, workspaceSize, stream);
        if (s != CUBLAS_STATUS_SUCCESS)
        {
            cuEventDestroy(start); cuEventDestroy(stop);
            fprintf(stderr, "cublasLtMatmul failed: status=%d\n", (int)s);
            return -1.0;
        }
    }
    cuEventRecord(stop, stream);
    cuEventSynchronize(stop);

    float ms = 0;
    cuEventElapsedTime(&ms, start, stop);
    cuEventDestroy(start);
    cuEventDestroy(stop);
    return (double)ms * 1000.0 / (double)n; // -> microseconds / iter
}

} // namespace

int CudaPeak::runCublas(CudaDevice &dev, benchmark_config_t &cfg)
{
    (void)cfg;

    log->print(NEWLINE TAB "cuBLASLt GEMM peak (TFLOPS)" NEWLINE);
    log->xmlOpenTag("cublas");

    const uint32_t D = pickGemmDim(dev.info);
    const uint32_t M = D, N = D, K = D;
    const double  flops_per_iter = 2.0 * (double)M * (double)N * (double)K;

    {
        std::stringstream ss; ss << M << "x" << N << "x" << K;
        log->xmlAppendAttribs("dim", ss.str());
    }
    log->xmlAppendAttribs("layout", "col-major-NN");
    log->xmlAppendAttribs("workspace", "128MB");

    // -----------------------------------------------------------------------
    // Allocate persistent buffers (A, B, C, workspace).  Worst-case input
    // size is fp32 = 4 bytes/elem; smaller dtypes can alias the same buffers
    // in later iterations.  All sizes are bytes for fp32.
    // -----------------------------------------------------------------------
    const size_t aBytes = (size_t)M * K * sizeof(float);
    const size_t bBytes = (size_t)K * N * sizeof(float);
    const size_t cBytes = (size_t)M * N * sizeof(float);
    const size_t wsBytes = 128ULL * 1024 * 1024; // 128 MB workspace

    CUdeviceptr dA = 0, dB = 0, dC = 0, dWS = 0;
    if (cuMemAlloc(&dA, aBytes) != CUDA_SUCCESS ||
        cuMemAlloc(&dB, bBytes) != CUDA_SUCCESS ||
        cuMemAlloc(&dC, cBytes) != CUDA_SUCCESS ||
        cuMemAlloc(&dWS, wsBytes) != CUDA_SUCCESS)
    {
        log->print(TAB TAB "Failed to allocate device buffers" NEWLINE);
        if (dA)  cuMemFree(dA);
        if (dB)  cuMemFree(dB);
        if (dC)  cuMemFree(dC);
        if (dWS) cuMemFree(dWS);
        log->xmlCloseTag();
        return -1;
    }

    // Fill A, B with non-zero garbage once (excluded from timing).  cuMemsetD32
    // is the fastest device-side fill that sticks with reasonable bit pattern;
    // 0x3f800000 == fp32 1.0f, so the GEMM produces deterministic finite output.
    cuMemsetD32(dA, 0x3f800000, aBytes / 4);
    cuMemsetD32(dB, 0x3f800000, bBytes / 4);
    cuMemsetD8(dC, 0, cBytes);

    // -----------------------------------------------------------------------
    // cuBLASLt handle
    // -----------------------------------------------------------------------
    cublasLtHandle_t lt = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS)
    {
        log->print(TAB TAB "cublasLtCreate failed" NEWLINE);
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dWS);
        log->xmlCloseTag();
        return -1;
    }

    auto runVariant = [&](const char *label,
                          cudaDataType_t abType, cudaDataType_t cType,
                          cublasComputeType_t computeType,
                          cudaDataType_t scaleType) -> int
    {
        log->print(TAB TAB);
        log->print(label);
        log->print(" : ");

        cublasLtMatmulDesc_t opDesc = nullptr;
        cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

        if (cublasLtMatmulDescCreate(&opDesc, computeType, scaleType) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Adesc, abType, M, K, M) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Bdesc, abType, K, N, K) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Cdesc, cType,  M, N, M) != CUBLAS_STATUS_SUCCESS)
        {
            log->print("descriptor create failed" NEWLINE);
            log->xmlRecord(label, 0.0f);
            if (opDesc) cublasLtMatmulDescDestroy(opDesc);
            if (Adesc)  cublasLtMatrixLayoutDestroy(Adesc);
            if (Bdesc)  cublasLtMatrixLayoutDestroy(Bdesc);
            if (Cdesc)  cublasLtMatrixLayoutDestroy(Cdesc);
            return -1;
        }

        // OP_N for both A and B (column-major NN), matching default cuBLAS layout.
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        // Heuristic: ask cuBLASLt to pick the best algo for our shape + dtype.
        // If returnedResults == 0, the path is unsupported on this device.
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes));

        cublasLtMatmulHeuristicResult_t heur = {};
        int returnedResults = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
            pref, 1, &heur, &returnedResults);
        cublasLtMatmulPreferenceDestroy(pref);

        if (hs != CUBLAS_STATUS_SUCCESS || returnedResults == 0)
        {
            log->print("unsupported on ");
            log->print(dev.info.archName);
            log->print(NEWLINE);
            log->xmlRecord(label, 0.0f);
            cublasLtMatmulDescDestroy(opDesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            return 0;
        }

        // alpha = 1, beta = 0; scaled in `scaleType`.  fp32 path uses float;
        // we pass a pair of f32 scalars and let cuBLASLt cast as needed.
        float alpha = 1.0f, beta = 0.0f;

        unsigned int warmup = warmupCount > 0 ? warmupCount : 2;
        double per_iter_us = timeCublasLt(lt, dev.stream, opDesc,
            &alpha, (void*)dA, Adesc, (void*)dB, Bdesc,
            &beta,  (void*)dC, Cdesc, (void*)dC, Cdesc,
            &heur.algo, (void*)dWS, wsBytes, warmup);

        if (per_iter_us <= 0.0)
        {
            log->print("timing probe failed" NEWLINE);
            log->xmlRecord(label, 0.0f);
            cublasLtMatmulDescDestroy(opDesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            return -1;
        }

        unsigned int iters = pickIters(per_iter_us, forceIters, specifiedIters);
        double mean_us = timeCublasLt(lt, dev.stream, opDesc,
            &alpha, (void*)dA, Adesc, (void*)dB, Bdesc,
            &beta,  (void*)dC, Cdesc, (void*)dC, Cdesc,
            &heur.algo, (void*)dWS, wsBytes, iters);

        double tops = flops_per_iter * 1.0e6 / mean_us / 1.0e12;
        log->print((float)tops);
        log->print(NEWLINE);
        log->xmlRecord(label, (float)tops);

        cublasLtMatmulDescDestroy(opDesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
        return 0;
    };

    // -----------------------------------------------------------------------
    // Iteration 1: fp32 only.
    // -----------------------------------------------------------------------
    runVariant("fp32", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtDestroy(lt);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dWS);
    log->xmlCloseTag(); // cublas
    return 0;
}

#endif // ENABLE_CUDA
