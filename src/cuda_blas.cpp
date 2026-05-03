#ifdef ENABLE_CUDA

// cuBLASLt-based GEMM peak benchmark.
//
// Uses NVIDIA's high-level cuBLASLt matmul API rather than hand-written WMMA
// to reach close to the vendor-advertised TFLOPS / TOPS.  H2D / D2H transfers
// are excluded from timing; we measure only the kernel launch + execution
// window via cuEvents.
//
// Note on consumer Blackwell (RTX 50 series, sm_120):
// NVIDIA caps several tensor-core dtypes on consumer parts as product
// segmentation vs the datacenter dies (B100 / B200).  Measured on RTX 5060:
//   * tf32 ~= fp32 (no tensor-core boost; TF32 runs on the regular ALUs)
//   * bf16 = 1/2 of fp16
//   * fp16 = full tensor-core rate
//   * fp8  = 2x fp16
// This is intentional, not a kernel-selection bug -- the numbers are at peak
// once the dtype/algo combo is correct.  Don't try to "fix" bf16 ~ 1/2 fp16
// on these cards.

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
    // Worst-case element size is fp64 = 8 bytes (DGEMM); shrink D if the
    // 3*D*D*8 footprint won't fit in 1/4 VRAM.
    while (D > 1024 && 3ULL * D * D * 8 > budget)
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

    const uint32_t D = pickGemmDim(dev.info);
    const uint32_t M = D, N = D, K = D;
    const double  flops_per_iter = 2.0 * (double)M * (double)N * (double)K;

    // Layout per variant: fp32 (CUDA cores) uses NN; tensor-core dtypes use TN
    // because that keeps the K axis contiguous for both A and B, matching the
    // MMA load pattern.
    std::stringstream dimStr; dimStr << M << "x" << N << "x" << K;

    // -----------------------------------------------------------------------
    // Allocate persistent buffers (A, B, C, workspace).  Worst-case input
    // size is fp64 = 8 bytes/elem (DGEMM); smaller dtypes alias the same
    // buffers.  All sizes are bytes for fp64.
    // -----------------------------------------------------------------------
    const size_t aBytes = (size_t)M * K * sizeof(double);
    const size_t bBytes = (size_t)K * N * sizeof(double);
    const size_t cBytes = (size_t)M * N * sizeof(double);
    const size_t wsBytes = 256ULL * 1024 * 1024; // 256 MB workspace

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
        return -1;
    }

    auto runVariantAB = [&](const char *label,
                            cudaDataType_t aType, cudaDataType_t bType,
                            cudaDataType_t cType,
                            cublasComputeType_t computeType,
                            cudaDataType_t scaleType,
                            const void *alphaPtr, const void *betaPtr,
                            bool useTN) -> int
    {
        log->print(TAB TAB);
        log->print(label);
        log->print(" : ");

        cublasLtMatmulDesc_t opDesc = nullptr;
        cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;

        // TN (TRANSA=T, TRANSB=N) puts the K axis on the contiguous side of
        // both stored A and stored B -- the MMA load pattern tensor-core
        // kernels are tuned around.  Stored A is K x M col-major, B is K x N
        // col-major.  NN is kept for fp32 because cuBLASLt's CUDA-core fp32
        // kernels are better tuned for the NN layout (TN measured ~50% slower
        // for fp32 on RTX 5060).
        const uint64_t Arows = useTN ? K : M;
        const uint64_t Acols = useTN ? M : K;
        const uint64_t Ald   = Arows;
        if (cublasLtMatmulDescCreate(&opDesc, computeType, scaleType) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Adesc, aType, Arows, Acols, Ald) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Bdesc, bType, K, N, K) != CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Cdesc, cType, M, N, M) != CUBLAS_STATUS_SUCCESS)
        {
            log->print("descriptor create failed" NEWLINE);
            log->xmlRecord(label, 0.0f);
            if (opDesc) cublasLtMatmulDescDestroy(opDesc);
            if (Adesc)  cublasLtMatrixLayoutDestroy(Adesc);
            if (Bdesc)  cublasLtMatrixLayoutDestroy(Bdesc);
            if (Cdesc)  cublasLtMatrixLayoutDestroy(Cdesc);
            return -1;
        }

        cublasOperation_t opA = useTN ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        // Heuristic: ask cuBLASLt for its top-K algo candidates.  The top-1
        // result is an estimator pick, not a measured pick; for some dtypes
        // (notably bf16 on RTX 5060 sm_120) the estimator's top algo is ~50%
        // slower than the measured-best.  We request K candidates, run a
        // short timing pass on each, and use the measured fastest for the
        // full timing run.
        const int kCandidates = 8;
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes, sizeof(wsBytes));

        cublasLtMatmulHeuristicResult_t heurs[kCandidates] = {};
        int returnedResults = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            lt, opDesc, Adesc, Bdesc, Cdesc, Cdesc,
            pref, kCandidates, heurs, &returnedResults);
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

        // Measured algo selection.  4 iters per candidate is enough to filter
        // out estimator outliers without burning a noticeable wall-clock budget
        // (top-K total is ~4 * K * per_iter_us; on a fast dtype that's well
        // under a second).  Skip candidates that fail to launch (CUBLAS_STATUS
        // not SUCCESS within timeCublasLt -- it returns a negative time).
        const unsigned int probeIters = 4;
        int bestIdx = -1;
        double bestProbeUs = 1e30;
        for (int i = 0; i < returnedResults; i++)
        {
            double t = timeCublasLt(lt, dev.stream, opDesc,
                alphaPtr, (void*)dA, Adesc, (void*)dB, Bdesc,
                betaPtr,  (void*)dC, Cdesc, (void*)dC, Cdesc,
                &heurs[i].algo, (void*)dWS, wsBytes, probeIters);
            if (t > 0.0 && t < bestProbeUs) { bestProbeUs = t; bestIdx = i; }
        }
        if (bestIdx < 0)
        {
            log->print("all candidate algos failed" NEWLINE);
            log->xmlRecord(label, 0.0f);
            cublasLtMatmulDescDestroy(opDesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            return -1;
        }

        // alpha / beta scalar bit-widths must match `scaleType`; the caller
        // supplies them by pointer so we don't have to fan out a switch here.
        unsigned int warmup = warmupCount > 0 ? warmupCount : 2;
        double per_iter_us = timeCublasLt(lt, dev.stream, opDesc,
            alphaPtr, (void*)dA, Adesc, (void*)dB, Bdesc,
            betaPtr,  (void*)dC, Cdesc, (void*)dC, Cdesc,
            &heurs[bestIdx].algo, (void*)dWS, wsBytes, warmup);

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
            alphaPtr, (void*)dA, Adesc, (void*)dB, Bdesc,
            betaPtr,  (void*)dC, Cdesc, (void*)dC, Cdesc,
            &heurs[bestIdx].algo, (void*)dWS, wsBytes, iters);

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
    // Floating-point variants.  All share col-major NN layout; cuBLASLt's
    // heuristic picks the best algo per dtype.  Where a dtype isn't supported
    // by the device's compute capability the heuristic returns 0 results and
    // the runner prints "unsupported on sm_<cc>".
    // -----------------------------------------------------------------------

    // Convenience wrapper: most variants use the same dtype for A and B.
    auto runVariant = [&](const char *label,
                          cudaDataType_t abType, cudaDataType_t cType,
                          cublasComputeType_t computeType,
                          cudaDataType_t scaleType,
                          const void *alphaPtr, const void *betaPtr,
                          bool useTN) -> int
    {
        return runVariantAB(label, abType, abType, cType, computeType, scaleType,
                            alphaPtr, betaPtr, useTN);
    };

    // Scale-type scalars.  Bit widths MUST match the cublasLtMatmul scaleType:
    //   * R_32F  -> float
    //   * R_16F  -> half (16-bit; we use uint16_t with the IEEE half bit
    //               pattern: 0x3c00 = 1.0, 0x0000 = 0.0)
    //   * R_32I  -> int32
    const float    alpha32   = 1.0f,  beta32   = 0.0f;
    const double   alpha64   = 1.0,   beta64   = 0.0;
    const uint16_t alpha16   = 0x3c00, beta16  = 0x0000;
    const int32_t  alpha32i  = 1,     beta32i  = 0;

    // -----------------------------------------------------------------------
    // Floating-point variants -- reported in TFLOPS.
    // -----------------------------------------------------------------------
    log->print(NEWLINE TAB "cuBLASLt GEMM peak (TFLOPS)" NEWLINE);
    // Test name carries no unit; the unit attribute is what the dump
    // pipeline (and the legacy-shim category derivation) reads.
    log->xmlOpenTag("cublas-fp");
    log->xmlAppendAttribs("unit", "tflops");
    log->xmlAppendAttribs("dim", dimStr.str());
    log->xmlAppendAttribs("workspace", "256MB");

    // fp32: full-precision GEMM on CUDA cores.  NN layout -- TN measured ~50%
    // slower for fp32 on RTX 5060 because cuBLASLt's heuristic falls off the
    // tuned kernel set for fp32 + TN.
    runVariant("fp32", CUDA_R_32F,  CUDA_R_32F, CUBLAS_COMPUTE_32F,
               CUDA_R_32F, &alpha32, &beta32, /*useTN=*/false);

    // fp64: DGEMM on CUDA cores (tensor-core fp64 acceleration kicks in
    // automatically on sm_80+; the heuristic picks the right algo).  NN
    // layout matches fp32 -- DGEMM kernels are tuned for that shape.
    runVariant("fp64", CUDA_R_64F,  CUDA_R_64F, CUBLAS_COMPUTE_64F,
               CUDA_R_64F, &alpha64, &beta64, /*useTN=*/false);

    // tf32: fp32 inputs, but cuBLASLt internally rounds to TF32 and runs on
    // tensor cores (Ampere+).  Inputs/outputs stay fp32.
    if (dev.info.tf32GemmSupported)
        runVariant("tf32", CUDA_R_32F,  CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
                   CUDA_R_32F, &alpha32, &beta32, /*useTN=*/true);

    // fp16: half inputs + half output + half compute.  scaleType is R_16F so
    // alpha/beta must be 16-bit half values, not float.
    if (dev.info.fp16Supported)
        runVariant("fp16", CUDA_R_16F,  CUDA_R_16F, CUBLAS_COMPUTE_16F,
                   CUDA_R_16F, &alpha16, &beta16, /*useTN=*/true);

    // bf16: bf16 inputs + fp32 output + fp32 compute (mixed-precision).
    // bf16 has no native accumulator format -- compute is always fp32.
    // R_32F output matches the dtype combo NVIDIA's published bf16 GEMM
    // peaks are quoted against.
    if (dev.info.bf16Supported)
        runVariant("bf16", CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                   CUDA_R_32F, &alpha32, &beta32, /*useTN=*/true);

    // fp8: cuBLASLt's fp8 matmul requires TN layout; inputs are 1-byte fp8,
    // accumulator + output are bf16, scale is fp32.  Gated on cc >= 8.9
    // (Ada / Hopper / Blackwell).
    //
    // cuBLASLt does NOT support both A and B being E5M2 -- the heuristic
    // returns 0 results for that combo on every CUDA version we've tested
    // (incl. 13.2 on sm_120).  The supported pairs are:
    //     E4M3 x E4M3   -- canonical "training" fp8 path
    //     E4M3 x E5M2   -- mixed
    //     E5M2 x E4M3   -- mixed (typical inference: e5m2 activations,
    //                              e4m3 weights)
    // We label the second variant "fp8_e5m2" and run E5M2 x E4M3 to measure
    // the throughput of fp8 paths that include an E5M2 input.  Hardware
    // throughput is the same as E4M3 x E4M3; the asymmetry comes from
    // cuBLASLt's algorithm coverage, not the tensor cores.
    if (dev.info.fp8MmaSupported)
    {
        runVariant("fp8_e4m3", CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F,
                   CUDA_R_32F, &alpha32, &beta32, /*useTN=*/true);
        runVariantAB("fp8_e5m2", CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF,
                     CUBLAS_COMPUTE_32F, CUDA_R_32F,
                     &alpha32, &beta32, /*useTN=*/true);
    }

    log->xmlCloseTag(); // cublas-tflops

    // -----------------------------------------------------------------------
    // Integer variants -- reported in TOPS (numerically equivalent to TFLOPS;
    // the unit just signals integer ops).
    // -----------------------------------------------------------------------
    log->print(NEWLINE TAB "cuBLASLt GEMM peak (TOPS)" NEWLINE);
    log->xmlOpenTag("cublas-int");
    log->xmlAppendAttribs("unit", "tops");
    log->xmlAppendAttribs("dim", dimStr.str());
    log->xmlAppendAttribs("workspace", "256MB");

    // int8: 1-byte signed inputs, int32 accumulator + output, int32 compute
    // and scale.  cc >= 7.5 (Turing) for IMMA tensor cores.
    if (dev.info.int8GemmSupported)
        runVariant("int8", CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I,
                   CUDA_R_32I, &alpha32i, &beta32i, /*useTN=*/true);

    // int4: packed 4-bit signed inputs, int32 accumulator + output.  cc >= 9.0
    // (Hopper) for the IMMA int4 path.  cuBLASLt may refuse this even when the
    // device's compute capability claims support -- the algo heuristic
    // returning 0 results is the canonical "unsupported" signal and we report
    // that line cleanly.
    if (dev.info.int4GemmSupported)
        runVariant("int4", CUDA_R_4I, CUDA_R_32I, CUBLAS_COMPUTE_32I,
                   CUDA_R_32I, &alpha32i, &beta32i, /*useTN=*/true);

    log->xmlCloseTag(); // cublas-tops

    cublasLtDestroy(lt);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dWS);
    return 0;
}

#endif // ENABLE_CUDA
