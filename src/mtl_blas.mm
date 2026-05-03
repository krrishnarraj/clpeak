#ifdef ENABLE_METAL

// MPS-based GEMM peak benchmark.
//
// Goal: get close to vendor-advertised TFLOPS using Apple's high-level MPS
// kernels rather than hand-written simdgroup_matrix code.  H2D / D2H
// transfers are excluded from timing; only the encode + GPU exec window
// is measured.
//
// We use TWO MPS APIs:
//   * MPSMatrixMultiplication (older, lower-level) for fp32 / fp16 -- it
//     consistently dispatches the simdgroup_matrix fast path from M1 (Apple7)
//     onward and reaches near-peak.  MPSGraph's matmul on Apple7 falls back
//     to a non-tile fp16 kernel that's ~10x slower, so it's the wrong tool
//     for that dtype on that hardware.
//   * MPSGraph for bf16 (gated to Apple9+ / macOS 14+).  MPSMatrix doesn't
//     support bf16; MPSGraph does, and on Apple9+ it lowers to bf16
//     simdgroup_matrix.
//
// int8 / int4 / fp8 are reported as `unsupported on this device` records.
// MPSGraph's matmul is float-only; MPSMatrix doesn't accept ints either;
// Apple silicon's hw int8 path is exposed via Core ML / MPSCNN
// convolutions, not a general GEMM, so they're outside scope here.

#include <mtl_peak.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstring>

// MetalDevice's impl is opaque in the public header; the full struct is
// only visible inside mtl_peak.mm.  Re-declare it here so we can reach
// dev.impl->device / queue without a header churn.  Layout MUST match
// the definition in mtl_peak.mm.
struct MetalDeviceImpl {
    id<MTLDevice>             device;
    id<MTLCommandQueue>       queue;
    NSMutableDictionary<NSValue*, id<MTLLibrary>>           *libraryCache;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>> *pipelineCache;
};

namespace {

// Pick a square GEMM dim D scaled to the GPU's compute budget so devices of
// different size finish in similar wall-clock windows.  ~4096 on M1 base
// (8 cores), ~10240 on M1 Max (32 cores), capped at 16384.  Also bounded by
// 25% of recommendedMaxWorkingSetSize using fp32 as worst case (3 buffers,
// 4 bytes/elem) so the same D works for every dtype variant.
uint32_t pickGemmDim(const mtl_device_info_t &info)
{
    uint32_t cores = info.gpuCoreCount;
    if (cores == 0) cores = 8;  // unknown -> assume base config
    uint64_t D = 2048 + (uint64_t)cores * 256;
    D = (D + 255) & ~uint64_t(255);
    if (D < 2048)  D = 2048;
    if (D > 16384) D = 16384;

    uint64_t budget = (info.recommendedMaxWorkingSetSize ? info.recommendedMaxWorkingSetSize
                                                         : (uint64_t)4 << 30) / 4;
    while (D > 1024 && 3ULL * D * D * 4 > budget)
        D /= 2;
    return (uint32_t)D;
}

id<MTLBuffer> makePrivateBuffer(id<MTLDevice> dev, uint64_t bytes)
{
    return [dev newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
}

// ---- MPSMatrixMultiplication path (fp32 / fp16) ---------------------------

// Encode `n` GEMMs into one MTLCommandBuffer; time the host commit+wait
// window.  Per-iter time in microseconds.
double timeMPSMatMul(id<MTLCommandQueue> queue,
                     MPSMatrixMultiplication *mm,
                     MPSMatrix *matA, MPSMatrix *matB, MPSMatrix *matC,
                     unsigned int n)
{
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        for (unsigned int i = 0; i < n; i++)
            [mm encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];

        auto t0 = std::chrono::steady_clock::now();
        [cb commit];
        [cb waitUntilCompleted];
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return us / (double)n;
    }
}

// ---- MPSGraph path (bf16; Apple9+ / macOS 14+) ----------------------------

double timeMPSGraph(id<MTLCommandQueue> queue,
                    MPSGraph *graph,
                    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
                    NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results,
                    unsigned int n)
{
    @autoreleasepool {
        MPSCommandBuffer *mcb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        for (unsigned int i = 0; i < n; i++)
        {
            [graph encodeToCommandBuffer:mcb
                                   feeds:feeds
                        targetOperations:nil
                       resultsDictionary:results
                     executionDescriptor:nil];
        }
        auto t0 = std::chrono::steady_clock::now();
        [mcb commit];
        [mcb waitUntilCompleted];
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        return us / (double)n;
    }
}

// ---- Auto-tune ------------------------------------------------------------

// Pick iter count from a calibration time, targeting ~5s steady-state.
unsigned int pickIters(double per_iter_us, bool forced, unsigned int forcedVal)
{
    if (forced && forcedVal > 0) return forcedVal;
    double want = 5.0e6 / per_iter_us;
    if (want < 8.0)     want = 8.0;
    if (want > 2000.0)  want = 2000.0;
    return (unsigned int)want;
}

} // namespace

int MetalPeak::runMpsGemm(MetalDevice &dev, benchmark_config_t &cfg)
{
    log->print(NEWLINE TAB "MPS GEMM peak (TFLOPS)" NEWLINE);
    log->resultScopeBegin("mps-gemm-fp");
    log->resultScopeAttribute("unit", "tflops");

    if (!dev.info.isAppleSilicon)
    {
        log->print(TAB TAB "MPS GEMM requires Apple silicon! Skipped" NEWLINE);
        log->resultScopeEnd();
        return 0;
    }

    const uint32_t D = pickGemmDim(dev.info);
    const uint32_t M = D, N = D, K = D;
    const double  flops_per_iter = 2.0 * (double)M * (double)N * (double)K;

    {
        std::stringstream ss; ss << M << "x" << N << "x" << K;
        log->resultScopeAttribute("dim", ss.str());
    }
    log->resultScopeAttribute("layout", "row-major");

    id<MTLDevice>       mtlDev = dev.impl->device;
    id<MTLCommandQueue> queue  = dev.impl->queue;

    // Pre-allocate the largest input set (fp32, 4 bytes) once; smaller dtypes
    // alias the same MTLBuffer with a different MPSMatrixDescriptor stride.
    const uint64_t maxInBytes = (uint64_t)M * K * 4;
    id<MTLBuffer> bufA = makePrivateBuffer(mtlDev, maxInBytes);
    id<MTLBuffer> bufB = makePrivateBuffer(mtlDev, (uint64_t)K * N * 4);
    if (!bufA || !bufB)
    {
        log->print(TAB TAB "Failed to allocate input buffers" NEWLINE);
        log->resultScopeEnd();
        return -1;
    }
    {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:bufA range:NSMakeRange(0, bufA.length) value:0x3f];
        [blit fillBuffer:bufB range:NSMakeRange(0, bufB.length) value:0x3f];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    auto runMatMul = [&](const char *label, MPSDataType dt, uint32_t elemSize)
    {
        log->print(TAB TAB);
        log->print(label);
        log->print(" : ");
        @autoreleasepool {
            const uint64_t outBytes = (uint64_t)M * N * elemSize;
            id<MTLBuffer> bufC = makePrivateBuffer(mtlDev, outBytes);
            if (!bufC)
            {
                log->print("output alloc failed" NEWLINE);
                log->resultRecord(label, 0.0f);
                return;
            }

            MPSMatrixDescriptor *aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                               columns:K
                                                                              rowBytes:(NSUInteger)K * elemSize
                                                                              dataType:dt];
            MPSMatrixDescriptor *bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                               columns:N
                                                                              rowBytes:(NSUInteger)N * elemSize
                                                                              dataType:dt];
            MPSMatrixDescriptor *cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                               columns:N
                                                                              rowBytes:(NSUInteger)N * elemSize
                                                                              dataType:dt];

            MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:aDesc];
            MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:bDesc];
            MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:cDesc];

            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
                initWithDevice:mtlDev
                 transposeLeft:NO
                transposeRight:NO
                    resultRows:M
                 resultColumns:N
               interiorColumns:K
                         alpha:1.0
                          beta:0.0];

            unsigned int warmup = warmupCount > 0 ? warmupCount : 2;
            double per_iter_us = timeMPSMatMul(queue, mm, matA, matB, matC, warmup);
            if (per_iter_us <= 0.0)
            {
                log->print("timing probe failed" NEWLINE);
                log->resultRecord(label, 0.0f);
                return;
            }

            unsigned int iters = pickIters(per_iter_us, forceIters, specifiedIters);
            double mean_us = timeMPSMatMul(queue, mm, matA, matB, matC, iters);
            double tops = flops_per_iter * 1.0e6 / mean_us / 1.0e12;

            log->print((float)tops);
            log->print(NEWLINE);
            log->resultRecord(label, (float)tops);
        }
    };

    auto runGraphMatMul = [&](const char *label, MPSDataType dt, uint32_t elemSize)
    {
        log->print(TAB TAB);
        log->print(label);
        log->print(" : ");
        @autoreleasepool {
            const uint64_t outBytes = (uint64_t)M * N * elemSize;
            id<MTLBuffer> bufC = makePrivateBuffer(mtlDev, outBytes);
            if (!bufC)
            {
                log->print("output alloc failed" NEWLINE);
                log->resultRecord(label, 0.0f);
                return;
            }

            MPSGraph *g = [MPSGraph new];
            MPSGraphTensor *A = [g placeholderWithShape:@[@(M),@(K)] dataType:dt name:@"A"];
            MPSGraphTensor *B = [g placeholderWithShape:@[@(K),@(N)] dataType:dt name:@"B"];
            MPSGraphTensor *C = [g matrixMultiplicationWithPrimaryTensor:A
                                                         secondaryTensor:B
                                                                    name:@"C"];

            MPSGraphTensorData *aData = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:bufA shape:@[@(M),@(K)] dataType:dt];
            MPSGraphTensorData *bData = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:bufB shape:@[@(K),@(N)] dataType:dt];
            MPSGraphTensorData *cData = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:bufC shape:@[@(M),@(N)] dataType:dt];

            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
                @{ A: aData, B: bData };
            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                @{ C: cData };

            unsigned int warmup = warmupCount > 0 ? warmupCount : 2;
            double per_iter_us = timeMPSGraph(queue, g, feeds, results, warmup);
            if (per_iter_us <= 0.0)
            {
                log->print("timing probe failed" NEWLINE);
                log->resultRecord(label, 0.0f);
                return;
            }

            unsigned int iters = pickIters(per_iter_us, forceIters, specifiedIters);
            double mean_us = timeMPSGraph(queue, g, feeds, results, iters);
            double tops = flops_per_iter * 1.0e6 / mean_us / 1.0e12;

            log->print((float)tops);
            log->print(NEWLINE);
            log->resultRecord(label, (float)tops);
        }
    };

    auto reportUnsupported = [&](const char *label, const char *msg)
    {
        log->print(TAB TAB);
        log->print(label);
        log->print(" : ");
        log->print(msg);
        log->print(NEWLINE);
        log->resultRecord(label, 0.0f);
    };

    // ---- Run the dtype matrix --------------------------------------------

    runMatMul("fp32", MPSDataTypeFloat32, 4);
    runMatMul("fp16", MPSDataTypeFloat16, 2);

    // bf16: only via MPSGraph, and only on Apple9+ / macOS 14+ where the
    // path actually lowers to bf16 simdgroup_matrix.
    if (@available(macOS 14.0, *))
    {
        if (dev.info.mpsGraphBF16Supported)
            runGraphMatMul("bf16", MPSDataTypeBFloat16, 2);
        else
            reportUnsupported("bf16", "bf16 requires Apple9 (M3) -- unsupported on this device");
    }
    else
    {
        reportUnsupported("bf16", "bf16 requires macOS 14 -- unsupported on this device");
    }

    log->resultScopeEnd(); // mps-gemm
    return 0;
}

// ----- Integer-domain MPS GEMM (TOPS) ---------------------------------------

int MetalPeak::runMpsGemmInt(MetalDevice &dev, benchmark_config_t &cfg)
{
    log->print(NEWLINE TAB "MPS GEMM peak (TOPS)" NEWLINE);
    log->resultScopeBegin("mps-gemm-int");
    log->resultScopeAttribute("unit", "tops");

    if (!dev.info.isAppleSilicon)
    {
        log->print(TAB TAB "MPS GEMM requires Apple silicon! Skipped" NEWLINE);
        log->resultScopeEnd();
        return 0;
    }

    const uint32_t D = pickGemmDim(dev.info);
    const uint32_t M = D, N = D, K = D;
    const double  ops_per_iter = 2.0 * (double)M * (double)N * (double)K;

    {
        std::stringstream ss; ss << M << "x" << N << "x" << K;
        log->resultScopeAttribute("dim", ss.str());
    }
    log->resultScopeAttribute("layout", "row-major");

    id<MTLDevice>       mtlDev = dev.impl->device;
    id<MTLCommandQueue> queue  = dev.impl->queue;

    // ---- int8 via MPSGraph (cast to int32 for the matmul) -----------------
    log->print(TAB TAB "int8 : ");
    if (!dev.info.simdgroupMatrixInt8Supported)
    {
        log->print("requires Apple9 (M3) or newer -- unsupported on this device" NEWLINE);
        log->resultRecord("int8", 0.0f);
    }
    else
    {
        @autoreleasepool {
            const uint64_t inBytes  = (uint64_t)M * K; // int8 = 1 byte
            const uint64_t outBytes = (uint64_t)M * N * 4; // int32 result

            id<MTLBuffer> bufA = makePrivateBuffer(mtlDev, inBytes);
            id<MTLBuffer> bufB = makePrivateBuffer(mtlDev, (uint64_t)K * N);
            id<MTLBuffer> bufC = makePrivateBuffer(mtlDev, outBytes);
            if (!bufA || !bufB || !bufC)
            {
                log->print("buffer alloc failed" NEWLINE);
                log->resultRecord("int8", 0.0f);
            }
            else
            {
                {
                    id<MTLCommandBuffer> cb = [queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                    [blit fillBuffer:bufA range:NSMakeRange(0, bufA.length) value:1];
                    [blit fillBuffer:bufB range:NSMakeRange(0, bufB.length) value:1];
                    [blit endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                }

                MPSGraph *g = [MPSGraph new];
                MPSGraphTensor *A8 = [g placeholderWithShape:@[@(M),@(K)]
                                                    dataType:MPSDataTypeInt8 name:@"A"];
                MPSGraphTensor *B8 = [g placeholderWithShape:@[@(K),@(N)]
                                                    dataType:MPSDataTypeInt8 name:@"B"];
                // MPSGraph matmul requires float-or-same-dtype operands; cast
                // to int32 so the engine takes the int simdgroup_matrix path
                // on Apple9+.
                MPSGraphTensor *A32 = [g castTensor:A8 toType:MPSDataTypeInt32 name:@"A32"];
                MPSGraphTensor *B32 = [g castTensor:B8 toType:MPSDataTypeInt32 name:@"B32"];
                MPSGraphTensor *C   = [g matrixMultiplicationWithPrimaryTensor:A32
                                                              secondaryTensor:B32
                                                                          name:@"C"];

                MPSGraphTensorData *aData = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:bufA shape:@[@(M),@(K)] dataType:MPSDataTypeInt8];
                MPSGraphTensorData *bData = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:bufB shape:@[@(K),@(N)] dataType:MPSDataTypeInt8];
                MPSGraphTensorData *cData = [[MPSGraphTensorData alloc]
                    initWithMTLBuffer:bufC shape:@[@(M),@(N)] dataType:MPSDataTypeInt32];

                NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
                    @{ A8: aData, B8: bData };
                NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                    @{ C: cData };

                unsigned int warmup = warmupCount > 0 ? warmupCount : 2;
                double per_iter_us = timeMPSGraph(queue, g, feeds, results, warmup);
                if (per_iter_us <= 0.0)
                {
                    log->print("timing probe failed" NEWLINE);
                    log->resultRecord("int8", 0.0f);
                }
                else
                {
                    unsigned int iters = pickIters(per_iter_us, forceIters, specifiedIters);
                    double mean_us = timeMPSGraph(queue, g, feeds, results, iters);
                    double tops = ops_per_iter * 1.0e6 / mean_us / 1.0e12;
                    log->print((float)tops);
                    log->print(NEWLINE);
                    log->resultRecord("int8", (float)tops);
                }
            }
        }
    }

    log->resultScopeEnd(); // mps-gemm-int
    return 0;
}

#endif // ENABLE_METAL
