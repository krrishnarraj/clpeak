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
//   * MPSGraph for bf16 (gated to Apple9+ / OS support).  MPSMatrix doesn't
//     support bf16; MPSGraph does, and on Apple9+ it lowers to bf16
//     simdgroup_matrix.
//
// int8 / int4 / fp8 are reported as `unsupported on this device` records.
// MPSGraph's matmul is float-only; MPSMatrix doesn't accept ints either;
// Apple silicon's hw int8 path is exposed via Core ML / MPSCNN
// convolutions, not a general GEMM, so they're outside scope here.

#include <metal/mtl_peak.h>
#include <common/common.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <TargetConditionals.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cmath>
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

// Target wall-clock window for ONE full-size GEMM iteration.
constexpr double kGemmTargetIterUs = 100000.0;   // 100 ms

// Probe size: large enough that MPS dispatches the same tiled kernel the
// full-size run uses (so the timing extrapolates), small enough to cost a few
// ms on any Apple GPU.
constexpr uint32_t kGemmProbeDim = 2048;

// D when the probe cannot run at all (allocation or timing failure).
constexpr uint32_t kGemmFallbackDim = 4096;

// Bound D by what the device can actually hold: three square fp32 matrices
// (the worst-case dtype, so one D works for every variant) inside 25% of the
// working set, and any ONE of them inside maxBufferLength -- each matrix is a
// single MTLBuffer, and iOS reports a much smaller cap than the Mac.
uint32_t clampGemmDim(const mtl_device_info_t &info, uint64_t D)
{
    uint64_t budget = (info.recommendedMaxWorkingSetSize ? info.recommendedMaxWorkingSetSize
                                                         : (uint64_t)4 << 30) / 4;
    while (D > 1024 && 3ULL * D * D * 4 > budget)
        D /= 2;
    while (D > 1024 && info.maxBufferLength && D * D * 4 > info.maxBufferLength)
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

// Choose the square GEMM dim by timing one small GEMM and extrapolating.
//
// MPS matmul is O(D^3), so t(D) = t(D0) * (D/D0)^3 and the D landing on the
// target window is D0 * cbrt(target / t0).  Measured rather than derived from
// gpuCoreCount because that count is unavailable on iOS (IOKit is not reachable
// from a sandboxed app, so every iPhone and iPad looked like an 8-core base
// config) and because a core count says nothing about clock or thermal state.
// It also actually delivers the equal-wall-clock-window goal: work grows as
// D^3, so scaling D linearly with core count stretched the window from ~50 ms
// on an M1 base to ~550 ms on an M1 Ultra.
uint32_t measureGemmDim(id<MTLDevice> mtlDev, id<MTLCommandQueue> queue,
                        const mtl_device_info_t &info)
{
    const uint32_t D0 = kGemmProbeDim;
    const uint32_t fallback = clampGemmDim(info, kGemmFallbackDim);

    @autoreleasepool {
        const uint64_t bytes = (uint64_t)D0 * D0 * 4;
        id<MTLBuffer> a = makePrivateBuffer(mtlDev, bytes);
        id<MTLBuffer> b = makePrivateBuffer(mtlDev, bytes);
        id<MTLBuffer> c = makePrivateBuffer(mtlDev, bytes);
        if (!a || !b || !c)
        {
            CLPEAK_VLOG("mps-gemm: probe alloc failed, using D=%u\n", fallback);
            return fallback;
        }
        {
            // Same benign 0x3f fill as the measured run: timing should not be
            // read off whatever bit patterns the allocator happened to leave.
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            for (id<MTLBuffer> buf in @[a, b])
                [blit fillBuffer:buf range:NSMakeRange(0, buf.length) value:0x3f];
            [blit endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        MPSMatrixDescriptor *desc =
            [MPSMatrixDescriptor matrixDescriptorWithRows:D0 columns:D0
                                                 rowBytes:(NSUInteger)D0 * 4
                                                 dataType:MPSDataTypeFloat32];
        MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:a descriptor:desc];
        MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:b descriptor:desc];
        MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:c descriptor:desc];
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:mtlDev
             transposeLeft:NO
            transposeRight:NO
                resultRows:D0
             resultColumns:D0
           interiorColumns:D0
                     alpha:1.0
                      beta:0.0];

        timeMPSMatMul(queue, mm, mA, mB, mC, 1);       // discard: first encode compiles
        double t0_us = timeMPSMatMul(queue, mm, mA, mB, mC, 3);
        if (t0_us <= 0.0)
        {
            CLPEAK_VLOG("mps-gemm: probe timing failed, using D=%u\n", fallback);
            return fallback;
        }

        uint64_t D = (uint64_t)((double)D0 * std::cbrt(kGemmTargetIterUs / t0_us));

        // Round to 1024, not 256: cbrt() damps timing noise but does not remove
        // it, and D sets the reported TFLOPS, so a fine step would let a device
        // drift between neighbouring sizes run to run.  Coarse buckets keep one
        // device on one size.
        D = ((D + 512) / 1024) * 1024;
        if (D < 2048)  D = 2048;
        if (D > 16384) D = 16384;

        const uint32_t chosen = clampGemmDim(info, D);
        CLPEAK_VLOG("mps-gemm: probe D=%u took %.2f ms/iter -> D=%u\n",
                    D0, t0_us / 1000.0, chosen);
        return chosen;
    }
}

// ---- MPSGraph path (bf16; Apple9+ / OS support) ---------------------------

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

} // namespace

int MetalPeak::runMpsGemm(MetalDevice &dev, benchmark_config_t &cfg)
{
    auto test = currentDeviceScope->beginTest(
        {"mps_gemm", "MPS GEMM peak", "flops", Category::Unknown,
         "Matrix-multiply speed through Apple's own tuned GPU library, on a "
         "large square problem.  Where the compute rows show what the hardware "
         "can do in principle, this shows what Apple's shipping code actually "
         "reaches on the operation most graphics and AI work is built from.",
         TestShape::Heterogeneous, "data type"});

    if (!dev.info.isAppleSilicon)
    {
        test.skipAll({"fp32", "fp16", "bf16"}, ResultStatus::Unsupported,
                      "MPS GEMM requires Apple silicon");
        return 0;
    }

    id<MTLDevice>       mtlDev = dev.impl->device;
    id<MTLCommandQueue> queue  = dev.impl->queue;

    const uint32_t D = measureGemmDim(mtlDev, queue, dev.info);
    const uint32_t M = D, N = D, K = D;
    const double  flops_per_iter = 2.0 * (double)M * (double)N * (double)K;

    // One note per dtype row, threaded through every emit / skip site below.
    const char *fp32Note = "Full 32-bit precision, the accurate but slowest option.";
    const char *fp16Note = "16-bit inputs, which the GPU's matrix hardware runs at "
                           "several times the 32-bit rate.";
    const char *bf16Note = "bfloat16 inputs -- 16 bits arranged for AI work, trading "
                           "digits of accuracy for a wider number range.  Needs an M3 "
                           "or newer GPU.";

    // Pre-allocate the largest input set (fp32, 4 bytes) once; smaller dtypes
    // alias the same MTLBuffer with a different MPSMatrixDescriptor stride.
    const uint64_t maxInBytes = (uint64_t)M * K * 4;
    id<MTLBuffer> bufA = makePrivateBuffer(mtlDev, maxInBytes);
    id<MTLBuffer> bufB = makePrivateBuffer(mtlDev, (uint64_t)K * N * 4);
    if (!bufA || !bufB)
    {
        test.skip("fp32", ResultStatus::Error, "Failed to allocate input buffers", fp32Note);
        test.skip("fp16", ResultStatus::Error, "Failed to allocate input buffers", fp16Note);
        test.skip("bf16", ResultStatus::Error, "Failed to allocate input buffers", bf16Note);
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

    auto runMatMul = [&](const char *label, const char *note, MPSDataType dt,
                         uint32_t elemSize)
    {
        @autoreleasepool {
            const uint64_t outBytes = (uint64_t)M * N * elemSize;
            id<MTLBuffer> bufC = makePrivateBuffer(mtlDev, outBytes);
            if (!bufC)
            {
                test.skip(label, ResultStatus::Error, "output alloc failed", note);
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
                test.skip(label, ResultStatus::Error, "timing probe failed", note);
                return;
            }

            unsigned int iters = pickIters(per_iter_us, 5000000u, forceIters ? specifiedIters : 0);
            double mean_us = timeMPSMatMul(queue, mm, matA, matB, matC, iters);
            double tops = flops_per_iter * 1.0e6 / mean_us;

            test.emit(label, (float)tops, note);
        }
    };

    auto runGraphMatMul = [&](const char *label, const char *note, MPSDataType dt,
                              uint32_t elemSize)
    {
        @autoreleasepool {
            const uint64_t outBytes = (uint64_t)M * N * elemSize;
            id<MTLBuffer> bufC = makePrivateBuffer(mtlDev, outBytes);
            if (!bufC)
            {
                test.skip(label, ResultStatus::Error, "output alloc failed", note);
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
                test.skip(label, ResultStatus::Error, "timing probe failed", note);
                return;
            }

            unsigned int iters = pickIters(per_iter_us, 5000000u, forceIters ? specifiedIters : 0);
            double mean_us = timeMPSGraph(queue, g, feeds, results, iters);
            double tops = flops_per_iter * 1.0e6 / mean_us;

            test.emit(label, (float)tops, note);
        }
    };

    auto reportUnsupported = [&](const char *label, const char *note, const char *msg)
    {
        test.skip(label, ResultStatus::Unsupported, msg, note);
    };

    // ---- Run the dtype matrix --------------------------------------------

    runMatMul("fp32", fp32Note, MPSDataTypeFloat32, 4);
    runMatMul("fp16", fp16Note, MPSDataTypeFloat16, 2);

    // bf16: only via MPSGraph, and only on Apple9+ / OS support where the
    // path actually lowers to bf16 simdgroup_matrix.
#if TARGET_OS_IPHONE
    if (@available(iOS 17.0, *))
#else
    if (@available(macOS 14.0, *))
#endif
    {
        if (!dev.info.mpsGraphSupported)
            reportUnsupported("bf16", bf16Note,
                              "MPSGraph is unavailable on this device (iOS Simulator)");
        else if (dev.info.mpsGraphBF16Supported)
            runGraphMatMul("bf16", bf16Note, MPSDataTypeBFloat16, 2);
        else
            reportUnsupported("bf16", bf16Note,
                              "bf16 requires Apple9 (M3) -- unsupported on this device");
    }
    else
    {
#if TARGET_OS_IPHONE
        reportUnsupported("bf16", bf16Note, "bf16 requires iOS 17 -- unsupported on this device");
#else
        reportUnsupported("bf16", bf16Note, "bf16 requires macOS 14 -- unsupported on this device");
#endif
    }

    return 0;
}

// ---------------------------------------------------------------------------
// MPSGraph scaled-dot-product-attention peak (transformer-shaped composite:
// softmax(scale * QK^T) V).  This is the op Apple tunes for LLM inference
// (WWDC24); it complements the raw GEMM rows the way coopmat complements the
// FMA chains.  fp16 only -- the dtype every shipping attention path uses.
// FLOPs counted as the two matmuls (2*H*N^2*F each); softmax is ignored by
// the usual convention, so the row reads slightly below raw GEMM peak.
// ---------------------------------------------------------------------------

int MetalPeak::runMpsAttention(MetalDevice &dev, benchmark_config_t &cfg)
{
    (void)cfg;
    // B=1, H=16 heads, seq N=4096, head dim F=128 -- llama-class shape.
    const uint32_t H = 16, N = 4096, F = 128;

    auto test = currentDeviceScope->beginTest(
        {"mps_attention", "MPS attention SDPA (H16 S4096 D128)", "flops",
         Category::Unknown,
         "Speed of the attention step -- the operation a language model spends "
         "most of its time in, deciding which earlier words each word should "
         "look at.  Apple's library runs it as one fused unit; the shape is "
         "fixed at a small-LLM size so the number compares across devices.",
         TestShape::Homogeneous});

    if (!dev.info.isAppleSilicon)
    {
        test.skip("fp16", ResultStatus::Unsupported, "MPS attention requires Apple silicon");
        return 0;
    }

    // The whole test is MPSGraph; on a device MPSGraph cannot wrap, even
    // building the first MPSGraphTensorData throws an uncaught ObjC exception.
    if (!dev.info.mpsGraphSupported)
    {
        test.skip("fp16", ResultStatus::Unsupported,
                  "MPSGraph is unavailable on this device (iOS Simulator)");
        return 0;
    }

    // The shape is FIXED (not scaled to the device like pickGemmDim) so the
    // number is comparable across Macs and iPhones -- but that means a small
    // device must be able to hold it.  If MPSGraph lowers to an unfused
    // attention it materializes the [1,H,N,N] fp16 score matrix, so budget for
    // that worst case (H*N*N*2 = 512 MB here) plus the Q/K/V/O buffers; skip
    // rather than risk an allocation failure mid-encode on a phone.  The score
    // matrix is one buffer, so maxBufferLength bounds it too -- and that is the
    // only bound left when recommendedMaxWorkingSetSize reads 0 (unknown).
    {
        const uint64_t scoreBytes = (uint64_t)H * N * N * 2;
        const uint64_t needBytes  = scoreBytes + 4ULL * H * N * F * 2;
        const uint64_t budget     = dev.info.recommendedMaxWorkingSetSize;
        const uint64_t maxBuf     = dev.info.maxBufferLength;
        if ((budget && needBytes > budget / 2) || (maxBuf && scoreBytes > maxBuf))
        {
            test.skip("fp16", ResultStatus::Unsupported,
                      "device memory limits too small for the fixed H16/S4096/D128 shape");
            return 0;
        }
    }

#if TARGET_OS_IPHONE
    if (@available(iOS 18.0, *))
#else
    if (@available(macOS 15.0, *))
#endif
    {
        @autoreleasepool {
            id<MTLDevice>       mtlDev = dev.impl->device;
            id<MTLCommandQueue> queue  = dev.impl->queue;

            const uint64_t elems = (uint64_t)H * N * F;
            id<MTLBuffer> bufQ = makePrivateBuffer(mtlDev, elems * 2);
            id<MTLBuffer> bufK = makePrivateBuffer(mtlDev, elems * 2);
            id<MTLBuffer> bufV = makePrivateBuffer(mtlDev, elems * 2);
            id<MTLBuffer> bufO = makePrivateBuffer(mtlDev, elems * 2);
            if (!bufQ || !bufK || !bufV || !bufO)
            {
                test.skip("fp16", ResultStatus::Error, "Failed to allocate Q/K/V buffers");
                return -1;
            }
            {
                // 0x3c3c is a benign ~1.06 in fp16; softmax renormalizes, so
                // the numbers stay bounded regardless of content.
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
                for (id<MTLBuffer> b in @[bufQ, bufK, bufV])
                    [blit fillBuffer:b range:NSMakeRange(0, b.length) value:0x3c];
                [blit endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            NSArray<NSNumber *> *shape = @[@1, @(H), @(N), @(F)];
            MPSGraph *g = [MPSGraph new];
            MPSGraphTensor *Q = [g placeholderWithShape:shape dataType:MPSDataTypeFloat16 name:@"Q"];
            MPSGraphTensor *K = [g placeholderWithShape:shape dataType:MPSDataTypeFloat16 name:@"K"];
            MPSGraphTensor *V = [g placeholderWithShape:shape dataType:MPSDataTypeFloat16 name:@"V"];
            MPSGraphTensor *O = [g scaledDotProductAttentionWithQueryTensor:Q
                                                                  keyTensor:K
                                                                valueTensor:V
                                                                      scale:1.0f / std::sqrt((float)F)
                                                                       name:@"O"];

            auto tdata = [&](id<MTLBuffer> b) {
                return [[MPSGraphTensorData alloc] initWithMTLBuffer:b
                                                               shape:shape
                                                            dataType:MPSDataTypeFloat16];
            };
            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds =
                @{ Q: tdata(bufQ), K: tdata(bufK), V: tdata(bufV) };
            NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
                @{ O: tdata(bufO) };

            unsigned int warmup = warmupCount > 0 ? warmupCount : 2;
            double per_iter_us = timeMPSGraph(queue, g, feeds, results, warmup);
            if (per_iter_us <= 0.0)
            {
                test.skip("fp16", ResultStatus::Error, "timing probe failed");
                return -1;
            }
            unsigned int iters = pickIters(per_iter_us, 5000000u, forceIters ? specifiedIters : 0);
            double mean_us = timeMPSGraph(queue, g, feeds, results, iters);

            // QK^T and PV: 2 * (2*H*N*N*F) flops.
            double flops = 4.0 * (double)H * (double)N * (double)N * (double)F;
            test.emit("fp16", (float)(flops * 1.0e6 / mean_us));
        }
    }
    else
    {
#if TARGET_OS_IPHONE
        test.skip("fp16", ResultStatus::Unsupported, "SDPA op requires iOS 18");
#else
        test.skip("fp16", ResultStatus::Unsupported, "SDPA op requires macOS 15");
#endif
    }
    return 0;
}

#endif // ENABLE_METAL
