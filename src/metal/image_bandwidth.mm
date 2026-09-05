#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (Metal -- MTLTexture + sampler)
// ---------------------------------------------------------------------------

int MetalPeak::runImageBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    auto test = currentDeviceScope->beginTest(
        {"image_memory_bandwidth", "Image memory bandwidth", "bps",
         Category::Unknown,
         "How many bytes per second the GPU reads through its texture units, "
         "which take a different path to memory than plain buffer reads.  Each "
         "reading uses a different pixel format, so they differ in how many "
         "bytes one pixel costs.",
         TestShape::Heterogeneous, "pixel format"});

    const NSUInteger imgW = 4096, imgH = 4096;
    const uint32_t tgSize = 256;
    // Size the dispatch so each pixel is read exactly once per launch,
    // eliminating cache reuse that inflates apparent bandwidth.
    uint32_t numGroups = ((uint32_t)imgW * (uint32_t)imgH) / IMAGE_FETCH_PER_WI / tgSize;
    if (numGroups == 0) numGroups = 1;
    uint64_t globalThreads = (uint64_t)numGroups * tgSize;

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:globalThreads * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        test.skipAll({"rgba32f", "rgba16f", "rgba8", "r32f"},
                      ResultStatus::Error, "Output buffer alloc failed");
        return -1;
    }

    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    struct V {
        const char     *label;
        const char     *kname;
        MTLPixelFormat  fmt;
        uint32_t        bytesPerPixel;
        const char     *note;
    };
    const V vs[] = {
        { "rgba32f", "image_bandwidth",       MTLPixelFormatRGBA32Float, 16,
          "Four colour channels at full 32-bit precision: 16 bytes a pixel, the "
          "heaviest format here." },
        { "rgba16f", "image_bandwidth_half4", MTLPixelFormatRGBA16Float, 8,
          "Four channels at half precision: 8 bytes a pixel." },
        { "rgba8",   "image_bandwidth",       MTLPixelFormatRGBA8Unorm,  4,
          "Four 8-bit channels: 4 bytes a pixel, the format ordinary images use." },
        { "r32f",    "image_bandwidth_r32f",  MTLPixelFormatR32Float,    4,
          "A single 32-bit channel: also 4 bytes a pixel, but one value instead "
          "of four." },
    };

    for (const auto &v : vs)
    {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:v.fmt
                                         width:imgW height:imgH mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [dev.impl->device newTextureWithDescriptor:td];
        if (!tex)
        {
            test.skip(v.label, ResultStatus::Error, "Texture alloc failed", v.note);
            continue;
        }

        // Fill texture with pseudo-random data to defeat hardware compression.
        {
            NSUInteger numPixels = imgW * imgH;
            NSUInteger numFloats = numPixels * v.bytesPerPixel / sizeof(float);
            float *staging = new float[numFloats];
            populate(staging, numFloats);
            MTLRegion region = MTLRegionMake2D(0, 0, imgW, imgH);
            [tex replaceRegion:region
                  mipmapLevel:0
                    withBytes:staging
                  bytesPerRow:imgW * v.bytesPerPixel];
            delete[] staging;
        }

        id<MTLComputePipelineState> pso = mtlGetPipeline(dev,
            mtl_kernels::image_bandwidth_src,
            mtl_kernels::image_bandwidth_name, v.kname);
        if (!pso) {
            test.skip(v.label, ResultStatus::Error, "Kernel compile failed", v.note);
            continue;
        }

        int walk = 0;
        auto enqueue = [&](unsigned int n) -> id<MTLCommandBuffer> {
            id<MTLCommandBuffer> cb = [dev.impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setTexture:tex atIndex:0];
            [enc setBuffer:outBuf offset:0 atIndex:0];
            [enc setBytes:&walk length:sizeof(walk) atIndex:1];
            for (unsigned int i = 0; i < n; i++)
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSizeM];
            [enc endEncoding];
            [cb commit];
            return cb;
        };
        // Phase 1: untimed warmup.
        for (unsigned int i = 0; i < warmupCount; i++)
        {
            id<MTLCommandBuffer> w = enqueue(1);
            [w waitUntilCompleted];
        }
        // Phase 2: timed calibration probe.
        unsigned int probeIters = 1;
        double probeT0 = [NSProcessInfo processInfo].systemUptime;
        id<MTLCommandBuffer> p = enqueue(probeIters);
        [p waitUntilCompleted];
        double probeT1 = [NSProcessInfo processInfo].systemUptime;
        CFTimeInterval gpuProbeSec = p.GPUEndTime - p.GPUStartTime;
        CFTimeInterval wallProbeSec = probeT1 - probeT0;
        double gpuProbeUs = gpuProbeSec * 1e6;
        if (gpuProbeSec <= 0.0 || gpuProbeSec < wallProbeSec * 0.01)
            gpuProbeUs = wallProbeSec * 1e6;
        double per_iter_us = gpuProbeUs / (double)probeIters;
        unsigned int iters = pickIters(per_iter_us, cfg.targetTimeUs,
                                       forceIters ? specifiedIters : 0);
        // Phase 3: real timed run.
        double t0 = [NSProcessInfo processInfo].systemUptime;
        id<MTLCommandBuffer> t = enqueue(iters);
        [t waitUntilCompleted];
        double t1 = [NSProcessInfo processInfo].systemUptime;
        CFTimeInterval gpuTimedSec = t.GPUEndTime - t.GPUStartTime;
        CFTimeInterval wallTimedSec = t1 - t0;
        double gpuTimedUs = gpuTimedSec * 1e6;
        if (gpuTimedSec <= 0.0 || gpuTimedSec < wallTimedSec * 0.01)
            gpuTimedUs = wallTimedSec * 1e6;
        float us = (float)(gpuTimedUs / iters);
        // Two walk orders are raced and the faster reported -- why, and why
        // neither can flatter the result: the image-bandwidth block in
        // include/common/common.h.
        walk = 1;
        double c0 = [NSProcessInfo processInfo].systemUptime;
        id<MTLCommandBuffer> c = enqueue(iters);
        [c waitUntilCompleted];
        double c1 = [NSProcessInfo processInfo].systemUptime;
        CFTimeInterval gpuColSec = c.GPUEndTime - c.GPUStartTime;
        CFTimeInterval wallColSec = c1 - c0;
        double gpuColUs = gpuColSec * 1e6;
        if (gpuColSec <= 0.0 || gpuColSec < wallColSec * 0.01)
            gpuColUs = wallColSec * 1e6;
        float colUs = (float)(gpuColUs / iters);
        walk = 0;

        uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * v.bytesPerPixel * globalThreads;
        float rowBps = (float)bytes / us * 1e6f;
        float colBps = colUs > 0.0f ? (float)bytes / colUs * 1e6f : 0.0f;
        CLPEAK_VLOG("image_memory_bandwidth %s: row-major %.1f, column-major %.1f B/s\n",
                    v.label, rowBps, colBps);
        test.emit(v.label, std::max(rowBps, colBps), v.note);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Texture sample rate (bilinear texels/s) -- TMU throughput, not bandwidth.
// A small cache-resident texture is sampled with forced-fractional bilinear
// coordinates, so the filter units are the limiter rather than DRAM (that is
// what image_bandwidth measures).  Apple's TBDR parts sustain very high
// filtered-fetch rates from the SLC; this row makes that visible.
// ---------------------------------------------------------------------------

int MetalPeak::runTextureSampleRate(MetalDevice &dev, benchmark_config_t &cfg)
{
    auto test = currentDeviceScope->beginTest(
        {"texture_sample_rate", "Texture sample rate (bilinear)", "texels",
         Category::Bandwidth,
         "How many filtered texture lookups per second the GPU's sampling "
         "hardware performs.  Every lookup falls between pixels, so the hardware "
         "must blend four of them -- the basic operation of drawing any textured "
         "surface.  The texture is small enough to stay cached, so this measures "
         "the sampling units rather than memory.",
         TestShape::Heterogeneous, "pixel format"});

    const unsigned int SAMPLES_PER_WI = 64;   // must match texture_sample.metal
    const NSUInteger imgW = 1024, imgH = 1024;  // 4 MB rgba8 -- SLC-resident
    const uint32_t tgSize = 256;
    uint64_t globalThreads = mtlTargetGlobalThreads(dev.info);
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);
    if (numGroups == 0) numGroups = 1;
    globalThreads = (uint64_t)numGroups * tgSize;

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:globalThreads * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        test.skipAll({"rgba8", "rgba16f"}, ResultStatus::Error, "Output buffer alloc failed");
        return -1;
    }

    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    struct V {
        const char     *label;
        const char     *kname;
        MTLPixelFormat  fmt;
        uint32_t        bytesPerPixel;
        const char     *note;
    };
    const V vs[] = {
        { "rgba8",   "texture_sample_rgba8",   MTLPixelFormatRGBA8Unorm,  4,
          "Blending four 8-bit-per-channel pixels, the common case in games and UI." },
        { "rgba16f", "texture_sample_rgba16f", MTLPixelFormatRGBA16Float, 8,
          "Blending four half-precision pixels -- twice the data per lookup, so "
          "the rate typically drops." },
    };

    for (const auto &v : vs)
    {
        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:v.fmt
                                         width:imgW height:imgH mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [dev.impl->device newTextureWithDescriptor:td];
        if (!tex)
        {
            test.skip(v.label, ResultStatus::Error, "Texture alloc failed", v.note);
            continue;
        }
        {
            // Pseudo-random fill so hardware lossless compression can't dodge
            // the actual texel reads.
            NSUInteger numBytes = imgW * imgH * v.bytesPerPixel;
            float *staging = new float[numBytes / sizeof(float)];
            populate(staging, numBytes / sizeof(float));
            [tex replaceRegion:MTLRegionMake2D(0, 0, imgW, imgH)
                   mipmapLevel:0
                     withBytes:staging
                   bytesPerRow:imgW * v.bytesPerPixel];
            delete[] staging;
        }

        id<MTLComputePipelineState> pso = mtlGetPipeline(dev,
            mtl_kernels::texture_sample_src,
            mtl_kernels::texture_sample_name, v.kname);
        if (!pso) {
            test.skip(v.label, ResultStatus::Error, "Kernel compile failed", v.note);
            continue;
        }

        auto enqueue = [&](unsigned int n) -> id<MTLCommandBuffer> {
            id<MTLCommandBuffer> cb = [dev.impl->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setTexture:tex atIndex:0];
            [enc setBuffer:outBuf offset:0 atIndex:0];
            for (unsigned int i = 0; i < n; i++)
                [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSizeM];
            [enc endEncoding];
            [cb commit];
            return cb;
        };
        for (unsigned int i = 0; i < warmupCount; i++)
        {
            id<MTLCommandBuffer> w = enqueue(1);
            [w waitUntilCompleted];
        }
        unsigned int probeIters = 1;
        double probeT0 = [NSProcessInfo processInfo].systemUptime;
        id<MTLCommandBuffer> p = enqueue(probeIters);
        [p waitUntilCompleted];
        double probeT1 = [NSProcessInfo processInfo].systemUptime;
        CFTimeInterval gpuProbeSec = p.GPUEndTime - p.GPUStartTime;
        CFTimeInterval wallProbeSec = probeT1 - probeT0;
        double gpuProbeUs = gpuProbeSec * 1e6;
        if (gpuProbeSec <= 0.0 || gpuProbeSec < wallProbeSec * 0.01)
            gpuProbeUs = wallProbeSec * 1e6;
        double per_iter_us = gpuProbeUs / (double)probeIters;
        unsigned int iters = pickIters(per_iter_us, cfg.targetTimeUs,
                                       forceIters ? specifiedIters : 0);
        double t0 = [NSProcessInfo processInfo].systemUptime;
        id<MTLCommandBuffer> t = enqueue(iters);
        [t waitUntilCompleted];
        double t1 = [NSProcessInfo processInfo].systemUptime;
        CFTimeInterval gpuTimedSec = t.GPUEndTime - t.GPUStartTime;
        CFTimeInterval wallTimedSec = t1 - t0;
        double gpuTimedUs = gpuTimedSec * 1e6;
        if (gpuTimedSec <= 0.0 || gpuTimedSec < wallTimedSec * 0.01)
            gpuTimedUs = wallTimedSec * 1e6;
        float us = (float)(gpuTimedUs / iters);
        uint64_t samples = (uint64_t)SAMPLES_PER_WI * globalThreads;
        float texels = (float)samples / us * 1e6f;   // samples/us -> samples/s
        test.emit(v.label, texels, v.note);
    }

    return 0;
}

#endif // ENABLE_METAL
