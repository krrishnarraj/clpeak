#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (Metal -- MTLTexture + sampler)
// ---------------------------------------------------------------------------

int MetalPeak::runImageBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    auto test = currentDeviceScope->beginTest({"image_memory_bandwidth", "Image memory bandwidth (GBPS)", "gbps"});

    const NSUInteger imgW = 4096, imgH = 4096;
    const uint32_t tgSize = 256;
    uint64_t globalThreads = mtlTargetGlobalThreads(dev.info);
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);

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
    };
    const V vs[] = {
        { "rgba32f", "image_bandwidth",       MTLPixelFormatRGBA32Float, 16 },
        { "rgba16f", "image_bandwidth_half4", MTLPixelFormatRGBA16Float, 8  },
        { "rgba8",   "image_bandwidth",       MTLPixelFormatRGBA8Unorm,  4  },
        { "r32f",    "image_bandwidth_r32f",  MTLPixelFormatR32Float,    4  },
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
            test.skip(v.label, ResultStatus::Error, "Texture alloc failed");
            continue;
        }

        id<MTLComputePipelineState> pso = mtlGetPipeline(dev,
            mtl_kernels::image_bandwidth_src,
            mtl_kernels::image_bandwidth_name, v.kname);
        if (!pso) {
            test.skip(v.label, ResultStatus::Error, "Kernel compile failed");
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
        // Phase 1: untimed warmup.
        for (unsigned int i = 0; i < warmupCount; i++)
        {
            id<MTLCommandBuffer> w = enqueue(1);
            [w waitUntilCompleted];
        }
        // Phase 2: timed calibration probe.
        unsigned int probeIters = 1;
        id<MTLCommandBuffer> p = enqueue(probeIters);
        [p waitUntilCompleted];
        double per_iter_us = (p.GPUEndTime - p.GPUStartTime) * 1e6 / (double)probeIters;
        unsigned int iters = pickIters(per_iter_us, cfg.targetTimeUs,
                                       forceIters ? specifiedIters : 0);
        // Phase 3: real timed run.
        id<MTLCommandBuffer> t = enqueue(iters);
        [t waitUntilCompleted];
        float us = (float)((t.GPUEndTime - t.GPUStartTime) * 1e6 / iters);
        uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * v.bytesPerPixel * globalThreads;
        float gbps = (float)bytes / us / 1e3f;
        test.emit(v.label, gbps);
    }

    return 0;
}


#endif // ENABLE_METAL
