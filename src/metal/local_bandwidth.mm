#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Local memory bandwidth (Metal -- threadgroup memory)
// ---------------------------------------------------------------------------

int MetalPeak::runLocalBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    auto test = currentDeviceScope->beginTest({"local_memory_bandwidth", "Local memory bandwidth", "gbps"});

    const uint32_t tgSize = 256;
    uint64_t globalThreads = mtlTargetGlobalThreads(dev.info);
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:globalThreads * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        test.skipAll({"float", "float2", "float4", "float8"},
                      ResultStatus::Error, "Buffer alloc failed");
        return -1;
    }

    struct V { const char *label; const char *kname; uint32_t width; };
    const V vs[] = {
        {"float",  "local_bandwidth_v1", 1},
        {"float2", "local_bandwidth_v2", 2},
        {"float4", "local_bandwidth_v4", 4},
        {"float8", "local_bandwidth_v8", 8},
    };
    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);
    for (const auto &v : vs)
    {
        id<MTLComputePipelineState> pso = mtlGetPipeline(dev,
            mtl_kernels::local_bandwidth_src,
            mtl_kernels::local_bandwidth_name, v.kname);
        if (!pso) {
            test.skip(v.label, ResultStatus::Error, "Kernel compile failed");
            continue;
        }
        float us = mtlRunDispatches(dev, pso, outBuf, nullptr, 0, nil,
                                 gridSize, tgSizeM, warmupCount,
                                 cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.width * sizeof(float) * globalThreads;
        float gbps = (float)bytes / us / 1e3f;
        test.emit(v.label, gbps);
    }

    return 0;
}


#endif // ENABLE_METAL
