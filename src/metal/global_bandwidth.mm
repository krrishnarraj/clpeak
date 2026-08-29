#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Global bandwidth (Metal)
// ---------------------------------------------------------------------------

int MetalPeak::runGlobalBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    const uint32_t tgSize = 256;

    auto test = currentDeviceScope->beginTest(
        {"global_memory_bandwidth", "Global memory bandwidth", "gbps",
         Category::Unknown,
         "How many bytes per second the GPU can stream out of main memory, "
         "reading a buffer far too large to cache.  Each reading fetches a "
         "different number of values per instruction, since wider fetches "
         "usually pull more through before the memory system saturates."});

    // Reserve enough scalar floats so the widest variant (v16 = 16 floats per
    // logical "WI" element) still aligns to (tg * FETCH_PER_WI * 16).
    const uint32_t maxVecScale = 16;
    uint64_t maxItems = dev.info.maxBufferLength / sizeof(float);
    uint64_t align    = (uint64_t)tgSize * FETCH_PER_WI * maxVecScale;
    uint64_t numItems = (maxItems / align) * align;
    if (numItems > cfg.globalBWMaxSize / sizeof(float))
        numItems = (cfg.globalBWMaxSize / sizeof(float) / align) * align;

    // The one number that decides whether this test measured memory or cache:
    // the timed phase re-reads the same buffer, so a working set that fits
    // behind the last-level cache reports the cache.  Metal exposes no cache
    // size and the SLC is memory-side, so this stays on the shared default --
    // print the working set so an implausible reading can be checked without
    // a rebuild.
    CLPEAK_VLOG("global_memory_bandwidth: working set %llu MB (Metal reports no cache size)\n",
                (unsigned long long)(numItems * sizeof(float) >> 20));

    id<MTLBuffer> inBuf  = [dev.impl->device newBufferWithLength:numItems * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:numItems * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!inBuf || !outBuf)
    {
        test.skipAll({"float", "float2", "float4", "float8", "float16"},
                      ResultStatus::Error, "Failed to allocate buffers");
        return -1;
    }
    // Touch the input with pseudo-random data so we don't measure
    // zero-page reads on copy-on-write or defeat hardware compression.
    populate((float *)inBuf.contents, numItems);

    struct V { const char *label; const char *kname; uint32_t width; };
    const V vs[] = {
        { "float",   "global_bandwidth",     1  },
        { "float2",  "global_bandwidth_v2",  2  },
        { "float4",  "global_bandwidth_v4",  4  },
        { "float8",  "global_bandwidth_v8",  8  },
        { "float16", "global_bandwidth_v16", 16 },
    };

    MTLSize tgSizeM = MTLSizeMake(tgSize, 1, 1);

    for (const auto &v : vs)
    {
        id<MTLComputePipelineState> pso = mtlGetPipeline(dev,
            mtl_kernels::global_bandwidth_src,
            mtl_kernels::global_bandwidth_name,
            v.kname);
        if (!pso) {
            test.skip(v.label, ResultStatus::Error, "Kernel compile failed",
                      mtlWidthNote(v.width));
            continue;
        }

        // Each threadgroup reads FETCH_PER_WI * tgSize logical-vec elements,
        // i.e. FETCH_PER_WI * tgSize * width scalar floats.
        uint64_t scalarsPerGroup = (uint64_t)tgSize * FETCH_PER_WI * v.width;
        uint32_t numGroups = (uint32_t)(numItems / scalarsPerGroup);
        if (numGroups == 0) numGroups = 1;
        MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);

        float us = mtlRunDispatches(dev, pso, inBuf, nullptr, 0, outBuf,
                                 gridSize, tgSizeM, warmupCount,
                                 cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        uint64_t bytesRead = (uint64_t)numGroups * scalarsPerGroup * sizeof(float);
        float gbps = (float)bytesRead / us / 1e3f;
        test.emit(v.label, gbps, mtlWidthNote(v.width));
    }

    return 0;
}


#endif // ENABLE_METAL
