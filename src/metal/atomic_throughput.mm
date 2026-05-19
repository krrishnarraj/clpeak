#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Atomic throughput (Metal -- global + local atomics)
// ---------------------------------------------------------------------------

int MetalPeak::runAtomicThroughput(MetalDevice &dev, benchmark_config_t &cfg)
{
    auto test = currentDeviceScope->beginTest({"atomic_throughput", "Atomic throughput", "gops"});

    const uint32_t tgSize = 256;
    uint64_t globalThreads = mtlTargetGlobalThreads(dev.info);
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);
    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    auto runOne = [&](const char *src, const char *srcName, const char *fnName,
                      NSUInteger bufBytes) -> float
    {
        id<MTLBuffer> buf = [dev.impl->device newBufferWithLength:bufBytes
                                                          options:MTLResourceStorageModeShared];
        if (!buf) return -1.0f;
        memset(buf.contents, 0, bufBytes);
        id<MTLComputePipelineState> pso = mtlGetPipeline(dev, src, srcName, fnName);
        if (!pso) return -1.0f;
        float us = mtlRunDispatches(dev, pso, buf, nullptr, 0, nil,
                                 gridSize, tgSizeM, warmupCount,
                                 cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        return us;
    };

    auto reportOne = [&](const std::string &resultKey, const char *src, const char *srcName,
                         const char *fnName, NSUInteger bufBytes,
                         bool skip, const char *skipMsg)
    {
        if (skip)
        {
            test.skip(resultKey, ResultStatus::Unsupported,
                       skipMsg ? skipMsg : "Skipped");
            return;
        }
        float us = runOne(src, srcName, fnName, bufBytes);
        if (us > 0)
        {
            float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
            test.emit(resultKey, gops);
        }
        else
        {
            test.skip(resultKey, ResultStatus::Error, "Kernel compile failed");
        }
    };

    auto report = [&](const char *label, const char *src, const char *srcName,
                      const char *gFn, const char *lFn, uint32_t elemSize,
                      bool skip, const char *skipMsg)
    {
        reportOne(std::string(label) + "_global",
                  src, srcName, gFn, globalThreads * elemSize, skip, skipMsg);
        if (lFn)
        {
            reportOne(std::string(label) + "_local",
                      src, srcName, lFn, numGroups * elemSize, skip, skipMsg);
        }
    };

    report("int",  mtl_kernels::atomic_throughput_src,
           mtl_kernels::atomic_throughput_name,
           "atomic_throughput_global",      "atomic_throughput_local",
           sizeof(int), false, nullptr);
    report("uint",  mtl_kernels::atomic_throughput_src,
           mtl_kernels::atomic_throughput_name,
           "atomic_throughput_global_uint", "atomic_throughput_local_uint",
           sizeof(uint32_t), false, nullptr);
    // atomic_ulong: 64-bit atomic add isn't accepted by every MSL/SDK combo.
    // Skipped on Apple7; on Apple8+ we still let mtlGetPipeline report
    // compile/load failed if the SDK rejects it.
    report("ulong",  mtl_kernels::atomic_throughput_ulong_src,
           mtl_kernels::atomic_throughput_ulong_name,
           "atomic_throughput_global_ulong", "atomic_throughput_local_ulong",
           sizeof(uint64_t),
           !dev.info.atomic64Supported,
           "skipped (requires Apple8 / M2 or newer)");

    return 0;
}

int MetalPeak::runAtomicThroughputFp(MetalDevice &dev, benchmark_config_t &cfg)
{
    auto test = currentDeviceScope->beginTest({"atomic_throughput", "Atomic throughput", "gflops"});

    const uint32_t tgSize = 256;
    uint64_t globalThreads = mtlTargetGlobalThreads(dev.info);
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);
    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);
    auto runOne = [&](const char *src, const char *srcName, const char *fnName,
                      NSUInteger bufBytes) -> float
    {
        id<MTLBuffer> buf = [dev.impl->device newBufferWithLength:bufBytes
                                                          options:MTLResourceStorageModeShared];
        if (!buf) return -1.0f;
        memset(buf.contents, 0, bufBytes);
        id<MTLComputePipelineState> pso = mtlGetPipeline(dev, src, srcName, fnName);
        if (!pso) return -1.0f;
        return mtlRunDispatches(dev, pso, buf, nullptr, 0, nil,
                             gridSize, tgSizeM, warmupCount,
                             cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    };

    auto reportOne = [&](const std::string &resultKey, const char *src, const char *srcName,
                         const char *fnName, NSUInteger bufBytes)
    {
        float us = runOne(src, srcName, fnName, bufBytes);
        if (us > 0)
        {
            float gflops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
            test.emit(resultKey, gflops);
        }
        else
        {
            test.skip(resultKey, ResultStatus::Error, "Kernel compile failed");
        }
    };

    reportOne("float_global",
              mtl_kernels::atomic_throughput_float_src,
              mtl_kernels::atomic_throughput_float_name,
              "atomic_throughput_global_float",
              globalThreads * sizeof(float));
    return 0;
}

// Free-function enumeration used by --list-devices and the Android JNI surface.
// Mirrors the device-discovery logic at the top of MetalPeak::runAll() but
// stops at name extraction.

#endif // ENABLE_METAL
