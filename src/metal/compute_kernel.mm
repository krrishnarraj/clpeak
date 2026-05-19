#ifdef ENABLE_METAL

#include "mtl_internal.h"

// Time a kernel batched as `iters` dispatches inside one MTLCommandBuffer,
// where `iters` is calibrated from a one-shot warmup so the timed phase lands
// at ~targetTimeUs.  Returns mean per-iter GPU time in microseconds (uses
// cmdBuf.GPUStartTime/GPUEndTime).  forcedIters != 0 short-circuits
// calibration (matches --iters).
float mtlRunDispatches(MetalDevice &dev, id<MTLComputePipelineState> pso,
                           id<MTLBuffer> outBuf, const void *scalarArg, uint32_t scalarSize,
                           id<MTLBuffer> secondBuf,
                           MTLSize gridSize, MTLSize tgSize,
                           unsigned int warmup,
                           unsigned int targetTimeUs, unsigned int forcedIters)
{
    auto enqueue = [&](unsigned int n) -> id<MTLCommandBuffer> {
        id<MTLCommandBuffer> cmdBuf = [dev.impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        if (outBuf)    [enc setBuffer:outBuf    offset:0 atIndex:0];
        if (secondBuf) [enc setBuffer:secondBuf offset:0 atIndex:1];
        if (scalarArg && scalarSize > 0)
            [enc setBytes:scalarArg length:scalarSize atIndex:1];
        for (unsigned int i = 0; i < n; i++)
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cmdBuf commit];
        return cmdBuf;
    };

    auto runBatch = [&](unsigned int n) -> float {
        NSProcessInfo *pi = [NSProcessInfo processInfo];
        double t0 = pi.systemUptime;
        id<MTLCommandBuffer> b = enqueue(n);
        [b waitUntilCompleted];
        double t1 = pi.systemUptime;
        CFTimeInterval gpuTime = b.GPUEndTime - b.GPUStartTime;
        CFTimeInterval wallTime = t1 - t0;
        // On the iOS simulator (or any configuration where the Metal driver
        // does not implement GPU profiling timestamps), GPUStartTime /
        // GPUEndTime may be zero or a tiny fraction of wall time.  Fall
        // back to host wall-clock time when the GPU time looks implausible
        // (< 1% of wall time, or zero).
        if (gpuTime > 0.0 && gpuTime >= wallTime * 0.01)
            return (float)(gpuTime * 1e6);
        return (float)(wallTime * 1e6);
    };

    // Phase 1: untimed warmup. Keep each warmup as its own completed command
    // buffer so slow kernels do not get batched before calibration.
    for (unsigned int i = 0; i < warmup; i++)
    {
        id<MTLCommandBuffer> w = enqueue(1);
        [w waitUntilCompleted];
    }

    // Phase 2: timed calibration probe. Keep this to one dispatch so warmup
    // does not force a multi-dispatch command buffer on slow kernels.
    unsigned int probeIters = 1;
    float probeUs = runBatch(probeIters);
    double per_iter_us = (double)probeUs / (double)probeIters;

    // Phase 3: real timed run with calibrated iter count.
    unsigned int iters = pickIters(per_iter_us, targetTimeUs, forcedIters);
    float totalUs = runBatch(iters);
    return totalUs / (float)iters;
}

// ---------------------------------------------------------------------------
// Shared compute-peak driver.  Mirrors vkPeak::runComputeKernel /
// CudaPeak::runComputeKernel: one output buffer, dispatch each variant.
// ---------------------------------------------------------------------------

int MetalPeak::runComputeKernel(MetalDevice &dev, benchmark_config_t &cfg,
                                const mtl_compute_desc_t &d)
{
    auto test = currentDeviceScope->beginTest({d.resultTag, d.title, d.unit});

    if (d.skip)
    {
        std::vector<std::string> skipLabels;
        if (d.variants && d.numVariants > 0)
        {
            for (uint32_t i = 0; i < d.numVariants; i++)
            {
                std::string label(d.variants[i].label);
                while (!label.empty() && label.back() == ' ')
                    label.pop_back();
                skipLabels.push_back(label);
            }
        }
        else
        {
            skipLabels.push_back(d.metricLabel);
        }
        for (const auto &label : skipLabels)
            test.skip(label, ResultStatus::Unsupported,
                       d.skipMsg ? d.skipMsg : "Skipped");
        return 0;
    }

    struct Variant { const char *label; const char *kernelName; const char *src; const char *srcName; };
    std::vector<Variant> variants;
    if (d.variants && d.numVariants > 0)
        for (uint32_t i = 0; i < d.numVariants; i++)
            variants.push_back({d.variants[i].label, d.variants[i].kernelName,
                                d.variants[i].src,   d.variants[i].srcName});
    else
        variants.push_back({d.metricLabel, d.kernelName, d.src, d.srcName});

    const uint32_t tgSize = d.threadsPerGroup ? d.threadsPerGroup : 256;
    const uint32_t outPerGroup = d.outElemsPerGroup ? d.outElemsPerGroup : tgSize;
    uint64_t globalThreads = mtlTargetGlobalThreads(dev.info);
    uint64_t bytesPerGroup = (uint64_t)outPerGroup * d.elemSize;
    uint64_t maxGroups  = dev.info.maxBufferLength / bytesPerGroup;
    uint64_t wantGroups = globalThreads / tgSize;
    uint32_t numGroups  = (uint32_t)((wantGroups < maxGroups) ? wantGroups : maxGroups);
    uint64_t bufferBytes = (uint64_t)numGroups * bytesPerGroup;

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:bufferBytes
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        std::vector<std::string> skipLabels;
        if (d.variants && d.numVariants > 0)
        {
            for (uint32_t i = 0; i < d.numVariants; i++)
            {
                std::string label(d.variants[i].label);
                while (!label.empty() && label.back() == ' ')
                    label.pop_back();
                skipLabels.push_back(label);
            }
        }
        else
        {
            skipLabels.push_back(d.metricLabel);
        }
        for (const auto &label : skipLabels)
            test.skip(label, ResultStatus::Error, "Failed to allocate output buffer");
        return -1;
    }

    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    for (const auto &v : variants)
    {
        id<MTLComputePipelineState> pso = mtlGetPipeline(dev, v.src, v.srcName, v.kernelName);
        if (!pso)
        {
            std::string metricTag(v.label);
            while (!metricTag.empty() && metricTag.back() == ' ')
                metricTag.pop_back();
            test.skip(metricTag, ResultStatus::Error, "Kernel compile failed");
            continue;
        }

        float us = mtlRunDispatches(dev, pso, outBuf, d.scalarArg, d.scalarSize, nil,
                                 gridSize, tgSizeM, warmupCount,
                                 cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        uint64_t totalThreads = (uint64_t)numGroups * tgSize;
        double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
        float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us / divider);

        // Strip the right-padding used for stdout column alignment so the
        // metric tag stored in the dump (and shown in the Android UI) is
        // a clean canonical label.
        std::string metricTag(v.label);
        while (!metricTag.empty() && metricTag.back() == ' ')
            metricTag.pop_back();
        test.emit(metricTag, value);
    }

    return 0;
}

#endif // ENABLE_METAL
