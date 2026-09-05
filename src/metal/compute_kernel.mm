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
    auto test = currentDeviceScope->beginTest(
        {d.resultTag, d.title, d.unit, Category::Unknown,
         d.description ? d.description : "",
         d.shape, d.axis ? d.axis : ""});

    struct Variant { const char *label; const char *kernelName; const char *src;
                     const char *srcName; const char *description;
                     const char *altKernelName; const char *skipMsg; };
    std::vector<Variant> variants;
    if (d.variants && d.numVariants > 0)
        for (uint32_t i = 0; i < d.numVariants; i++)
            variants.push_back({d.variants[i].label, d.variants[i].kernelName,
                                d.variants[i].src,   d.variants[i].srcName,
                                d.variants[i].description,
                                d.variants[i].altKernelName,
                                d.variants[i].skipMsg});
    else
        variants.push_back({d.metricLabel, d.kernelName, d.src, d.srcName, nullptr,
                            nullptr, nullptr});

    // Labels carry right-padding for stdout column alignment; the metric tag
    // stored in the dump (and shown in the GUI) must be the clean label.
    auto metricTag = [](const char *label) {
        std::string s(label);
        while (!s.empty() && s.back() == ' ')
            s.pop_back();
        return s;
    };
    auto note = [](const char *text) { return text ? std::string(text) : std::string(); };

    if (d.skip)
    {
        for (const auto &v : variants)
            test.skip(metricTag(v.label), ResultStatus::Unsupported,
                      d.skipMsg ? d.skipMsg : "Skipped", note(v.description));
        return 0;
    }

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
        for (const auto &v : variants)
            test.skip(metricTag(v.label), ResultStatus::Error,
                      "Failed to allocate output buffer", note(v.description));
        return -1;
    }

    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    for (const auto &v : variants)
    {
        // One reading the hardware cannot take, in a test the rest of which it
        // can: the family stays one test and this row moves to the GUI's
        // unavailable section.
        if (v.skipMsg)
        {
            test.skip(metricTag(v.label), ResultStatus::Unsupported, v.skipMsg,
                      note(v.description));
            continue;
        }

        id<MTLComputePipelineState> pso = mtlGetPipeline(dev, v.src, v.srcName, v.kernelName);
        if (!pso)
        {
            test.skip(metricTag(v.label), ResultStatus::Error, "Kernel compile failed",
                      note(v.description));
            continue;
        }

        float us = mtlRunDispatches(dev, pso, outBuf, d.scalarArg, d.scalarSize, nil,
                                 gridSize, tgSizeM, warmupCount,
                                 cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        uint64_t totalThreads = (uint64_t)numGroups * tgSize;
        float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us);

        // Race the affine chain and keep the faster reading.  A failure here is
        // not an error -- the squaring chain already produced one.
        if (v.altKernelName)
        {
            id<MTLComputePipelineState> altPso =
                mtlGetPipeline(dev, v.src, v.srcName, v.altKernelName);
            if (altPso)
            {
                float altUs = mtlRunDispatches(dev, altPso, outBuf, d.scalarArg, d.scalarSize,
                                               nil, gridSize, tgSizeM, warmupCount,
                                               cfg.targetTimeUs,
                                               forceIters ? specifiedIters : 0);
                if (altUs > 0.0f)
                {
                    float altValue = (float)((double)totalThreads * (double)d.workPerWI
                                             * 1e6 / altUs);
                    CLPEAK_VLOG("%s %s: squaring chain %.1f, alt chain %.1f %s\n",
                                d.resultTag, metricTag(v.label).c_str(), value,
                                altValue, d.unit);
                    if (altValue > value * MAX_ALT_CHAIN_RATIO)
                        CLPEAK_VLOG("%s %s: alt chain %.1fx faster -- rejecting it as a "
                                    "compiler fold\n", d.resultTag,
                                    metricTag(v.label).c_str(), altValue / value);
                    else if (altValue > value)
                        value = altValue;
                }
            }
        }

        test.emit(metricTag(v.label), value, note(v.description).c_str());
    }

    return 0;
}

#endif // ENABLE_METAL
