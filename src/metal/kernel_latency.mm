#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Kernel launch latency (Metal)
// ---------------------------------------------------------------------------

int MetalPeak::runKernelLatency(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = forceIters ? specifiedIters
                                    : (cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000);

    auto test = currentDeviceScope->beginTest({"kernel_launch_latency", "Kernel launch latency", "us"});

    id<MTLComputePipelineState> pso = mtlGetPipeline(dev,
        mtl_kernels::kernel_latency_src,
        mtl_kernels::kernel_latency_name,
        "kernel_latency_noop");
    if (!pso)
    {
        test.skip("dispatch", ResultStatus::Error, "Pipeline create failed");
        test.skip("roundtrip", ResultStatus::Error, "Pipeline create failed");
        return -1;
    }

    // Two metrics:
    //   dispatch  = (GPU kernelStartTime) - (host systemUptime captured just
    //               before [cb commit]).  Both live in the mach_absolute_time
    //               / CFTimeInterval domain, so this is the exact one-way
    //               host->driver->GPU dispatch latency, equivalent to
    //               OpenCL's CL_PROFILING_COMMAND_QUEUED -> COMMAND_START.
    //   roundtrip = host time around [cb commit] + waitUntilCompleted, i.e.
    //               full submit -> GPU complete -> signal back.
    NSProcessInfo *pi = [NSProcessInfo processInfo];

    auto enqueueOne = [&](double &commitTimeOut) -> id<MTLCommandBuffer> {
        id<MTLCommandBuffer> cb = [dev.impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];
        commitTimeOut = pi.systemUptime;
        [cb commit];
        return cb;
    };

    bool submitFailed = false;

    // Warmup
    for (unsigned int w = 0; w < warmupCount; w++)
    {
        double t;
        id<MTLCommandBuffer> cb = enqueueOne(t);
        [cb waitUntilCompleted];
        (void)t;
        if (cb.status != MTLCommandBufferStatusCompleted) { submitFailed = true; break; }
    }

    double totalDispatchSec  = 0;
    double totalRoundtripSec = 0;
    if (!submitFailed)
    {
        for (unsigned int i = 0; i < iters; i++)
        {
            double commitTime = 0;
            id<MTLCommandBuffer> cb = enqueueOne(commitTime);
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) { submitFailed = true; break; }
            double doneTime = pi.systemUptime;
            totalDispatchSec  += (cb.kernelStartTime - commitTime);
            totalRoundtripSec += (doneTime - commitTime);
        }
    }
    if (submitFailed)
    {
        test.skip("dispatch",  ResultStatus::Error, "MTLCommandBuffer execution failed");
        test.skip("roundtrip", ResultStatus::Error, "MTLCommandBuffer execution failed");
    }
    else
    {
        float dispatchUs = (float)(totalDispatchSec * 1e6 / iters);
        test.emit("dispatch", dispatchUs);
        float roundtripUs = (float)(totalRoundtripSec * 1e6 / iters);
        test.emit("roundtrip", roundtripUs);
    }

    return 0;
}


#endif // ENABLE_METAL
