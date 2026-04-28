#ifdef ENABLE_METAL

#include <mtl_peak.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

// Workload constant: simdgroup_matrix kernels do 1024 outer iters of 4
// independent 8x8x8 matmul chains per simdgroup.  Per simdgroup ops =
// 1024 * 4 * 8*8*8*2 = 4,194,304; per thread (32 threads/simdgroup) =
// 131,072 ops.  Distinct from COOPMAT_WORK_PER_WI (which assumes 16x16x16)
// because Apple silicon's simdgroup_matrix is fixed at 8x8x8.
static const uint32_t MTL_SIMDGROUP_WORK_PER_WI = 131072;

// ---------------------------------------------------------------------------
// Pimpls
// ---------------------------------------------------------------------------

struct MetalDeviceImpl {
    id<MTLDevice>             device;
    id<MTLCommandQueue>       queue;
    NSMutableDictionary<NSValue*, id<MTLLibrary>>           *libraryCache;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>> *pipelineCache;
};

struct MetalPeakImpl {
    NSArray<id<MTLDevice>> *allDevices;
};

// ---------------------------------------------------------------------------
// MetalDevice
// ---------------------------------------------------------------------------

MetalDevice::MetalDevice() : impl(nullptr) {}
MetalDevice::~MetalDevice() { cleanup(); }

bool MetalDevice::init(int devIndex)
{
    NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
    if (devs.count == 0)
    {
        // On macOS the default-device call is the most reliable way to grab
        // the integrated Apple-silicon GPU.
        id<MTLDevice> def = MTLCreateSystemDefaultDevice();
        if (!def) return false;
        devs = @[def];
    }
    if ((NSUInteger)devIndex >= devs.count) return false;

    impl = new MetalDeviceImpl();
    impl->device = devs[devIndex];
    impl->queue  = [impl->device newCommandQueue];
    impl->libraryCache  = [NSMutableDictionary new];
    impl->pipelineCache = [NSMutableDictionary new];

    info.deviceName = std::string([impl->device.name UTF8String]);
    info.recommendedMaxWorkingSetSize = (uint64_t)impl->device.recommendedMaxWorkingSetSize;
    info.maxBufferLength              = (uint64_t)impl->device.maxBufferLength;
    info.maxThreadsPerThreadgroup     = (uint32_t)impl->device.maxThreadsPerThreadgroup.width;

    NSOperatingSystemVersion v = [[NSProcessInfo processInfo] operatingSystemVersion];
    {
        std::stringstream ss;
        ss << "macOS " << v.majorVersion << "." << v.minorVersion << "." << v.patchVersion;
        info.osVersion = ss.str();
    }

    // Probe Apple silicon family.  Apple7 = M1 baseline; Apple9 = M3
    // (bf16 simdgroup_matrix lights up here); Apple10 = M4.
    info.appleFamily = 0;
    info.isAppleSilicon = false;
    for (int f = 7; f <= 10; f++)
    {
        if ([impl->device supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + f - 1)])
        {
            info.appleFamily = (uint32_t)f;
            info.isAppleSilicon = true;
        }
    }

    // Capability bits.  Apple silicon always has fp16; fp16 simdgroup_matrix
    // is M1+; bf16 simdgroup_matrix is M3+ (Apple9).
    info.fp16Supported                = info.isAppleSilicon;
    info.simdgroupMatrixFP16Supported = info.appleFamily >= 7;
    info.simdgroupMatrixBF16Supported = info.appleFamily >= 9;

    return true;
}

void MetalDevice::cleanup()
{
    if (impl)
    {
        // ARC releases the strong refs (device, queue, dictionaries) when
        // the impl struct's destructor fires; we just delete the C++ wrapper.
        delete impl;
        impl = nullptr;
    }
}

// Get an MTLLibrary for the given source text, compiling on first miss.
static id<MTLLibrary> getLibrary(MetalDevice &dev, const char *src, const char *srcName)
{
    NSValue *key = [NSValue valueWithPointer:src];
    id<MTLLibrary> lib = dev.impl->libraryCache[key];
    if (lib) return lib;

    NSError *err = nil;
    NSString *srcStr = [NSString stringWithUTF8String:src];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    // languageVersion is a property on MTLCompileOptions; pin to 3.0
    // (macOS 13+) so simdgroup_matrix compiles even when the SDK default
    // is older.  bf16 needs 3.1, set conditionally below.
    opts.languageVersion = MTLLanguageVersion3_0;
    if (@available(macOS 14.0, *))
        opts.languageVersion = MTLLanguageVersion3_1;

    lib = [dev.impl->device newLibraryWithSource:srcStr options:opts error:&err];
    if (!lib)
    {
        NSLog(@"Metal compile of %s failed: %@", srcName, err);
        return nil;
    }
    dev.impl->libraryCache[key] = lib;
    return lib;
}

static id<MTLComputePipelineState> getPipeline(MetalDevice &dev, const char *src,
                                               const char *srcName, const char *fnName)
{
    NSString *cacheKey = [NSString stringWithFormat:@"%p#%s", (void*)src, fnName];
    id<MTLComputePipelineState> pso = dev.impl->pipelineCache[cacheKey];
    if (pso) return pso;

    id<MTLLibrary> lib = getLibrary(dev, src, srcName);
    if (!lib) return nil;

    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fnName]];
    if (!fn)
    {
        NSLog(@"Metal: function %s not found in %s", fnName, srcName);
        return nil;
    }

    NSError *err = nil;
    pso = [dev.impl->device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
    {
        NSLog(@"Metal: pipeline create for %s failed: %@", fnName, err);
        return nil;
    }
    dev.impl->pipelineCache[cacheKey] = pso;
    return pso;
}

// ---------------------------------------------------------------------------
// MetalPeak
// ---------------------------------------------------------------------------

MetalPeak::MetalPeak()
  : warmupCount(2), specifiedIters(0), forceIters(false), listDevices(false),
    impl(nullptr) {}

MetalPeak::~MetalPeak() { delete impl; impl = nullptr; }

// Time `iters` dispatches batched into one MTLCommandBuffer.  Returns mean
// per-iter GPU time in microseconds (uses cmdBuf.GPUStartTime/GPUEndTime).
static float runDispatches(MetalDevice &dev, id<MTLComputePipelineState> pso,
                           id<MTLBuffer> outBuf, const void *scalarArg, uint32_t scalarSize,
                           id<MTLBuffer> secondBuf,
                           MTLSize gridSize, MTLSize tgSize,
                           unsigned int warmup, unsigned int iters)
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

    if (warmup > 0)
    {
        id<MTLCommandBuffer> w = enqueue(warmup);
        [w waitUntilCompleted];
    }
    id<MTLCommandBuffer> t = enqueue(iters);
    [t waitUntilCompleted];

    CFTimeInterval gpuTime = t.GPUEndTime - t.GPUStartTime;
    return (float)(gpuTime * 1e6 / (double)iters);
}

int MetalPeak::runAll()
{
    log->print(NEWLINE "=== Metal backend ===" NEWLINE);

    impl = new MetalPeakImpl();
    impl->allDevices = MTLCopyAllDevices();
    if (impl->allDevices.count == 0)
    {
        id<MTLDevice> def = MTLCreateSystemDefaultDevice();
        if (def) impl->allDevices = @[def];
    }
    if (impl->allDevices.count == 0)
    {
        log->print("Metal: no devices found" NEWLINE);
        return -1;
    }

    if (listDevices)
    {
        for (NSUInteger i = 0; i < impl->allDevices.count; i++)
        {
            id<MTLDevice> d = impl->allDevices[i];
            std::stringstream ss;
            ss << "  Metal Device " << i << ": " << [d.name UTF8String] << NEWLINE;
            log->print(ss.str());
        }
        return 0;
    }

    log->xmlOpenTag("clpeak");
    log->xmlAppendAttribs("os", OS_NAME);
    log->xmlOpenTag("platform");
    log->xmlAppendAttribs("name", "Metal");
    log->xmlAppendAttribs("backend", "Metal");

    for (NSUInteger d = 0; d < impl->allDevices.count; d++)
    {
        MetalDevice dev;
        if (!dev.init((int)d))
        {
            log->print(NEWLINE "Metal: failed to init device " + std::to_string(d) + NEWLINE);
            continue;
        }
        if (!dev.info.isAppleSilicon)
        {
            log->print(NEWLINE "Metal: skipping " + dev.info.deviceName +
                       " -- requires Apple silicon (M1 or newer)" NEWLINE);
            continue;
        }

        benchmark_config_t cfg = benchmark_config_t::forDevice(CL_DEVICE_TYPE_GPU);
        if (forceIters)
        {
            cfg.computeIters       = specifiedIters;
            cfg.globalBWIters      = specifiedIters;
            cfg.kernelLatencyIters = specifiedIters;
        }

        log->print(NEWLINE "Metal Device: " + dev.info.deviceName + NEWLINE);
        log->print(TAB "OS            : " + dev.info.osVersion + NEWLINE);
        log->print(TAB "Apple family  : ");
        log->print((unsigned int)dev.info.appleFamily);
        log->print(NEWLINE);
        log->print(TAB "Working set   : ");
        log->print((unsigned int)(dev.info.recommendedMaxWorkingSetSize / (1024 * 1024)));
        log->print(" MB" NEWLINE);

        log->xmlOpenTag("device");
        log->xmlAppendAttribs("name", dev.info.deviceName);
        log->xmlAppendAttribs("api", "metal");
        {
            std::stringstream ss; ss << "Apple" << dev.info.appleFamily;
            log->xmlAppendAttribs("family", ss.str());
        }

        runComputeSP(dev, cfg);
        runComputeHP(dev, cfg);
        runComputeMP(dev, cfg);
        runComputeInt8DP(dev, cfg);
        runComputeInt4Packed(dev, cfg);
        runSimdgroupMatrix(dev, cfg);
        runGlobalBandwidth(dev, cfg);
        runLocalBandwidth(dev, cfg);
        runImageBandwidth(dev, cfg);
        runAtomicThroughput(dev, cfg);
        runKernelLatency(dev, cfg);

        log->print(NEWLINE);
        log->xmlCloseTag(); // device
    }

    log->xmlCloseTag(); // platform
    log->xmlCloseTag(); // clpeak
    return 0;
}

// ---------------------------------------------------------------------------
// Shared compute-peak driver.  Mirrors vkPeak::runComputeKernel /
// CudaPeak::runComputeKernel: one output buffer, dispatch each variant.
// ---------------------------------------------------------------------------

int MetalPeak::runComputeKernel(MetalDevice &dev, benchmark_config_t &cfg,
                                const mtl_compute_desc_t &d)
{
    log->print(NEWLINE TAB);
    log->print(d.title);
    log->print(NEWLINE);
    log->xmlOpenTag(d.xmlTag);
    log->xmlAppendAttribs("unit", d.unit);
    if (d.extraAttribKey && d.extraAttribVal)
        log->xmlAppendAttribs(d.extraAttribKey, d.extraAttribVal);

    if (d.skip)
    {
        log->print(TAB TAB);
        log->print(d.skipMsg ? d.skipMsg : "Skipped");
        log->print(NEWLINE);
        log->xmlCloseTag();
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
    uint64_t globalThreads = 32ULL * 1024 * 1024;
    uint64_t bytesPerGroup = (uint64_t)outPerGroup * d.elemSize;
    uint64_t maxGroups  = dev.info.maxBufferLength / bytesPerGroup;
    uint64_t wantGroups = globalThreads / tgSize;
    uint32_t numGroups  = (uint32_t)((wantGroups < maxGroups) ? wantGroups : maxGroups);
    uint64_t bufferBytes = (uint64_t)numGroups * bytesPerGroup;

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:bufferBytes
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        log->print(TAB TAB "Failed to allocate output buffer" NEWLINE);
        log->xmlCloseTag();
        return -1;
    }

    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    for (const auto &v : variants)
    {
        log->print(TAB TAB);
        log->print(v.label);
        log->print(" : ");

        id<MTLComputePipelineState> pso = getPipeline(dev, v.src, v.srcName, v.kernelName);
        if (!pso)
        {
            log->print("compile/load failed" NEWLINE);
            continue;
        }

        float us = runDispatches(dev, pso, outBuf, d.scalarArg, d.scalarSize, nil,
                                 gridSize, tgSizeM, warmupCount, cfg.computeIters);
        uint64_t totalThreads = (uint64_t)numGroups * tgSize;
        float value = ((float)totalThreads * (float)d.workPerWI) / us / 1e3f;

        log->print(value);
        log->print(NEWLINE);
        log->xmlRecord(v.label, value);
    }

    log->xmlCloseTag();
    return 0;
}

// ---------------------------------------------------------------------------
// Per-benchmark methods
// ---------------------------------------------------------------------------

int MetalPeak::runComputeSP(MetalDevice &dev, benchmark_config_t &cfg)
{
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Single-precision compute (GFLOPS)";
    d.xmlTag      = "single_precision_compute";
    d.unit        = "gflops";
    d.metricLabel = "float";
    d.kernelName  = "compute_sp";
    d.src         = mtl_kernels::compute_sp_src;
    d.srcName     = mtl_kernels::compute_sp_name;
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeHP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "half",  "compute_hp",  mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
        { "half2", "compute_hp2", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Half-precision compute (GFLOPS)";
    d.xmlTag      = "half_precision_compute";
    d.unit        = "gflops";
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeMP(MetalDevice &dev, benchmark_config_t &cfg)
{
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Mixed-precision compute fp16xfp16+fp32 (GFLOPS)";
    d.xmlTag      = "mixed_precision_compute";
    d.unit        = "gflops";
    d.metricLabel = "mp";
    d.kernelName  = "compute_mp";
    d.src         = mtl_kernels::compute_mp_src;
    d.srcName     = mtl_kernels::compute_mp_name;
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeInt8DP(MetalDevice &dev, benchmark_config_t &cfg)
{
    int A = 4;
    mtl_compute_desc_t d = {};
    d.title          = "INT8 dot-product compute (emulated) (GOPS)";
    d.xmlTag         = "integer_compute_int8_dp";
    d.unit           = "gops";
    d.metricLabel    = "int8_dp";
    d.kernelName     = "compute_int8_dp";
    d.src            = mtl_kernels::compute_int8_dp_src;
    d.srcName        = mtl_kernels::compute_int8_dp_name;
    d.workPerWI      = COMPUTE_INT8_DP_WORK_PER_WI;
    d.elemSize       = sizeof(int);
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.extraAttribKey = "emulated";
    d.extraAttribVal = "true";
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeInt4Packed(MetalDevice &dev, benchmark_config_t &cfg)
{
    int A = 3;
    mtl_compute_desc_t d = {};
    d.title          = "Packed INT4 compute (emulated) (GOPS)";
    d.xmlTag         = "int4_packed_compute";
    d.unit           = "gops";
    d.metricLabel    = "int4_packed";
    d.kernelName     = "compute_int4_packed";
    d.src            = mtl_kernels::compute_int4_packed_src;
    d.srcName        = mtl_kernels::compute_int4_packed_name;
    d.workPerWI      = COMPUTE_INT4_PACKED_WORK_PER_WI;
    d.elemSize       = sizeof(int);
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.extraAttribKey = "emulated";
    d.extraAttribVal = "true";
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runSimdgroupMatrix(MetalDevice &dev, benchmark_config_t &cfg)
{
    if (!dev.info.simdgroupMatrixFP16Supported)
    {
        log->print(NEWLINE TAB "simdgroup_matrix tensor compute (GFLOPS)" NEWLINE);
        log->xmlOpenTag("simdgroup_matrix");
        log->print(TAB TAB "simdgroup_matrix requires Apple7 (M1) or newer! Skipped" NEWLINE);
        log->xmlCloseTag();
        return 0;
    }

    {
        float A = 1.3f;
        mtl_compute_desc_t d = {};
        d.title            = "simdgroup_matrix fp16xfp16+fp32 8x8x8 (GFLOPS)";
        d.xmlTag           = "simdgroup_matrix_fp16";
        d.unit             = "gflops";
        d.metricLabel      = "simdgroup_fp16";
        d.kernelName       = "simdgroup_matrix_fp16";
        d.src              = mtl_kernels::simdgroup_matrix_fp16_src;
        d.srcName          = mtl_kernels::simdgroup_matrix_fp16_name;
        d.workPerWI        = MTL_SIMDGROUP_WORK_PER_WI;
        d.elemSize         = sizeof(float);
        d.threadsPerGroup  = 32;          // one simdgroup
        d.outElemsPerGroup = 64;          // 8x8 tile
        d.scalarArg        = &A;
        d.scalarSize       = sizeof(A);
        d.extraAttribKey   = "tile";
        d.extraAttribVal   = "8x8x8";
        runComputeKernel(dev, cfg, d);
    }
    {
        float A = 1.3f;
        mtl_compute_desc_t d = {};
        d.title            = "simdgroup_matrix bf16xbf16+fp32 8x8x8 (GFLOPS)";
        d.xmlTag           = "simdgroup_matrix_bf16";
        d.unit             = "gflops";
        d.metricLabel      = "simdgroup_bf16";
        d.kernelName       = "simdgroup_matrix_bf16";
        d.src              = mtl_kernels::simdgroup_matrix_bf16_src;
        d.srcName          = mtl_kernels::simdgroup_matrix_bf16_name;
        d.workPerWI        = MTL_SIMDGROUP_WORK_PER_WI;
        d.elemSize         = sizeof(float);
        d.threadsPerGroup  = 32;
        d.outElemsPerGroup = 64;
        d.scalarArg        = &A;
        d.scalarSize       = sizeof(A);
        d.skip             = !dev.info.simdgroupMatrixBF16Supported;
        d.skipMsg          = "bf16 simdgroup_matrix requires Apple9 (M3) or newer! Skipped";
        d.extraAttribKey   = "tile";
        d.extraAttribVal   = "8x8x8";
        runComputeKernel(dev, cfg, d);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Global bandwidth (Metal)
// ---------------------------------------------------------------------------

int MetalPeak::runGlobalBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.globalBWIters;
    const uint32_t tgSize = 256;

    uint64_t maxItems = dev.info.maxBufferLength / sizeof(float);
    uint64_t numItems = (maxItems / (tgSize * FETCH_PER_WI)) * (tgSize * FETCH_PER_WI);
    if (numItems > cfg.globalBWMaxSize / sizeof(float))
        numItems = (cfg.globalBWMaxSize / sizeof(float) / (tgSize * FETCH_PER_WI)) * (tgSize * FETCH_PER_WI);

    uint32_t numGroups = (uint32_t)(numItems / FETCH_PER_WI / tgSize);
    if (numGroups == 0) numGroups = 1;

    log->print(NEWLINE TAB "Global memory bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("global_memory_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    id<MTLBuffer> inBuf  = [dev.impl->device newBufferWithLength:numItems * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:numItems * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!inBuf || !outBuf)
    {
        log->print(TAB TAB "Failed to allocate buffers" NEWLINE);
        log->xmlCloseTag();
        return -1;
    }
    // Touch the input so we don't measure zero-page reads on copy-on-write.
    memset(inBuf.contents, 0x3f, numItems * sizeof(float));

    id<MTLComputePipelineState> pso = getPipeline(dev,
        mtl_kernels::global_bandwidth_src,
        mtl_kernels::global_bandwidth_name,
        "global_bandwidth");
    if (!pso)
    {
        log->print(TAB TAB "Pipeline create failed" NEWLINE);
        log->xmlCloseTag();
        return -1;
    }

    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    log->print(TAB TAB "float   : ");
    float us = runDispatches(dev, pso, inBuf, nullptr, 0, outBuf,
                             gridSize, tgSizeM, warmupCount, iters);
    float gbps = ((float)numItems * sizeof(float)) / us / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("float", gbps);

    log->xmlCloseTag();
    return 0;
}

// ---------------------------------------------------------------------------
// Kernel launch latency (Metal)
// ---------------------------------------------------------------------------

int MetalPeak::runKernelLatency(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000;

    log->print(NEWLINE TAB "Kernel launch latency (us)" NEWLINE);
    log->xmlOpenTag("kernel_launch_latency");
    log->xmlAppendAttribs("unit", "us");

    id<MTLComputePipelineState> pso = getPipeline(dev,
        mtl_kernels::kernel_latency_src,
        mtl_kernels::kernel_latency_name,
        "kernel_latency_noop");
    if (!pso)
    {
        log->print(TAB TAB "Pipeline create failed" NEWLINE);
        log->xmlCloseTag();
        return -1;
    }

    // Submit each iter as its own MTLCommandBuffer so what we time is the
    // per-cmdBuf submit + complete latency, not encoder reuse.  Sum the
    // GPU-time deltas reported by each cmdBuf.
    auto enqueueOne = [&]() -> id<MTLCommandBuffer> {
        id<MTLCommandBuffer> cb = [dev.impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
        [enc endEncoding];
        [cb commit];
        return cb;
    };

    // Warmup
    for (unsigned int w = 0; w < warmupCount; w++)
    {
        id<MTLCommandBuffer> cb = enqueueOne();
        [cb waitUntilCompleted];
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    id<MTLCommandBuffer> last = nil;
    for (unsigned int i = 0; i < iters; i++)
    {
        id<MTLCommandBuffer> cb = enqueueOne();
        [cb waitUntilCompleted];
        last = cb;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)last;

    double total_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    float us = (float)(total_us / iters);
    log->print(TAB TAB "noop : ");
    log->print(us);
    log->print(NEWLINE);
    log->xmlRecord("noop", us);

    log->xmlCloseTag();
    return 0;
}

// ---------------------------------------------------------------------------
// Local memory bandwidth (Metal -- threadgroup memory)
// ---------------------------------------------------------------------------

int MetalPeak::runLocalBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.computeIters;
    log->print(NEWLINE TAB "Local memory bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("local_memory_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    const uint32_t tgSize = 256;
    uint64_t globalThreads = 32ULL * 1024 * 1024;
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:globalThreads * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        log->print(TAB TAB "Buffer alloc failed" NEWLINE);
        log->xmlCloseTag();
        return -1;
    }

    struct V { const char *label; const char *kname; uint32_t width; };
    const V vs[] = {
        {"float  ", "local_bandwidth_v1", 1},
        {"float2 ", "local_bandwidth_v2", 2},
        {"float4 ", "local_bandwidth_v4", 4},
        {"float8 ", "local_bandwidth_v8", 8},
    };
    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);
    for (const auto &v : vs)
    {
        log->print(TAB TAB);
        log->print(v.label);
        log->print(": ");
        id<MTLComputePipelineState> pso = getPipeline(dev,
            mtl_kernels::local_bandwidth_src,
            mtl_kernels::local_bandwidth_name, v.kname);
        if (!pso) { log->print("compile/load failed" NEWLINE); continue; }
        float us = runDispatches(dev, pso, outBuf, nullptr, 0, nil,
                                 gridSize, tgSizeM, warmupCount, iters);
        uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.width * sizeof(float) * globalThreads;
        float gbps = (float)bytes / us / 1e3f;
        log->print(gbps); log->print(NEWLINE);
        std::string key(v.label);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        log->xmlRecord(key, gbps);
    }

    log->xmlCloseTag();
    return 0;
}

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (Metal -- MTLTexture + sampler)
// ---------------------------------------------------------------------------

int MetalPeak::runImageBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.globalBWIters;
    log->print(NEWLINE TAB "Image memory bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("image_memory_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    const NSUInteger imgW = 4096, imgH = 4096;
    const uint32_t tgSize = 256;
    uint64_t globalThreads = 32ULL * 1024 * 1024;
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);

    MTLTextureDescriptor *td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:imgW height:imgH mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev.impl->device newTextureWithDescriptor:td];
    if (!tex)
    {
        log->print(TAB TAB "Texture create failed" NEWLINE);
        log->xmlCloseTag();
        return -1;
    }
    // Contents undefined is fine for a bandwidth measurement.

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:globalThreads * sizeof(float)
                                                         options:MTLResourceStorageModeShared];

    id<MTLComputePipelineState> pso = getPipeline(dev,
        mtl_kernels::image_bandwidth_src,
        mtl_kernels::image_bandwidth_name, "image_bandwidth");
    if (!pso)
    {
        log->print(TAB TAB "Pipeline create failed" NEWLINE);
        log->xmlCloseTag();
        return -1;
    }

    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    log->print(TAB TAB "float4 : ");
    // Custom dispatch -- runDispatches doesn't know about textures.
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
    if (warmupCount > 0)
    {
        id<MTLCommandBuffer> w = enqueue(warmupCount);
        [w waitUntilCompleted];
    }
    id<MTLCommandBuffer> t = enqueue(iters);
    [t waitUntilCompleted];
    float us = (float)((t.GPUEndTime - t.GPUStartTime) * 1e6 / iters);
    uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;
    float gbps = (float)bytes / us / 1e3f;
    log->print(gbps); log->print(NEWLINE);
    log->xmlRecord("float4", gbps);

    log->xmlCloseTag();
    return 0;
}

// ---------------------------------------------------------------------------
// Atomic throughput (Metal -- global + local atomics)
// ---------------------------------------------------------------------------

int MetalPeak::runAtomicThroughput(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.computeIters;
    log->print(NEWLINE TAB "Atomic throughput (GOPS)" NEWLINE);
    log->xmlOpenTag("atomic_throughput");
    log->xmlAppendAttribs("unit", "gops");

    const uint32_t tgSize = 256;
    uint64_t globalThreads = 32ULL * 1024 * 1024;
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);
    MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);
    MTLSize tgSizeM  = MTLSizeMake(tgSize, 1, 1);

    auto runOne = [&](const char *fnName, NSUInteger bufBytes) -> float
    {
        id<MTLBuffer> buf = [dev.impl->device newBufferWithLength:bufBytes
                                                          options:MTLResourceStorageModeShared];
        if (!buf) return -1.0f;
        memset(buf.contents, 0, bufBytes);
        id<MTLComputePipelineState> pso = getPipeline(dev,
            mtl_kernels::atomic_throughput_src,
            mtl_kernels::atomic_throughput_name, fnName);
        if (!pso) return -1.0f;
        float us = runDispatches(dev, pso, buf, nullptr, 0, nil,
                                 gridSize, tgSizeM, warmupCount, iters);
        return us;
    };

    log->print(TAB TAB "global : ");
    float us_g = runOne("atomic_throughput_global", globalThreads * sizeof(int));
    if (us_g > 0)
    {
        float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us_g / 1e3f;
        log->print(gops); log->print(NEWLINE);
        log->xmlRecord("global", gops);
    }
    else
    {
        log->print("failed" NEWLINE);
    }

    log->print(TAB TAB "local  : ");
    float us_l = runOne("atomic_throughput_local", numGroups * sizeof(int));
    if (us_l > 0)
    {
        float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us_l / 1e3f;
        log->print(gops); log->print(NEWLINE);
        log->xmlRecord("local", gops);
    }
    else
    {
        log->print("failed" NEWLINE);
    }

    log->xmlCloseTag();
    return 0;
}

#endif // ENABLE_METAL
