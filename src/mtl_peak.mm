#ifdef ENABLE_METAL

#include <mtl_peak.h>
#include <inventory.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
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
    info.simdgroupMatrixInt8Supported = info.appleFamily >= 9;
    // 64-bit integer atomics (atomic_ulong / atomic_long fetch_add) light up
    // on Apple8 (M2) and newer.  Apple7 lacks the hardware path entirely; the
    // MSL compile would succeed but the dispatch produces wrong results.
    info.atomic64Supported            = info.appleFamily >= 8;

    // MPSGraph bf16 dtype was added in macOS 14 (Sonoma) and only lights up
    // on Apple9+ (M3); below that it falls back to a slow software path.
    info.mpsGraphBF16Supported = false;
    if (@available(macOS 14.0, *))
        info.mpsGraphBF16Supported = info.appleFamily >= 9;

    // Best-effort GPU core count via IORegistry (Apple silicon exposes
    // "gpu-core-count" on the AGXAccelerator service).  Used to scale the
    // GEMM dim so similar-class GPUs land in similar wall-clock windows.
    info.gpuCoreCount = 0;
#if __has_include(<IOKit/IOKitLib.h>)
    {
        io_iterator_t it = 0;
        if (IOServiceGetMatchingServices(kIOMainPortDefault,
                                         IOServiceMatching("AGXAccelerator"),
                                         &it) == KERN_SUCCESS)
        {
            io_object_t obj;
            while ((obj = IOIteratorNext(it)) != 0)
            {
                CFTypeRef p = IORegistryEntryCreateCFProperty(obj,
                    CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
                if (p)
                {
                    if (CFGetTypeID(p) == CFNumberGetTypeID())
                    {
                        int v = 0;
                        CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &v);
                        if (v > 0) info.gpuCoreCount = (uint32_t)v;
                    }
                    CFRelease(p);
                }
                IOObjectRelease(obj);
                if (info.gpuCoreCount) break;
            }
            IOObjectRelease(it);
        }
    }
#endif

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
  : warmupCount(2), specifiedIters(0), forceIters(false),
    deviceIndex(-1),
    impl(nullptr)
{
  enabledTests.set();
  enabledCategories.set();
}

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

    log->resultScopeBegin("clpeak");
    log->resultScopeAttribute("os", OS_NAME);
    log->resultScopeBegin("platform");
    log->resultScopeAttribute("name", "Metal");
    log->resultScopeAttribute("backend", "Metal");

    for (NSUInteger d = 0; d < impl->allDevices.count; d++)
    {
        if (deviceIndex >= 0 && static_cast<NSUInteger>(deviceIndex) != d)
            continue;

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
        if (dev.info.gpuCoreCount)
        {
            log->print(TAB "GPU cores     : ");
            log->print((unsigned int)dev.info.gpuCoreCount);
            log->print(NEWLINE);
        }

        log->resultScopeBegin("device");
        log->resultScopeAttribute("name", dev.info.deviceName);
        log->resultScopeAttribute("api", "metal");
        {
            std::stringstream ss; ss << "Apple" << dev.info.appleFamily;
            log->resultScopeAttribute("family", ss.str());
        }

        // ---- Phase 1: floating-point compute (GFLOPS / TFLOPS) -----------
        if (isAllowed(Benchmark::ComputeSP))         runComputeSP(dev, cfg);
        if (isAllowed(Benchmark::ComputeHP))         runComputeHP(dev, cfg);
        if (isAllowed(Benchmark::ComputeMP))         runComputeMP(dev, cfg);
        if (isAllowedAs(Benchmark::SimdgroupMatrix, Category::FpCompute))
            runSimdgroupMatrix(dev, cfg);
        if (isAllowedAs(Benchmark::MpsGemm, Category::FpCompute))
            runMpsGemm(dev, cfg);
        if (isAllowedAs(Benchmark::AtomicThroughput, Category::FpCompute))
            runAtomicThroughputFp(dev, cfg);

        // ---- Phase 2: integer compute (GOPS / TOPS) ----------------------
        if (isAllowed(Benchmark::ComputeInt8DP))     runComputeInt8DP(dev, cfg);
        if (isAllowed(Benchmark::ComputeInt4Packed)) runComputeInt4Packed(dev, cfg);
        if (isAllowedAs(Benchmark::SimdgroupMatrix, Category::IntCompute))
            runSimdgroupMatrixInt(dev, cfg);
        if (isAllowedAs(Benchmark::MpsGemm, Category::IntCompute))
            runMpsGemmInt(dev, cfg);
        if (isAllowedAs(Benchmark::AtomicThroughput, Category::IntCompute))
            runAtomicThroughput(dev, cfg);

        // ---- Phase 3: bandwidth (GBPS) -----------------------------------
        if (isAllowed(Benchmark::GlobalBW))          runGlobalBandwidth(dev, cfg);
        if (isAllowed(Benchmark::LocalBW))           runLocalBandwidth(dev, cfg);
        if (isAllowed(Benchmark::ImageBW))           runImageBandwidth(dev, cfg);

        // ---- Phase 4: latency (us) ---------------------------------------
        if (isAllowed(Benchmark::KernelLatency))     runKernelLatency(dev, cfg);

        log->print(NEWLINE);
        log->resultScopeEnd(); // device
    }

    log->resultScopeEnd(); // platform
    log->resultScopeEnd(); // clpeak
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
    log->resultScopeBegin(d.resultTag);
    log->resultScopeAttribute("unit", d.unit);
    if (d.extraAttribKey && d.extraAttribVal)
        log->resultScopeAttribute(d.extraAttribKey, d.extraAttribVal);

    if (d.skip)
    {
        log->print(TAB TAB);
        log->print(d.skipMsg ? d.skipMsg : "Skipped");
        log->print(NEWLINE);
        log->resultScopeEnd();
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
        log->resultScopeEnd();
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
        double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
        float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us / divider);

        log->print(value);
        log->print(NEWLINE);

        // Strip the right-padding used for stdout column alignment so the
        // metric tag stored in the dump (and shown in the Android UI) is
        // a clean canonical label.
        std::string metricTag(v.label);
        while (!metricTag.empty() && metricTag.back() == ' ')
            metricTag.pop_back();
        log->resultRecord(metricTag, value);
    }

    log->resultScopeEnd();
    return 0;
}

// ---------------------------------------------------------------------------
// Per-benchmark methods
// ---------------------------------------------------------------------------

int MetalPeak::runComputeSP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "float ", "compute_sp",  mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
        { "float2", "compute_sp2", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
        { "float4", "compute_sp4", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
        { "float8", "compute_sp8", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Single-precision compute (GFLOPS)";
    d.resultTag      = "single_precision_compute";
    d.unit        = "gflops";
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeHP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "half ", "compute_hp",  mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
        { "half2", "compute_hp2", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
        { "half4", "compute_hp4", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
        { "half8", "compute_hp8", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Half-precision compute (GFLOPS)";
    d.resultTag      = "half_precision_compute";
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
    static const mtl_compute_variant_t variants[] = {
        { "mp ", "compute_mp",  mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name },
        { "mp2", "compute_mp2", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name },
        { "mp4", "compute_mp4", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Mixed-precision compute fp16xfp16+fp32 (GFLOPS)";
    d.resultTag      = "mixed_precision_compute";
    d.unit        = "gflops";
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeInt8DP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "int8_dp ", "compute_int8_dp",  mtl_kernels::compute_int8_dp_src, mtl_kernels::compute_int8_dp_name },
        { "int8_dp2", "compute_int8_dp2", mtl_kernels::compute_int8_dp_src, mtl_kernels::compute_int8_dp_name },
        { "int8_dp4", "compute_int8_dp4", mtl_kernels::compute_int8_dp_src, mtl_kernels::compute_int8_dp_name },
    };
    int A = 4;
    mtl_compute_desc_t d = {};
    d.title          = "INT8 dot-product compute (emulated) (GOPS)";
    d.resultTag         = "integer_compute_int8_dp";
    d.unit           = "gops";
    d.variants       = variants;
    d.numVariants    = sizeof(variants) / sizeof(variants[0]);
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
    static const mtl_compute_variant_t variants[] = {
        { "int4_packed ", "compute_int4_packed",  mtl_kernels::compute_int4_packed_src, mtl_kernels::compute_int4_packed_name },
        { "int4_packed2", "compute_int4_packed2", mtl_kernels::compute_int4_packed_src, mtl_kernels::compute_int4_packed_name },
        { "int4_packed4", "compute_int4_packed4", mtl_kernels::compute_int4_packed_src, mtl_kernels::compute_int4_packed_name },
    };
    int A = 3;
    mtl_compute_desc_t d = {};
    d.title          = "Packed INT4 compute (emulated) (GOPS)";
    d.resultTag         = "int4_packed_compute";
    d.unit           = "gops";
    d.variants       = variants;
    d.numVariants    = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI      = COMPUTE_INT4_PACKED_WORK_PER_WI;
    d.elemSize       = sizeof(int);
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.extraAttribKey = "emulated";
    d.extraAttribVal = "true";
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runSimdgroupMatrixInt(MetalDevice &dev, benchmark_config_t &cfg)
{
    int A = 1;
    mtl_compute_desc_t d = {};
    d.title            = "simdgroup_matrix int8xint8+int32 8x8x8 (TOPS)";
    d.resultTag           = "simdgroup_matrix_int8";
    d.unit             = "tops";
    d.unitDivider      = 1e12;
    d.metricLabel      = "simdgroup_int8";
    d.kernelName       = "simdgroup_matrix_int8";
    d.src              = mtl_kernels::simdgroup_matrix_int8_src;
    d.srcName          = mtl_kernels::simdgroup_matrix_int8_name;
    d.workPerWI        = MTL_SIMDGROUP_WORK_PER_WI;
    d.elemSize         = sizeof(int);
    d.threadsPerGroup  = 32;
    d.outElemsPerGroup = 64;
    d.scalarArg        = &A;
    d.scalarSize       = sizeof(A);
    d.skip             = !dev.info.simdgroupMatrixInt8Supported;
    d.skipMsg          = "int8 simdgroup_matrix requires Apple9 (M3) or newer! Skipped";
    d.extraAttribKey   = "tile";
    d.extraAttribVal   = "8x8x8";
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runSimdgroupMatrix(MetalDevice &dev, benchmark_config_t &cfg)
{
    if (!dev.info.simdgroupMatrixFP16Supported)
    {
        log->print(NEWLINE TAB "simdgroup_matrix tensor compute (TFLOPS)" NEWLINE);
        log->resultScopeBegin("simdgroup_matrix");
        log->print(TAB TAB "simdgroup_matrix requires Apple7 (M1) or newer! Skipped" NEWLINE);
        log->resultScopeEnd();
        return 0;
    }

    {
        float A = 1.3f;
        mtl_compute_desc_t d = {};
        d.title            = "simdgroup_matrix fp16xfp16+fp32 8x8x8 (TFLOPS)";
        d.resultTag           = "simdgroup_matrix_fp16";
        d.unit             = "tflops";
        d.unitDivider      = 1e12;
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
        d.title            = "simdgroup_matrix bf16xbf16+fp32 8x8x8 (TFLOPS)";
        d.resultTag           = "simdgroup_matrix_bf16";
        d.unit             = "tflops";
        d.unitDivider      = 1e12;
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

    log->print(NEWLINE TAB "Global memory bandwidth (GBPS)" NEWLINE);
    log->resultScopeBegin("global_memory_bandwidth");
    log->resultScopeAttribute("unit", "gbps");

    // Reserve enough scalar floats so the widest variant (v16 = 16 floats per
    // logical "WI" element) still aligns to (tg * FETCH_PER_WI * 16).
    const uint32_t maxVecScale = 16;
    uint64_t maxItems = dev.info.maxBufferLength / sizeof(float);
    uint64_t align    = (uint64_t)tgSize * FETCH_PER_WI * maxVecScale;
    uint64_t numItems = (maxItems / align) * align;
    if (numItems > cfg.globalBWMaxSize / sizeof(float))
        numItems = (cfg.globalBWMaxSize / sizeof(float) / align) * align;

    id<MTLBuffer> inBuf  = [dev.impl->device newBufferWithLength:numItems * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:numItems * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!inBuf || !outBuf)
    {
        log->print(TAB TAB "Failed to allocate buffers" NEWLINE);
        log->resultScopeEnd();
        return -1;
    }
    // Touch the input so we don't measure zero-page reads on copy-on-write.
    memset(inBuf.contents, 0x3f, numItems * sizeof(float));

    struct V { const char *label; const char *kname; uint32_t width; };
    const V vs[] = {
        { "float  ", "global_bandwidth",     1  },
        { "float2 ", "global_bandwidth_v2",  2  },
        { "float4 ", "global_bandwidth_v4",  4  },
        { "float8 ", "global_bandwidth_v8",  8  },
        { "float16", "global_bandwidth_v16", 16 },
    };

    MTLSize tgSizeM = MTLSizeMake(tgSize, 1, 1);

    for (const auto &v : vs)
    {
        log->print(TAB TAB);
        log->print(v.label);
        log->print(": ");

        id<MTLComputePipelineState> pso = getPipeline(dev,
            mtl_kernels::global_bandwidth_src,
            mtl_kernels::global_bandwidth_name,
            v.kname);
        if (!pso) { log->print("compile/load failed" NEWLINE); continue; }

        // Each threadgroup reads FETCH_PER_WI * tgSize logical-vec elements,
        // i.e. FETCH_PER_WI * tgSize * width scalar floats.
        uint64_t scalarsPerGroup = (uint64_t)tgSize * FETCH_PER_WI * v.width;
        uint32_t numGroups = (uint32_t)(numItems / scalarsPerGroup);
        if (numGroups == 0) numGroups = 1;
        MTLSize gridSize = MTLSizeMake(numGroups, 1, 1);

        float us = runDispatches(dev, pso, inBuf, nullptr, 0, outBuf,
                                 gridSize, tgSizeM, warmupCount, iters);
        uint64_t bytesRead = (uint64_t)numGroups * scalarsPerGroup * sizeof(float);
        float gbps = (float)bytesRead / us / 1e3f;
        log->print(gbps);
        log->print(NEWLINE);
        std::string key(v.label);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        log->resultRecord(key, gbps);
    }

    log->resultScopeEnd();
    return 0;
}

// ---------------------------------------------------------------------------
// Kernel launch latency (Metal)
// ---------------------------------------------------------------------------

int MetalPeak::runKernelLatency(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000;

    log->print(NEWLINE TAB "Kernel launch latency (us)" NEWLINE);
    log->resultScopeBegin("kernel_launch_latency");
    log->resultScopeAttribute("unit", "us");

    id<MTLComputePipelineState> pso = getPipeline(dev,
        mtl_kernels::kernel_latency_src,
        mtl_kernels::kernel_latency_name,
        "kernel_latency_noop");
    if (!pso)
    {
        log->print(TAB TAB "Pipeline create failed" NEWLINE);
        log->resultScopeEnd();
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

    // Warmup
    for (unsigned int w = 0; w < warmupCount; w++)
    {
        double t;
        id<MTLCommandBuffer> cb = enqueueOne(t);
        [cb waitUntilCompleted];
        (void)t;
    }

    double totalDispatchSec  = 0;
    double totalRoundtripSec = 0;
    for (unsigned int i = 0; i < iters; i++)
    {
        double commitTime = 0;
        id<MTLCommandBuffer> cb = enqueueOne(commitTime);
        [cb waitUntilCompleted];
        double doneTime = pi.systemUptime;
        totalDispatchSec  += (cb.kernelStartTime - commitTime);
        totalRoundtripSec += (doneTime - commitTime);
    }
    float dispatchUs  = (float)(totalDispatchSec  * 1e6 / iters);
    float roundtripUs = (float)(totalRoundtripSec * 1e6 / iters);
    log->print(TAB TAB TAB "dispatch  : ");
    log->print(dispatchUs);
    log->print(NEWLINE TAB TAB TAB "roundtrip : ");
    log->print(roundtripUs);
    log->print(NEWLINE);
    log->resultRecord("dispatch",  dispatchUs);
    log->resultRecord("roundtrip", roundtripUs);

    log->resultScopeEnd();
    return 0;
}

// ---------------------------------------------------------------------------
// Local memory bandwidth (Metal -- threadgroup memory)
// ---------------------------------------------------------------------------

int MetalPeak::runLocalBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.computeIters;
    log->print(NEWLINE TAB "Local memory bandwidth (GBPS)" NEWLINE);
    log->resultScopeBegin("local_memory_bandwidth");
    log->resultScopeAttribute("unit", "gbps");

    const uint32_t tgSize = 256;
    uint64_t globalThreads = 32ULL * 1024 * 1024;
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:globalThreads * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        log->print(TAB TAB "Buffer alloc failed" NEWLINE);
        log->resultScopeEnd();
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
        log->resultRecord(key, gbps);
    }

    log->resultScopeEnd();
    return 0;
}

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (Metal -- MTLTexture + sampler)
// ---------------------------------------------------------------------------

int MetalPeak::runImageBandwidth(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.globalBWIters;
    log->print(NEWLINE TAB "Image memory bandwidth (GBPS)" NEWLINE);
    log->resultScopeBegin("image_memory_bandwidth");
    log->resultScopeAttribute("unit", "gbps");

    const NSUInteger imgW = 4096, imgH = 4096;
    const uint32_t tgSize = 256;
    uint64_t globalThreads = 32ULL * 1024 * 1024;
    uint32_t numGroups = (uint32_t)(globalThreads / tgSize);

    id<MTLBuffer> outBuf = [dev.impl->device newBufferWithLength:globalThreads * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    if (!outBuf)
    {
        log->print(TAB TAB "Output buffer alloc failed" NEWLINE);
        log->resultScopeEnd();
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
        { "rgba8  ", "image_bandwidth",       MTLPixelFormatRGBA8Unorm,  4  },
        { "r32f   ", "image_bandwidth_r32f",  MTLPixelFormatR32Float,    4  },
    };

    for (const auto &v : vs)
    {
        log->print(TAB TAB);
        log->print(v.label);
        log->print(": ");

        MTLTextureDescriptor *td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:v.fmt
                                         width:imgW height:imgH mipmapped:NO];
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [dev.impl->device newTextureWithDescriptor:td];
        if (!tex)
        {
            log->print("texture alloc failed" NEWLINE);
            continue;
        }

        id<MTLComputePipelineState> pso = getPipeline(dev,
            mtl_kernels::image_bandwidth_src,
            mtl_kernels::image_bandwidth_name, v.kname);
        if (!pso) { log->print("compile/load failed" NEWLINE); continue; }

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
        uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * v.bytesPerPixel * globalThreads;
        float gbps = (float)bytes / us / 1e3f;
        log->print(gbps); log->print(NEWLINE);
        std::string key(v.label);
        while (!key.empty() && key.back() == ' ') key.pop_back();
        log->resultRecord(key, gbps);
    }

    log->resultScopeEnd();
    return 0;
}

// ---------------------------------------------------------------------------
// Atomic throughput (Metal -- global + local atomics)
// ---------------------------------------------------------------------------

int MetalPeak::runAtomicThroughput(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.computeIters;
    log->print(NEWLINE TAB "Atomic throughput (GOPS)" NEWLINE);
    log->resultScopeBegin("atomic_throughput");
    log->resultScopeAttribute("unit", "gops");

    const uint32_t tgSize = 256;
    uint64_t globalThreads = 32ULL * 1024 * 1024;
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
        id<MTLComputePipelineState> pso = getPipeline(dev, src, srcName, fnName);
        if (!pso) return -1.0f;
        float us = runDispatches(dev, pso, buf, nullptr, 0, nil,
                                 gridSize, tgSizeM, warmupCount, iters);
        return us;
    };

    auto reportOne = [&](const char *resultKey, const char *src, const char *srcName,
                         const char *fnName, NSUInteger bufBytes,
                         bool skip, const char *skipMsg)
    {
        if (skip)
        {
            log->print(skipMsg); log->print(NEWLINE);
            log->resultRecord(resultKey, 0.0f);
            return;
        }
        float us = runOne(src, srcName, fnName, bufBytes);
        if (us > 0)
        {
            float gops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
            log->print(gops); log->print(NEWLINE);
            log->resultRecord(resultKey, gops);
        }
        else
        {
            log->print("compile/load failed" NEWLINE);
            log->resultRecord(resultKey, 0.0f);
        }
    };

    auto report = [&](const char *label, const char *src, const char *srcName,
                      const char *gFn, const char *lFn, uint32_t elemSize,
                      bool skip, const char *skipMsg)
    {
        // Trim trailing/internal padding so the persisted metric key is valid.
        std::string trimmed(label);
        while (!trimmed.empty() && trimmed.back() == ' ') trimmed.pop_back();
        log->print(TAB TAB); log->print(trimmed.c_str()); log->print("_global : ");
        reportOne((trimmed + "_global").c_str(),
                  src, srcName, gFn, globalThreads * elemSize, skip, skipMsg);
        if (lFn)
        {
            log->print(TAB TAB); log->print(trimmed.c_str()); log->print("_local  : ");
            reportOne((trimmed + "_local").c_str(),
                      src, srcName, lFn, numGroups * elemSize, skip, skipMsg);
        }
    };

    report("int  ",  mtl_kernels::atomic_throughput_src,
           mtl_kernels::atomic_throughput_name,
           "atomic_throughput_global",      "atomic_throughput_local",
           sizeof(int), false, nullptr);
    report("uint ",  mtl_kernels::atomic_throughput_src,
           mtl_kernels::atomic_throughput_name,
           "atomic_throughput_global_uint", "atomic_throughput_local_uint",
           sizeof(uint32_t), false, nullptr);
    // atomic_ulong: 64-bit atomic add isn't accepted by every MSL/SDK combo.
    // Skipped on Apple7; on Apple8+ we still let getPipeline report
    // compile/load failed if the SDK rejects it.
    report("ulong",  mtl_kernels::atomic_throughput_ulong_src,
           mtl_kernels::atomic_throughput_ulong_name,
           "atomic_throughput_global_ulong", "atomic_throughput_local_ulong",
           sizeof(uint64_t),
           !dev.info.atomic64Supported,
           "skipped (requires Apple8 / M2 or newer)");

    log->resultScopeEnd();
    return 0;
}

int MetalPeak::runAtomicThroughputFp(MetalDevice &dev, benchmark_config_t &cfg)
{
    unsigned int iters = cfg.computeIters;
    log->print(NEWLINE TAB "Atomic throughput (GFLOPS)" NEWLINE);
    log->resultScopeBegin("atomic_throughput");
    log->resultScopeAttribute("unit", "gflops");

    const uint32_t tgSize = 256;
    uint64_t globalThreads = 32ULL * 1024 * 1024;
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
        id<MTLComputePipelineState> pso = getPipeline(dev, src, srcName, fnName);
        if (!pso) return -1.0f;
        return runDispatches(dev, pso, buf, nullptr, 0, nil,
                             gridSize, tgSizeM, warmupCount, iters);
    };

    auto reportOne = [&](const char *resultKey, const char *src, const char *srcName,
                         const char *fnName, NSUInteger bufBytes)
    {
        log->print(TAB TAB); log->print(resultKey); log->print(" : ");
        float us = runOne(src, srcName, fnName, bufBytes);
        if (us > 0)
        {
            float gflops = ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f;
            log->print(gflops); log->print(NEWLINE);
            log->resultRecord(resultKey, gflops);
        }
        else
        {
            log->print("compile/load failed" NEWLINE);
            log->resultRecord(resultKey, 0.0f);
        }
    };

    reportOne("float_global",
              mtl_kernels::atomic_throughput_float_src,
              mtl_kernels::atomic_throughput_float_name,
              "atomic_throughput_global_float",
              globalThreads * sizeof(float));
    log->resultScopeEnd();
    return 0;
}

// Free-function enumeration used by --list-devices and the Android JNI surface.
// Mirrors the device-discovery logic at the top of MetalPeak::runAll() but
// stops at name extraction.
BackendInventory enumerateMetal()
{
    BackendInventory inv;
    inv.backend = "Metal";

    NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
    if (devs.count == 0)
    {
        id<MTLDevice> def = MTLCreateSystemDefaultDevice();
        if (def) devs = @[def];
    }
    if (devs.count == 0) return inv;

    inv.available = true;
    InventoryPlatform plat;
    plat.index = 0;
    plat.name  = "Metal";

    for (NSUInteger i = 0; i < devs.count; i++)
    {
        InventoryDevice dev;
        dev.index = static_cast<int>(i);
        dev.name  = [devs[i].name UTF8String];
        plat.devices.push_back(std::move(dev));
    }

    inv.platforms.push_back(std::move(plat));
    return inv;
}

#endif // ENABLE_METAL
