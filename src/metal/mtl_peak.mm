#ifdef ENABLE_METAL

#include "mtl_internal.h"

namespace
{
NSArray<id<MTLDevice>> *copyClpeakMetalDevices()
{
#if TARGET_OS_IPHONE
    id<MTLDevice> def = MTLCreateSystemDefaultDevice();
    return def ? @[def] : @[];
#else
    NSArray<id<MTLDevice>> *devs = MTLCopyAllDevices();
    if (devs.count == 0)
    {
        // On macOS the default-device call is the most reliable way to grab
        // the integrated Apple-silicon GPU.
        id<MTLDevice> def = MTLCreateSystemDefaultDevice();
        if (def) devs = @[def];
    }
    return devs;
#endif
}
}

// ---------------------------------------------------------------------------
// MetalDevice
// ---------------------------------------------------------------------------

MetalDevice::MetalDevice() : impl(nullptr) {}
MetalDevice::~MetalDevice() { cleanup(); }

bool MetalDevice::init(int devIndex)
{
    NSArray<id<MTLDevice>> *devs = copyClpeakMetalDevices();
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
#if TARGET_OS_IPHONE
        ss << "iOS " << v.majorVersion << "." << v.minorVersion << "." << v.patchVersion;
#else
        ss << "macOS " << v.majorVersion << "." << v.minorVersion << "." << v.patchVersion;
#endif
        info.osVersion = ss.str();
    }

    // Probe Apple GPU family.  Apple7 = M1/A15 baseline for simdgroup_matrix;
    // Apple9 = M3/A17 generation where bf16/int8 paths light up.
    info.appleFamily = 0;
    info.isAppleSilicon = false;
    for (int f = 1; f <= 10; f++)
    {
        if ([impl->device supportsFamily:(MTLGPUFamily)(MTLGPUFamilyApple1 + f - 1)])
        {
            info.appleFamily = (uint32_t)f;
        }
    }
    info.isAppleSilicon = info.appleFamily >= 7;
#if TARGET_OS_IPHONE
    info.isAppleSilicon = info.appleFamily > 0;
#endif

    // Capability bits.  Apple silicon always has fp16; fp16 simdgroup_matrix
    // is M1+; bf16 simdgroup_matrix is M3+ (Apple9).
    info.fp16Supported                = info.isAppleSilicon;
    info.deviceType                   = DeviceType::Gpu;  // Apple Silicon GPU
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
#if TARGET_OS_IPHONE
    if (@available(iOS 17.0, *))
        info.mpsGraphBF16Supported = info.appleFamily >= 9;
#else
    if (@available(macOS 14.0, *))
        info.mpsGraphBF16Supported = info.appleFamily >= 9;
#endif

    // Best-effort GPU core count via IORegistry (Apple silicon exposes
    // "gpu-core-count" on the AGXAccelerator service).  Used to scale the
    // GEMM dim so similar-class GPUs land in similar wall-clock windows.
    info.gpuCoreCount = 0;
#if __has_include(<IOKit/IOKitLib.h>) && !TARGET_OS_IPHONE
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
id<MTLLibrary> mtlGetLibrary(MetalDevice &dev, const char *src, const char *srcName)
{
    NSValue *key = [NSValue valueWithPointer:src];
    id<MTLLibrary> lib = dev.impl->libraryCache[key];
    if (lib) return lib;

    NSError *err = nil;
    NSString *srcStr = [NSString stringWithUTF8String:src];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    // languageVersion is a property on MTLCompileOptions; pin to 3.0 so
    // simdgroup_matrix compiles even when the SDK default is older. bf16
    // needs 3.1, set conditionally below.
    opts.languageVersion = MTLLanguageVersion3_0;
#if TARGET_OS_IPHONE
    if (@available(iOS 17.0, *))
        opts.languageVersion = MTLLanguageVersion3_1;
#else
    if (@available(macOS 14.0, *))
        opts.languageVersion = MTLLanguageVersion3_1;
#endif

    lib = [dev.impl->device newLibraryWithSource:srcStr options:opts error:&err];
    if (!lib)
    {
        NSLog(@"Metal compile of %s failed: %@", srcName, err);
        return nil;
    }
    dev.impl->libraryCache[key] = lib;
    return lib;
}

id<MTLComputePipelineState> mtlGetPipeline(MetalDevice &dev, const char *src,
                                               const char *srcName, const char *fnName)
{
    NSString *cacheKey = [NSString stringWithFormat:@"%p#%s", (void*)src, fnName];
    id<MTLComputePipelineState> pso = dev.impl->pipelineCache[cacheKey];
    if (pso) return pso;

    id<MTLLibrary> lib = mtlGetLibrary(dev, src, srcName);
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
  : deviceIndex(-1),
    impl(nullptr)
{
}

MetalPeak::~MetalPeak() { delete impl; impl = nullptr; }

void MetalPeak::applyOptions(const CliOptions &opts)
{
    Peak::applyOptions(opts);
    deviceIndex = opts.mtlDeviceIndex;
}

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

int MetalPeak::runAll()
{
    impl = new MetalPeakImpl();
    impl->allDevices = copyClpeakMetalDevices();
    if (impl->allDevices.count == 0)
    {
        log->note("Metal: no devices found");
        return -1;
    }

    auto backendScope = log->beginBackend("Metal");

    for (NSUInteger d = 0; d < impl->allDevices.count; d++)
    {
        if (deviceIndex >= 0 && static_cast<NSUInteger>(deviceIndex) != d)
            continue;

        MetalDevice dev;
        if (!dev.init((int)d))
        {
            log->note("Metal: failed to init device " + std::to_string(d));
            continue;
        }
#if !TARGET_OS_IPHONE
        if (!dev.info.isAppleSilicon)
        {
            log->note("Metal: skipping " + dev.info.deviceName +
                      " -- requires Apple silicon (M1 or newer)");
            continue;
        }
#endif

        benchmark_config_t cfg = benchmark_config_t::forDevice(DeviceType::Gpu);
        cfg.targetTimeUs = targetTimeUs;
        if (forceIters)
            cfg.kernelLatencyIters = specifiedIters;

        std::stringstream familySS;
        familySS << "Apple" << dev.info.appleFamily;

        auto deviceScope = backendScope.beginDevice({
            dev.info.deviceName,
            "",
            dev.info.osVersion,
            {{"Apple family", familySS.str()},
             {"Working set", std::to_string(dev.info.recommendedMaxWorkingSetSize / (1024*1024)) + " MB"},
             {"GPU cores", std::to_string(dev.info.gpuCoreCount)}},
            -1,
            static_cast<int>(d)
        });
        currentDeviceScope = &deviceScope;

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

        currentDeviceScope = nullptr;
    }

    return 0;
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

// ---------------------------------------------------------------------------
// Benchmark methods live in separate category files:
//   compute_float.mm    compute_int.mm    simdgroup.mm
//   global_bandwidth.mm local_bandwidth.mm image_bandwidth.mm
//   kernel_latency.mm   atomic_throughput.mm
//   mtl_blas.mm (MPSGraph GEMM)
// ---------------------------------------------------------------------------

// Free-function enumeration used by --list-devices and the Android JNI surface.
// Mirrors the device-discovery logic at the top of MetalPeak::runAll() but
// stops at name extraction.
BackendInventory MetalPeak::enumerate()
{
    BackendInventory inv;
    inv.backend = "Metal";

    NSArray<id<MTLDevice>> *devs = copyClpeakMetalDevices();
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

void MetalPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
    os << "\n=== Metal backend ===\n";
    if (!b.available)
    {
        os << "Metal: no devices found\n";
        return;
    }
    for (const auto &plat : b.platforms)
        for (const auto &d : plat.devices)
            os << "  Metal Device " << d.index << ": " << d.name << "\n";
}

#endif // ENABLE_METAL
