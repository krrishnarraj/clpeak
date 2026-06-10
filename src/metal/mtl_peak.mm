#ifdef ENABLE_METAL

#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// MetalPeak
// ---------------------------------------------------------------------------

MetalPeak::MetalPeak()
  : impl(nullptr)
{
}

MetalPeak::~MetalPeak() { delete impl; impl = nullptr; }

void MetalPeak::applyOptions(const CliOptions &opts)
{
    Peak::applyOptions(opts);
    deviceIndices = opts.mtlDeviceIndices;
}

// ---------------------------------------------------------------------------
// Benchmark methods live in separate files:
//   mtl_device.mm          compute_kernel.mm     mtl_utils.mm
//   compute_float.mm       compute_int.mm        simdgroup.mm
//   mtl_blas.mm            global_bandwidth.mm   local_bandwidth.mm
//   image_bandwidth.mm     kernel_latency.mm
// ---------------------------------------------------------------------------

int MetalPeak::runAll()
{
    impl = new MetalPeakImpl();
    impl->allDevices = copyClpeakMetalDevices();
    if (impl->allDevices.count == 0)
    {
        log->note("Metal: no devices found\n");
        return 0;
    }

    auto backendScope = log->beginBackend("Metal");

    for (NSUInteger d = 0; d < impl->allDevices.count; d++)
    {
        if (!deviceIndices.empty() &&
            std::find(deviceIndices.begin(), deviceIndices.end(), static_cast<int>(d)) == deviceIndices.end())
            continue;

        MetalDevice dev;
        if (!dev.init((int)d))
        {
            log->note("Metal: failed to init device " + std::to_string(d) + "\n");
            continue;
        }
#if !TARGET_OS_IPHONE
        if (!dev.info.isAppleSilicon)
        {
            log->note("Metal: skipping " + dev.info.deviceName +
                      " -- requires Apple silicon (M1 or newer)\n");
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
            {
#if !TARGET_OS_IPHONE
                {"Apple family", familySS.str()},
                {"Working set", std::to_string(dev.info.recommendedMaxWorkingSetSize / (1024*1024)) + " MB"},
                {"GPU cores", std::to_string(dev.info.gpuCoreCount)}
#endif
             },
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

        // ---- Phase 2: bandwidth (GBPS) -----------------------------------
        if (isAllowed(Benchmark::GlobalBW))          runGlobalBandwidth(dev, cfg);
        if (isAllowed(Benchmark::LocalBW))           runLocalBandwidth(dev, cfg);
        if (isAllowed(Benchmark::ImageBW))           runImageBandwidth(dev, cfg);

        // ---- Phase 4: latency (us) ---------------------------------------
        if (isAllowed(Benchmark::KernelLatency))     runKernelLatency(dev, cfg);

        currentDeviceScope = nullptr;
    }

    return 0;
}

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
