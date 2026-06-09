#ifdef ENABLE_METAL

#include "mtl_internal.h"

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

#endif // ENABLE_METAL
