// What only IOKit and Foundation answer, for host_info.cpp (host_info.h):
// the product's names from the device tree, the internal battery and the
// power source on macOS, and the low-power mode and thermal state on macOS
// and iOS.  iOS reaches no IOKit -- its power source is UIKit's, which the
// app passes in -- so those read nothing there.
#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

#if TARGET_OS_OSX
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/pwr_mgt/IOPM.h>
#endif

#include <string>

namespace xpupeak {
namespace apple {

// A string property of IODeviceTree:/product, the node Apple silicon's
// device tree describes the product in: "product-name" is the marketing name
// ("MacBook Pro (14-inch, 2021)"), "product-soc-name" the SoC ("Apple M1
// Pro").  Only named keys are read: the node also holds the unit's
// product-id, which a result never carries.
std::string productProperty(const char *key)
{
#if TARGET_OS_OSX
    // MACH_PORT_NULL is the default main port, on every macOS.
    io_registry_entry_t product = IORegistryEntryFromPath(MACH_PORT_NULL, "IODeviceTree:/product");
    if (product == MACH_PORT_NULL) return "";
    std::string out;
    CFStringRef name = CFStringCreateWithCString(kCFAllocatorDefault, key, kCFStringEncodingUTF8);
    CFTypeRef value = name ? IORegistryEntryCreateCFProperty(product, name, kCFAllocatorDefault, 0)
                           : nullptr;
    if (value && CFGetTypeID(value) == CFDataGetTypeID())
    {
        // Device-tree strings are NUL-terminated bytes.
        const auto *bytes = reinterpret_cast<const char *>(CFDataGetBytePtr((CFDataRef)value));
        const CFIndex length = CFDataGetLength((CFDataRef)value);
        CFIndex end = 0;
        while (end < length && bytes[end] != '\0') end++;
        out.assign(bytes, static_cast<size_t>(end));
    }
    else if (value && CFGetTypeID(value) == CFStringGetTypeID())
    {
        char buf[256];
        if (CFStringGetCString((CFStringRef)value, buf, sizeof buf, kCFStringEncodingUTF8))
            out = buf;
    }
    if (value) CFRelease(value);
    if (name) CFRelease(name);
    IOObjectRelease(product);
    return out;
#else
    (void)key;
    return "";
#endif
}

// 1 when IOKit's power sources list an internal battery, 0 when they list
// none (a UPS is not one), -1 when they cannot be read.
int internalBattery()
{
#if TARGET_OS_OSX
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (!info) return -1;
    CFArrayRef list = IOPSCopyPowerSourcesList(info);
    if (!list)
    {
        CFRelease(info);
        return -1;
    }
    int found = 0;
    for (CFIndex i = 0; i < CFArrayGetCount(list) && !found; i++)
    {
        CFDictionaryRef source = IOPSGetPowerSourceDescription(info, CFArrayGetValueAtIndex(list, i));
        if (!source) continue;
        const void *type = CFDictionaryGetValue(source, CFSTR(kIOPSTypeKey));
        if (type && CFGetTypeID(type) == CFStringGetTypeID() &&
            CFEqual(type, CFSTR(kIOPSInternalBatteryType)))
            found = 1;
    }
    CFRelease(list);
    CFRelease(info);
    return found;
#else
    return -1;
#endif
}

// What the Mac is drawing from: "ac" or "battery".  A UPS is neither word,
// and stays unsaid.
std::string providingPowerSource()
{
#if TARGET_OS_OSX
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (!info) return "";
    std::string out;
    if (CFStringRef type = IOPSGetProvidingPowerSourceType(info))   // not retained
    {
        if (CFEqual(type, CFSTR(kIOPMACPowerKey)))
            out = "ac";
        else if (CFEqual(type, CFSTR(kIOPMBatteryPowerKey)))
            out = "battery";
    }
    CFRelease(info);
    return out;
#else
    return "";
#endif
}

// Low Power Mode: macOS 12 and later, iOS 9 and later.
int lowPowerMode()
{
    @autoreleasepool
    {
        if (@available(macOS 12.0, iOS 9.0, *))
            return [NSProcessInfo processInfo].isLowPowerModeEnabled ? 1 : 0;
        return -1;
    }
}

// NSProcessInfo's thermal state, in its own words.
std::string thermalState()
{
    @autoreleasepool
    {
        switch ([NSProcessInfo processInfo].thermalState)
        {
        case NSProcessInfoThermalStateNominal:  return "nominal";
        case NSProcessInfoThermalStateFair:     return "fair";
        case NSProcessInfoThermalStateSerious:  return "serious";
        case NSProcessInfoThermalStateCritical: return "critical";
        }
        return "";
    }
}

} // namespace apple
} // namespace xpupeak
