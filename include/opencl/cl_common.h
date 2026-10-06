#ifndef CL_COMMON_H
#define CL_COMMON_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <CL/opencl.hpp>
#include <string>
#include <cstdint>
#include <sstream>
#include <common/benchmark_enums.h>
#include <common/common.h>

// Immutable device properties queried from OpenCL.  Every field starts at
// zero: runAll fills one per device into the same storage, so a field the
// queries skip on one device would otherwise keep the last device's value --
// the int8 packed-dot flag did, carrying an Intel runtime's "yes" onto an
// NVIDIA GPU behind it.
struct device_info_t
{
    std::string deviceName;
    std::string driverVersion;

    unsigned int numCUs = 0;
    unsigned int maxWGSize = 0;
    uint64_t maxAllocSize = 0;
    uint64_t maxGlobalSize = 0;
    unsigned int maxClockFreq = 0;

    bool halfSupported = false;
    bool doubleSupported = false;
    bool int8DotProductSupported = false;
    bool int8DotProductPackedSupported = false;
    // The compiler takes -cl-std=CL3.0 (OpenCL C 3.0 among
    // CL_DEVICE_OPENCL_C_ALL_VERSIONS).  Without it a compiler builds OpenCL C
    // 1.2, where the 3.0 feature macros the int8 dot builtins hang off are
    // never defined.
    bool openclC30 = false;
    cl_device_type clDeviceType = 0; // original OpenCL device type
    DeviceType deviceType = DeviceType::Unknown; // neutral equivalent

    uint64_t localMemSize = 0;
    // CL_DEVICE_LOCAL_MEM_TYPE == CL_LOCAL.  CL_GLOBAL means the device has no
    // scratchpad and __local is carved out of ordinary global memory.
    bool localMemDedicated = false;
    // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE -- the last level of cache in front of
    // global memory.  0 when the runtime does not report one.
    uint64_t globalMemCacheSize = 0;

    bool imageSupported = false;
    uint64_t image2dMaxWidth = 0;
    uint64_t image2dMaxHeight = 0;
};

device_info_t getDeviceInfo(cl::Device &d);

#endif // CL_COMMON_H
