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

// Immutable device properties queried from OpenCL.
struct device_info_t
{
    std::string deviceName;
    std::string driverVersion;

    unsigned int numCUs;
    unsigned int maxWGSize;
    uint64_t maxAllocSize;
    uint64_t maxGlobalSize;
    unsigned int maxClockFreq;

    bool halfSupported;
    bool doubleSupported;
    bool int8DotProductSupported;
    bool int8DotProductPackedSupported;
    // The compiler takes -cl-std=CL3.0 (OpenCL C 3.0 among
    // CL_DEVICE_OPENCL_C_ALL_VERSIONS).  Without it a compiler builds OpenCL C
    // 1.2, where the 3.0 feature macros the int8 dot builtins hang off are
    // never defined.
    bool openclC30;
    cl_device_type clDeviceType; // original OpenCL device type
    DeviceType deviceType;       // neutral equivalent

    uint64_t localMemSize;
    // CL_DEVICE_LOCAL_MEM_TYPE == CL_LOCAL.  CL_GLOBAL means the device has no
    // scratchpad and __local is carved out of ordinary global memory.
    bool localMemDedicated;
    // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE -- the last level of cache in front of
    // global memory.  0 when the runtime does not report one.
    uint64_t globalMemCacheSize;

    bool imageSupported;
    uint64_t image2dMaxWidth;
    uint64_t image2dMaxHeight;
};

device_info_t getDeviceInfo(cl::Device &d);

#endif // CL_COMMON_H
