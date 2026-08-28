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

// Return time in us for the given event.
float timeInUS(cl::Event &timeEvent);

#endif // CL_COMMON_H
