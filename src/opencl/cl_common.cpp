#include <opencl/cl_common.h>
#include <common.h>
#include <benchmark_constants.h>
#include <vector>
#include <cctype>

device_info_t getDeviceInfo(cl::Device &d)
{
    device_info_t devInfo;

    devInfo.deviceName = d.getInfo<CL_DEVICE_NAME>();
    devInfo.driverVersion = d.getInfo<CL_DRIVER_VERSION>();
    trimString(devInfo.deviceName);
    trimString(devInfo.driverVersion);

    devInfo.numCUs = (unsigned int)d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    std::vector<size_t> maxWIPerDim;
    maxWIPerDim = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    devInfo.maxWGSize = (unsigned int)maxWIPerDim[0];

    // Cap work-group size to what hardware reports (up to MAX_WG_SIZE)
    devInfo.maxWGSize = std::min(devInfo.maxWGSize, (unsigned int)MAX_WG_SIZE);

    // FIXME limit max-workgroup size for qualcomm platform to 128
    // Kernel launch fails for workgroup size 256(CL_DEVICE_MAX_WORK_ITEM_SIZES)
    std::string vendor = d.getInfo<CL_DEVICE_VENDOR>();
    std::string vendorLower = vendor;
    std::transform(vendorLower.begin(), vendorLower.end(), vendorLower.begin(), ::tolower);
    if (vendorLower.find("qualcomm") != std::string::npos)
    {
        devInfo.maxWGSize = std::min(devInfo.maxWGSize, (unsigned int)128);
    }

    devInfo.maxAllocSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
    devInfo.localMemSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
    devInfo.maxGlobalSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>());

    devInfo.imageSupported = (d.getInfo<CL_DEVICE_IMAGE_SUPPORT>() == CL_TRUE);
    devInfo.image2dMaxWidth  = devInfo.imageSupported ? static_cast<uint64_t>(d.getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>())  : 0;
    devInfo.image2dMaxHeight = devInfo.imageSupported ? static_cast<uint64_t>(d.getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>()) : 0;
    devInfo.maxClockFreq = static_cast<unsigned int>(d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
    devInfo.doubleSupported = false;
    devInfo.halfSupported = false;
    devInfo.int8DotProductSupported = false;

    std::string extns = d.getInfo<CL_DEVICE_EXTENSIONS>();

    if ((extns.find("cl_khr_fp16") != std::string::npos))
        devInfo.halfSupported = true;

    if ((extns.find("cl_khr_fp64") != std::string::npos) || (extns.find("cl_amd_fp64") != std::string::npos))
        devInfo.doubleSupported = true;

    if (extns.find("cl_khr_integer_dot_product") != std::string::npos)
        devInfo.int8DotProductSupported = true;

    devInfo.clDeviceType = d.getInfo<CL_DEVICE_TYPE>();

    // Convert to neutral DeviceType
    if (devInfo.clDeviceType & CL_DEVICE_TYPE_GPU)
        devInfo.deviceType = DeviceType::Gpu;
    else if (devInfo.clDeviceType & CL_DEVICE_TYPE_CPU)
        devInfo.deviceType = DeviceType::Cpu;
    else if (devInfo.clDeviceType & CL_DEVICE_TYPE_ACCELERATOR)
        devInfo.deviceType = DeviceType::Accelerator;

    return devInfo;
}

float timeInUS(cl::Event &timeEvent)
{
    cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
    cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;

    return (float)(end - start);
}
