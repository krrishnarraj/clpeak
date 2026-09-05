#include <opencl/cl_common.h>
#include <opencl/cl_utils.h>
#include <vector>

// cl_khr_integer_dot_product enums, defined here when the platform's OpenCL
// headers predate the extension -- the OpenCL 2.2 headers that ship with ROCm
// do, and clpeak builds against whatever headers the system provides.  The
// values are fixed by the extension spec, so defining them locally is safe;
// querying them on a device that lacks the extension is not, which is why the
// call below stays behind the extension-string check.
#ifndef CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR
#define CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR 0x1073
#endif
#ifndef CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR
#define CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR (1 << 1)
#endif
#ifndef CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR
#define CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR (1 << 0)
#endif

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

    // Per-kernel work-group limits (CL_KERNEL_WORK_GROUP_SIZE) are enforced at
    // launch time by clPeak::clampToKernelWG, which supersedes the old
    // Qualcomm-specific 128 cap for kernels that could not run at the device max.

    devInfo.maxAllocSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
    devInfo.localMemSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>());
    devInfo.localMemDedicated = (d.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() == CL_LOCAL);
    devInfo.globalMemCacheSize = static_cast<uint64_t>(d.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>());
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

    // cl_khr_integer_dot_product advertises two independent input forms, and a
    // device may implement only the packed one.  The kernels here call
    // dot_acc_sat(char4, char4, int), which needs the unpacked 4x8-bit form, so
    // gating on the extension string alone turns an honest "unsupported" into
    // five clCreateKernel errors -- as seen on Intel Arc (-44) and Intel's CPU
    // runtime (-46), both of which advertise the extension.
    if (extns.find("cl_khr_integer_dot_product") != std::string::npos)
    {
        cl_bitfield dpCaps = 0;
        if (clGetDeviceInfo(d(), CL_DEVICE_INTEGER_DOT_PRODUCT_CAPABILITIES_KHR,
                            sizeof(dpCaps), &dpCaps, nullptr) == CL_SUCCESS)
        {
            devInfo.int8DotProductSupported =
                (dpCaps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR) != 0;
            devInfo.int8DotProductPackedSupported =
                (dpCaps & CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_PACKED_KHR) != 0;
        }
    }

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
