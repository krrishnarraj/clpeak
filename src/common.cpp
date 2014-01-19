#include <common.h>
#include <math.h>
#include <iostream>
#include <string>

using namespace std;

device_info_t getDeviceInfo(cl::Device &d)
{
    device_info_t devInfo;

    devInfo.deviceName = d.getInfo<CL_DEVICE_NAME>();
    devInfo.driverVersion = d.getInfo<CL_DRIVER_VERSION>();

    devInfo.numCUs = (uint)d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    devInfo.maxWGSize = (uint)d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    // Limiting max work-group size to 256
    #define MAX_WG_SIZE 256
    devInfo.maxWGSize = MIN(devInfo.maxWGSize, MAX_WG_SIZE);

    devInfo.maxAllocSize = (uint)d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    devInfo.maxGlobalSize = (uint)d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    devInfo.doubleSupported = false;

    std::string extns = d.getInfo<CL_DEVICE_EXTENSIONS>();
    if((extns.find("cl_khr_fp64") != std::string::npos) || (extns.find("cl_amd_fp64") != std::string::npos))
        devInfo.doubleSupported = true;

    devInfo.deviceType = d.getInfo<CL_DEVICE_TYPE>();

    if(devInfo.deviceType & CL_DEVICE_TYPE_CPU) {
        devInfo.gloalBWIters = 20;
        devInfo.computeWgsPerCU = 512;
        devInfo.computeIters = 10;
    } else {            // GPU
        devInfo.gloalBWIters = 50;
        devInfo.computeWgsPerCU = 2048;
        devInfo.computeIters = 30;
    }
    devInfo.transferBWIters = 20;
    devInfo.kernelLatencyIters = 20000;

    return devInfo;
}


float timeInUS(cl::Event &timeEvent)
{
    cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
    cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;

    return (float)((int)end - (int)start);
}


void Timer::start()
{
    tick = chrono::high_resolution_clock::now();
}


float Timer::stopAndTime()
{
    tock = chrono::high_resolution_clock::now();
    return (float)(chrono::duration_cast<chrono::microseconds>(tock - tick).count());
}


void populate(float *ptr, uint N)
{
    srand((unsigned int)time(NULL));

    for(int i=0; i<(int)N; i++)
    {
        //ptr[i] = (float)rand();
        ptr[i] = (float)i;
    }
}

void populate(double *ptr, uint N)
{
    srand((unsigned int)time(NULL));

    for(int i=0; i<(int)N; i++)
    {
        //ptr[i] = (double)rand();
        ptr[i] = (double)i;
    }
}


uint roundToPowOf2(uint number, int maxPower)
{
    double logd = log(number) / log(2);
    logd = floor(logd);

    // If maximum power limit was specified
    if(maxPower > 0) {
        logd = MIN(logd, ((double)maxPower));
    }

    return (uint)pow(2, (int)logd);
}

