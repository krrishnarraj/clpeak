
#include <common.h>
#include <math.h>

using namespace std;

device_info_t getDeviceInfo(cl::Device &d)
{
    device_info_t devInfo;
    
    devInfo.numCUs = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    devInfo.maxWGSize = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    devInfo.maxAllocSize = d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    devInfo.maxGlobalSize = d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    
    return devInfo;
}


void Timer::start()
{
    tick = chrono::high_resolution_clock::now();
}

float Timer::stopAndTime()
{
    tock = chrono::high_resolution_clock::now();
    return (chrono::duration_cast<chrono::microseconds>(tock - tick).count());
}



void populate(float *ptr, uint N)
{
    srand(time(NULL));

    for(int i=0; i<(int)N; i++)
    {
        ptr[i] = (float)rand();
    }
}

void populate(double *ptr, uint N)
{
    srand(time(NULL));

    for(int i=0; i<(int)N; i++)
    {
        ptr[i] = (double)rand();
    }
}

uint roundToPowOf2(uint number)
{
    float logd = log2(number);
    logd = floor(logd);
    
    return pow(2, (int)logd);
}

