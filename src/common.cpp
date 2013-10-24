
#include <common.h>

device_info_t getDeviceInfo(cl::Device &d)
{
    device_info_t devInfo;
    
    devInfo.numCUs = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    devInfo.maxWGSize = d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    
    return devInfo;
}

int populate(float *ptr, uint N)
{
    srand(time(NULL));

    for(int i=0; i<N; i++)
    {
        ptr[i] = (float)rand();
    }
}

int populate(double *ptr, uint N)
{
    srand(time(NULL));

    for(int i=0; i<N; i++)
    {
        ptr[i] = (double)rand();
    }
}
