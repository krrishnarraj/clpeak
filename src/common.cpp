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
  trimString(devInfo.deviceName);
  trimString(devInfo.driverVersion);

  devInfo.numCUs = (uint)d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  vector<size_t> maxWIPerDim;
  maxWIPerDim = d.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  devInfo.maxWGSize = (uint)maxWIPerDim[0];

  // Limiting max work-group size to 256
#define MAX_WG_SIZE 256
  devInfo.maxWGSize = MIN(devInfo.maxWGSize, MAX_WG_SIZE);

  // FIXME limit max-workgroup size for qualcomm platform to 128
  // Kernel launch fails for workgroup size 256(CL_DEVICE_MAX_WORK_ITEM_SIZES)
  string vendor = d.getInfo<CL_DEVICE_VENDOR>();
  if( (vendor.find("QUALCOMM") != std::string::npos) ||
      (vendor.find("qualcomm") != std::string::npos) )
  {
    devInfo.maxWGSize = MIN(devInfo.maxWGSize, 128);
  }

  devInfo.maxAllocSize = (ulong)d.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
  devInfo.maxGlobalSize = (ulong)d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  devInfo.maxClockFreq = (uint)d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
  devInfo.doubleSupported = false;
  devInfo.halfSupported = false;

  std::string extns = d.getInfo<CL_DEVICE_EXTENSIONS>();

  if((extns.find("cl_khr_fp16") != std::string::npos))
    devInfo.halfSupported = true;

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


uint roundToMultipleOf(uint number, const uint base, int maxValue)
{
  if(maxValue > 0 && number > static_cast<uint>(maxValue))
    return (maxValue / base) * base;

  return (number / base) * base;
}

void trimString(std::string &str)
{
  str.erase(str.find('\0'));
}
