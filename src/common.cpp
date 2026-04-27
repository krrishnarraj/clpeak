#include <common.h>
#include <benchmark_constants.h>
#include <iostream>
#include <string>
#include <cctype>

benchmark_config_t benchmark_config_t::forDevice(cl_device_type type)
{
  benchmark_config_t cfg;

  if (type & CL_DEVICE_TYPE_CPU)
  {
    cfg.globalBWIters = 20;
    cfg.globalBWMaxSize = 1 << 27;
    cfg.computeWgsPerCU = 512;
    cfg.computeDPWgsPerCU = 256;
    cfg.computeIters = 10;
    cfg.localBWIters = 20;
    cfg.imageBWIters = 20;
    cfg.transferBWMaxSize = 1 << 27;
  }
  else
  { // GPU
    cfg.globalBWIters = 50;
    cfg.globalBWMaxSize = 1 << 29;
    cfg.computeWgsPerCU = 2048;
    cfg.computeDPWgsPerCU = 512;
    cfg.computeIters = 30;
    cfg.localBWIters = 50;
    cfg.imageBWIters = 50;
    cfg.transferBWMaxSize = 1 << 29;
  }
  cfg.transferBWIters = 20;
  cfg.kernelLatencyIters = 20000;

  return cfg;
}

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

  devInfo.deviceType = d.getInfo<CL_DEVICE_TYPE>();

  return devInfo;
}

float timeInUS(cl::Event &timeEvent)
{
  cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
  cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;

  return (float)(end - start);
}

void Timer::start()
{
  tick = std::chrono::high_resolution_clock::now();
}

float Timer::stopAndTime()
{
  tock = std::chrono::high_resolution_clock::now();
  return (float)(std::chrono::duration_cast<std::chrono::microseconds>(tock - tick).count());
}

void populate(float *ptr, uint64_t N)
{
  for (uint64_t i = 0; i < N; i++)
  {
    ptr[i] = (float)i;
  }
}

void populate(double *ptr, uint64_t N)
{
  for (uint64_t i = 0; i < N; i++)
  {
    ptr[i] = (double)i;
  }
}

uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue)
{
  uint64_t n = (number > maxValue) ? maxValue : number;
  return (n / base) * base;
}

void trimString(std::string &str)
{
  size_t pos = str.find('\0');

  if (pos != std::string::npos)
  {
    str.erase(pos);
  }
}
