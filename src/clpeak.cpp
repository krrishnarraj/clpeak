#include <clpeak.h>
#include <cstring>

#define MSTRINGIFY(...) #__VA_ARGS__

static const std::string stringifiedKernels =
#include "global_bandwidth_kernels.cl"
#include "compute_sp_kernels.cl"
#include "compute_hp_kernels.cl"
#include "compute_dp_kernels.cl"
#include "compute_int24_kernels.cl"
#include "compute_integer_kernels.cl"
#include "compute_char_kernels.cl"
#include "compute_short_kernels.cl"
    ;

#ifdef USE_STUB_OPENCL
// Prototype
extern "C"
{
  void stubOpenclReset();
}
#endif

clPeak::clPeak() : forcePlatform(false), forcePlatformName(false), forceDevice(false),
                   forceDeviceName(false), forceTest(false), useEventTimer(false),
                   isGlobalBW(true), isComputeHP(true), isComputeSP(true), isComputeDP(true),
                   isComputeIntFast(true), isComputeInt(true),
                   isTransferBW(true), isKernelLatency(true), isComputeChar(true), isComputeShort(true),
                   specifiedPlatform(0), specifiedDevice(0),
                   specifiedPlatformName(0), specifiedDeviceName(0), specifiedTestName(0)
{
}

clPeak::~clPeak()
{
  if (log)
  {
    delete log;
  }
}

int clPeak::runAll()
{
  try
  {
#ifdef USE_STUB_OPENCL
    stubOpenclReset();
#endif
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    log->xmlOpenTag("clpeak");
    log->xmlAppendAttribs("os", OS_NAME);
    for (size_t p = 0; p < platforms.size(); p++)
    {
      if (forcePlatform && (p != specifiedPlatform))
        continue;

      std::string platformName = platforms[p].getInfo<CL_PLATFORM_NAME>();
      trimString(platformName);

      if (forcePlatformName && !(strcmp(platformName.c_str(), specifiedPlatformName) == 0))
        continue;

      log->print(NEWLINE "Platform: " + platformName + NEWLINE);
      log->xmlOpenTag("platform");
      log->xmlAppendAttribs("name", platformName);

      cl_context_properties cps[3] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)(platforms[p])(),
          0};

      cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
      vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
      cl::Program::Sources source(1, stringifiedKernels);
      cl::Program prog = cl::Program(ctx, source);

      for (size_t d = 0; d < devices.size(); d++)
      {
        if (forceDevice && (d != specifiedDevice))
          continue;

        device_info_t devInfo = getDeviceInfo(devices[d]);

        if (forceDeviceName && !(strcmp(devInfo.deviceName.c_str(), specifiedDeviceName) == 0))
          continue;

        log->print(TAB "Device: " + devInfo.deviceName + NEWLINE);
        log->print(TAB TAB "Driver version  : ");
        log->print(devInfo.driverVersion);
        log->print(" (" OS_NAME ")" NEWLINE);
        log->print(TAB TAB "Compute units   : ");
        log->print(devInfo.numCUs);
        log->print(NEWLINE);
        log->print(TAB TAB "Clock frequency : ");
        log->print(devInfo.maxClockFreq);
        log->print(" MHz" NEWLINE);
        log->xmlOpenTag("device");
        log->xmlAppendAttribs("name", devInfo.deviceName);
        log->xmlAppendAttribs("driver_version", devInfo.driverVersion);
        log->xmlAppendAttribs("compute_units", devInfo.numCUs);
        log->xmlAppendAttribs("clock_frequency", devInfo.maxClockFreq);
        log->xmlAppendAttribs("clock_frequency_unit", "MHz");

        try
        {
          vector<cl::Device> dev = {devices[d]};
          prog.build(dev, BUILD_OPTIONS);
        }
        catch (cl::Error &error)
        {
          UNUSED(error);
          log->print(TAB TAB "Build Log: " + prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[d]) + NEWLINE NEWLINE);
          continue;
        }

        cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], CL_QUEUE_PROFILING_ENABLE);

        runGlobalBandwidthTest(queue, prog, devInfo);
        runComputeSP(queue, prog, devInfo);
        runComputeHP(queue, prog, devInfo);
        runComputeDP(queue, prog, devInfo);
        runComputeInteger(queue, prog, devInfo);
        runComputeIntFast(queue, prog, devInfo);
        runComputeChar(queue, prog, devInfo);
        runComputeShort(queue, prog, devInfo);
        runTransferBandwidthTest(queue, prog, devInfo);
        runKernelLatency(queue, prog, devInfo);

        log->print(NEWLINE);
        log->xmlCloseTag(); // device
      }
      log->xmlCloseTag(); // platform
    }
    log->xmlCloseTag(); // clpeak
  }
  catch (cl::Error &error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE;

    log->print(ss.str());

    // skip error for no platform
    if (strcmp(error.what(), "clGetPlatformIDs") == 0)
    {
      log->print("no platforms found" NEWLINE);
    }
    else
    {
      return -1;
    }
  }

  return 0;
}

float clPeak::run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, uint iters)
{
  float timed = 0;

  // Dummy calls
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.finish();

  if (useEventTimer)
  {
    for (uint i = 0; i < iters; i++)
    {
      cl::Event timeEvent;

      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
      queue.finish();
      timed += timeInUS(timeEvent);
    }
  }
  else // std timer
  {
    Timer timer;

    timer.start();
    for (uint i = 0; i < iters; i++)
    {
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
      queue.flush();
    }
    queue.finish();
    timed = timer.stopAndTime();
  }

  return (timed / static_cast<float>(iters));
}
