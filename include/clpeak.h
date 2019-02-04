#ifndef CLPEAK_HPP
#define CLPEAK_HPP

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <common.h>
#include <logger.h>

#define BUILD_OPTIONS           " -cl-mad-enable "

using namespace std;

class clPeak
{
public:

  bool forcePlatform, forceDevice, useEventTimer;
  bool isGlobalBW, isComputeSP, isComputeDP, isComputeInt, isTransferBW, isKernelLatency;
  ulong specifiedPlatform, specifiedDevice;
  logger *log;

  clPeak();
  ~clPeak();

  int parseArgs(int argc, char **argv);

  // Return avg time in us
  float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, uint iters);

  int runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeHP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeInteger(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runAll();
};

#endif  // CLPEAK_HPP
