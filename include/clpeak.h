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
  bool isRuntimeOverheadTest, isVerbose;
  int specifiedPlatform, specifiedDevice;
  logger *log;

  clPeak();
  ~clPeak();

  int parseArgs(int argc, char **argv);

  // Return avg time in us
  float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, int iters);

  int runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeHP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeInteger(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runAll();

  void runRuntimeOverheadTests(const cl::Context &ctx, const cl::Device &device, cl::Program &prog);

private:
  void printHeader(size_t queueSize, uint32_t threadSize, const cl::NDRange& globalWorkSize, const cl::NDRange& localWorkSize);

  template <uint32_t enqueueIterations, uint32_t flushIterations>
  void printRecords(size_t enqueuesPerGivenTime, const char *lineStart);

  template <uint32_t enqueueIterations, uint32_t flushIterations>
  void printRecords(size_t enqueuesPerGivenTime, const performanceStatisticsPackVec &statistics,
                    microsecondsT warmupDuration = 0.0f, const char *lineStart = TAB TAB TAB TAB TAB);

  void printRecords(const performanceStatistics &statistics, microsecondsT warmupDuration = 0.0f, const char *lineStart = TAB TAB TAB TAB TAB TAB TAB);

  template <uint32_t enqueueIterations, uint32_t flushIterations>
  void logRecords(size_t enqueuesPerGivenTime, const performanceStatisticsPackVec &statistics,
                  microsecondsT warmupDuration = 0.0f);

  void logRecords(const char* tagName, const performanceStatistics &statistics, microsecondsT warmupDuration = 0.0f);

  template <uint32_t batchSize, uint32_t enqueueIterations, uint32_t flushIterations>
  void runKernel(cl::CommandQueue &queue, cl::Kernel &kernel, durationTimesVec &times, const cl::NDRange &globalOffsetSize,
                 const cl::NDRange &globalSize, const cl::NDRange &localSize);

  microsecondsT runKernel(cl::CommandQueue &queue, cl::Kernel &kernel, const cl::NDRange &globalOffsetSize,
                          const cl::NDRange &globalSize, const cl::NDRange &localSize);

  microsecondsT flushQueue(cl::CommandQueue &queue);

  template <uint32_t batchSize, uint32_t enqueueIterations, uint32_t flushIterations>
  void generateTestCase(cl::CommandQueue &queue, cl::Kernel &kernel, const cl::NDRange &globalOffsetSize, const cl::NDRange &globalSize, const cl::NDRange &localSize);
};

#endif  // CLPEAK_HPP
