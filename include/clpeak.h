#ifndef CLPEAK_HPP
#define CLPEAK_HPP

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <bitset>
#include <memory>
#include <common.h>
#include <benchmark_constants.h>
#include <logger.h>

#define BUILD_OPTIONS " -cl-mad-enable "

enum class Benchmark : unsigned int {
  GlobalBW = 0,
  LocalBW,
  ImageBW,
  ComputeHP,
  ComputeMP,
  ComputeSP,
  ComputeDP,
  ComputeInt,
  ComputeIntFast,
  ComputeChar,
  ComputeShort,
  ComputeInt8DP,
  ComputeInt4Packed,
  ComputeBF16,
  AtomicThroughput,
  TransferBW,
  KernelLatency,
  COUNT
};

class clPeak
{
public:
  bool forcePlatform, forcePlatformName, forceDevice, forceDeviceName, forceTest, forceIters, useEventTimer;
  std::bitset<static_cast<size_t>(Benchmark::COUNT)> enabledTests;
  unsigned long specifiedPlatform, specifiedDevice;
  std::string specifiedPlatformName;
  std::string specifiedDeviceName;
  std::string specifiedTestName;
  unsigned int specifiedIters;
  unsigned int warmupCount;

  // Output format options
  bool enableJson;
  std::string jsonFileName;
  bool enableCsv;
  std::string csvFileName;

  // Baseline compare
  std::string compareFileName;

  // List devices mode
  bool listDevices;

  std::unique_ptr<logger> log;

  clPeak();
  ~clPeak() = default;

  int parseArgs(int argc, char **argv);

  bool isTestEnabled(Benchmark b) const { return enabledTests.test(static_cast<size_t>(b)); }
  void enableTest(Benchmark b)  { enabledTests.set(static_cast<size_t>(b)); }
  void disableAll()             { enabledTests.reset(); }
  void enableAll()              { enabledTests.set(); }

  float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, unsigned int iters);

  // Unified compute benchmark helper -- replaces 7 nearly-identical runCompute* methods
  int runComputeTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo,
                     benchmark_config_t &cfg, Benchmark which,
                     const std::string &displayName, const std::string &xmlTag,
                     const std::string &kernelPrefix, const std::string &typeName,
                     const std::string &unit, unsigned int workPerWI,
                     unsigned int wgsPerCU, size_t elemSize);

  int runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg);

  int runLocalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg);

  int runImageBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg);

  int runAtomicThroughputTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg);

  int runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg);

  int runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg);

  int runAll();
};

#endif // CLPEAK_HPP
