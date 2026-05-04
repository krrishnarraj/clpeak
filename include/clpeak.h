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

#include <result_store.h>  // Category enum

struct CliOptions; // forward decl

#define BUILD_OPTIONS " -cl-mad-enable "

// Benchmark enum must be defined early (before backend_gating.h uses it).
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
    CoopMatrix,
    Wmma,
    Bmma,                 // CUDA binary tensor cores (BMMA), int_compute primary
    SimdgroupMatrix,
    MpsGemm,
    Cublas,
    AtomicThroughput,
    TransferBW,
    KernelLatency,
    COUNT
};

// Primary category of a benchmark.  Used for both CLI gating and the
// run-order phase loop.  Tensor / vendor-library tests that span both fp
// and int variants (Wmma, CoopMatrix, SimdgroupMatrix, Cublas, MpsGemm)
// are listed under their fp form here; backends iterate them again in the
// int_compute phase emitting only int variants there. AtomicThroughput is
// primarily integer, with Metal's atomic_float variant emitted explicitly in
// the fp_compute phase.
inline Category categoryOf(Benchmark b)
{
    switch (b) {
    case Benchmark::GlobalBW:
    case Benchmark::LocalBW:
    case Benchmark::ImageBW:
    case Benchmark::TransferBW:
        return Category::Bandwidth;

    case Benchmark::ComputeSP:
    case Benchmark::ComputeHP:
    case Benchmark::ComputeDP:
    case Benchmark::ComputeMP:
    case Benchmark::ComputeBF16:
    case Benchmark::Wmma:
    case Benchmark::CoopMatrix:
    case Benchmark::SimdgroupMatrix:
    case Benchmark::Cublas:
    case Benchmark::MpsGemm:
        return Category::FpCompute;

    case Benchmark::ComputeInt:
    case Benchmark::ComputeIntFast:
    case Benchmark::ComputeChar:
    case Benchmark::ComputeShort:
    case Benchmark::ComputeInt8DP:
    case Benchmark::ComputeInt4Packed:
    case Benchmark::AtomicThroughput:
    case Benchmark::Bmma:
        return Category::IntCompute;

    case Benchmark::KernelLatency:
        return Category::Latency;

    case Benchmark::COUNT:
        break;
    }
    return Category::Unknown;
}

#include <backend_gating.h>  // centralized benchmark gates (uses Benchmark & Category)

class clPeak
{
public:
  bool forcePlatform, forcePlatformName, forceDevice, forceDeviceName, forceTest, forceIters, useEventTimer;
  BackendGating gating;
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

  std::unique_ptr<logger> log;

  clPeak();
  ~clPeak() = default;

  void applyOptions(const CliOptions &opts);

  float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, unsigned int iters);

  // Unified compute benchmark helper -- replaces 7 nearly-identical runCompute* methods
  int runComputeTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo,
                     benchmark_config_t &cfg, Benchmark which,
                     const std::string &displayName, const std::string &resultTag,
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
