#ifndef ONEAPI_PEAK_H
#define ONEAPI_PEAK_H

#ifdef ENABLE_ONEAPI

#include <sycl/sycl.hpp>
#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>
#include <functional>
#include <string>
#include <vector>

struct CliOptions;

struct oneapi_device_info_t {
  std::string deviceName;
  std::string vendor;
  std::string driverVersion;
  std::string backendName;        // L0 / OpenCL / CUDA (SYCL backend behind the device)

  DeviceType deviceType = DeviceType::Unknown;

  int      numCUs = 0;            // sycl::info::device::max_compute_units
  size_t   maxWorkGroupSize = 0;
  uint64_t totalGlobalMem = 0;
  int      clockRateMHz = 0;
  uint32_t preferredSubGroupSize = 0;
  std::vector<size_t> subGroupSizes;

  bool fp16Supported = false;
  bool fp64Supported = false;
  bool bf16Supported = false;     // queried via sycl::aspect::bf16
  bool xmxSupported = false;      // Xe-cores Matrix eXtensions (Arc / PVC / Battlemage)
};

class OneapiDevice
{
public:
  int deviceIndex = -1;
  sycl::device dev;
  sycl::queue  stream;
  oneapi_device_info_t info;

  OneapiDevice();
  ~OneapiDevice();

  bool init(int devIndex, const sycl::device &d);
  void cleanup();
};

class OneapiPeak : public Peak
{
public:
  int deviceIndex;

  OneapiPeak();
  ~OneapiPeak();

  void applyOptions(const CliOptions &opts) override;
  int  runAll() override;

  static BackendInventory enumerate();
  static void printInventory(const BackendInventory &inv, std::ostream &os);

  int runComputeSP(OneapiDevice &dev, benchmark_config_t &cfg);
  int runComputeHP(OneapiDevice &dev, benchmark_config_t &cfg);
  int runComputeDP(OneapiDevice &dev, benchmark_config_t &cfg);
  int runComputeMP(OneapiDevice &dev, benchmark_config_t &cfg);
  int runComputeBF16(OneapiDevice &dev, benchmark_config_t &cfg);
  int runComputeInt32(OneapiDevice &dev, benchmark_config_t &cfg);
  int runComputeInt8DP(OneapiDevice &dev, benchmark_config_t &cfg);
  int runComputeInt4Packed(OneapiDevice &dev, benchmark_config_t &cfg);
  int runJointMatrix(OneapiDevice &dev, benchmark_config_t &cfg, Category category);
  int runOnemkl(OneapiDevice &dev, benchmark_config_t &cfg);
  int runGlobalBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);
  int runAtomicThroughput(OneapiDevice &dev, benchmark_config_t &cfg);
  int runKernelLatency(OneapiDevice &dev, benchmark_config_t &cfg);

  // Timed launcher used by every compute / bandwidth benchmark.  Runs
  // warmups + one probe + `pickIters(...)` timed launches via the supplied
  // submitter, returning the mean dispatch time in microseconds (or a
  // negative value on submission failure).
  using KernelSubmitter = std::function<sycl::event(sycl::queue &)>;
  float runKernel(OneapiDevice &dev,
                  const KernelSubmitter &submit,
                  unsigned int targetTimeUsLocal,
                  unsigned int forcedIters);

  logger::DeviceScope *currentDeviceScope = nullptr;

private:
  bool initialised;
  std::vector<sycl::device> devices;

  bool initRuntime();
};

#endif // ENABLE_ONEAPI
#endif // ONEAPI_PEAK_H
