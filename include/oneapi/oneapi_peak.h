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

// Shared note for one reading of a vector-width sweep.  NOT for the int8-dot
// rows: those are independent chains -- see oneapiChainNote().
static inline const char *oneapiWidthNote(int width)
{
  switch (width)
  {
  case 1:  return "One value per work-item at a time -- the plain, unvectorised case.";
  case 2:  return "Two values per work-item at a time, as one 2-wide vector.";
  case 4:  return "Four values per work-item at a time, as one 4-wide vector.";
  case 8:  return "Eight values per work-item at a time, as one 8-wide vector.";
  case 16: return "Sixteen values per work-item at a time, the widest vector SYCL offers.";
  default: return "";
  }
}

// Shared note for one reading of the int8 dot-product sweep: these variants
// are independent chains, not wider vectors.  Work per item is identical.
static inline const char *oneapiChainNote(int chains)
{
  switch (chains)
  {
  case 1: return "One chain of dot products, each waiting on the one before it.";
  case 2: return "Two independent chains, so the device has a second dot product "
                 "to get on with while the first is still finishing.";
  case 4: return "Four independent chains.";
  case 8: return "Eight independent chains.  Where this stops improving on four, "
                 "the hardware itself is the limit, not the waiting.";
  default: return "";
  }
}

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
  // sycl::info::device::global_mem_cache_size -- the last level of cache in
  // front of global memory.  0 when the runtime does not report one.
  uint64_t globalMemCacheSize = 0;
  // sycl::info::device::local_mem_type == local.  `global` means the device has
  // no scratchpad and the runtime carves local memory out of global memory.
  bool localMemDedicated = true;
  int      clockRateMHz = 0;
  uint32_t preferredSubGroupSize = 0;
  std::vector<size_t> subGroupSizes;

  bool fp16Supported = false;
  bool fp64Supported = false;
  bool bf16Supported = false;     // assumed true — all Intel SYCL GPUs have native bf16
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

  // Recreate the in-order queue on the same (default) context.  Used to recover
  // from a poisoned queue: on some runtimes a failed submission leaves the
  // in-order queue in an error state that makes every subsequent launch fail,
  // cascading errors across unrelated benchmarks.  USM allocations bound to the
  // device's default context stay valid across this reset.  Returns success.
  bool resetQueue();
};

class OneapiPeak : public Peak
{
public:
  std::vector<int> deviceIndices;  // empty = run all

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
  int runJointMatrix(OneapiDevice &dev, benchmark_config_t &cfg, Category category);
  int runOnemkl(OneapiDevice &dev, benchmark_config_t &cfg, Category category);
  int runGlobalBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(OneapiDevice &dev, benchmark_config_t &cfg);

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
