#ifndef ROCM_PEAK_H
#define ROCM_PEAK_H

#ifdef ENABLE_ROCM

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct CliOptions;

struct rocm_device_info_t {
  std::string deviceName;
  std::string driverVersion;
  std::string runtimeVersion;
  std::string archName;

  DeviceType deviceType = DeviceType::Unknown;

  int numCUs = 0;
  int maxThreadsPerBlock = 0;
  uint64_t totalGlobalMem = 0;
  int clockRateKHz = 0;
  int warpSize = 0;

  bool fp16Supported = false;
  bool bf16Supported = false;
  bool rocwmmaSupported = false;
};

class RocmDevice
{
public:
  int deviceIndex;
  hipStream_t stream;
  rocm_device_info_t info;

  RocmDevice();
  ~RocmDevice();

  bool init(int devIndex);
  void cleanup();

  bool getKernel(const char *src, const char *srcName,
                 const char *kernelName, hipFunction_t &fn,
                 const std::vector<const char *> &extraOpts = {});

private:
  std::unordered_map<const char *, hipModule_t> moduleCache;
};

struct rocm_compute_variant_t
{
  const char *label;
  const char *kernelName;
  const char *src;
  const char *srcName;
};

struct rocm_compute_desc_t
{
  const char *title;
  const char *resultTag;
  const char *unit;
  double      unitDivider;

  const char *metricLabel;
  const char *kernelName;
  const char *src;
  const char *srcName;

  const rocm_compute_variant_t *variants;
  uint32_t numVariants;

  uint32_t workPerWI;
  uint32_t elemSize;
  uint32_t blockSize;
  uint32_t outElemsPerBlock;

  const void *scalarArg;
  uint32_t    scalarSize;

  bool        skip;
  const char *skipMsg;

  const char *const *extraHiprtcOpts;
  uint32_t           numExtraHiprtcOpts;
};

class RocmPeak : public Peak
{
public:
  int deviceIndex;

  RocmPeak();
  ~RocmPeak();

  void applyOptions(const CliOptions &opts) override;
  int runAll() override;

  static BackendInventory enumerate();
  static void printInventory(const BackendInventory &inv, std::ostream &os);

  int runComputeSP(RocmDevice &dev, benchmark_config_t &cfg);
  int runComputeHP(RocmDevice &dev, benchmark_config_t &cfg);
  int runComputeDP(RocmDevice &dev, benchmark_config_t &cfg);
  int runComputeMP(RocmDevice &dev, benchmark_config_t &cfg);
  int runComputeBF16(RocmDevice &dev, benchmark_config_t &cfg);
  int runComputeInt32(RocmDevice &dev, benchmark_config_t &cfg);
  int runComputeInt8DP(RocmDevice &dev, benchmark_config_t &cfg);
  int runComputeInt4Packed(RocmDevice &dev, benchmark_config_t &cfg);
  int runRocwmma(RocmDevice &dev, benchmark_config_t &cfg, Category category);
  int runMfma(RocmDevice &dev, benchmark_config_t &cfg, Category category);
  int runRocblas(RocmDevice &dev, benchmark_config_t &cfg);
  int runGlobalBandwidth(RocmDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(RocmDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(RocmDevice &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(RocmDevice &dev, benchmark_config_t &cfg);
  int runAtomicThroughput(RocmDevice &dev, benchmark_config_t &cfg);
  int runKernelLatency(RocmDevice &dev, benchmark_config_t &cfg);

private:
  bool initialised;
  std::vector<int> devIndices;
  logger::DeviceScope *currentDeviceScope = nullptr;

  bool initRuntime();

  float runKernel(RocmDevice &dev, hipFunction_t fn,
                  uint32_t gridX, uint32_t blockX,
                  void **args,
                  unsigned int targetTimeUs, unsigned int forcedIters);

  int runComputeKernel(RocmDevice &dev, benchmark_config_t &cfg,
                       const rocm_compute_desc_t &d);
};

namespace rocm_kernels {
  extern const char *compute_sp_src;
  extern const char *compute_sp_name;
  extern const char *compute_hp_src;
  extern const char *compute_hp_name;
  extern const char *compute_dp_src;
  extern const char *compute_dp_name;
  extern const char *compute_mp_src;
  extern const char *compute_mp_name;
  extern const char *compute_bf16_src;
  extern const char *compute_bf16_name;
  extern const char *compute_int32_src;
  extern const char *compute_int32_name;
  extern const char *compute_int4_packed_src;
  extern const char *compute_int4_packed_name;
  extern const char *compute_int8_dp_src;
  extern const char *compute_int8_dp_name;
  extern const char *rocwmma_fp16_src;
  extern const char *rocwmma_fp16_name;
  extern const char *rocwmma_int8_src;
  extern const char *rocwmma_int8_name;
  extern const char *mfma_fp16_src;
  extern const char *mfma_fp16_name;
  extern const char *mfma_bf16_src;
  extern const char *mfma_bf16_name;
  extern const char *mfma_int8_src;
  extern const char *mfma_int8_name;
  extern const char *mfma_fp8_src;
  extern const char *mfma_fp8_name;
  extern const char *global_bandwidth_src;
  extern const char *global_bandwidth_name;
  extern const char *local_bandwidth_src;
  extern const char *local_bandwidth_name;
  extern const char *image_bandwidth_src;
  extern const char *image_bandwidth_name;
  extern const char *atomic_throughput_src;
  extern const char *atomic_throughput_name;
  extern const char *kernel_latency_src;
  extern const char *kernel_latency_name;
}

#endif // ENABLE_ROCM
#endif // ROCM_PEAK_H
