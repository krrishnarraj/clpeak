#ifndef WEBGPU_PEAK_H
#define WEBGPU_PEAK_H

#ifdef ENABLE_WEBGPU

#include <atomic>
#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct CliOptions;
struct WebgpuRuntime;

struct WGPUInstanceImpl;
struct WGPUAdapterImpl;
struct WGPUDeviceImpl;
struct WGPUQueueImpl;
struct WGPUBufferImpl;
struct WGPUTextureImpl;
struct WGPUTextureDescriptor;
struct WGPUComputePipelineImpl;
struct WGPUPipelineLayoutImpl;
struct WGPUBindGroupLayoutImpl;
struct WGPUBindGroupImpl;
struct WGPUCommandBufferImpl;
struct WGPUConstantEntry;

struct wgpu_device_info_t
{
  std::string deviceName;
  std::string vendor;
  std::string architecture;
  std::string driverDescription;
  std::string backendName;
  std::string adapterTypeStr;
  uint32_t vendorID = 0;
  uint32_t deviceID = 0;
  DeviceType deviceType = DeviceType::Unknown;
  bool isIntegratedOrCpu = false;

  uint32_t maxWGSize = 256;
  uint32_t maxWGCount = 65535;
  uint32_t maxTextureDimension2D = 4096;
  uint32_t maxWorkgroupStorageSize = 16384;
  uint64_t maxStorageBufferBindingSize = 128ULL << 20;
  uint64_t maxBufferSize = 256ULL << 20;

  bool float16Supported = false;
  bool packed4x8DotProductSupported = false;
};

static inline uint64_t targetWebgpuGlobalThreads(const wgpu_device_info_t &info)
{
  // WebGPU does not expose a compute-unit count.  Integrated/mobile GPUs and
  // CPU adapters use a smaller floor so the calibration probe never becomes a
  // watchdog-sized dispatch; discrete GPUs use targetGlobalThreads(0).
  if (info.isIntegratedOrCpu)
    return 2ULL << 20;
  return targetGlobalThreads(0);
}

static inline const char *wgpuWidthNote(uint32_t width)
{
  switch (width)
  {
  case 1: return "Scalar, one value per thread.";
  case 2: return "2-wide vector per thread.";
  case 4: return "4-wide vector per thread.";
  default: return "";
  }
}

class WebgpuDevice
{
public:
  // Shared callback state kept alive until the owning WGPUInstance is released,
  // preventing data races or use-after-free if spontaneous device-lost or
  // uncaptured-error callbacks fire from another thread or during teardown.
  struct CallbackState
  {
    std::atomic<bool> deviceLost{false};
    std::atomic<bool> hadError{false};
  };

  const WebgpuRuntime *rt = nullptr;
  WGPUInstanceImpl *instance = nullptr;
  WGPUAdapterImpl *adapter = nullptr;
  WGPUDeviceImpl *device = nullptr;
  WGPUQueueImpl *queue = nullptr;
  wgpu_device_info_t info;
  bool timedWaitAny = false;
  std::shared_ptr<CallbackState> cbState;

  WebgpuDevice() = default;
  ~WebgpuDevice();

  WebgpuDevice(const WebgpuDevice &) = delete;
  WebgpuDevice &operator=(const WebgpuDevice &) = delete;

  bool init(const WebgpuRuntime *runtime, WGPUInstanceImpl *inst,
            WGPUAdapterImpl *adp, bool hasTimedWaitAny,
            std::vector<std::shared_ptr<CallbackState>> &statePool);
  void cleanup();

  bool isLost() const
  {
    return cbState && cbState->deviceLost.load(std::memory_order_acquire);
  }

  bool hasError() const
  {
    return cbState && cbState->hadError.load(std::memory_order_acquire);
  }

  void clearError()
  {
    if (cbState)
      cbState->hadError.store(false, std::memory_order_release);
  }

  WGPUBufferImpl *createBuffer(uint64_t size, uint64_t usage,
                               bool mappedAtCreation = false);

  WGPUTextureImpl *createTexture(const WGPUTextureDescriptor *desc);

  WGPUComputePipelineImpl *createComputePipeline(
      const std::string &wgsl,
      WGPUPipelineLayoutImpl *pipeLayout,
      const WGPUConstantEntry *constants = nullptr,
      size_t constantCount = 0);

  bool submitAndWait(WGPUCommandBufferImpl *cmdBuf);

private:
  void pushErrorScopes();
  bool popErrorScope(std::string *msg);
  bool popErrorScopes(std::string *msg);
};

struct wgpu_compute_variant_t
{
  const char *label;
  std::string wgsl;
  const char *description;
  std::string altWgsl;
};

struct wgpu_compute_desc_t
{
  const char *title = nullptr;
  const char *resultTag = nullptr;
  const char *unit = nullptr;
  const char *description = nullptr;
  TestShape shape = TestShape::Homogeneous;
  const char *axis = nullptr;

  std::vector<wgpu_compute_variant_t> variants;

  uint32_t workPerWI = 0;
  uint32_t elemSize = 4;
  uint32_t wgSize = 256;

  // 16-byte uniform payload (vec4<f32> or vec4<i32>) bound at @binding(1).
  const void *uniformData = nullptr;
  uint32_t uniformSize = 0;

  bool skip = false;
  const char *skipMsg = nullptr;
};

class WebgpuPeak : public Peak
{
public:
  WebgpuPeak();
  ~WebgpuPeak() override;

  static constexpr Backend kBackend = Backend::WebGPU;
  Backend backend() const override { return kBackend; }
  int runAll() override;

  int runComputeSP(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runComputeHP(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runComputeMP(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runComputeInt32(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runComputeInt8DP(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runGlobalBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg);
  int runKernelLatency(WebgpuDevice &dev, benchmark_config_t &cfg);

  static BackendInventory enumerate();

private:
  float runKernel(WebgpuDevice &dev,
                  WGPUComputePipelineImpl *pipeline,
                  WGPUBindGroupImpl *bindGroup,
                  uint32_t groupCountX,
                  unsigned int targetTimeUs,
                  unsigned int forcedIters,
                  bool serialize);

  int runComputeKernel(WebgpuDevice &dev, benchmark_config_t &cfg,
                       const wgpu_compute_desc_t &d);

  logger::DeviceScope *currentDeviceScope = nullptr;
};

// Choose which WebGPU shared library to load ahead of the platform defaults;
// backs `--webgpu-lib`.  Between runs only.
void webgpuSetLibraryOverride(const std::string &path);

// Why the runtime failed to load, ready to show a user; empty when loaded.
std::string webgpuLoadDiagnostic();

#endif // ENABLE_WEBGPU
#endif // WEBGPU_PEAK_H
