#ifndef VK_PEAK_H
#define VK_PEAK_H

#ifdef ENABLE_VULKAN

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <bitset>
#include <common/common.h>
#include <common/logger.h>
#include <common/peak.h>
#include <common/inventory.h>

struct CliOptions;       // forward decl
struct BackendInventory; // forward decl

// Convenience: defined if any cooperative-matrix shader compiled.  Used by
// vk_peak.cpp to gate extension / feature enablement and dispatch.
#if defined(VK_HAS_COOPMAT_FP8_E4M3) || defined(VK_HAS_COOPMAT_FP8_E5M2)
#define VK_HAS_ANY_COOPMAT_FP8 1
#endif
#if defined(VK_HAS_COOPMAT_FP16) || defined(VK_HAS_COOPMAT_BF16) || defined(VK_HAS_COOPMAT_INT8) || defined(VK_HAS_ANY_COOPMAT_FP8) || defined(VK_HAS_COOPMAT_FP32)
#define VK_HAS_ANY_COOPMAT 1
#endif

// One cooperative-matrix tile (subgroup scope) selected for a given dtype.
// M/N/K are whatever the driver advertised via
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR -- the shaders take these
// as specialization constants, so a single SPIR-V module runs any advertised
// shape (NVIDIA 8-bit types use K=32, fp16/bf16 use K=16, etc.).
struct coopmat_tile_t {
  bool     supported = false;
  uint32_t M = 0, N = 0, K = 0;
};

// Vulkan device info (mirrors OpenCL device_info_t for display)
struct vk_device_info_t {
  std::string deviceName;
  std::string driverVersion;
  std::string apiVersion;

  unsigned int numCUs;            // maxComputeWorkGroupCount[0] as proxy
  unsigned int maxWGSize;
  uint64_t maxAllocSize;          // maxStorageBufferRange
  uint64_t heapSize;              // device-local heap
  unsigned int maxClockFreq;      // not always available (0 if unknown)

  VkPhysicalDeviceType vkDeviceType;
  DeviceType deviceType = DeviceType::Unknown;
  uint32_t computeQueueFamily;

  // Optional feature / extension gates
  bool int8DotProductSupported;   // VK_KHR_shader_integer_dot_product + shaderInt8
  bool float16Supported;          // VK_KHR_shader_float16_int8::shaderFloat16
  bool float64Supported;          // VkPhysicalDeviceFeatures::shaderFloat64
  bool bfloat16Supported;         // VK_KHR_shader_bfloat16::shaderBFloat16Type
  bool cooperativeMatrixSupported;// VK_KHR_cooperative_matrix + cooperativeMatrix
  bool fp8Supported;              // VK_EXT_shader_float8 + shaderFloat8CoopMatrix
  bool calibratedTimestampsSupported; // VK_EXT_calibrated_timestamps

  // Canonical cooperative-matrix tile selected per dtype from
  // vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.  We don't assume a
  // fixed shape: whatever subgroup-scope tile the driver advertises for an
  // input/accumulator combination is recorded here, and the coopmat shaders
  // take M/N/K as specialization constants so one SPIR-V module runs it.
  // This is how fp8/int8 land on NVIDIA's K=32 tiles while fp16/bf16 stay at
  // K=16 -- no per-K shader variants needed.  .supported == false means no
  // matching subgroup-scope property was advertised.
  coopmat_tile_t coopmatFP32;     // fp32 A/B,    fp32 C
  coopmat_tile_t coopmatFP16;     // fp16 A/B,    fp32 C
  coopmat_tile_t coopmatBF16;     // bf16 A/B,    fp32 C
  coopmat_tile_t coopmatFP8E4M3;  // fp8 E4M3 A/B, fp32 C
  coopmat_tile_t coopmatFP8E5M2;  // fp8 E5M2 A/B, fp32 C
  coopmat_tile_t coopmatINT8;     // int8 A/B,    int32 C
};

// Dispatch-sizing helper used by runComputeKernel and several benchmark
// files (local_bandwidth, image_bandwidth, etc.).
static inline uint64_t targetVulkanGlobalThreads(const vk_device_info_t &info)
{
  if (info.numCUs > 0)
    return targetGlobalThreads(info.numCUs);

  // Mobile/integrated Vulkan drivers often do not expose a vendor CU-count
  // property.  The desktop 32M fallback can make the calibration probe itself
  // a multi-second dispatch on those GPUs, so start smaller and let timed
  // calibration batch more dispatches when the kernel is fast enough.
  if (info.vkDeviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
      info.vkDeviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
    return 2ULL << 20;

  return targetGlobalThreads(0);
}

// Manages a single Vulkan device for benchmarking
class VulkanDevice
{
public:
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkQueue computeQueue;
  VkCommandPool commandPool;
  vk_device_info_t info;

  VulkanDevice();
  ~VulkanDevice();

  bool init(VkInstance inst, VkPhysicalDevice physDev);
  void cleanup();

  // Allocate a device-local buffer
  bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags memProps,
                    VkBuffer &buffer, VkDeviceMemory &memory);

  // Create compute pipeline from SPIR-V.  specInfo (optional) supplies
  // specialization constants -- used by the coopmat shaders to bind the
  // selected M/N/K tile + loop count at pipeline-creation time.
  bool createComputePipeline(const uint32_t *spirv, size_t spirvSize,
                             VkDescriptorSetLayout dsLayout,
                             VkPipelineLayout pipeLayout,
                             VkPipeline &pipeline,
                             const VkSpecializationInfo *specInfo = nullptr);

  // Submit a command buffer and wait.  Returns the worst VkResult seen
  // across vkQueueSubmit / vkQueueWaitIdle so callers can detect and skip
  // out-of-spec drivers (e.g. Adreno/Turnip silently losing the device on
  // shaders that use advertised-but-unsupported features).
  VkResult submitAndWait(VkCommandBuffer cmdBuf);

  // Clear a transfer-dst buffer to zero and make the writes visible to
  // following compute shader dispatches.
  bool zeroBuffer(VkBuffer buffer);

private:
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

// Describes a single compute-peak benchmark: just the bits that differ
// between runComputeSP / MP / INT8-DP / etc.  The Vulkan
// scaffolding (buffer, descriptors, pipeline, dispatch loop, cleanup) is
// identical across all of them and lives in vkPeak::runComputeKernel.
//
// A benchmark can declare one or more *variants* (typically vector widths
// v1/v2/v4 matching the OpenCL kernels); they share the output buffer and
// descriptor set but each has its own SPIR-V module and metric row.  If
// variants == nullptr, runComputeKernel falls back to the single-variant
// (metricLabel + spirv + spirvSize) fields.
struct vk_compute_variant_t
{
  const char *label;         // column + result metric, e.g. "mp", "mp2", "mp4"
  const uint32_t *spirv;
  size_t spirvSize;
};

struct vk_compute_desc_t
{
  // Display / reporting
  const char *title;         // e.g. "Single-precision compute (GFLOPS)"
  const char *resultTag;        // e.g. "single_precision_compute"
  const char *metricLabel;   // used when variants==nullptr
  const char *unit;          // "gflops" / "gops" / "tflops" / "tops"
  double      unitDivider;   // 1e9 = G* (default when 0), 1e12 = T*

  // Single-variant shader (used when variants == nullptr)
  const uint32_t *spirv;
  size_t spirvSize;

  // Multi-variant shader list (takes precedence over single-variant fields)
  const vk_compute_variant_t *variants;
  uint32_t numVariants;

  // Scaling
  uint32_t workPerWI;        // matches the kernel's per-WI op budget
  uint32_t elemSize;         // output element size (sizeof float / int32 / ...)
  uint32_t wgSize;           // local_size_x in the shader; 0 => use default 256.
                             // Cooperative-matrix shaders use 32 (one subgroup).
  uint32_t outElemsPerWG;    // number of output buffer elements the shader
                             // writes per work-group.  0 => defaults to wgSize
                             // (one element per WI).  Coopmat shaders write an
                             // M*N tile (the selected tile's M*N) per WG.

  // Push-constant payload.  nullptr => no push constants bound.
  const void *pushData;
  uint32_t pushSize;

  // Optional specialization constants, applied to every variant's pipeline.
  // Used by the coopmat shaders to bind the selected M/N/K tile + loop count.
  // nullptr => none.
  const VkSpecializationInfo *specInfo;

  // Optional feature gate.  If skip==true, emit skipMsg and close the tag.
  bool skip;
  const char *skipMsg;

};

// Top-level Vulkan benchmark runner
class vkPeak : public Peak
{
public:
  std::vector<int> deviceIndices; // empty = run all

  vkPeak();
  ~vkPeak();

  void applyOptions(const CliOptions &opts) override;
  int runAll() override;

  // Individual benchmarks
  int runComputeSP(VulkanDevice &dev, benchmark_config_t &cfg);
#ifdef VK_HAS_COMPUTE_HP_V1
  int runComputeHP(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef VK_HAS_COMPUTE_DP_V1
  int runComputeDP(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef VK_HAS_COMPUTE_INT32_V1
  int runComputeInt32(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef VK_HAS_COMPUTE_MP_V1
  int runComputeMP(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
  int runComputeInt8DP(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef VK_HAS_COMPUTE_BF16_V1
  int runComputeBF16(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
  // Cooperative matrix (tensor-core) umbrella -- runs each advertised dtype.
  int runCoopMatrix(VulkanDevice &dev, benchmark_config_t &cfg, bool intPart = false);
  int runGlobalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);
  int runKernelLatency(VulkanDevice &dev, benchmark_config_t &cfg);

  static BackendInventory enumerate();
  static void printInventory(const BackendInventory &inv, std::ostream &os);

private:
  VkInstance instance;
  std::vector<VkPhysicalDevice> physicalDevices;

  bool initInstance();
  void cleanup();

  // Time a compute dispatch batched as `iters` dispatches, where `iters` is
  // calibrated from a one-shot warmup so the timed phase lands at
  // ~targetTimeUs.  Returns mean per-iter time in microseconds.  forcedIters
  // != 0 short-circuits calibration (matches --iters).
  float runKernel(VulkanDevice &dev, VkPipeline pipeline,
                  VkPipelineLayout pipeLayout,
                  VkDescriptorSet descriptorSet,
                  uint32_t groupCountX,
                  unsigned int targetTimeUs, unsigned int forcedIters,
                  const void *pushData = nullptr, uint32_t pushSize = 0);

  // Shared implementation of the single-buffer compute-peak pattern
  // used by every runCompute* benchmark.  Returns 0 on success (including
  // a clean skip) and -1 if buffer allocation itself failed.
  int runComputeKernel(VulkanDevice &dev, benchmark_config_t &cfg,
                       const vk_compute_desc_t &d);

  // Points to the current device scope during runAll().  Individual
  // benchmark methods use it to open TestScopes.
  logger::DeviceScope *currentDeviceScope = nullptr;
};

// Embedded SPIR-V shader data (generated at build time)
namespace vk_shaders {
  extern const uint32_t compute_sp_v1[];
  extern const size_t   compute_sp_v1_size;
#ifdef VK_HAS_COMPUTE_SP_V2
  extern const uint32_t compute_sp_v2[];
  extern const size_t   compute_sp_v2_size;
#endif
#ifdef VK_HAS_COMPUTE_SP_V4
  extern const uint32_t compute_sp_v4[];
  extern const size_t   compute_sp_v4_size;
#endif
#ifdef VK_HAS_COMPUTE_HP_V1
  extern const uint32_t compute_hp_v1[];
  extern const size_t   compute_hp_v1_size;
#endif
#ifdef VK_HAS_COMPUTE_HP_V2
  extern const uint32_t compute_hp_v2[];
  extern const size_t   compute_hp_v2_size;
#endif
#ifdef VK_HAS_COMPUTE_HP_V4
  extern const uint32_t compute_hp_v4[];
  extern const size_t   compute_hp_v4_size;
#endif
#ifdef VK_HAS_COMPUTE_DP_V1
  extern const uint32_t compute_dp_v1[];
  extern const size_t   compute_dp_v1_size;
#endif
#ifdef VK_HAS_COMPUTE_DP_V2
  extern const uint32_t compute_dp_v2[];
  extern const size_t   compute_dp_v2_size;
#endif
#ifdef VK_HAS_COMPUTE_DP_V4
  extern const uint32_t compute_dp_v4[];
  extern const size_t   compute_dp_v4_size;
#endif
#ifdef VK_HAS_COMPUTE_INT32_V1
  extern const uint32_t compute_int32_v1[];
  extern const size_t   compute_int32_v1_size;
#endif
#ifdef VK_HAS_COMPUTE_INT32_V2
  extern const uint32_t compute_int32_v2[];
  extern const size_t   compute_int32_v2_size;
#endif
#ifdef VK_HAS_COMPUTE_INT32_V4
  extern const uint32_t compute_int32_v4[];
  extern const size_t   compute_int32_v4_size;
#endif
  extern const uint32_t global_bandwidth_v1[];
  extern const size_t   global_bandwidth_v1_size;
#ifdef VK_HAS_GLOBAL_BANDWIDTH_V2
  extern const uint32_t global_bandwidth_v2[];
  extern const size_t   global_bandwidth_v2_size;
#endif
#ifdef VK_HAS_GLOBAL_BANDWIDTH_V4
  extern const uint32_t global_bandwidth_v4[];
  extern const size_t   global_bandwidth_v4_size;
#endif
  extern const uint32_t local_bandwidth_v1[];
  extern const size_t   local_bandwidth_v1_size;
  extern const uint32_t local_bandwidth_v2[];
  extern const size_t   local_bandwidth_v2_size;
  extern const uint32_t local_bandwidth_v4[];
  extern const size_t   local_bandwidth_v4_size;
  extern const uint32_t image_bandwidth_v1[];
  extern const size_t   image_bandwidth_v1_size;
  extern const uint32_t kernel_latency[];
  extern const size_t   kernel_latency_size;
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
  extern const uint32_t compute_int8_dp_v1[];
  extern const size_t   compute_int8_dp_v1_size;
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V2
  extern const uint32_t compute_int8_dp_v2[];
  extern const size_t   compute_int8_dp_v2_size;
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V4
  extern const uint32_t compute_int8_dp_v4[];
  extern const size_t   compute_int8_dp_v4_size;
#endif
#ifdef VK_HAS_COMPUTE_MP_V1
  extern const uint32_t compute_mp_v1[];
  extern const size_t   compute_mp_v1_size;
#endif
#ifdef VK_HAS_COMPUTE_MP_V2
  extern const uint32_t compute_mp_v2[];
  extern const size_t   compute_mp_v2_size;
#endif
#ifdef VK_HAS_COMPUTE_MP_V4
  extern const uint32_t compute_mp_v4[];
  extern const size_t   compute_mp_v4_size;
#endif
#ifdef VK_HAS_COMPUTE_BF16_V1
  extern const uint32_t compute_bf16_v1[];
  extern const size_t   compute_bf16_v1_size;
#endif
#ifdef VK_HAS_COMPUTE_BF16_V2
  extern const uint32_t compute_bf16_v2[];
  extern const size_t   compute_bf16_v2_size;
#endif
#ifdef VK_HAS_COMPUTE_BF16_V4
  extern const uint32_t compute_bf16_v4[];
  extern const size_t   compute_bf16_v4_size;
#endif
#ifdef VK_HAS_COOPMAT_FP16
  extern const uint32_t coopmat_fp16[];
  extern const size_t   coopmat_fp16_size;
#endif
#ifdef VK_HAS_COOPMAT_BF16
  extern const uint32_t coopmat_bf16[];
  extern const size_t   coopmat_bf16_size;
#endif
#ifdef VK_HAS_COOPMAT_INT8
  extern const uint32_t coopmat_int8[];       // M/N/K bound via spec constants
  extern const size_t   coopmat_int8_size;
#endif
#ifdef VK_HAS_COOPMAT_FP8_E4M3
  extern const uint32_t coopmat_fp8_e4m3[];
  extern const size_t   coopmat_fp8_e4m3_size;
#endif
#ifdef VK_HAS_COOPMAT_FP8_E5M2
  extern const uint32_t coopmat_fp8_e5m2[];
  extern const size_t   coopmat_fp8_e5m2_size;
#endif
#ifdef VK_HAS_COOPMAT_FP32
  extern const uint32_t coopmat_fp32[];
  extern const size_t   coopmat_fp32_size;
#endif
}

#endif // ENABLE_VULKAN
#endif // VK_PEAK_H
