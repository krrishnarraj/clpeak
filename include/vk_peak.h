#ifndef VK_PEAK_H
#define VK_PEAK_H

#ifdef ENABLE_VULKAN

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <bitset>
#include <common.h>
#include <benchmark_constants.h>
#include <logger.h>
#include <clpeak.h>      // Benchmark enum

struct CliOptions; // forward decl

// Convenience: defined if any cooperative-matrix shader compiled.  Used by
// vk_peak.cpp to gate extension / feature enablement and dispatch.
#if defined(CLPEAK_VK_HAS_COOPMAT_FP8_E4M3) || defined(CLPEAK_VK_HAS_COOPMAT_FP8_E5M2)
#define CLPEAK_VK_HAS_ANY_COOPMAT_FP8 1
#endif
#if defined(CLPEAK_VK_HAS_COOPMAT_FP16) || defined(CLPEAK_VK_HAS_COOPMAT_BF16) || defined(CLPEAK_VK_HAS_COOPMAT_INT8) || defined(CLPEAK_VK_HAS_ANY_COOPMAT_FP8)
#define CLPEAK_VK_HAS_ANY_COOPMAT 1
#endif

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

  VkPhysicalDeviceType deviceType;
  uint32_t computeQueueFamily;

  // Optional feature / extension gates
  bool int8DotProductSupported;   // VK_KHR_shader_integer_dot_product + shaderInt8
  bool float16Supported;          // VK_KHR_shader_float16_int8::shaderFloat16
  bool bfloat16Supported;         // VK_KHR_shader_bfloat16::shaderBFloat16Type
  bool cooperativeMatrixSupported;// VK_KHR_cooperative_matrix + cooperativeMatrix
  bool fp8Supported;              // VK_EXT_shader_float8 + shaderFloat8CoopMatrix

  // Cached subset of vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR used
  // to gate individual coopmat dtype tests.  Each flag means "a subgroup-
  // scope property exists advertising this input/accumulator combination
  // at a tile we have a pre-compiled shader for".
  //
  // FP16/BF16/FP8 tests are compiled at 16x16x16 (widely supported on
  // NVIDIA/AMD/Intel).  INT8 is trickier: NVIDIA tensor cores advertise
  // INT8 only at K=32 (typically 16x16x32), reflecting how DP4a lanes
  // naturally accumulate four INT8s per 32-bit slot -- so we ship two
  // INT8 shader variants (K=16, K=32) and select whichever matches.
  bool coopmatFP16Supported;      // fp16 A/B, fp32 C  @ 16x16x16
  bool coopmatBF16Supported;      // bf16 A/B, fp32 C  @ 16x16x16
  bool coopmatFP8E4M3Supported;   // fp8 E4M3 A/B, fp32 C  @ 16x16x16
  bool coopmatFP8E5M2Supported;   // fp8 E5M2 A/B, fp32 C  @ 16x16x16
  // INT8: store the selected K so host can pick the right shader variant.
  // 0 = unsupported; 16 or 32 = supported at 16x16xK.
  uint32_t coopmatINT8K;
};

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

  // Create compute pipeline from SPIR-V
  bool createComputePipeline(const uint32_t *spirv, size_t spirvSize,
                             VkDescriptorSetLayout dsLayout,
                             VkPipelineLayout pipeLayout,
                             VkPipeline &pipeline);

  // Submit a command buffer and wait
  void submitAndWait(VkCommandBuffer cmdBuf);

private:
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

// Describes a single compute-peak benchmark: just the bits that differ
// between runComputeSP / MP / INT8-DP / INT4-packed / etc.  The Vulkan
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
  const char *label;         // column + xmlRecord key, e.g. "mp", "mp2", "mp4"
  const uint32_t *spirv;
  size_t spirvSize;
};

struct vk_compute_desc_t
{
  // Display / reporting
  const char *title;         // e.g. "Single-precision compute (GFLOPS)"
  const char *xmlTag;        // e.g. "single_precision_compute"
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
                             // (one element per WI).  Coopmat shaders write
                             // an M*N tile = 256 elements per WG.

  // Push-constant payload.  nullptr => no push constants bound.
  const void *pushData;
  uint32_t pushSize;

  // Optional feature gate.  If skip==true, emit skipMsg and close the tag.
  bool skip;
  const char *skipMsg;

  // Optional extra xml attribute (e.g. emulated="true" for packed INT4).
  const char *extraAttribKey;
  const char *extraAttribVal;
};

// Top-level Vulkan benchmark runner
class vkPeak
{
public:
  std::unique_ptr<logger> log;
  unsigned int warmupCount;
  unsigned int specifiedIters;
  bool forceIters;
  bool listDevices;
  int  deviceIndex; // -1 = run all

  std::bitset<static_cast<size_t>(Benchmark::COUNT)> enabledTests;
  bool isTestEnabled(Benchmark b) const
  { return enabledTests.test(static_cast<size_t>(b)); }

  vkPeak();
  ~vkPeak();

  void applyOptions(const CliOptions &opts);
  int runAll();

  // Individual benchmarks
  int runComputeSP(VulkanDevice &dev, benchmark_config_t &cfg);
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V1
  int runComputeMP(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
  int runComputeInt8DP(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT4_PACKED_V1
  int runComputeInt4Packed(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V1
  int runComputeBF16(VulkanDevice &dev, benchmark_config_t &cfg);
#endif
  // Cooperative matrix (tensor-core) umbrella -- runs each advertised dtype.
  int runCoopMatrix(VulkanDevice &dev, benchmark_config_t &cfg);
  int runGlobalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);
  int runAtomicThroughput(VulkanDevice &dev, benchmark_config_t &cfg);

private:
  VkInstance instance;
  std::vector<VkPhysicalDevice> physicalDevices;

  bool initInstance();
  void cleanup();

  // Timing helper: run a compute dispatch iters times, return avg time in us
  float runKernel(VulkanDevice &dev, VkPipeline pipeline,
                  VkPipelineLayout pipeLayout,
                  VkDescriptorSet descriptorSet,
                  uint32_t groupCountX, unsigned int iters,
                  const void *pushData = nullptr, uint32_t pushSize = 0);

  // Shared implementation of the single-buffer compute-peak pattern
  // used by every runCompute* benchmark.  Returns 0 on success (including
  // a clean skip) and -1 if buffer allocation itself failed.
  int runComputeKernel(VulkanDevice &dev, benchmark_config_t &cfg,
                       const vk_compute_desc_t &d);
};

// Embedded SPIR-V shader data (generated at build time)
namespace vk_shaders {
  extern const uint32_t compute_sp_v1[];
  extern const size_t   compute_sp_v1_size;
  extern const uint32_t global_bandwidth_v1[];
  extern const size_t   global_bandwidth_v1_size;
  extern const uint32_t local_bandwidth_v1[];
  extern const size_t   local_bandwidth_v1_size;
  extern const uint32_t local_bandwidth_v2[];
  extern const size_t   local_bandwidth_v2_size;
  extern const uint32_t local_bandwidth_v4[];
  extern const size_t   local_bandwidth_v4_size;
  extern const uint32_t local_bandwidth_v8[];
  extern const size_t   local_bandwidth_v8_size;
  extern const uint32_t image_bandwidth_v1[];
  extern const size_t   image_bandwidth_v1_size;
  extern const uint32_t atomic_throughput_global[];
  extern const size_t   atomic_throughput_global_size;
  extern const uint32_t atomic_throughput_local[];
  extern const size_t   atomic_throughput_local_size;
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
  extern const uint32_t compute_int8_dp_v1[];
  extern const size_t   compute_int8_dp_v1_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V2
  extern const uint32_t compute_int8_dp_v2[];
  extern const size_t   compute_int8_dp_v2_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V4
  extern const uint32_t compute_int8_dp_v4[];
  extern const size_t   compute_int8_dp_v4_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V1
  extern const uint32_t compute_mp_v1[];
  extern const size_t   compute_mp_v1_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V2
  extern const uint32_t compute_mp_v2[];
  extern const size_t   compute_mp_v2_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V4
  extern const uint32_t compute_mp_v4[];
  extern const size_t   compute_mp_v4_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT4_PACKED_V1
  extern const uint32_t compute_int4_packed_v1[];
  extern const size_t   compute_int4_packed_v1_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V1
  extern const uint32_t compute_bf16_v1[];
  extern const size_t   compute_bf16_v1_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V2
  extern const uint32_t compute_bf16_v2[];
  extern const size_t   compute_bf16_v2_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V4
  extern const uint32_t compute_bf16_v4[];
  extern const size_t   compute_bf16_v4_size;
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_FP16
  extern const uint32_t coopmat_fp16[];
  extern const size_t   coopmat_fp16_size;
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_BF16
  extern const uint32_t coopmat_bf16[];
  extern const size_t   coopmat_bf16_size;
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_INT8
  extern const uint32_t coopmat_int8[];       // 16x16x16 tile (AMD/Intel path)
  extern const size_t   coopmat_int8_size;
  extern const uint32_t coopmat_int8_k32[];   // 16x16x32 tile (NVIDIA tensor-core INT8)
  extern const size_t   coopmat_int8_k32_size;
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_FP8_E4M3
  extern const uint32_t coopmat_fp8_e4m3[];
  extern const size_t   coopmat_fp8_e4m3_size;
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_FP8_E5M2
  extern const uint32_t coopmat_fp8_e5m2[];
  extern const size_t   coopmat_fp8_e5m2_size;
#endif
}

#endif // ENABLE_VULKAN
#endif // VK_PEAK_H
