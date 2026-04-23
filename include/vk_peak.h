#ifndef VK_PEAK_H
#define VK_PEAK_H

#ifdef ENABLE_VULKAN

#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <memory>
#include <common.h>
#include <benchmark_constants.h>
#include <logger.h>

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

  bool init(VkPhysicalDevice physDev);
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

// Top-level Vulkan benchmark runner
class vkPeak
{
public:
  std::unique_ptr<logger> log;
  unsigned int warmupCount;
  unsigned int specifiedIters;
  bool forceIters;
  bool listDevices;

  vkPeak();
  ~vkPeak();

  int parseArgs(int argc, char **argv);
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
  int runGlobalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg);

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
};

// Embedded SPIR-V shader data (generated at build time)
namespace vk_shaders {
  extern const uint32_t compute_sp_v1[];
  extern const size_t   compute_sp_v1_size;
  extern const uint32_t global_bandwidth_v1[];
  extern const size_t   global_bandwidth_v1_size;
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
  extern const uint32_t compute_int8_dp_v1[];
  extern const size_t   compute_int8_dp_v1_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V1
  extern const uint32_t compute_mp_v1[];
  extern const size_t   compute_mp_v1_size;
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT4_PACKED_V1
  extern const uint32_t compute_int4_packed_v1[];
  extern const size_t   compute_int4_packed_v1_size;
#endif
}

#endif // ENABLE_VULKAN
#endif // VK_PEAK_H
