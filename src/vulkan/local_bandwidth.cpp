#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstring>

// ---------------------------------------------------------------------------
// Local memory bandwidth (Vulkan -- shared memory)
// ---------------------------------------------------------------------------
//
// Same single-output-buffer scaffolding as runComputeKernel; only the bytes-
// per-WI calculation differs per variant.  Width = 1/2/4/8 floats per slot.

int vkPeak::runLocalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  logger::TestSpec testSpec;
  testSpec.tag = "local_memory_bandwidth";
  testSpec.display = "Local memory bandwidth";
  testSpec.unit = "gbps";
  auto test = currentDeviceScope->beginTest(testSpec);

  const uint32_t wgSize = 256;
  uint64_t globalWIs = targetVulkanGlobalThreads(dev.info);
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);
  uint64_t bufferBytes = (uint64_t)globalWIs * sizeof(float);

  VkBuffer outBuf;
  VkDeviceMemory outMem;
  if (!dev.createBuffer(bufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem))
  {
    CLPEAK_VLOG("Failed to allocate buffer\n");
    return -1;
  }

  // Descriptor set layout: 1 storage buffer.
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslCI = {};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 1; dslCI.pBindings = &binding;
  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dslCI, nullptr, &dsLayout);

  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsLayout;
  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pipeLayout);

  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &ps;
  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool; dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &dsLayout;
  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorBufferInfo bi = {outBuf, 0, bufferBytes};
  VkWriteDescriptorSet w = {};
  w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w.dstSet = descSet; w.dstBinding = 0; w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);

  struct V { const char *label; const uint32_t *spv; size_t sz; uint32_t width; };
  const V variants[] = {
    {"float  ", vk_shaders::local_bandwidth_v1, vk_shaders::local_bandwidth_v1_size, 1},
    {"float2 ", vk_shaders::local_bandwidth_v2, vk_shaders::local_bandwidth_v2_size, 2},
    {"float4 ", vk_shaders::local_bandwidth_v4, vk_shaders::local_bandwidth_v4_size, 4},
  };
  for (const auto &v : variants)
  {
    VkPipeline pipe;
    if (!dev.createComputePipeline(v.spv, v.sz, dsLayout, pipeLayout, pipe))
    {
      std::string key(v.label);
      while (!key.empty() && key.back() == ' ') key.pop_back();
      test.skip(key, ResultStatus::Error, "Pipeline creation failed");
      continue;
    }
    float us = runKernel(dev, pipe, pipeLayout, descSet, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    // Strip padding from the display label for the result metric key.
    std::string key(v.label);
    while (!key.empty() && key.back() == ' ') key.pop_back();
    if (us <= 0.0f)
    {
      test.skip(key, ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
      vkDestroyPipeline(dev.device, pipe, nullptr);
      continue;
    }
    // Each rep: 1 write + 1 read per WI = 2 * width * sizeof(float) bytes.
    uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.width * sizeof(float) * globalWIs;
    float gbps = (float)bytes / us / 1e3f;
    test.emit(key, gbps);
    vkDestroyPipeline(dev.device, pipe, nullptr);
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outBuf, nullptr);
  vkFreeMemory(dev.device, outMem, nullptr);
  return 0;
}

#endif // ENABLE_VULKAN
