#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstring>
#include <chrono>

// ---------------------------------------------------------------------------
// Global memory bandwidth benchmark (Vulkan)
// ---------------------------------------------------------------------------

int vkPeak::runGlobalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  const uint32_t wgSize = 256;

  uint64_t maxItems = dev.info.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = (maxItems / (wgSize * FETCH_PER_WI)) * (wgSize * FETCH_PER_WI);
  if (numItems > cfg.globalBWMaxSize / sizeof(float))
    numItems = (cfg.globalBWMaxSize / sizeof(float) / (wgSize * FETCH_PER_WI)) * (wgSize * FETCH_PER_WI);

  uint32_t numGroups = (uint32_t)(numItems / FETCH_PER_WI / wgSize);
  if (numGroups == 0) numGroups = 1;

  logger::TestSpec testSpec;
  testSpec.tag = "global_memory_bandwidth";
  testSpec.display = "Global memory bandwidth";
  testSpec.unit = "gbps";
  auto test = currentDeviceScope->beginTest(testSpec);

  // Create input + output buffers
  VkBuffer inputBuf, outputBuf;
  VkDeviceMemory inputMem, outputMem;

  if (!dev.createBuffer(numItems * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        inputBuf, inputMem) ||
      !dev.createBuffer(numItems * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        outputBuf, outputMem))
  {
    log->note("Failed to allocate buffers\n");
    return -1;
  }

  // Fill the input buffer with pseudo-random data to defeat hardware memory
  // compression.  Upload via a host-visible staging buffer then copy to the
  // device-local input buffer.
  {
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (!dev.createBuffer(numItems * sizeof(float),
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          stagingBuf, stagingMem))
    {
      log->note("Failed to allocate staging buffer\n");
      vkDestroyBuffer(dev.device, inputBuf, nullptr);
      vkFreeMemory(dev.device, inputMem, nullptr);
      vkDestroyBuffer(dev.device, outputBuf, nullptr);
      vkFreeMemory(dev.device, outputMem, nullptr);
      return -1;
    }

    void *stagingMap = nullptr;
    vkMapMemory(dev.device, stagingMem, 0, numItems * sizeof(float), 0, &stagingMap);
    populate((float *)stagingMap, numItems);
    vkUnmapMemory(dev.device, stagingMem);

    VkCommandBufferAllocateInfo cmdAI = {};
    cmdAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAI.commandPool = dev.commandPool;
    cmdAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAI.commandBufferCount = 1;

    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(dev.device, &cmdAI, &cmdBuf);

    VkCommandBufferBeginInfo cmdBI = {};
    cmdBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &cmdBI);

    VkBufferCopy copyRegion = {};
    copyRegion.size = numItems * sizeof(float);
    vkCmdCopyBuffer(cmdBuf, stagingBuf, inputBuf, 1, &copyRegion);

    vkEndCommandBuffer(cmdBuf);
    dev.submitAndWait(cmdBuf);
    vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &cmdBuf);

    vkDestroyBuffer(dev.device, stagingBuf, nullptr);
    vkFreeMemory(dev.device, stagingMem, nullptr);
  }

  // Descriptor set layout (2 bindings: input + output)
  VkDescriptorSetLayoutBinding bindings[2] = {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dsLayoutCI = {};
  dsLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsLayoutCI.bindingCount = 2;
  dsLayoutCI.pBindings = bindings;

  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dsLayoutCI, nullptr, &dsLayout);

  VkPipelineLayoutCreateInfo pipeLayoutCI = {};
  pipeLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutCI.setLayoutCount = 1;
  pipeLayoutCI.pSetLayouts = &dsLayout;

  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &pipeLayoutCI, nullptr, &pipeLayout);

  struct GBVar { const char *label; const uint32_t *spv; size_t sz; uint32_t width; };
  const GBVar variants[] = {
    {"float ",  vk_shaders::global_bandwidth_v1, vk_shaders::global_bandwidth_v1_size, 1},
#ifdef VK_HAS_GLOBAL_BANDWIDTH_V2
    {"float2",  vk_shaders::global_bandwidth_v2, vk_shaders::global_bandwidth_v2_size, 2},
#endif
#ifdef VK_HAS_GLOBAL_BANDWIDTH_V4
    {"float4",  vk_shaders::global_bandwidth_v4, vk_shaders::global_bandwidth_v4_size, 4},
#endif
  };

  // Descriptor pool + set
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 2;

  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1;
  dpCI.poolSizeCount = 1;
  dpCI.pPoolSizes = &poolSize;

  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool;
  dsAI.descriptorSetCount = 1;
  dsAI.pSetLayouts = &dsLayout;

  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorBufferInfo bufInfos[2] = {};
  bufInfos[0].buffer = inputBuf;
  bufInfos[0].offset = 0;
  bufInfos[0].range = numItems * sizeof(float);
  bufInfos[1].buffer = outputBuf;
  bufInfos[1].offset = 0;
  bufInfos[1].range = numItems * sizeof(float);

  VkWriteDescriptorSet writes[2] = {};
  for (int i = 0; i < 2; i++)
  {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufInfos[i];
  }
  vkUpdateDescriptorSets(dev.device, 2, writes, 0, nullptr);

  // Run each variant.  Same buffer + descriptor set; only the pipeline
  // (i.e. the loaded shader) changes per row.
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
    // Match OpenCL: reduce work-group count by vector width so total bytes
    // touched stays at numItems * sizeof(float) across all variants.
    uint32_t variantGroups = numGroups / v.width;
    if (variantGroups == 0) variantGroups = 1;
    float timed = runKernel(dev, pipe, pipeLayout, descSet, variantGroups,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    std::string key(v.label);
    while (!key.empty() && key.back() == ' ') key.pop_back();
    if (timed <= 0.0f)
    {
      test.skip(key, ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
      vkDestroyPipeline(dev.device, pipe, nullptr);
      continue;
    }
    float gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
    test.emit(key, gbps);
    vkDestroyPipeline(dev.device, pipe, nullptr);
  }

  // Cleanup
  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, inputBuf, nullptr);
  vkFreeMemory(dev.device, inputMem, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  return 0;
}

#endif // ENABLE_VULKAN
