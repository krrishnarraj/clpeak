#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (Vulkan)
// ---------------------------------------------------------------------------
//
// Combined image-sampler descriptor + storage-buffer output.  Image is
// VK_FORMAT_R32G32B32A32_SFLOAT, sampled with NEAREST + CLAMP_TO_EDGE.

int vkPeak::runImageBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  logger::TestSpec testSpec;
  testSpec.tag = "image_memory_bandwidth";
  testSpec.display = "Image memory bandwidth";
  testSpec.unit = "gbps";
  auto test = currentDeviceScope->beginTest(testSpec);

  const uint32_t imgW = 4096, imgH = 4096;
  const uint32_t wgSize = 256;
  uint64_t globalWIs = targetVulkanGlobalThreads(dev.info);
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);
  uint64_t outBytes  = globalWIs * sizeof(float);

  // Create image (RGBA32F, sampled, transfer-dst so we can clear it).
  VkImageCreateInfo imgCI = {};
  imgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgCI.imageType = VK_IMAGE_TYPE_2D;
  imgCI.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
  imgCI.extent    = {imgW, imgH, 1};
  imgCI.mipLevels = 1; imgCI.arrayLayers = 1;
  imgCI.samples   = VK_SAMPLE_COUNT_1_BIT;
  imgCI.tiling    = VK_IMAGE_TILING_OPTIMAL;
  imgCI.usage     = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImage img;
  if (vkCreateImage(dev.device, &imgCI, nullptr, &img) != VK_SUCCESS)
  {
    log->note("Image create failed\n");
    return -1;
  }

  // Allocate device-local memory for the image.
  VkMemoryRequirements imReq;
  vkGetImageMemoryRequirements(dev.device, img, &imReq);
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(dev.physicalDevice, &memProps);
  uint32_t typeIdx = UINT32_MAX;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    if ((imReq.memoryTypeBits & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    { typeIdx = i; break; }

  VkMemoryAllocateInfo aI = {};
  aI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  aI.allocationSize = imReq.size;
  aI.memoryTypeIndex = typeIdx;
  VkDeviceMemory imgMem;
  vkAllocateMemory(dev.device, &aI, nullptr, &imgMem);
  vkBindImageMemory(dev.device, img, imgMem, 0);

  // Transition UNDEFINED -> SHADER_READ_ONLY_OPTIMAL.  We don't need to
  // upload any data; the image contents being unspecified is fine for a
  // bandwidth measurement (the cache lines still get fetched).
  VkCommandBufferAllocateInfo cbAI = {};
  cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAI.commandPool = dev.commandPool;
  cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAI.commandBufferCount = 1;
  VkCommandBuffer transCmd;
  vkAllocateCommandBuffers(dev.device, &cbAI, &transCmd);
  VkCommandBufferBeginInfo cbBI = {};
  cbBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbBI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(transCmd, &cbBI);
  VkImageMemoryBarrier b = {};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(transCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &b);
  vkEndCommandBuffer(transCmd);
  dev.submitAndWait(transCmd);
  vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &transCmd);

  VkImageViewCreateInfo ivCI = {};
  ivCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ivCI.image = img; ivCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ivCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  ivCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView imgView;
  vkCreateImageView(dev.device, &ivCI, nullptr, &imgView);

  VkSamplerCreateInfo smCI = {};
  smCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  smCI.magFilter = VK_FILTER_NEAREST;
  smCI.minFilter = VK_FILTER_NEAREST;
  smCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  smCI.addressModeU = smCI.addressModeV = smCI.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  smCI.unnormalizedCoordinates = VK_FALSE;
  VkSampler sampler;
  vkCreateSampler(dev.device, &smCI, nullptr, &sampler);

  VkBuffer outBuf; VkDeviceMemory outMem;
  if (!dev.createBuffer(outBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem))
  {
    log->note("Output buffer alloc failed\n");
    vkDestroySampler(dev.device, sampler, nullptr);
    vkDestroyImageView(dev.device, imgView, nullptr);
    vkDestroyImage(dev.device, img, nullptr);
    vkFreeMemory(dev.device, imgMem, nullptr);
    return -1;
  }

  VkDescriptorSetLayoutBinding bs[2] = {};
  bs[0].binding = 0; bs[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bs[0].descriptorCount = 1; bs[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bs[1].binding = 1; bs[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bs[1].descriptorCount = 1; bs[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslCI = {};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 2; dslCI.pBindings = bs;
  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dslCI, nullptr, &dsLayout);

  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsLayout;
  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pipeLayout);

  VkDescriptorPoolSize ps[2] = {
    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
  };
  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1; dpCI.poolSizeCount = 2; dpCI.pPoolSizes = ps;
  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool; dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &dsLayout;
  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorImageInfo ii = {};
  ii.imageView = imgView; ii.sampler = sampler;
  ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkDescriptorBufferInfo bi = {outBuf, 0, outBytes};
  VkWriteDescriptorSet ws[2] = {};
  ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  ws[0].dstSet = descSet; ws[0].dstBinding = 0; ws[0].descriptorCount = 1;
  ws[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  ws[0].pImageInfo = &ii;
  ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  ws[1].dstSet = descSet; ws[1].dstBinding = 1; ws[1].descriptorCount = 1;
  ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ws[1].pBufferInfo = &bi;
  vkUpdateDescriptorSets(dev.device, 2, ws, 0, nullptr);

  VkPipeline pipe;
  bool ok = dev.createComputePipeline(vk_shaders::image_bandwidth_v1,
      vk_shaders::image_bandwidth_v1_size, dsLayout, pipeLayout, pipe);
  if (!ok)
  {
    test.skip("float4", ResultStatus::Error, "Pipeline creation failed");
  }
  else
  {
    float us = runKernel(dev, pipe, pipeLayout, descSet, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      test.skip("float4", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
    }
    else
    {
      uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalWIs;
      float gbps = (float)bytes / us / 1e3f;
      test.emit("float4", gbps);
    }
    vkDestroyPipeline(dev.device, pipe, nullptr);
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outBuf, nullptr);
  vkFreeMemory(dev.device, outMem, nullptr);
  vkDestroySampler(dev.device, sampler, nullptr);
  vkDestroyImageView(dev.device, imgView, nullptr);
  vkDestroyImage(dev.device, img, nullptr);
  vkFreeMemory(dev.device, imgMem, nullptr);
  return 0;
}

#endif // ENABLE_VULKAN
