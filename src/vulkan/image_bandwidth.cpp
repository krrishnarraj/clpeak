#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <algorithm>
#include <cstddef>

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (Vulkan)
// ---------------------------------------------------------------------------
//
// Combined image-sampler descriptor + storage-buffer output.  Image is
// VK_FORMAT_R32G32B32A32_SFLOAT, read with texelFetch -- integer coordinates,
// no sampler -- to match what every other backend's image read compiles to.
// The sampler below exists only because GLSL's sampler2D carries one; the
// shader never samples through it.
//
// Four walk shapes are raced and the fastest reported -- why, and why none of
// them can flatter the result: the image-bandwidth block in
// include/common/common.h.

int vkPeak::runImageBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  logger::TestSpec testSpec;
  testSpec.tag = "image_memory_bandwidth";
  testSpec.display = "Image memory bandwidth";
  testSpec.unit = "bps";
  testSpec.description =
      "How many bytes per second the device reads through its texture units, "
      "which take a different path to memory than plain buffer reads.  Each "
      "pixel of the image is read exactly once, so caching cannot flatter the "
      "number.";
  testSpec.shape = TestShape::Homogeneous;
  auto test = currentDeviceScope->beginTest(testSpec);

  const uint32_t imgW = 4096, imgH = 4096;
  const uint32_t wgSize = 256;
  // Size the dispatch so each pixel is read exactly once per launch,
  // eliminating cache reuse that inflates apparent bandwidth.
  uint32_t numGroups = ((uint32_t)imgW * (uint32_t)imgH) / IMAGE_FETCH_PER_WI / wgSize;
  if (numGroups == 0) numGroups = 1;
  uint64_t globalWIs = (uint64_t)numGroups * wgSize;
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

  if (typeIdx == UINT32_MAX)
  {
    log->note("No device-local memory type for the image\n");
    vkDestroyImage(dev.device, img, nullptr);
    return -1;
  }

  VkMemoryAllocateInfo aI = {};
  aI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  aI.allocationSize = imReq.size;
  aI.memoryTypeIndex = typeIdx;
  VkDeviceMemory imgMem;
  // A quarter-gigabyte image is a real allocation on a phone: bail rather than
  // time a dispatch against unbound memory and report the result as bandwidth.
  if (vkAllocateMemory(dev.device, &aI, nullptr, &imgMem) != VK_SUCCESS)
  {
    log->note("Image memory alloc failed\n");
    vkDestroyImage(dev.device, img, nullptr);
    return -1;
  }
  if (vkBindImageMemory(dev.device, img, imgMem, 0) != VK_SUCCESS)
  {
    log->note("Image memory bind failed\n");
    vkDestroyImage(dev.device, img, nullptr);
    vkFreeMemory(dev.device, imgMem, nullptr);
    return -1;
  }

  // Upload pseudo-random data to defeat hardware memory compression.
  {
    VkDeviceSize stagingBytes = (VkDeviceSize)imgW * (VkDeviceSize)imgH * 16;
    VkBuffer stagingBuf;
    VkDeviceMemory stagingMem;
    if (!dev.createBuffer(stagingBytes,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          stagingBuf, stagingMem))
    {
      log->note("Staging buffer alloc failed\n");
      vkDestroyImage(dev.device, img, nullptr);
      vkFreeMemory(dev.device, imgMem, nullptr);
      return -1;
    }
    void *stagingMap = nullptr;
    vkMapMemory(dev.device, stagingMem, 0, stagingBytes, 0, &stagingMap);
    populate((float *)stagingMap, (size_t)imgW * (size_t)imgH * 4);
    vkUnmapMemory(dev.device, stagingMem);

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

    // UNDEFINED -> TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier b0 = {};
    b0.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b0.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b0.image = img;
    b0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b0.srcAccessMask = 0;
    b0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(transCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b0);

    // Copy staging buffer to image
    VkBufferImageCopy copyRegion = {};
    copyRegion.bufferOffset      = 0;
    copyRegion.bufferRowLength   = 0; // tightly packed
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.imageOffset       = {0, 0, 0};
    copyRegion.imageExtent       = {imgW, imgH, 1};
    vkCmdCopyBufferToImage(transCmd, stagingBuf, img,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    VkImageMemoryBarrier b1 = {};
    b1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b1.image = img;
    b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(transCmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b1);

    vkEndCommandBuffer(transCmd);
    dev.submitAndWait(transCmd);
    vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &transCmd);

    vkDestroyBuffer(dev.device, stagingBuf, nullptr);
    vkFreeMemory(dev.device, stagingMem, nullptr);
  }

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
  smCI.unnormalizedCoordinates = VK_TRUE;
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

  VkPushConstantRange walkRange = {};
  walkRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  walkRange.offset = 0;
  walkRange.size = sizeof(int32_t);

  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsLayout;
  plCI.pushConstantRangeCount = 1; plCI.pPushConstantRanges = &walkRange;
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

  // The image is RGBA32F, so one fetch returns a whole pixel: four 32-bit
  // values, hence the metric name.
  const char *fetchNote = "Each fetch returns one whole pixel -- four 32-bit "
                          "colour values, 16 bytes.";

  // The walk shapes raced.  A row-major sweep and its transpose both hand a
  // warp a 1D run of texels, which is what a linear surface wants; a driver
  // that stores the image swizzled instead -- Mali's 16x16 u-order blocks, where
  // a 64-byte line holds a 2x2 quad of RGBA32F texels -- serves either of them
  // half a line at a time.  The blocked shapes give a run of lanes a 2D block,
  // so one line is consumed by one warp whatever the swizzle granularity.  Each
  // shape reads every pixel exactly once, so the byte count is identical and
  // none of them can flatter the result.
  //
  // Which block size wins is the driver's swizzle granularity, and the two
  // measured devices want opposite ends of the range: a Mali G615 climbs with
  // block size (14.4 / 14.5 / 14.9 GBPS at 2x2 / 4x4 / 16x16) while an M1 Pro
  // falls (175.6 / 170.7 / 166.8).  So both ends are raced.  4x4 is not: it
  // landed strictly between them on both, which is what a shape that can never
  // win looks like.
  struct Shape { const char *label; int32_t tileW, tileH; int32_t walk; };
  const Shape shapes[] = {
    { "row",       (int32_t)imgW,  1, 0 },
    { "col",       (int32_t)imgW,  1, 1 },
    { "tile2x2",               2,  2, 0 },
    { "tile16x16",            16, 16, 0 },
  };

  const uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalWIs;
  float best = 0.0f;

  for (const Shape &s : shapes)
  {
    // Extent and block shape are specialization constants: that folds every
    // divide and modulo in the walk into a shift and a mask, which matters on
    // the GPUs without an integer-divide unit.
    struct { int32_t w, h, tw, th; } specData =
        { (int32_t)imgW, (int32_t)imgH, s.tileW, s.tileH };
    VkSpecializationMapEntry specEntries[4] = {
      { 0, (uint32_t)offsetof(decltype(specData), w),  sizeof(int32_t) },
      { 1, (uint32_t)offsetof(decltype(specData), h),  sizeof(int32_t) },
      { 2, (uint32_t)offsetof(decltype(specData), tw), sizeof(int32_t) },
      { 3, (uint32_t)offsetof(decltype(specData), th), sizeof(int32_t) },
    };
    VkSpecializationInfo specInfo = {};
    specInfo.mapEntryCount = 4;
    specInfo.pMapEntries   = specEntries;
    specInfo.dataSize      = sizeof(specData);
    specInfo.pData         = &specData;

    VkPipeline pipe;
    if (!dev.createComputePipeline(vk_shaders::image_bandwidth_v1,
                                   vk_shaders::image_bandwidth_v1_size,
                                   dsLayout, pipeLayout, pipe, &specInfo))
    {
      log->note(std::string("Image pipeline creation failed for ") + s.label + "\n");
      continue;
    }

    int32_t walk = s.walk;
    float us = runKernel(dev, pipe, pipeLayout, descSet, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0, true,
                         &walk, sizeof(walk));
    vkDestroyPipeline(dev.device, pipe, nullptr);

    float bps = us > 0.0f ? (float)bytes / us * 1e6f : 0.0f;
    best = std::max(best, bps);
    CLPEAK_VLOG("image_memory_bandwidth: %-16s %.1f B/s\n", s.label, bps);
  }

  if (best <= 0.0f)
    test.skip("float4", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed",
              fetchNote);
  else
    test.emit("float4", best, fetchNote);

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
