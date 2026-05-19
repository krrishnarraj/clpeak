#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Atomic throughput (Vulkan -- global + local atomics)
// ---------------------------------------------------------------------------

int vkPeak::runAtomicThroughput(VulkanDevice &dev, benchmark_config_t &cfg)
{
  logger::TestSpec testSpec;
  testSpec.tag = "atomic_throughput";
  testSpec.display = "Atomic throughput";
  testSpec.unit = "gops";
  auto test = currentDeviceScope->beginTest(testSpec);

  const uint32_t wgSize = 256;
  uint64_t globalWIs = targetVulkanGlobalThreads(dev.info);
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);

  // Helper: allocate a single-storage-buffer descriptor + dispatch + time.
  auto runOne = [&](const char *label, const uint32_t *spv, size_t spvSize,
                    uint64_t bufBytes) -> float
  {
    VkBuffer buf; VkDeviceMemory mem;
    if (!dev.createBuffer(bufBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem))
      return -1.0f;
    if (!dev.zeroBuffer(buf))
    {
      vkDestroyBuffer(dev.device, buf, nullptr);
      vkFreeMemory(dev.device, mem, nullptr);
      return -1.0f;
    }
    VkDescriptorSetLayoutBinding b = {};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslCI = {};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 1; dslCI.pBindings = &b;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev.device, &dslCI, nullptr, &dsl);
    VkPipelineLayoutCreateInfo plCI = {};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsl;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pl);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpCI = {};
    dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.maxSets = 1; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &ps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsAI = {};
    dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAI.descriptorPool = dp; dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev.device, &dsAI, &ds);
    VkDescriptorBufferInfo bi = {buf, 0, bufBytes};
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = ds; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);

    VkPipeline pipe;
    if (!dev.createComputePipeline(spv, spvSize, dsl, pl, pipe))
    {
      vkDestroyDescriptorPool(dev.device, dp, nullptr);
      vkDestroyPipelineLayout(dev.device, pl, nullptr);
      vkDestroyDescriptorSetLayout(dev.device, dsl, nullptr);
      vkDestroyBuffer(dev.device, buf, nullptr);
      vkFreeMemory(dev.device, mem, nullptr);
      return -1.0f;
    }
    float us = runKernel(dev, pipe, pl, ds, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    vkDestroyPipeline(dev.device, pipe, nullptr);
    vkDestroyDescriptorPool(dev.device, dp, nullptr);
    vkDestroyPipelineLayout(dev.device, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev.device, dsl, nullptr);
    vkDestroyBuffer(dev.device, buf, nullptr);
    vkFreeMemory(dev.device, mem, nullptr);
    (void)label;
    return us;
  };

  // Global atomics: one int counter per WI -> 128 MB output.
  float us_g = runOne("int_global", vk_shaders::atomic_throughput_global,
      vk_shaders::atomic_throughput_global_size, globalWIs * sizeof(int32_t));
  if (us_g > 0)
  {
    float gops = ((float)globalWIs * (float)ATOMIC_REPS) / us_g / 1e3f;
    test.emit("int_global", gops);
  }
  else
  {
    test.skip("int_global", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
  }

#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64
  if (dev.info.atomicInt64Supported)
  {
    float us = runOne("ulong_global", vk_shaders::atomic_throughput_global_uint64,
        vk_shaders::atomic_throughput_global_uint64_size, globalWIs * sizeof(uint64_t));
    if (us > 0)
    {
      float gops = ((float)globalWIs * (float)ATOMIC_REPS) / us / 1e3f;
      test.emit("ulong_global", gops);
    }
    else
    {
      test.skip("ulong_global", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
    }
  }
#endif

  // Local atomics: one int counter per workgroup.
  float us_l = runOne("int_local", vk_shaders::atomic_throughput_local,
      vk_shaders::atomic_throughput_local_size, (uint64_t)numGroups * sizeof(int32_t));
  if (us_l > 0)
  {
    float gops = ((float)globalWIs * (float)ATOMIC_REPS) / us_l / 1e3f;
    test.emit("int_local", gops);
  }
  else
  {
    test.skip("int_local", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
  }

  return 0;
}

// ---------------------------------------------------------------------------
// FP atomic throughput (Vulkan) -- emitted in the FpCompute phase, mirroring
// Metal's runAtomicThroughputFp.  Currently a single global_float row, gated
// on VK_EXT_shader_atomic_float + shaderBufferFloat32AtomicAdd.
// ---------------------------------------------------------------------------

#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT
int vkPeak::runAtomicThroughputFp(VulkanDevice &dev, benchmark_config_t &cfg)
{
  logger::TestSpec testSpec;
  testSpec.tag = "atomic_throughput";
  testSpec.display = "Atomic throughput";
  testSpec.unit = "gflops";
  auto test = currentDeviceScope->beginTest(testSpec);

  if (!dev.info.atomicFloat32Supported)
  {
    test.skip("float_global", ResultStatus::Unsupported,
              "VK_EXT_shader_atomic_float not supported");
    return 0;
  }

  const uint32_t wgSize = 256;
  uint64_t globalWIs = targetVulkanGlobalThreads(dev.info);
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);
  uint64_t bufBytes = globalWIs * sizeof(float);

  VkBuffer buf; VkDeviceMemory mem;
  if (!dev.createBuffer(bufBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem))
  {
    log->note("Failed to allocate buffer\n");
    return -1;
  }
  if (!dev.zeroBuffer(buf))
  {
    log->note("Failed to zero buffer\n");
    vkDestroyBuffer(dev.device, buf, nullptr);
    vkFreeMemory(dev.device, mem, nullptr);
    return -1;
  }
  VkDescriptorSetLayoutBinding b = {};
  b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslCI = {};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 1; dslCI.pBindings = &b;
  VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev.device, &dslCI, nullptr, &dsl);
  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsl;
  VkPipelineLayout pl; vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pl);
  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &ps;
  VkDescriptorPool dp; vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &dp);
  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = dp; dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &dsl;
  VkDescriptorSet ds; vkAllocateDescriptorSets(dev.device, &dsAI, &ds);
  VkDescriptorBufferInfo bi = {buf, 0, bufBytes};
  VkWriteDescriptorSet w = {};
  w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w.dstSet = ds; w.dstBinding = 0; w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);

  VkPipeline pipe;
  if (!dev.createComputePipeline(vk_shaders::atomic_throughput_global_float,
                                  vk_shaders::atomic_throughput_global_float_size,
                                  dsl, pl, pipe))
  {
    test.skip("float_global", ResultStatus::Error, "Pipeline creation failed");
  }
  else
  {
    float us = runKernel(dev, pipe, pl, ds, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      test.skip("float_global", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
    }
    else
    {
      float gflops = ((float)globalWIs * (float)ATOMIC_REPS) / us / 1e3f;
      test.emit("float_global", gflops);
    }
    vkDestroyPipeline(dev.device, pipe, nullptr);
  }

  vkDestroyDescriptorPool(dev.device, dp, nullptr);
  vkDestroyPipelineLayout(dev.device, pl, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsl, nullptr);
  vkDestroyBuffer(dev.device, buf, nullptr);
  vkFreeMemory(dev.device, mem, nullptr);
  return 0;
}
#endif

#endif // ENABLE_VULKAN
