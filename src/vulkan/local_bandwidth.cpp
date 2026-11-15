#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstring>

using xpupeak::Dim;

// ---------------------------------------------------------------------------
// Local memory bandwidth (Vulkan -- shared memory)
// ---------------------------------------------------------------------------
//
// Same single-output-buffer scaffolding as runComputeKernel; only the bytes-
// per-WI calculation differs per variant.  Width = 1/2/4 floats per slot.
//
// Under KosmicKrisp (Mesa's Vulkan driver over Metal) this reads about a
// quarter of Metal's figure on Apple silicon, and the cost is in the MSL the
// driver writes, not in these shaders.  It declares a shader's shared memory
// as one `threadgroup char shared_data[N]` and reaches it through
// `*(threadgroup T*)&shared_data[offset]` (emit_local_vars() and
// load_shared/store_shared in Mesa's src/kosmickrisp/compiler/nir_to_msl.c).
// A char array is 1-byte aligned, so Apple's compiler cannot prove those
// accesses aligned and issues unaligned threadgroup loads and stores.  No GLSL
// form avoids it: every shared access goes through that array.  On an M1 Pro
// (macOS 27.2.1; KosmicKrisp of Vulkan SDK 1.4.357.1, Mesa 26.2.99), float /
// float2 / float4 in TB/s, this test:
//
//   under KosmicKrisp                                  0.69  0.55  0.41
//   under MoltenVK, whose MSL (SPIRV-Cross) declares   2.85  2.73  2.67
//     typed threadgroup arrays
//   Metal's own test                                   2.94  2.77  2.71
//
// and KosmicKrisp's MSL for these shaders (MESA_KK_DEBUG=msl), dispatched
// through plain Metal on this test's grid, beside Metal's kernel there (2.85
// 2.73 2.67):
//
//   as the driver writes it                            0.69  0.55  0.41
//   with only alignas(16) added to shared_data         2.81  2.75  2.67

int vkPeak::runLocalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  logger::TestSpec testSpec;
  testSpec.tag = "local_memory_bandwidth";
  testSpec.display = "Local memory bandwidth";
  testSpec.unit = "bps";
  testSpec.description =
      "Bandwidth of shared memory, the scratchpad a workgroup shares.";
  testSpec.shape = TestShape::Homogeneous;
  testSpec.axis  = "vector width";
  // Revision 2: since it first shipped, a barrier went in between consecutive
  // dispatches (2.1.4) and came out again (4.0.0).  It held MoltenVK's float
  // reading a tenth low, and guarded nothing here: see runKernel's call below.
  testSpec.revision = 2;
  auto test = currentDeviceScope->beginTest(testSpec);

  struct V { const char *label; const uint32_t *spv; size_t sz; uint32_t width; };
  const V variants[] = {
    {"float",  vk_shaders::local_bandwidth_v1, vk_shaders::local_bandwidth_v1_size, 1},
    {"float2", vk_shaders::local_bandwidth_v2, vk_shaders::local_bandwidth_v2_size, 2},
    {"float4", vk_shaders::local_bandwidth_v4, vk_shaders::local_bandwidth_v4_size, 4},
  };

  // A setup failure measures no width, and every one still gets its row.
  auto skipEveryWidth = [&](const char *reason)
  {
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Error, reason, vkWidthNote(v.width),
                {{Dim::Width, v.width}});
  };

  const uint32_t wgSize = 256;
  uint64_t globalWIs = targetVulkanGlobalThreads(dev.info);
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);
  uint64_t bufferBytes = (uint64_t)globalWIs * sizeof(float);

  VkBuffer outBuf;
  VkDeviceMemory outMem;
  if (!dev.createBuffer(bufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem))
  {
    skipEveryWidth("Failed to allocate buffer");
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

  for (const auto &v : variants)
  {
    VkPipeline pipe;
    if (!dev.createComputePipeline(v.spv, v.sz, dsLayout, pipeLayout, pipe))
    {
      test.skip(v.label, ResultStatus::Error, "Pipeline creation failed",
                vkWidthNote(v.width), {{Dim::Width, v.width}});
      continue;
    }
    // No barrier between the batched dispatches (serialize = false), as in
    // the compute tests: shared memory is private to its workgroup, so
    // dispatches that overlap still move every byte once.  MoltenVK spent
    // ~45 us on each barrier on an M1 Pro (425 against 377 us a float
    // dispatch); without them it matches Metal's serial encoder.
    float us = runKernel(dev, pipe, pipeLayout, descSet, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0, false);
    if (us <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "vkQueueSubmit/WaitIdle failed",
                vkWidthNote(v.width), {{Dim::Width, v.width}});
      vkDestroyPipeline(dev.device, pipe, nullptr);
      continue;
    }
    // Each rep: 1 write + 1 read per WI = 2 * width * sizeof(float) bytes.
    uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.width * sizeof(float) * globalWIs;
    float bps = (float)bytes / us * 1e6f;
    test.emit(v.label, bps, vkWidthNote(v.width), {{Dim::Width, v.width}});
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
