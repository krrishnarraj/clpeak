#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <vector>

// ---------------------------------------------------------------------------
// Shared compute-peak driver.
//
// Every compute-peak benchmark (runComputeSP / MP / INT8-DP /
// coop-matrix / ...) shares the same Vulkan scaffolding: allocate a single
// device-local output buffer, build a one-binding descriptor set, create a
// pipeline from the shader's SPIR-V, dispatch repeatedly with a push
// constant, and report work-per-WI / elapsed time.  The only differences
// are the shader, the buffer-element size, the push-constant payload, and
// the strings used for display / result output.  All of those are bundled into
// vk_compute_desc_t so each concrete benchmark becomes a few-line wrapper.
// ---------------------------------------------------------------------------

int vkPeak::runComputeKernel(VulkanDevice &dev, benchmark_config_t &cfg,
                             const vk_compute_desc_t &d)
{
  logger::TestSpec testSpec;
  testSpec.tag = d.resultTag;
  testSpec.display = d.title;
  testSpec.unit = d.unit;
  if (d.description) testSpec.description = d.description;
  auto test = currentDeviceScope->beginTest(testSpec);

  // Collect variants.  Multi-variant path (e.g. fp16 v1/v2/v4) shares one
  // buffer + descriptor set and swaps only the pipeline between dispatches;
  // single-variant benchmarks materialize a one-entry list.
  struct Variant { const char *label; const uint32_t *spirv; size_t spirvSize;
                   const char *description;
                   const uint32_t *altSpirv; size_t altSpirvSize; };
  std::vector<Variant> variants;
  if (d.variants && d.numVariants > 0)
  {
    for (uint32_t i = 0; i < d.numVariants; i++)
      variants.push_back({d.variants[i].label, d.variants[i].spirv,
                          d.variants[i].spirvSize, d.variants[i].description,
                          d.variants[i].altSpirv, d.variants[i].altSpirvSize});
  }
  else
  {
    // Single-variant tests (coopmat): the metric name restates the title, so
    // the test-level description covers it.
    variants.push_back({d.metricLabel, d.spirv, d.spirvSize, nullptr, nullptr, 0});
  }

  auto note = [](const char *text) { return text ? std::string(text) : std::string(); };

  if (d.skip)
  {
    const char *msg = d.skipMsg ? d.skipMsg : "Skipped";
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Unsupported, msg, note(v.description));
    return 0;
  }

  // Size the dispatch to saturate the device and amortize submit overhead.
  // When Vulkan exposes a CU count, mirror OpenCL's
  // numCUs*2048*maxWGSize formula.  Unknown integrated/mobile GPUs use a
  // smaller floor so the calibration probe does not become a watchdog-sized
  // dispatch.  Cooperative-matrix shaders run 32 threads -- one subgroup,
  // pinned via requiredSubgroupSize where the device allows it; other compute
  // kernels use the classic 256.
  const uint32_t wgSize = d.wgSize ? d.wgSize : 256;
  const uint32_t outPerWG = d.outElemsPerWG ? d.outElemsPerWG : wgSize;
  uint64_t globalWIs = targetVulkanGlobalThreads(dev.info);
  // Buffer footprint = numGroups * outPerWG * elemSize.  Bound by allocation.
  uint64_t bytesPerWG = (uint64_t)outPerWG * d.elemSize;
  uint64_t maxWGs = dev.info.maxAllocSize / bytesPerWG;
  uint64_t wantWGs = globalWIs / wgSize;
  uint32_t numGroups = (uint32_t)std::min(wantWGs, maxWGs);
  globalWIs = (uint64_t)numGroups * wgSize;
  uint64_t bufferBytes = (uint64_t)numGroups * bytesPerWG;

  VkBuffer outputBuf;
  VkDeviceMemory outputMem;
  if (!dev.createBuffer(bufferBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        outputBuf, outputMem))
  {
    log->note("Failed to allocate buffer\n");
    return -1;
  }

  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dsLayoutCI = {};
  dsLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsLayoutCI.bindingCount = 1;
  dsLayoutCI.pBindings = &binding;

  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dsLayoutCI, nullptr, &dsLayout);

  VkPushConstantRange pushRange = {};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = d.pushSize;

  VkPipelineLayoutCreateInfo pipeLayoutCI = {};
  pipeLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutCI.setLayoutCount = 1;
  pipeLayoutCI.pSetLayouts = &dsLayout;
  if (d.pushSize > 0)
  {
    pipeLayoutCI.pushConstantRangeCount = 1;
    pipeLayoutCI.pPushConstantRanges = &pushRange;
  }

  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &pipeLayoutCI, nullptr, &pipeLayout);

  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1;

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

  VkDescriptorBufferInfo bufInfo = {};
  bufInfo.buffer = outputBuf;
  bufInfo.offset = 0;
  bufInfo.range = bufferBytes;

  VkWriteDescriptorSet write = {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descSet;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &bufInfo;

  vkUpdateDescriptorSets(dev.device, 1, &write, 0, nullptr);

  // Build + dispatch each variant's pipeline.  Variants failing pipeline
  // creation are skipped but don't abort the group -- some drivers accept
  // the v1 shader but choke on a wider packed variant.
  //
  // Time one shader, returning microseconds per dispatch (<= 0 on failure).
  // *built distinguishes "the driver rejected the stage" from "the dispatch
  // failed", which the caller reports differently.
  auto timeShape = [&](const uint32_t *spirv, size_t spirvSize,
                       bool *built) -> float {
    VkPipeline pipeline;
    *built = dev.createComputePipeline(spirv, spirvSize, dsLayout, pipeLayout,
                                       pipeline, d.specInfo, d.requiredSubgroupSize);
    // A pinned subgroup width is a preference, not a requirement: if the
    // driver won't compile the stage at that width, run it however it likes
    // rather than dropping the row.  Worth knowing about, though -- a coopmat
    // shader that lands on several subgroups per work-group has every one of
    // them recompute the same tile, so the reading comes out divided by that
    // factor (see coopmatRequiredSubgroupSize() in vk_peak.h).
    if (!*built && d.requiredSubgroupSize)
    {
      CLPEAK_VLOG("%s: subgroup size %u refused by the driver, "
                  "falling back to its own choice\n",
                  d.resultTag, d.requiredSubgroupSize);
      *built = dev.createComputePipeline(spirv, spirvSize, dsLayout, pipeLayout,
                                         pipeline, d.specInfo, 0);
    }
    if (!*built) return -1.0f;

    float timed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                            d.pushData, d.pushSize);
    vkDestroyPipeline(dev.device, pipeline, nullptr);
    return timed;
  };

  double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
  auto toValue = [&](float timed) {
    return (float)((double)globalWIs * (double)d.workPerWI * 1e6 / timed / divider);
  };

  for (const auto &v : variants)
  {
    bool built = false;
    float timed = timeShape(v.spirv, v.spirvSize, &built);
    if (!built)
    {
      test.skip(v.label, ResultStatus::Error, "Pipeline creation failed",
                note(v.description));
      continue;
    }
    if (timed <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "vkQueueSubmit/WaitIdle failed",
                note(v.description));
      continue;
    }
    float value = toValue(timed);

    // Race the affine build of the same shader and keep the faster.  A shape
    // that fails to build or run here is not an error -- the first shape
    // already produced a reading.
    if (v.altSpirv && v.altSpirvSize)
    {
      bool altBuilt = false;
      float altTimed = timeShape(v.altSpirv, v.altSpirvSize, &altBuilt);
      if (altBuilt && altTimed > 0.0f)
      {
        float altValue = toValue(altTimed);
        CLPEAK_VLOG("%s %s: squaring chain %.1f, alt chain %.1f %s\n",
                    d.resultTag, v.label, value, altValue, d.unit);
        if (altValue > value * MAX_ALT_CHAIN_RATIO)
          CLPEAK_VLOG("%s %s: alt chain %.1fx faster -- rejecting it as a "
                      "compiler fold\n", d.resultTag, v.label, altValue / value);
        else if (altValue > value)
          value = altValue;
      }
    }

    test.emit(v.label, value, {false, note(v.description)});
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  return 0;
}

#endif // ENABLE_VULKAN
