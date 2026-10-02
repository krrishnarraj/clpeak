#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <common/form_race.h>
#include <cstdio>
#include <memory>
#include <string>
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
  // The spec is built only when this desc opens its own test.  A desc that
  // writes into a caller's scope leaves the header fields null -- assigning
  // one of those to the spec's std::string is undefined, and did crash.
  std::unique_ptr<logger::TestScope> ownTest;
  if (!d.scope)
  {
    logger::TestSpec testSpec;
    testSpec.tag = d.resultTag;
    testSpec.display = d.title;
    testSpec.unit = d.unit;
    if (d.description)
      testSpec.description = d.description;
    testSpec.shape = d.shape;
    if (d.axis)
      testSpec.axis = d.axis;
    ownTest.reset(new logger::TestScope(currentDeviceScope->beginTest(testSpec)));
  }
  logger::TestScope &test = d.scope ? *d.scope : *ownTest;

  // Collect variants.  Multi-variant path (e.g. fp16 v1/v2/v4) shares one
  // buffer + descriptor set and swaps only the pipeline between dispatches;
  // single-variant benchmarks materialize a one-entry list.
  struct Variant
  {
    const char *label;
    const uint32_t *spirv;
    size_t spirvSize;
    const char *description;
    const uint32_t *altSpirv;
    size_t altSpirvSize;
  };
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
    // Single-variant tests: one reading, documented by d.metricDescription.
    // Coopmat uses this path once per data type, all into the same test.
    variants.push_back({d.metricLabel, d.spirv, d.spirvSize,
                        d.metricDescription, nullptr, 0});
  }

  auto note = [](const char *text)
  { return text ? std::string(text) : std::string(); };

  // Unit override for the single-variant path: an integer member of an
  // otherwise floating-point family carries its own.
  auto emitOpts = [&](const char *description)
  {
    logger::EmitOptions o;
    o.description = note(description);
    if (d.metricUnit)
      o.unit = d.metricUnit;
    return o;
  };

  if (d.skip)
  {
    const char *msg = d.skipMsg ? d.skipMsg : "Skipped";
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Unsupported, msg, emitOpts(v.description));
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
  // maxComputeWorkGroupCount is a hard limit, and dispatching past it is
  // invalid usage a driver may fault on rather than report.  Several vendors
  // report 65535 here while the buffer would happily hold far more groups, so
  // this is not slack -- without it the coopmat dispatch (32 threads per group,
  // so 16x the groups of a 256-thread kernel) is the first to go over.
  if (dev.info.maxWGCount)
    maxWGs = std::min(maxWGs, (uint64_t)dev.info.maxWGCount);
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

  // Every pipeline here asks for a subgroup width: the desc's when it names
  // one (coopmat names its tile's), else the width the device reports.  Left
  // to choose, Intel's Windows driver compiles compute shaders at SIMD16, and
  // Alchemist runs fp16 at double rate only at SIMD32: an Arc A380's half
  // read 4.93 TFLOPS unpinned and at SIMD16, 2.47 at SIMD8 and 9.50 at SIMD32
  // (half2/half4 9.46) against a 10.04 peak, whatever the chain shape, chain
  // count, work-group size or MAD spelling.  vkpeak reads 9.49 because ncnn
  // pins every pipeline to the reported width.  A no-op on NVIDIA, which
  // offers only 32, and wherever the device has no subgroup-size control.  A
  // group that would need more subgroups than the device allows runs unpinned.
  uint32_t subgroup = d.requiredSubgroupSize ? d.requiredSubgroupSize
                                             : dev.info.subgroupSize;
  if (subgroup && dev.info.maxComputeWorkgroupSubgroups &&
      (uint64_t)wgSize > (uint64_t)subgroup * dev.info.maxComputeWorkgroupSubgroups)
    subgroup = 0;

  // Build + dispatch each variant's pipeline.  Variants failing pipeline
  // creation are skipped but don't abort the group -- some drivers accept
  // the v1 shader but choke on a wider packed variant.
  //
  // Time one shader, returning microseconds per dispatch (<= 0 on failure).
  // *built distinguishes "the driver rejected the stage" from "the dispatch
  // failed", which the caller reports differently.

  // Phase markers.  A driver that faults inside its own shader compiler or on
  // submit takes the process with it, so the last line printed is the only
  // evidence of where it went -- which of the two below appears last says
  // whether pipeline creation or the dispatch was fatal.
  auto timeShape = [&](const uint32_t *spirv, size_t spirvSize,
                       const VkSpecializationInfo *spec, bool *built) -> float
  {
    VkPipeline pipeline;
    *built = dev.createComputePipeline(spirv, spirvSize, dsLayout, pipeLayout,
                                       pipeline, spec, subgroup);
    // A pinned subgroup width is a preference, not a requirement: if the
    // driver won't compile the stage at that width, run it however it likes
    // rather than dropping the row.  Worth knowing about, though -- a coopmat
    // shader that lands on several subgroups per work-group has every one of
    // them recompute the same tile, so the reading comes out divided by that
    // factor (see coopmatRequiredSubgroupSize() in vk_peak.h).
    if (!*built && subgroup)
    {
      CLPEAK_VLOG("%s: subgroup size %u refused by the driver, "
                  "falling back to its own choice\n",
                  d.resultTag, subgroup);
      *built = dev.createComputePipeline(spirv, spirvSize, dsLayout, pipeLayout,
                                         pipeline, spec, 0);
    }
    if (!*built)
      return -1.0f;

    // No barrier between dispatches: a compute peak reads nothing twice, so
    // overlapping launches cannot flatter it, and forbidding the overlap costs
    // ~2% of the reading.
    float timed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                            false, d.pushData, d.pushSize);
    vkDestroyPipeline(dev.device, pipeline, nullptr);
    return timed;
  };

  auto toValue = [&](float timed)
  {
    return (float)((double)globalWIs * (double)d.workPerWI * 1e6 / timed);
  };

  // The affine chain's two addends, as specializations of the one alt module:
  // MAD_CHAIN_LANE_B false is the uniform b, true the per-lane one.  Which is
  // faster depends on where the driver's allocator puts the operands -- on an
  // Arc A380 each won one of fp32 and mixed precision (shaders/mad_chain.glsl)
  // -- so they race width by width as a FormRace: where the addend makes no
  // difference they tie at the first width, and only the uniform one is timed
  // from there.  A family without an addend runs its alt build once, as built.
  const VkBool32 laneB[2] = {VK_FALSE, VK_TRUE};
  const VkSpecializationMapEntry laneBEntry = {VK_MAD_CHAIN_LANE_B_ID, 0,
                                               sizeof(VkBool32)};
  const VkSpecializationInfo addendSpec[2] = {
    {1, &laneBEntry, sizeof(VkBool32), &laneB[0]},
    {1, &laneBEntry, sizeof(VkBool32), &laneB[1]},
  };
  clpeak::FormRace addendRace;
  if (!d.raceAffineAddend)
    addendRace.drop(true);

  for (const auto &v : variants)
  {
    bool built = false;
    float timed = timeShape(v.spirv, v.spirvSize, d.specInfo, &built);
    if (!built)
    {
      test.skip(v.label, ResultStatus::Error, "Pipeline creation failed",
                emitOpts(v.description));
      continue;
    }
    if (timed <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "vkQueueSubmit/WaitIdle failed",
                emitOpts(v.description));
      continue;
    }
    float value = toValue(timed);

    // Race the alt build of the same shader -- both addends of the affine
    // chain, where the family has one -- and keep the fastest.  A shape that
    // fails to build or run here is not an error: the first shape already
    // produced a reading.
    if (v.altSpirv && v.altSpirvSize)
    {
      double altValue[2] = {0.0, 0.0};
      for (int f = 0; f < 2; f++)
      {
        if (!addendRace.runs(f == 1))
          continue;
        bool altBuilt = false;
        float altTimed = timeShape(v.altSpirv, v.altSpirvSize,
                                   d.raceAffineAddend ? &addendSpec[f] : d.specInfo,
                                   &altBuilt);
        if (altBuilt && altTimed > 0.0f)
          altValue[f] = toValue(altTimed);
        else if (d.raceAffineAddend)
          addendRace.drop(f == 1);
      }
      if (clpeak::verboseEnabled() && (altValue[0] > 0.0 || altValue[1] > 0.0))
      {
        std::string alts;
        for (int f = 0; f < 2; f++)
        {
          if (altValue[f] <= 0.0)
            continue;
          char reading[64];
          snprintf(reading, sizeof(reading), "%s%.1f%s", alts.empty() ? "" : ", ",
                   altValue[f], !d.raceAffineAddend ? ""
                                : f ? " (per-lane b)" : " (uniform b)");
          alts += reading;
        }
        CLPEAK_VLOG("%s %s: first shape %.1f, alt shape %s %s\n",
                    d.resultTag, v.label, value, alts.c_str(), d.unit);
      }
      if (d.raceAffineAddend && addendRace.runs(false) && addendRace.runs(true))
      {
        const double noBuildPreference[2] = {1.0, 1.0};
        addendRace.settle(altValue, noBuildPreference);
        if (!addendRace.runs(false) || !addendRace.runs(true))
          CLPEAK_VLOG("%s %s: alt addends settled -- the %s b alone from here\n",
                      d.resultTag, v.label,
                      addendRace.runs(true) ? "per-lane" : "uniform");
      }
      const double first = value;
      for (int f = 0; f < 2; f++)
      {
        if (altValue[f] > first * MAX_ALT_CHAIN_RATIO)
          CLPEAK_VLOG("%s %s: alt chain %.1fx faster -- rejecting it as a "
                      "compiler fold\n",
                      d.resultTag, v.label, altValue[f] / first);
        else if (altValue[f] > value)
          value = (float)altValue[f];
      }
    }

    test.emit(v.label, value, emitOpts(v.description));

    // TEMPORARY -- one tester round on the Arc A380.  Every shape again,
    // pinned to subgroup 16, logged and never reported: whether SIMD32's
    // register-bank pairing is what holds Alchemist's affine chain at 70-96%
    // of its rate.  Comes out once the round is read.
    if (d.raceAffineAddend && clpeak::verboseEnabled() &&
        dev.info.subgroupSizeControl && subgroup != 16 &&
        dev.info.minSubgroupSize <= 16 && 16 <= dev.info.maxSubgroupSize &&
        (!dev.info.maxComputeWorkgroupSubgroups ||
         (uint64_t)wgSize <= 16ull * dev.info.maxComputeWorkgroupSubgroups))
    {
      auto timeAt16 = [&](const uint32_t *spirv, size_t spirvSize,
                          const VkSpecializationInfo *spec) -> float
      {
        VkPipeline pipeline;
        if (!spirv || !dev.createComputePipeline(spirv, spirvSize, dsLayout,
                                                 pipeLayout, pipeline, spec, 16))
          return 0.0f;
        float probeTimed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups,
                                     cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                                     false, d.pushData, d.pushSize);
        vkDestroyPipeline(dev.device, pipeline, nullptr);
        return probeTimed > 0.0f ? toValue(probeTimed) : 0.0f;
      };
      float first16 = timeAt16(v.spirv, v.spirvSize, d.specInfo);
      float uniform16 = timeAt16(v.altSpirv, v.altSpirvSize, &addendSpec[0]);
      float lane16 = timeAt16(v.altSpirv, v.altSpirvSize, &addendSpec[1]);
      CLPEAK_VLOG("%s %s probe subgroup 16: first shape %.1f, alt shape %.1f "
                  "(uniform b), %.1f (per-lane b) %s\n",
                  d.resultTag, v.label, first16, uniform16, lane16, d.unit);
    }
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  return 0;
}

#endif // ENABLE_VULKAN
