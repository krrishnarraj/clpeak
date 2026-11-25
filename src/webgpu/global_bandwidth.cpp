#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include "wgsl_shaders.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <algorithm>
#include <string>

// ---------------------------------------------------------------------------
// Global memory bandwidth benchmark (WebGPU)
// ---------------------------------------------------------------------------

int WebgpuPeak::runGlobalBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const WebgpuRuntime &rt = *dev.rt;
  const uint32_t wgSize = 256;

  // Opened before sizing diagnostics so logs attach to this test's header.
  logger::TestSpec testSpec;
  testSpec.tag = "global_memory_bandwidth";
  testSpec.display = "Global memory bandwidth";
  testSpec.unit = "bps";
  testSpec.description =
      "Read bandwidth from device memory, over a buffer too large to cache.";
  testSpec.shape = TestShape::Homogeneous;
  testSpec.axis = "vector width";
  auto test = currentDeviceScope->beginTest(testSpec);

  uint64_t maxBind = std::min(dev.info.maxStorageBufferBindingSize, dev.info.maxBufferSize);
  // Leave headroom for input + output + staging within system memory budget.
  uint64_t capBytes = std::min(cfg.globalBWMaxSize, maxBind);
  capBytes = std::min(capBytes, clpeak::memoryBudget(cfg.globalBWMaxSize, 4));
  // Ensure alignment to wgSize * FETCH_PER_WI * 4 floats (so float4 divides evenly).
  const uint64_t alignItems = static_cast<uint64_t>(wgSize) * FETCH_PER_WI * 4ULL;
  uint64_t numItems = (capBytes / sizeof(float) / alignItems) * alignItems;
  if (numItems == 0)
    numItems = alignItems;

  uint32_t numGroups = static_cast<uint32_t>(numItems / FETCH_PER_WI / wgSize);
  if (dev.info.maxWGCount && numGroups > dev.info.maxWGCount)
  {
    numGroups = (dev.info.maxWGCount / 4) * 4;
    if (numGroups == 0) numGroups = 1;
    numItems = static_cast<uint64_t>(numGroups) * wgSize * FETCH_PER_WI;
  }

  const uint64_t inputBytes = numItems * sizeof(float);
  const uint64_t outputBytes = static_cast<uint64_t>(numGroups) * wgSize * sizeof(float);

  CLPEAK_VLOG("global_memory_bandwidth: working set %llu MB (maxStorageBinding %llu MB)\n",
              static_cast<unsigned long long>(inputBytes >> 20),
              static_cast<unsigned long long>(dev.info.maxStorageBufferBindingSize >> 20));

  struct GBVar { const char *label; uint32_t width; };
  const GBVar variants[] = {
    {"float",  1},
    {"float2", 2},
    {"float4", 4},
  };

  auto skipAll = [&](const char *reason)
  {
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Error, reason, wgpuWidthNote(v.width));
  };

  WGPUBuffer inputBuf = dev.createBuffer(
      inputBytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
  WGPUBuffer outputBuf = dev.createBuffer(outputBytes, WGPUBufferUsage_Storage);
  if (!inputBuf || !outputBuf)
  {
    if (inputBuf) { rt.api.wgpuBufferDestroy(inputBuf); rt.api.wgpuBufferRelease(inputBuf); }
    if (outputBuf) { rt.api.wgpuBufferDestroy(outputBuf); rt.api.wgpuBufferRelease(outputBuf); }
    CLPEAK_VLOG("Failed to allocate WebGPU global bandwidth buffers\n");
    skipAll("Buffer allocation failed");
    return -1;
  }

  // Fill input buffer with pseudo-random data via a mapped staging buffer to
  // defeat hardware memory compression.
  {
    WGPUBuffer stagingBuf = dev.createBuffer(
        inputBytes, WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
        /*mappedAtCreation=*/true);
    if (!stagingBuf)
    {
      rt.api.wgpuBufferDestroy(inputBuf);
      rt.api.wgpuBufferRelease(inputBuf);
      rt.api.wgpuBufferDestroy(outputBuf);
      rt.api.wgpuBufferRelease(outputBuf);
      CLPEAK_VLOG("Failed to allocate WebGPU staging buffer\n");
      skipAll("Staging buffer allocation failed");
      return -1;
    }
    void *mapped = rt.api.wgpuBufferGetMappedRange(stagingBuf, 0, inputBytes);
    if (!mapped)
    {
      rt.api.wgpuBufferUnmap(stagingBuf);
      rt.api.wgpuBufferDestroy(stagingBuf);
      rt.api.wgpuBufferRelease(stagingBuf);
      rt.api.wgpuBufferDestroy(inputBuf);
      rt.api.wgpuBufferRelease(inputBuf);
      rt.api.wgpuBufferDestroy(outputBuf);
      rt.api.wgpuBufferRelease(outputBuf);
      CLPEAK_VLOG("Failed to map WebGPU staging buffer\n");
      skipAll("Staging buffer mapping failed");
      return -1;
    }
    populate(static_cast<float *>(mapped), numItems);
    rt.api.wgpuBufferUnmap(stagingBuf);

    WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder enc = rt.api.wgpuDeviceCreateCommandEncoder(dev.device, &encDesc);
    rt.api.wgpuCommandEncoderCopyBufferToBuffer(enc, stagingBuf, 0, inputBuf, 0, inputBytes);
    WGPUCommandBufferDescriptor cbDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd = rt.api.wgpuCommandEncoderFinish(enc, &cbDesc);
    rt.api.wgpuCommandEncoderRelease(enc);
    bool ok = dev.submitAndWait(cmd);
    rt.api.wgpuCommandBufferRelease(cmd);
    rt.api.wgpuBufferDestroy(stagingBuf);
    rt.api.wgpuBufferRelease(stagingBuf);
    if (!ok)
    {
      rt.api.wgpuBufferDestroy(inputBuf);
      rt.api.wgpuBufferRelease(inputBuf);
      rt.api.wgpuBufferDestroy(outputBuf);
      rt.api.wgpuBufferRelease(outputBuf);
      skipAll("Staging buffer upload failed");
      return -1;
    }
  }

  WGPUBindGroupLayoutEntry bglEntries[2] = {
    WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
    WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
  };
  bglEntries[0].binding = 0;
  bglEntries[0].visibility = WGPUShaderStage_Compute;
  bglEntries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
  bglEntries[1].binding = 1;
  bglEntries[1].visibility = WGPUShaderStage_Compute;
  bglEntries[1].buffer.type = WGPUBufferBindingType_Storage;

  WGPUBindGroupLayoutDescriptor bglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  bglDesc.entryCount = 2;
  bglDesc.entries = bglEntries;
  WGPUBindGroupLayout bgl = rt.api.wgpuDeviceCreateBindGroupLayout(dev.device, &bglDesc);

  WGPUPipelineLayoutDescriptor plDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  plDesc.bindGroupLayoutCount = 1;
  plDesc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeLayout = rt.api.wgpuDeviceCreatePipelineLayout(dev.device, &plDesc);

  WGPUBindGroupEntry bgEntries[2] = {
    WGPU_BIND_GROUP_ENTRY_INIT,
    WGPU_BIND_GROUP_ENTRY_INIT,
  };
  bgEntries[0].binding = 0;
  bgEntries[0].buffer = inputBuf;
  bgEntries[0].offset = 0;
  bgEntries[0].size = inputBytes;
  bgEntries[1].binding = 1;
  bgEntries[1].buffer = outputBuf;
  bgEntries[1].offset = 0;
  bgEntries[1].size = outputBytes;

  WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bgDesc.layout = bgl;
  bgDesc.entryCount = 2;
  bgDesc.entries = bgEntries;
  WGPUBindGroup bindGroup = rt.api.wgpuDeviceCreateBindGroup(dev.device, &bgDesc);

  for (const auto &v : variants)
  {
    WGPUComputePipeline pipe = dev.createComputePipeline(
        wgpu_shaders::makeGlobalBandwidthWgsl(v.width), pipeLayout);
    if (!pipe)
    {
      test.skip(v.label, ResultStatus::Error, "Pipeline creation failed",
                wgpuWidthNote(v.width));
      continue;
    }
    uint32_t variantGroups = numGroups / v.width;
    if (variantGroups == 0) variantGroups = 1;
    float timed = runKernel(dev, pipe, bindGroup, variantGroups,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                            /*serialize=*/true);
    rt.api.wgpuComputePipelineRelease(pipe);
    if (timed <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "WebGPU dispatch failed",
                wgpuWidthNote(v.width));
      continue;
    }
    float bps = static_cast<float>(static_cast<double>(inputBytes) / timed * 1e6);
    test.emit(v.label, bps, wgpuWidthNote(v.width));
  }

  rt.api.wgpuBindGroupRelease(bindGroup);
  rt.api.wgpuPipelineLayoutRelease(pipeLayout);
  rt.api.wgpuBindGroupLayoutRelease(bgl);
  rt.api.wgpuBufferDestroy(outputBuf);
  rt.api.wgpuBufferRelease(outputBuf);
  rt.api.wgpuBufferDestroy(inputBuf);
  rt.api.wgpuBufferRelease(inputBuf);
  return 0;
}

#endif // ENABLE_WEBGPU
