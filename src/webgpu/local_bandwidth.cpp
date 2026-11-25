#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include "wgsl_shaders.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <algorithm>

// ---------------------------------------------------------------------------
// Local (workgroup shared) memory bandwidth benchmark (WebGPU)
// ---------------------------------------------------------------------------

int WebgpuPeak::runLocalBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const WebgpuRuntime &rt = *dev.rt;

  logger::TestSpec testSpec;
  testSpec.tag = "local_memory_bandwidth";
  testSpec.display = "Local memory bandwidth";
  testSpec.unit = "bps";
  testSpec.description =
      "Bandwidth of shared memory, the scratchpad a workgroup shares.";
  testSpec.shape = TestShape::Homogeneous;
  testSpec.axis = "vector width";
  auto test = currentDeviceScope->beginTest(testSpec);

  const uint32_t wgSize = 256;
  uint64_t globalWIs = targetWebgpuGlobalThreads(dev.info);
  uint64_t maxBind = std::min(dev.info.maxStorageBufferBindingSize, dev.info.maxBufferSize);
  uint64_t maxWGs = maxBind / (wgSize * sizeof(float));
  if (dev.info.maxWGCount)
    maxWGs = std::min(maxWGs, static_cast<uint64_t>(dev.info.maxWGCount));
  uint32_t numGroups = static_cast<uint32_t>(std::max<uint64_t>(1, std::min(globalWIs / wgSize, maxWGs)));
  globalWIs = static_cast<uint64_t>(numGroups) * wgSize;
  const uint64_t bufferBytes = globalWIs * sizeof(float);

  WGPUBuffer outBuf = dev.createBuffer(bufferBytes, WGPUBufferUsage_Storage);
  if (!outBuf)
  {
    CLPEAK_VLOG("Failed to allocate WebGPU local bandwidth buffer\n");
    return -1;
  }

  WGPUBindGroupLayoutEntry bglEntry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
  bglEntry.binding = 0;
  bglEntry.visibility = WGPUShaderStage_Compute;
  bglEntry.buffer.type = WGPUBufferBindingType_Storage;

  WGPUBindGroupLayoutDescriptor bglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
  bglDesc.entryCount = 1;
  bglDesc.entries = &bglEntry;
  WGPUBindGroupLayout bgl = rt.api.wgpuDeviceCreateBindGroupLayout(dev.device, &bglDesc);

  WGPUPipelineLayoutDescriptor plDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  plDesc.bindGroupLayoutCount = 1;
  plDesc.bindGroupLayouts = &bgl;
  WGPUPipelineLayout pipeLayout = rt.api.wgpuDeviceCreatePipelineLayout(dev.device, &plDesc);

  WGPUBindGroupEntry bgEntry = WGPU_BIND_GROUP_ENTRY_INIT;
  bgEntry.binding = 0;
  bgEntry.buffer = outBuf;
  bgEntry.offset = 0;
  bgEntry.size = bufferBytes;

  WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bgDesc.layout = bgl;
  bgDesc.entryCount = 1;
  bgDesc.entries = &bgEntry;
  WGPUBindGroup bindGroup = rt.api.wgpuDeviceCreateBindGroup(dev.device, &bgDesc);

  struct V { const char *label; uint32_t width; };
  const V variants[] = {
    {"float",  1},
    {"float2", 2},
    {"float4", 4},
  };

  for (const auto &v : variants)
  {
    const uint32_t neededShared = wgSize * v.width * sizeof(float);
    if (dev.info.maxWorkgroupStorageSize && neededShared > dev.info.maxWorkgroupStorageSize)
    {
      test.skip(v.label, ResultStatus::Unsupported,
                "Exceeds maxComputeWorkgroupStorageSize", wgpuWidthNote(v.width));
      continue;
    }

    WGPUComputePipeline pipe = dev.createComputePipeline(
        wgpu_shaders::makeLocalBandwidthWgsl(v.width), pipeLayout);
    if (!pipe)
    {
      test.skip(v.label, ResultStatus::Error, "Pipeline creation failed",
                wgpuWidthNote(v.width));
      continue;
    }
    float us = runKernel(dev, pipe, bindGroup, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                         /*serialize=*/true);
    rt.api.wgpuComputePipelineRelease(pipe);
    if (us <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "WebGPU dispatch failed",
                wgpuWidthNote(v.width));
      continue;
    }
    uint64_t bytes = static_cast<uint64_t>(LMEM_REPS) * 2 * v.width * sizeof(float) * globalWIs;
    float bps = static_cast<float>(static_cast<double>(bytes) / us * 1e6);
    test.emit(v.label, bps, wgpuWidthNote(v.width));
  }

  rt.api.wgpuBindGroupRelease(bindGroup);
  rt.api.wgpuPipelineLayoutRelease(pipeLayout);
  rt.api.wgpuBindGroupLayoutRelease(bgl);
  rt.api.wgpuBufferDestroy(outBuf);
  rt.api.wgpuBufferRelease(outBuf);
  return 0;
}

#endif // ENABLE_WEBGPU
