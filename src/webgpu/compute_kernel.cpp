#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <algorithm>
#include <cstdio>
#include <string>

// ---------------------------------------------------------------------------
// Shared WebGPU compute-peak driver.
//
// Allocates one output storage buffer (@binding(0)) and one 16-byte uniform
// seed buffer (@binding(1)), builds each variant's pipeline (racing squaring
// and alt chain shapes with the MAX_ALT_CHAIN_RATIO fold guard), and emits
// the best rate.
// ---------------------------------------------------------------------------

int WebgpuPeak::runComputeKernel(WebgpuDevice &dev, benchmark_config_t &cfg,
                                 const wgpu_compute_desc_t &d)
{
  const WebgpuRuntime &rt = *dev.rt;

  logger::TestSpec testSpec;
  testSpec.tag = d.resultTag;
  testSpec.display = d.title;
  testSpec.unit = d.unit;
  if (d.description)
    testSpec.description = d.description;
  testSpec.shape = d.shape;
  if (d.axis)
    testSpec.axis = d.axis;
  auto test = currentDeviceScope->beginTest(testSpec);

  auto note = [](const char *text) -> const char *
  { return text ? text : ""; };

  if (d.skip)
  {
    const char *msg = d.skipMsg ? d.skipMsg : "Skipped";
    for (const auto &v : d.variants)
      test.skip(v.label, ResultStatus::Unsupported, msg, note(v.description));
    return 0;
  }

  const uint32_t wgSize = d.wgSize ? d.wgSize : 256;
  uint64_t globalWIs = targetWebgpuGlobalThreads(dev.info);
  uint64_t bytesPerWG = static_cast<uint64_t>(wgSize) * d.elemSize;
  uint64_t maxBind = std::min(dev.info.maxStorageBufferBindingSize, dev.info.maxBufferSize);
  uint64_t maxWGs = maxBind / bytesPerWG;
  if (dev.info.maxWGCount)
    maxWGs = std::min(maxWGs, static_cast<uint64_t>(dev.info.maxWGCount));
  uint64_t wantWGs = globalWIs / wgSize;
  uint32_t numGroups = static_cast<uint32_t>(std::max<uint64_t>(1, std::min(wantWGs, maxWGs)));
  globalWIs = static_cast<uint64_t>(numGroups) * wgSize;
  uint64_t bufferBytes = static_cast<uint64_t>(numGroups) * bytesPerWG;

  WGPUBuffer outputBuf = dev.createBuffer(bufferBytes, WGPUBufferUsage_Storage);
  if (!outputBuf)
  {
    log->note("Failed to allocate WebGPU output buffer\n");
    return -1;
  }

  const uint64_t uniformBytes = std::max<uint64_t>(16, d.uniformSize);
  WGPUBuffer uniformBuf = dev.createBuffer(
      uniformBytes, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
  if (!uniformBuf)
  {
    rt.api.wgpuBufferDestroy(outputBuf);
    rt.api.wgpuBufferRelease(outputBuf);
    log->note("Failed to allocate WebGPU uniform buffer\n");
    return -1;
  }
  if (d.uniformData && d.uniformSize > 0)
    rt.api.wgpuQueueWriteBuffer(dev.queue, uniformBuf, 0, d.uniformData, d.uniformSize);

  // Bind group layout: binding 0 = read/write storage, binding 1 = uniform.
  WGPUBindGroupLayoutEntry bglEntries[2] = {
    WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
    WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
  };
  bglEntries[0].binding = 0;
  bglEntries[0].visibility = WGPUShaderStage_Compute;
  bglEntries[0].buffer.type = WGPUBufferBindingType_Storage;
  bglEntries[1].binding = 1;
  bglEntries[1].visibility = WGPUShaderStage_Compute;
  bglEntries[1].buffer.type = WGPUBufferBindingType_Uniform;

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
  bgEntries[0].buffer = outputBuf;
  bgEntries[0].offset = 0;
  bgEntries[0].size = bufferBytes;
  bgEntries[1].binding = 1;
  bgEntries[1].buffer = uniformBuf;
  bgEntries[1].offset = 0;
  bgEntries[1].size = uniformBytes;

  WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bgDesc.layout = bgl;
  bgDesc.entryCount = 2;
  bgDesc.entries = bgEntries;
  WGPUBindGroup bindGroup = rt.api.wgpuDeviceCreateBindGroup(dev.device, &bgDesc);

  auto timeShape = [&](const std::string &wgsl, bool *built) -> float
  {
    WGPUComputePipeline pipe = dev.createComputePipeline(wgsl, pipeLayout);
    *built = (pipe != nullptr);
    if (!pipe)
      return -1.0f;
    float timed = runKernel(dev, pipe, bindGroup, numGroups,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                            /*serialize=*/false);
    rt.api.wgpuComputePipelineRelease(pipe);
    return timed;
  };

  auto toValue = [&](float timed) -> float
  {
    return static_cast<float>(
        static_cast<double>(globalWIs) * static_cast<double>(d.workPerWI) * 1e6 / timed);
  };

  for (const auto &v : d.variants)
  {
    bool built = false;
    float timed = timeShape(v.wgsl, &built);
    if (!built)
    {
      test.skip(v.label, ResultStatus::Error, "Pipeline creation failed", note(v.description));
      continue;
    }
    if (timed <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "WebGPU dispatch failed", note(v.description));
      continue;
    }
    float value = toValue(timed);

    if (!v.altWgsl.empty())
    {
      bool altBuilt = false;
      float altTimed = timeShape(v.altWgsl, &altBuilt);
      if (altBuilt && altTimed > 0.0f)
      {
        double altValue = toValue(altTimed);
        CLPEAK_VLOG("%s %s: first shape %.1f, alt shape %.1f %s\n",
                    d.resultTag, v.label, value, altValue, d.unit);
        if (altValue > static_cast<double>(value) * MAX_ALT_CHAIN_RATIO)
        {
          CLPEAK_VLOG("%s %s: alt chain %.1fx faster -- rejecting it as a compiler fold\n",
                      d.resultTag, v.label, altValue / value);
        }
        else if (altValue > value)
        {
          value = static_cast<float>(altValue);
        }
      }
    }

    test.emit(v.label, value, note(v.description));
  }

  rt.api.wgpuBindGroupRelease(bindGroup);
  rt.api.wgpuPipelineLayoutRelease(pipeLayout);
  rt.api.wgpuBindGroupLayoutRelease(bgl);
  rt.api.wgpuBufferDestroy(uniformBuf);
  rt.api.wgpuBufferRelease(uniformBuf);
  rt.api.wgpuBufferDestroy(outputBuf);
  rt.api.wgpuBufferRelease(outputBuf);
  return 0;
}

#endif // ENABLE_WEBGPU
