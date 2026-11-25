#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include "wgsl_shaders.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <algorithm>
#include <string>

// ---------------------------------------------------------------------------
// Image (texture) bandwidth benchmark (WebGPU)
//
// 4096x4096 RGBA32Float 2D texture read with textureLoad (integer coordinates,
// no sampler).  Races the same four walk shapes as Vulkan/OpenCL using WGSL
// pipeline override constants so power-of-two divides fold into shifts/masks.
// ---------------------------------------------------------------------------

int WebgpuPeak::runImageBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const WebgpuRuntime &rt = *dev.rt;

  logger::TestSpec testSpec;
  testSpec.tag = "image_memory_bandwidth";
  testSpec.display = "Image memory bandwidth";
  testSpec.unit = "bps";
  testSpec.description =
      "Image read bandwidth, reading each pixel once.";
  testSpec.shape = TestShape::Homogeneous;
  auto test = currentDeviceScope->beginTest(testSpec);

  const char *fetchNote = "One RGBA fp32 pixel (16 bytes) per fetch.";

  uint32_t imgW = 4096, imgH = 4096;
  while (imgW > dev.info.maxTextureDimension2D && imgW > 256) imgW /= 2;
  while (imgH > dev.info.maxTextureDimension2D && imgH > 256) imgH /= 2;
  // Ensure texture + staging buffer fit within maxBufferSize and system memory budget.
  uint64_t maxTexBytes = clpeak::memoryBudget(static_cast<uint64_t>(imgW) * imgH * 16ULL, 4);
  if (dev.info.maxBufferSize > 0)
    maxTexBytes = std::min(maxTexBytes, dev.info.maxBufferSize);
  while (static_cast<uint64_t>(imgW) * imgH * 16ULL > maxTexBytes)
  {
    if (imgH > 256)
      imgH /= 2;
    else if (imgW > 256)
      imgW /= 2;
    else
      break;
  }

  const uint32_t wgSize = 256;
  uint32_t numGroups = (imgW * imgH) / IMAGE_FETCH_PER_WI / wgSize;
  if (numGroups == 0) numGroups = 1;
  uint64_t globalWIs = static_cast<uint64_t>(numGroups) * wgSize;
  uint64_t outBytes = globalWIs * sizeof(float);
  uint64_t texBytes = static_cast<uint64_t>(imgW) * imgH * 16ULL;

  WGPUTextureDescriptor texDesc = WGPU_TEXTURE_DESCRIPTOR_INIT;
  texDesc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
  texDesc.dimension = WGPUTextureDimension_2D;
  texDesc.size = {imgW, imgH, 1};
  texDesc.format = WGPUTextureFormat_RGBA32Float;
  texDesc.mipLevelCount = 1;
  texDesc.sampleCount = 1;

  WGPUTexture tex = dev.createTexture(&texDesc);
  if (!tex)
  {
    test.skip("float4", ResultStatus::Error, "Texture create failed", fetchNote);
    return -1;
  }

  // Upload pseudo-random data via a staging buffer to defeat compression.
  {
    WGPUBuffer staging = dev.createBuffer(
        texBytes, WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
        /*mappedAtCreation=*/true);
    if (!staging)
    {
      rt.api.wgpuTextureDestroy(tex);
      rt.api.wgpuTextureRelease(tex);
      test.skip("float4", ResultStatus::Error, "Staging buffer alloc failed", fetchNote);
      return -1;
    }
    void *mapped = rt.api.wgpuBufferGetMappedRange(staging, 0, texBytes);
    if (!mapped)
    {
      rt.api.wgpuBufferUnmap(staging);
      rt.api.wgpuBufferDestroy(staging);
      rt.api.wgpuBufferRelease(staging);
      rt.api.wgpuTextureDestroy(tex);
      rt.api.wgpuTextureRelease(tex);
      test.skip("float4", ResultStatus::Error, "Staging buffer mapping failed", fetchNote);
      return -1;
    }
    populate(static_cast<float *>(mapped), static_cast<uint64_t>(imgW) * imgH * 4ULL);
    rt.api.wgpuBufferUnmap(staging);

    WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder enc = rt.api.wgpuDeviceCreateCommandEncoder(dev.device, &encDesc);
    WGPUTexelCopyBufferInfo src = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    src.buffer = staging;
    src.layout.offset = 0;
    src.layout.bytesPerRow = imgW * 16;
    src.layout.rowsPerImage = imgH;
    WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    dst.texture = tex;
    dst.mipLevel = 0;
    dst.origin = {0, 0, 0};
    dst.aspect = WGPUTextureAspect_All;
    WGPUExtent3D extent = {imgW, imgH, 1};
    rt.api.wgpuCommandEncoderCopyBufferToTexture(enc, &src, &dst, &extent);
    WGPUCommandBufferDescriptor cbDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd = rt.api.wgpuCommandEncoderFinish(enc, &cbDesc);
    rt.api.wgpuCommandEncoderRelease(enc);
    bool ok = dev.submitAndWait(cmd);
    rt.api.wgpuCommandBufferRelease(cmd);
    rt.api.wgpuBufferDestroy(staging);
    rt.api.wgpuBufferRelease(staging);
    if (!ok)
    {
      rt.api.wgpuTextureDestroy(tex);
      rt.api.wgpuTextureRelease(tex);
      test.skip("float4", ResultStatus::Error, "Texture upload failed", fetchNote);
      return -1;
    }
  }

  WGPUBuffer outBuf = dev.createBuffer(outBytes, WGPUBufferUsage_Storage);
  if (!outBuf)
  {
    rt.api.wgpuTextureDestroy(tex);
    rt.api.wgpuTextureRelease(tex);
    test.skip("float4", ResultStatus::Error, "Output buffer creation failed", fetchNote);
    return -1;
  }
  WGPUTextureView texView = rt.api.wgpuTextureCreateView(tex, nullptr);

  WGPUBindGroupLayoutEntry bglEntries[2] = {
    WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
    WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
  };
  bglEntries[0].binding = 0;
  bglEntries[0].visibility = WGPUShaderStage_Compute;
  bglEntries[0].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
  bglEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D;
  bglEntries[0].texture.multisampled = WGPU_FALSE;
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
  bgEntries[0].textureView = texView;
  bgEntries[1].binding = 1;
  bgEntries[1].buffer = outBuf;
  bgEntries[1].offset = 0;
  bgEntries[1].size = outBytes;

  WGPUBindGroupDescriptor bgDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
  bgDesc.layout = bgl;
  bgDesc.entryCount = 2;
  bgDesc.entries = bgEntries;
  WGPUBindGroup bindGroup = rt.api.wgpuDeviceCreateBindGroup(dev.device, &bgDesc);

  struct Shape { const char *label; uint32_t tileW, tileH, walk; };
  const Shape shapes[] = {
    {"row",       imgW, 1,  0},
    {"col",       imgW, 1,  1},
    {"tile2x2",   2,    2,  0},
    {"tile16x16", 16,   16, 0},
  };

  const std::string wgsl = wgpu_shaders::makeImageBandwidthWgsl();
  const uint64_t bytes = static_cast<uint64_t>(IMAGE_FETCH_PER_WI) * 4 * sizeof(float) * globalWIs;
  float best = 0.0f;

  for (const Shape &s : shapes)
  {
    WGPUConstantEntry constants[5] = {
      WGPU_CONSTANT_ENTRY_INIT, WGPU_CONSTANT_ENTRY_INIT,
      WGPU_CONSTANT_ENTRY_INIT, WGPU_CONSTANT_ENTRY_INIT,
      WGPU_CONSTANT_ENTRY_INIT,
    };
    constants[0].key = wgpuStrView("IMG_W");  constants[0].value = static_cast<double>(imgW);
    constants[1].key = wgpuStrView("IMG_H");  constants[1].value = static_cast<double>(imgH);
    constants[2].key = wgpuStrView("TILE_W"); constants[2].value = static_cast<double>(s.tileW);
    constants[3].key = wgpuStrView("TILE_H"); constants[3].value = static_cast<double>(s.tileH);
    constants[4].key = wgpuStrView("WALK");   constants[4].value = static_cast<double>(s.walk);

    WGPUComputePipeline pipe = dev.createComputePipeline(wgsl, pipeLayout, constants, 5);
    if (!pipe)
    {
      log->note(std::string("Image pipeline creation failed for ") + s.label + "\n");
      continue;
    }

    float us = runKernel(dev, pipe, bindGroup, numGroups,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                         /*serialize=*/true);
    rt.api.wgpuComputePipelineRelease(pipe);

    float bps = (us > 0.0f) ? static_cast<float>(static_cast<double>(bytes) / us * 1e6) : 0.0f;
    best = std::max(best, bps);
    CLPEAK_VLOG("image_memory_bandwidth: %-16s %.1f B/s\n", s.label, bps);
  }

  if (best <= 0.0f)
    test.skip("float4", ResultStatus::Error, "WebGPU dispatch failed", fetchNote);
  else
    test.emit("float4", best, fetchNote);

  rt.api.wgpuBindGroupRelease(bindGroup);
  rt.api.wgpuPipelineLayoutRelease(pipeLayout);
  rt.api.wgpuBindGroupLayoutRelease(bgl);
  rt.api.wgpuBufferDestroy(outBuf);
  rt.api.wgpuBufferRelease(outBuf);
  rt.api.wgpuTextureViewRelease(texView);
  rt.api.wgpuTextureDestroy(tex);
  rt.api.wgpuTextureRelease(tex);
  return 0;
}

#endif // ENABLE_WEBGPU
