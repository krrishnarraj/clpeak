#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <algorithm>
#include <chrono>

// ---------------------------------------------------------------------------
// Host<->device transfer bandwidth (WebGPU)
//
// Measures copyBufferToBuffer bandwidth between WebGPU host-mappable staging
// buffers (MAP_WRITE|COPY_SRC for h2d, MAP_READ|COPY_DST for d2h) and a
// device-side storage buffer (STORAGE|COPY_SRC|COPY_DST).
// ---------------------------------------------------------------------------

int WebgpuPeak::runTransferBandwidth(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const WebgpuRuntime &rt = *dev.rt;

  uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ULL << 27);
  if (dev.info.maxBufferSize > 0)
    bytes = std::min(bytes, dev.info.maxBufferSize);
  bytes = std::min(bytes, clpeak::memoryBudget(bytes, 4));
  bytes &= ~255ULL;
  if (bytes == 0)
    bytes = 256;

  unsigned int forced = forceIters ? specifiedIters : 0;

  logger::TestSpec testSpec;
  testSpec.tag = "transfer_bandwidth";
  testSpec.display = "Transfer bandwidth";
  testSpec.unit = "bps";
  testSpec.description =
      "Copy bandwidth between host and device memory.";
  testSpec.shape = TestShape::Heterogeneous;
  testSpec.axis = "direction";
  auto test = currentDeviceScope->beginTest(testSpec);

  CLPEAK_VLOG("transfer_bandwidth: buffer size %llu MB\n",
              static_cast<unsigned long long>(bytes >> 20));

  WGPUBuffer devBuf = dev.createBuffer(
      bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
  if (!devBuf)
  {
    log->note("Failed to allocate WebGPU device transfer buffer\n");
    return -1;
  }

  auto runCopy = [&](WGPUBuffer src, WGPUBuffer dst) -> float
  {
    auto runBatch = [&](unsigned int n) -> float
    {
      WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
      WGPUCommandEncoder enc = rt.api.wgpuDeviceCreateCommandEncoder(dev.device, &encDesc);
      for (unsigned int i = 0; i < n; i++)
        rt.api.wgpuCommandEncoderCopyBufferToBuffer(enc, src, 0, dst, 0, bytes);
      WGPUCommandBufferDescriptor cbDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
      WGPUCommandBuffer cmd = rt.api.wgpuCommandEncoderFinish(enc, &cbDesc);
      rt.api.wgpuCommandEncoderRelease(enc);

      auto start = std::chrono::high_resolution_clock::now();
      bool ok = dev.submitAndWait(cmd);
      auto end = std::chrono::high_resolution_clock::now();
      rt.api.wgpuCommandBufferRelease(cmd);
      if (!ok)
        return -1.0f;
      return static_cast<float>(
          std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    };

    for (unsigned int w = 0; w < warmupCount; w++)
    {
      if (runBatch(1) < 0.0f)
        return -1.0f;
    }

    float probeUs = runBatch(1);
    if (probeUs < 0.0f)
      return -1.0f;
    if (probeUs <= 0.0f)
      probeUs = 1.0f;
    unsigned int iters = pickIters(static_cast<double>(probeUs), cfg.targetTimeUs, forced);

    float totalUs = runBatch(iters);
    if (totalUs <= 0.0f)
      return -1.0f;
    return totalUs / static_cast<float>(iters);
  };

  auto reportCopy = [&](const char *metric, const char *note, float us)
  {
    if (us <= 0.0f)
    {
      test.skip(metric, ResultStatus::Error, "WebGPU copy failed", note);
      return;
    }
    float bps = static_cast<float>(static_cast<double>(bytes) / us * 1e6);
    test.emit(metric, bps, note);
  };

  // H2D: MAP_WRITE|COPY_SRC -> devBuf (populated with pseudo-random data to
  // defeat compression; also initializes devBuf for the subsequent D2H test).
  {
    WGPUBuffer h2dBuf = dev.createBuffer(
        bytes, WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc,
        /*mappedAtCreation=*/true);
    if (!h2dBuf)
    {
      test.skip("h2d", ResultStatus::Error, "Staging buffer alloc failed", "Host to device.");
    }
    else
    {
      void *mapped = rt.api.wgpuBufferGetMappedRange(h2dBuf, 0, bytes);
      if (!mapped)
      {
        rt.api.wgpuBufferUnmap(h2dBuf);
        test.skip("h2d", ResultStatus::Error, "Staging buffer mapping failed", "Host to device.");
      }
      else
      {
        populate(static_cast<float *>(mapped), bytes / sizeof(float));
        rt.api.wgpuBufferUnmap(h2dBuf);
        reportCopy("h2d", "Host to device.", runCopy(h2dBuf, devBuf));
      }
      rt.api.wgpuBufferDestroy(h2dBuf);
      rt.api.wgpuBufferRelease(h2dBuf);
    }
  }

  // D2H: devBuf -> MAP_READ|COPY_DST.
  {
    WGPUBuffer d2hBuf = dev.createBuffer(
        bytes, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);
    if (!d2hBuf)
    {
      test.skip("d2h", ResultStatus::Error, "Readback buffer alloc failed", "Device to host.");
    }
    else
    {
      reportCopy("d2h", "Device to host.", runCopy(devBuf, d2hBuf));
      rt.api.wgpuBufferDestroy(d2hBuf);
      rt.api.wgpuBufferRelease(d2hBuf);
    }
  }

  rt.api.wgpuBufferDestroy(devBuf);
  rt.api.wgpuBufferRelease(devBuf);
  return 0;
}

#endif // ENABLE_WEBGPU
