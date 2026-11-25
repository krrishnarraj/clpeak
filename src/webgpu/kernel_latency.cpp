#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include "wgsl_shaders.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <chrono>

// ---------------------------------------------------------------------------
// Kernel launch latency (WebGPU)
//
//   dispatch  : Skipped — WebGPU does not expose calibrated host/device clocks.
//   roundtrip : Wall clock around wgpuQueueSubmit + wgpuQueueOnSubmittedWorkDone
//               for a 1x1x1 no-op compute dispatch.
// ---------------------------------------------------------------------------

int WebgpuPeak::runKernelLatency(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const WebgpuRuntime &rt = *dev.rt;
  unsigned int iters = forceIters ? specifiedIters
                                  : (cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000);

  logger::TestSpec testSpec;
  testSpec.tag = "kernel_launch_latency";
  testSpec.display = "Kernel launch latency";
  testSpec.unit = "s";
  testSpec.description =
      "Time to launch an empty kernel, the fixed cost every dispatch pays.";
  testSpec.shape = TestShape::Heterogeneous;
  auto test = currentDeviceScope->beginTest(testSpec);

  const char *dispatchNote = "Host submit to kernel start, one way.";
  const char *roundtripNote = "Host submit to completion seen by the host.";

  WGPUPipelineLayoutDescriptor plDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
  plDesc.bindGroupLayoutCount = 0;
  plDesc.bindGroupLayouts = nullptr;
  WGPUPipelineLayout pipeLayout = rt.api.wgpuDeviceCreatePipelineLayout(dev.device, &plDesc);

  WGPUComputePipeline pipeline = dev.createComputePipeline(
      wgpu_shaders::makeKernelLatencyWgsl(), pipeLayout);
  if (!pipeline)
  {
    rt.api.wgpuPipelineLayoutRelease(pipeLayout);
    test.skip("dispatch", ResultStatus::Unsupported,
              "WebGPU does not expose calibrated host/device timestamps", dispatchNote);
    test.skip("roundtrip", ResultStatus::Error, "Pipeline creation failed", roundtripNote);
    return -1;
  }

  auto makeEmptyDispatch = [&]() -> WGPUCommandBuffer
  {
    WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder enc = rt.api.wgpuDeviceCreateCommandEncoder(dev.device, &encDesc);
    WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUComputePassEncoder pass = rt.api.wgpuCommandEncoderBeginComputePass(enc, &passDesc);
    rt.api.wgpuComputePassEncoderSetPipeline(pass, pipeline);
    rt.api.wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    rt.api.wgpuComputePassEncoderEnd(pass);
    rt.api.wgpuComputePassEncoderRelease(pass);
    WGPUCommandBufferDescriptor cbDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmd = rt.api.wgpuCommandEncoderFinish(enc, &cbDesc);
    rt.api.wgpuCommandEncoderRelease(enc);
    return cmd;
  };

  bool submitFailed = false;
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    WGPUCommandBuffer cmd = makeEmptyDispatch();
    bool ok = dev.submitAndWait(cmd);
    rt.api.wgpuCommandBufferRelease(cmd);
    if (!ok)
    {
      submitFailed = true;
      break;
    }
  }

  double totalRoundtripUs = 0.0;
  if (!submitFailed)
  {
    for (unsigned int i = 0; i < iters; i++)
    {
      WGPUCommandBuffer cmd = makeEmptyDispatch();
      auto start = std::chrono::high_resolution_clock::now();
      bool ok = dev.submitAndWait(cmd);
      auto end = std::chrono::high_resolution_clock::now();
      rt.api.wgpuCommandBufferRelease(cmd);
      if (!ok)
      {
        submitFailed = true;
        break;
      }
      totalRoundtripUs +=
          static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) /
          1000.0;
    }
  }

  test.skip("dispatch", ResultStatus::Unsupported,
            "WebGPU does not expose calibrated host/device timestamps", dispatchNote);
  if (submitFailed)
  {
    test.skip("roundtrip", ResultStatus::Error, "WebGPU submit failed", roundtripNote);
  }
  else
  {
    double avgRoundtripUs = totalRoundtripUs / static_cast<double>(iters);
    test.emit("roundtrip", static_cast<float>(avgRoundtripUs * 1e-6), roundtripNote);
  }

  rt.api.wgpuComputePipelineRelease(pipeline);
  rt.api.wgpuPipelineLayoutRelease(pipeLayout);
  return 0;
}

#endif // ENABLE_WEBGPU
