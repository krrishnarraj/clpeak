#ifdef ENABLE_WEBGPU

#include "wgsl_shaders.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Floating-point compute benchmarks (WebGPU / WGSL).
// ---------------------------------------------------------------------------

int WebgpuPeak::runComputeSP(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const float params[4] = {1.3f, 0.0f, 0.0f, 0.0f};
  wgpu_compute_desc_t d;
  d.title = "Single-precision compute";
  d.resultTag = "single_precision_compute";
  d.unit = "flops";
  d.description = "Peak fp32 arithmetic rate of the device's ALUs, with no memory traffic.";
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.variants = {
    {"float",  wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F32, 1, false),
     wgpuWidthNote(1), wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F32, 1, true)},
    {"float2", wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F32, 2, false),
     wgpuWidthNote(2), wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F32, 2, true)},
    {"float4", wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F32, 4, false),
     wgpuWidthNote(4), wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F32, 4, true)},
  };
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.uniformData = params;
  d.uniformSize = sizeof(params);
  return runComputeKernel(dev, cfg, d);
}

int WebgpuPeak::runComputeHP(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const float params[4] = {1.3f, 0.0f, 0.0f, 0.0f};
  wgpu_compute_desc_t d;
  d.title = "Half-precision compute fp16xfp16+fp16";
  d.resultTag = "half_precision_compute";
  d.unit = "flops";
  d.description = "Peak fp16 arithmetic rate, with fp16 inputs and accumulator.";
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.variants = {
    {"half",  wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F16, 1, false),
     wgpuWidthNote(1), wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F16, 1, true)},
    {"half2", wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F16, 2, false),
     wgpuWidthNote(2), wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F16, 2, true)},
    {"half4", wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F16, 4, false),
     wgpuWidthNote(4), wgpu_shaders::makeComputeFpWgsl(wgpu_shaders::ScalarType::F16, 4, true)},
  };
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.uniformData = params;
  d.uniformSize = sizeof(params);
  d.skip = !dev.info.float16Supported;
  d.skipMsg = "shader-f16 feature not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int WebgpuPeak::runComputeMP(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const float params[4] = {1.3f, 0.0f, 0.0f, 0.0f};
  wgpu_compute_desc_t d;
  d.title = "Mixed-precision compute fp16xfp16+fp32";
  d.resultTag = "mixed_precision_compute";
  d.unit = "flops";
  d.description = "Peak rate of fp16 multiplies accumulated in fp32, without the matrix engine.";
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.variants = {
    {"mp",  wgpu_shaders::makeComputeMpWgsl(1, false),
     wgpuWidthNote(1), wgpu_shaders::makeComputeMpWgsl(1, true)},
    {"mp2", wgpu_shaders::makeComputeMpWgsl(2, false),
     wgpuWidthNote(2), wgpu_shaders::makeComputeMpWgsl(2, true)},
    {"mp4", wgpu_shaders::makeComputeMpWgsl(4, false),
     wgpuWidthNote(4), wgpu_shaders::makeComputeMpWgsl(4, true)},
  };
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.uniformData = params;
  d.uniformSize = sizeof(params);
  d.skip = !dev.info.float16Supported;
  d.skipMsg = "shader-f16 feature not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_WEBGPU
