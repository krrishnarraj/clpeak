#ifdef ENABLE_WEBGPU

#include "wgsl_shaders.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Integer compute benchmarks (WebGPU / WGSL).
// ---------------------------------------------------------------------------

int WebgpuPeak::runComputeInt32(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const int32_t params[4] = {4, 0, 0, 0};
  wgpu_compute_desc_t d;
  d.title = "Integer compute int32";
  d.resultTag = "integer_compute";
  d.unit = "ops";
  d.description = "Peak 32-bit integer arithmetic rate.";
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.variants = {
    {"int",  wgpu_shaders::makeComputeInt32Wgsl(1, false),
     wgpuWidthNote(1), wgpu_shaders::makeComputeInt32Wgsl(1, true)},
    {"int2", wgpu_shaders::makeComputeInt32Wgsl(2, false),
     wgpuWidthNote(2), wgpu_shaders::makeComputeInt32Wgsl(2, true)},
    {"int4", wgpu_shaders::makeComputeInt32Wgsl(4, false),
     wgpuWidthNote(4), wgpu_shaders::makeComputeInt32Wgsl(4, true)},
  };
  d.workPerWI = COMPUTE_INT_WORK_PER_WI;
  d.elemSize = sizeof(int32_t);
  d.uniformData = params;
  d.uniformSize = sizeof(params);
  return runComputeKernel(dev, cfg, d);
}

int WebgpuPeak::runComputeInt8DP(WebgpuDevice &dev, benchmark_config_t &cfg)
{
  const int32_t params[4] = {4, 0, 0, 0};
  wgpu_compute_desc_t d;
  d.title = "INT8 dot-product compute";
  d.resultTag = "integer_compute_int8_dp";
  d.unit = "ops";
  d.description = "Peak rate of the 4-way int8 dot-product instruction, without the matrix engine.";
  d.shape = TestShape::Homogeneous;
  d.axis = "chains in flight";
  d.variants = {
    {"int8_dp",  wgpu_shaders::makeComputeInt8DpWgsl(1, false),
     "One dependent chain of dot products.",
     wgpu_shaders::makeComputeInt8DpWgsl(1, true)},
    {"int8_dp2", wgpu_shaders::makeComputeInt8DpWgsl(2, false),
     "2 independent chains.",
     wgpu_shaders::makeComputeInt8DpWgsl(2, true)},
    {"int8_dp4", wgpu_shaders::makeComputeInt8DpWgsl(4, false),
     "4 independent chains.",
     wgpu_shaders::makeComputeInt8DpWgsl(4, true)},
  };
  d.workPerWI = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize = sizeof(int32_t);
  d.uniformData = params;
  d.uniformSize = sizeof(params);
  d.skip = !dev.info.packed4x8DotProductSupported;
  d.skipMsg = "packed_4x8_integer_dot_product WGSL feature not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_WEBGPU
