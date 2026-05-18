#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Integer compute benchmarks.
// Each is a thin wrapper that fills a vk_compute_desc_t and delegates to
// vkPeak::runComputeKernel.
// ---------------------------------------------------------------------------

#ifdef VK_HAS_COMPUTE_INT32_V1
int vkPeak::runComputeInt32(VulkanDevice &dev, benchmark_config_t &cfg)
{
  static const vk_compute_variant_t variants[] = {
    { "int",   vk_shaders::compute_int32_v1, vk_shaders::compute_int32_v1_size },
#ifdef VK_HAS_COMPUTE_INT32_V2
    { "int2",  vk_shaders::compute_int32_v2, vk_shaders::compute_int32_v2_size },
#endif
#ifdef VK_HAS_COMPUTE_INT32_V4
    { "int4",  vk_shaders::compute_int32_v4, vk_shaders::compute_int32_v4_size },
#endif
  };
  int32_t A = 4;
  vk_compute_desc_t d = {};
  d.title       = "Integer compute int32 (GOPS)";
  d.resultTag   = "integer_compute";
  d.unit        = "gops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_INT_WORK_PER_WI;
  d.elemSize    = sizeof(int32_t);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef VK_HAS_COMPUTE_INT4_PACKED_V1
int vkPeak::runComputeInt4Packed(VulkanDevice &dev, benchmark_config_t &cfg)
{
  int32_t A = 3;
  vk_compute_desc_t d = {};
  d.title        = "Packed INT4 compute (emulated) (GOPS)";
  d.resultTag    = "int4_packed_compute";
  d.metricLabel  = "int4_packed";
  d.unit         = "gops";
  d.spirv        = vk_shaders::compute_int4_packed_v1;
  d.spirvSize    = vk_shaders::compute_int4_packed_v1_size;
  d.workPerWI    = COMPUTE_INT4_PACKED_WORK_PER_WI;
  d.elemSize     = sizeof(int32_t);
  d.pushData     = &A;
  d.pushSize     = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef VK_HAS_COMPUTE_INT8_DP_V1
int vkPeak::runComputeInt8DP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // v1 = single dp4a chain (serial through REPACK; dep-stall bound).
  // v2 = two parallel dp4a chains (double the independent work for the
  //       instruction issue queue to pipeline).
  // v4 = four parallel chains (enough to saturate dp4a issue rate on
  //       NVIDIA Turing+ / AMD RDNA2+ / Intel Xe+).
  static const vk_compute_variant_t variants[] = {
    { "int8_dp",  vk_shaders::compute_int8_dp_v1, vk_shaders::compute_int8_dp_v1_size },
#ifdef VK_HAS_COMPUTE_INT8_DP_V2
    { "int8_dp2", vk_shaders::compute_int8_dp_v2, vk_shaders::compute_int8_dp_v2_size },
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V4
    { "int8_dp4", vk_shaders::compute_int8_dp_v4, vk_shaders::compute_int8_dp_v4_size },
#endif
  };
  int32_t A = 4;
  vk_compute_desc_t d = {};
  d.title       = "INT8 dot-product compute (GOPS)";
  d.resultTag   = "integer_compute_int8_dp";
  d.unit        = "gops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize    = sizeof(int32_t);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.int8DotProductSupported;
  d.skipMsg     = "VK_KHR_shader_integer_dot_product / shaderInt8 not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#endif // ENABLE_VULKAN
