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
    { "int",   vk_shaders::compute_int32_v1, vk_shaders::compute_int32_v1_size, vkWidthNote(1),
      VK_ALT_SHADER(compute_int32_v1) },
#ifdef VK_HAS_COMPUTE_INT32_V2
    { "int2",  vk_shaders::compute_int32_v2, vk_shaders::compute_int32_v2_size, vkWidthNote(2),
      VK_ALT_SHADER(compute_int32_v2) },
#endif
#ifdef VK_HAS_COMPUTE_INT32_V4
    { "int4",  vk_shaders::compute_int32_v4, vk_shaders::compute_int32_v4_size, vkWidthNote(4),
      VK_ALT_SHADER(compute_int32_v4) },
#endif
  };
  int32_t A = 4;
  vk_compute_desc_t d = {};
  d.title       = "Integer compute int32";
  d.resultTag   = "integer_compute";
  d.unit        = "ops";
  d.description = "Peak 32-bit integer arithmetic rate.";
  d.shape       = TestShape::Homogeneous;
  d.axis        = "vector width";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_INT_WORK_PER_WI;
  d.elemSize    = sizeof(int32_t);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef VK_HAS_COMPUTE_INT8_DP_V1
int vkPeak::runComputeInt8DP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // v1 = one dp4a chain per thread, v2/v4 = two/four independent ones.  Each
  // is raced in both of shaders/dp4a_chain.glsl's shapes, the second at two
  // subgroup widths too.
  static const vk_compute_variant_t variants[] = {
    { "int8_dp",  vk_shaders::compute_int8_dp_v1, vk_shaders::compute_int8_dp_v1_size,
      "One dependent chain of dot products.",
      VK_ALT_SHADER(compute_int8_dp_v1) },
#ifdef VK_HAS_COMPUTE_INT8_DP_V2
    { "int8_dp2", vk_shaders::compute_int8_dp_v2, vk_shaders::compute_int8_dp_v2_size,
      "2 independent chains.",
      VK_ALT_SHADER(compute_int8_dp_v2) },
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V4
    { "int8_dp4", vk_shaders::compute_int8_dp_v4, vk_shaders::compute_int8_dp_v4_size,
      "4 independent chains.",
      VK_ALT_SHADER(compute_int8_dp_v4) },
#endif
  };
  int32_t A = 4;
  vk_compute_desc_t d = {};
  d.title       = "INT8 dot-product compute";
  d.resultTag   = "integer_compute_int8_dp";
  d.unit        = "ops";
  d.description = "Peak rate of the 4-way int8 dot-product instruction, "
                  "without the matrix engine.";
  d.shape       = TestShape::Homogeneous;
  // Independent chains, not wider vectors: each reading gives the hardware
  // more dot products to have in flight at once.
  d.axis        = "chains in flight";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize    = sizeof(int32_t);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.raceHalfSubgroup = true;
  d.skip        = !dev.info.int8DotProductSupported;
  d.skipMsg     = "VK_KHR_shader_integer_dot_product / shaderInt8 not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#endif // ENABLE_VULKAN
