#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Floating-point compute benchmarks.
// Each is a thin wrapper that fills a vk_compute_desc_t and delegates to
// vkPeak::runComputeKernel.
// ---------------------------------------------------------------------------

int vkPeak::runComputeSP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  static const vk_compute_variant_t variants[] = {
    { "float",  vk_shaders::compute_sp_v1, vk_shaders::compute_sp_v1_size },
#ifdef VK_HAS_COMPUTE_SP_V2
    { "float2", vk_shaders::compute_sp_v2, vk_shaders::compute_sp_v2_size },
#endif
#ifdef VK_HAS_COMPUTE_SP_V4
    { "float4", vk_shaders::compute_sp_v4, vk_shaders::compute_sp_v4_size },
#endif
  };
  float A = 1.3f;
  vk_compute_desc_t d = {};
  d.title       = "Single-precision compute (GFLOPS)";
  d.resultTag   = "single_precision_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

#ifdef VK_HAS_COMPUTE_HP_V1
int vkPeak::runComputeHP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  static const vk_compute_variant_t variants[] = {
    { "half",   vk_shaders::compute_hp_v1, vk_shaders::compute_hp_v1_size },
#ifdef VK_HAS_COMPUTE_HP_V2
    { "half2",  vk_shaders::compute_hp_v2, vk_shaders::compute_hp_v2_size },
#endif
#ifdef VK_HAS_COMPUTE_HP_V4
    { "half4",  vk_shaders::compute_hp_v4, vk_shaders::compute_hp_v4_size },
#endif
  };
  float A = 1.3f;
  vk_compute_desc_t d = {};
  d.title       = "Half-precision compute fp16xfp16+fp16 (GFLOPS)";
  d.resultTag   = "half_precision_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.float16Supported;
  d.skipMsg     = "shaderFloat16 not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef VK_HAS_COMPUTE_DP_V1
int vkPeak::runComputeDP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  static const vk_compute_variant_t variants[] = {
    { "double",  vk_shaders::compute_dp_v1, vk_shaders::compute_dp_v1_size },
#ifdef VK_HAS_COMPUTE_DP_V2
    { "double2", vk_shaders::compute_dp_v2, vk_shaders::compute_dp_v2_size },
#endif
#ifdef VK_HAS_COMPUTE_DP_V4
    { "double4", vk_shaders::compute_dp_v4, vk_shaders::compute_dp_v4_size },
#endif
  };
  double A = 1.3;
  vk_compute_desc_t d = {};
  d.title       = "Double-precision compute (GFLOPS)";
  d.resultTag   = "double_precision_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_DP_WORK_PER_WI;
  d.elemSize    = sizeof(double);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.float64Supported;
  d.skipMsg     = "shaderFloat64 not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef VK_HAS_COMPUTE_MP_V1
int vkPeak::runComputeMP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // v1 = scalar fp16 (baseline; no HFMA2 packing).
  // v2 = f16vec2  (unlocks NVIDIA HFMA2 at 2x FP32 rate on shader cores).
  // v4 = f16vec4  (wider packing; informs AMD/Intel where issue rate
  //                exceeds two lanes per slot).
  static const vk_compute_variant_t variants[] = {
    { "mp",  vk_shaders::compute_mp_v1, vk_shaders::compute_mp_v1_size },
#ifdef VK_HAS_COMPUTE_MP_V2
    { "mp2", vk_shaders::compute_mp_v2, vk_shaders::compute_mp_v2_size },
#endif
#ifdef VK_HAS_COMPUTE_MP_V4
    { "mp4", vk_shaders::compute_mp_v4, vk_shaders::compute_mp_v4_size },
#endif
  };
  float A = 1.3f;
  vk_compute_desc_t d = {};
  d.title       = "Mixed-precision compute fp16xfp16+fp32 (GFLOPS)";
  d.resultTag   = "mixed_precision_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.float16Supported;
  d.skipMsg     = "shaderFloat16 not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef VK_HAS_COMPUTE_BF16_V1
int vkPeak::runComputeBF16(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // v1 / v2 / v4: same packing story as MP.  NVIDIA shader-core BF16
  // peaks at bf16vec2 via BMMA2-style packed multiply.
  static const vk_compute_variant_t variants[] = {
    { "bf16",  vk_shaders::compute_bf16_v1, vk_shaders::compute_bf16_v1_size },
#ifdef VK_HAS_COMPUTE_BF16_V2
    { "bf16_2", vk_shaders::compute_bf16_v2, vk_shaders::compute_bf16_v2_size },
#endif
#ifdef VK_HAS_COMPUTE_BF16_V4
    { "bf16_4", vk_shaders::compute_bf16_v4, vk_shaders::compute_bf16_v4_size },
#endif
  };
  float A = 1.3f;
  vk_compute_desc_t d = {};
  d.title       = "BF16 compute bf16xbf16+fp32 (GFLOPS)";
  d.resultTag   = "bfloat16_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.bfloat16Supported;
  d.skipMsg     = "VK_KHR_shader_bfloat16 / shaderBFloat16Type not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#endif // ENABLE_VULKAN
