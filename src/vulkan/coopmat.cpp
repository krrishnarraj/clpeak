#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Cooperative matrix (tensor-core) umbrella.
//
// Runs every dtype combination the driver advertises at the canonical
// 16x16x16 subgroup-scope tile (queried once in VulkanDevice::init).  Each
// dtype shares the same scaffolding via runComputeKernel -- only the shader,
// buffer element type, and label strings differ.  Adding FP8 / INT4 in
// Phase 2 reduces to: compile a coopmat_fp8.comp, query the matching
// component-type enums, and add one more entry here.
// ---------------------------------------------------------------------------

int vkPeak::runCoopMatrix(VulkanDevice &dev, benchmark_config_t &cfg, bool intPart)
{
  // Coopmat shape constants: shaders hard-code 16x16x16 with 256 iters and
  // local_size_x=32 (one subgroup per work-group).  See COOPMAT_WORK_PER_WI.
  const uint32_t coopWGSize  = 32;
  const uint32_t coopOutElems = 16 * 16;  // M*N tile written per WG
  const uint32_t coopWork    = COOPMAT_WORK_PER_WI;

  if (!intPart) {
#ifdef VK_HAS_COOPMAT_FP32
    {
      float A = 1.3f;
      vk_compute_desc_t d = {};
      d.title          = "Cooperative-matrix fp32xfp32+fp32 16x16x16 (TFLOPS)";
      d.resultTag      = "coopmat_fp32";
      d.metricLabel    = "coopmat_fp32";
      d.unit           = "tflops";
      d.unitDivider    = 1e12;
      d.spirv          = vk_shaders::coopmat_fp32;
      d.spirvSize      = vk_shaders::coopmat_fp32_size;
      d.workPerWI      = coopWork;
      d.elemSize       = sizeof(float);
      d.wgSize         = coopWGSize;
      d.outElemsPerWG  = coopOutElems;
      d.pushData       = &A;
      d.pushSize       = sizeof(A);
      d.skip           = !dev.info.coopmatFP32Supported;
      d.skipMsg        = "No 16x16x16 fp32xfp32+fp32 coopmat property! Skipped";
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP16
    {
      float A = 1.3f;
      vk_compute_desc_t d = {};
      d.title          = "Cooperative-matrix fp16xfp16+fp32 16x16x16 (TFLOPS)";
      d.resultTag      = "coopmat_fp16";
      d.metricLabel    = "coopmat_fp16";
      d.unit           = "tflops";
      d.unitDivider    = 1e12;
      d.spirv          = vk_shaders::coopmat_fp16;
      d.spirvSize      = vk_shaders::coopmat_fp16_size;
      d.workPerWI      = coopWork;
      d.elemSize       = sizeof(float);
      d.wgSize         = coopWGSize;
      d.outElemsPerWG  = coopOutElems;
      d.pushData       = &A;
      d.pushSize       = sizeof(A);
      d.skip           = !dev.info.coopmatFP16Supported;
      d.skipMsg        = "No 16x16x16 fp16xfp16+fp32 coopmat property! Skipped";
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_BF16
    {
      float A = 1.3f;
      vk_compute_desc_t d = {};
      d.title          = "Cooperative-matrix bf16xbf16+fp32 16x16x16 (TFLOPS)";
      d.resultTag      = "coopmat_bf16";
      d.metricLabel    = "coopmat_bf16";
      d.unit           = "tflops";
      d.unitDivider    = 1e12;
      d.spirv          = vk_shaders::coopmat_bf16;
      d.spirvSize      = vk_shaders::coopmat_bf16_size;
      d.workPerWI      = coopWork;
      d.elemSize       = sizeof(float);
      d.wgSize         = coopWGSize;
      d.outElemsPerWG  = coopOutElems;
      d.pushData       = &A;
      d.pushSize       = sizeof(A);
      d.skip           = !dev.info.coopmatBF16Supported;
      d.skipMsg        = "No 16x16x16 bf16xbf16+fp32 coopmat property! Skipped";
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E4M3
    {
      float A = 1.3f;
      vk_compute_desc_t d = {};
      d.title          = "Cooperative-matrix fp8(E4M3)xfp8(E4M3)+fp32 16x16x16 (TFLOPS)";
      d.resultTag      = "coopmat_fp8_e4m3";
      d.metricLabel    = "coopmat_fp8_e4m3";
      d.unit           = "tflops";
      d.unitDivider    = 1e12;
      d.spirv          = vk_shaders::coopmat_fp8_e4m3;
      d.spirvSize      = vk_shaders::coopmat_fp8_e4m3_size;
      d.workPerWI      = coopWork;
      d.elemSize       = sizeof(float);
      d.wgSize         = coopWGSize;
      d.outElemsPerWG  = coopOutElems;
      d.pushData       = &A;
      d.pushSize       = sizeof(A);
      d.skip           = !(dev.info.fp8Supported && dev.info.coopmatFP8E4M3Supported);
      d.skipMsg        = "No fp8-E4M3 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E5M2
    {
      float A = 1.3f;
      vk_compute_desc_t d = {};
      d.title          = "Cooperative-matrix fp8(E5M2)xfp8(E5M2)+fp32 16x16x16 (TFLOPS)";
      d.resultTag      = "coopmat_fp8_e5m2";
      d.metricLabel    = "coopmat_fp8_e5m2";
      d.unit           = "tflops";
      d.unitDivider    = 1e12;
      d.spirv          = vk_shaders::coopmat_fp8_e5m2;
      d.spirvSize      = vk_shaders::coopmat_fp8_e5m2_size;
      d.workPerWI      = coopWork;
      d.elemSize       = sizeof(float);
      d.wgSize         = coopWGSize;
      d.outElemsPerWG  = coopOutElems;
      d.pushData       = &A;
      d.pushSize       = sizeof(A);
      d.skip           = !(dev.info.fp8Supported && dev.info.coopmatFP8E5M2Supported);
      d.skipMsg        = "No fp8-E5M2 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
#endif
  } // !intPart

#if defined(VK_HAS_COOPMAT_INT8) || defined(VK_HAS_COOPMAT_INT8_K32)
  if (intPart) {
    // Select the shader variant matching whichever INT8 tile the driver
    // advertised.  K=16 is the generic path; NVIDIA tensor cores need K=32.
    int32_t A = 3;
    vk_compute_desc_t d = {};
    d.resultTag      = "coopmat_int8";
    d.metricLabel    = "coopmat_int8";
    d.unit           = "tops";
    d.unitDivider    = 1e12;
    d.workPerWI      = coopWork;
    d.elemSize       = sizeof(int32_t);
    d.wgSize         = coopWGSize;
    d.outElemsPerWG  = coopOutElems;
    d.pushData       = &A;
    d.pushSize       = sizeof(A);

    const char *titleK16 = "Cooperative-matrix int8xint8+int32 16x16x16 (TOPS)";
    const char *titleK32 = "Cooperative-matrix int8xint8+int32 16x16x32 (TOPS)";
    bool haveShaderK16 = false, haveShaderK32 = false;
#ifdef VK_HAS_COOPMAT_INT8
    haveShaderK16 = true;
#endif
#ifdef VK_HAS_COOPMAT_INT8_K32
    haveShaderK32 = true;
#endif

    if (dev.info.coopmatINT8K == 16 && haveShaderK16)
    {
#ifdef VK_HAS_COOPMAT_INT8
      d.title          = titleK16;
      d.spirv          = vk_shaders::coopmat_int8;
      d.spirvSize      = vk_shaders::coopmat_int8_size;
#endif
    }
    else if (dev.info.coopmatINT8K == 32 && haveShaderK32)
    {
#ifdef VK_HAS_COOPMAT_INT8_K32
      d.title          = titleK32;
      d.spirv          = vk_shaders::coopmat_int8_k32;
      d.spirvSize      = vk_shaders::coopmat_int8_k32_size;
#endif
    }
    else
    {
      // Driver advertised neither 16x16x16 nor 16x16x32 INT8 -- or the
      // corresponding shader didn't compile in this build.  Skip with a
      // label that names the probed tiles so the reason is obvious.
      d.title          = titleK16;
      d.skip           = true;
      d.skipMsg        = "No 16x16x{16,32} int8xint8+int32 coopmat property! Skipped";
      d.spirv          = nullptr;
      d.spirvSize      = 0;
    }
    runComputeKernel(dev, cfg, d);
  } // if (intPart)
#endif
  return 0;
}

#endif // ENABLE_VULKAN
