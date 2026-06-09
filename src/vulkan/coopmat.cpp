#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstddef>   // offsetof
#include <string>

// ---------------------------------------------------------------------------
// Cooperative matrix (tensor-core) umbrella.
//
// Runs every dtype combination the driver advertises.  The tile shape
// (M/N/K) is whatever vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
// reported for that dtype -- selected once in vkPeak::enumerate and carried
// in dev.info.coopmat* -- and is bound into the shader as specialization
// constants here, so a single SPIR-V module per dtype runs whatever shape
// the hardware exposes (K=16 for fp16/bf16, K=32 for NVIDIA's 8-bit types,
// and anything else a driver chooses to advertise).  Each dtype shares the
// same scaffolding via runComputeKernel -- only the shader, buffer element
// type, push value, and label strings differ.
// ---------------------------------------------------------------------------

namespace {

// Number of explicit independent accumulator chains in the coopmat shaders
// (c0..c3).  Exposes ILP so the tensor unit issues back-to-back MMAs instead
// of stalling on a single accumulator's dependency -- single-chain is
// latency-bound far below peak on un-throttled paths (fp16/bf16 on NVIDIA).
// MUST match the explicit accumulator count hard-coded in the .comp shaders.
const uint32_t COOPMAT_CHAINS = 4;

// Plain-old-data spec-constant payload (constant_id 0..3 in the shaders).
struct CoopSpecData { uint32_t M, N, K; int32_t iters; };

// Spec-constant storage for one tile.  Must outlive the runComputeKernel call
// that consumes specInfo, so callers declare it in the dispatching scope.
struct CoopTileRun {
  CoopSpecData             data;
  VkSpecializationMapEntry entries[4];
  VkSpecializationInfo     specInfo;
  std::string              title;
};

// Bind a selected tile into a desc: build the spec constants, scale the outer
// loop so total work-per-WI (across all COOPMAT_CHAINS chains) stays
// ~COOPMAT_WORK_PER_WI regardless of tile volume, and label the row with the
// actual MxNxK that runs.
void bindCoopTile(CoopTileRun &r, vk_compute_desc_t &d,
                  const coopmat_tile_t &t, uint32_t wgSize,
                  const char *dtypeLabel)
{
  const uint64_t volume = (uint64_t)t.M * t.N * t.K;   // MACs per coopMatMulAdd
  // Total MMAs/WI = COOPMAT_CHAINS*iters; hold total work ~= COOPMAT_WORK_PER_WI.
  uint64_t iters = ((uint64_t)COOPMAT_WORK_PER_WI * wgSize) / (volume * 2 * COOPMAT_CHAINS);
  if (iters < 1) iters = 1;

  r.data = { t.M, t.N, t.K, (int32_t)iters };
  r.entries[0] = { 0, (uint32_t)offsetof(CoopSpecData, M),     sizeof(uint32_t) };
  r.entries[1] = { 1, (uint32_t)offsetof(CoopSpecData, N),     sizeof(uint32_t) };
  r.entries[2] = { 2, (uint32_t)offsetof(CoopSpecData, K),     sizeof(uint32_t) };
  r.entries[3] = { 3, (uint32_t)offsetof(CoopSpecData, iters), sizeof(int32_t) };
  r.specInfo.mapEntryCount = 4;
  r.specInfo.pMapEntries   = r.entries;
  r.specInfo.dataSize      = sizeof(r.data);
  r.specInfo.pData         = &r.data;
  r.title  = std::string("Cooperative-matrix ") + dtypeLabel + " " +
             std::to_string(t.M) + "x" + std::to_string(t.N) + "x" + std::to_string(t.K);

  d.specInfo      = &r.specInfo;
  d.title         = r.title.c_str();
  d.wgSize        = wgSize;
  d.outElemsPerWG = t.M * t.N;
  // Reported work per WI = 2*MACs*COOPMAT_CHAINS*ITERS / subgroup-size; exact
  // since M*N is a multiple of the subgroup width for every advertised tile.
  d.workPerWI     = (uint32_t)((volume * 2 * COOPMAT_CHAINS * iters) / wgSize);
}

} // namespace

int vkPeak::runCoopMatrix(VulkanDevice &dev, benchmark_config_t &cfg, bool intPart)
{
  // One subgroup per work-group: each subgroup collectively computes one MxN
  // output tile.  32 matches NVIDIA / AMD RDNA3+ / Intel Arc subgroup widths.
  const uint32_t coopWGSize = 32;

  if (!intPart) {
#ifdef VK_HAS_COOPMAT_FP32
    {
      float A = 1.3f;
      CoopTileRun r;
      vk_compute_desc_t d = {};
      d.resultTag   = "coopmat_fp32";
      d.metricLabel = "coopmat_fp32";
      d.unit        = "tflops";
      d.unitDivider = 1e12;
      d.elemSize    = sizeof(float);
      d.pushData    = &A;
      d.pushSize    = sizeof(A);
      if (dev.info.coopmatFP32.supported) {
        d.spirv     = vk_shaders::coopmat_fp32;
        d.spirvSize = vk_shaders::coopmat_fp32_size;
        bindCoopTile(r, d, dev.info.coopmatFP32, coopWGSize, "fp32xfp32+fp32");
      } else {
        d.title   = "Cooperative-matrix fp32xfp32+fp32";
        d.skip    = true;
        d.skipMsg = "No fp32xfp32+fp32 coopmat property! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP16
    {
      float A = 1.3f;
      CoopTileRun r;
      vk_compute_desc_t d = {};
      d.resultTag   = "coopmat_fp16";
      d.metricLabel = "coopmat_fp16";
      d.unit        = "tflops";
      d.unitDivider = 1e12;
      d.elemSize    = sizeof(float);
      d.pushData    = &A;
      d.pushSize    = sizeof(A);
      if (dev.info.coopmatFP16.supported) {
        d.spirv     = vk_shaders::coopmat_fp16;
        d.spirvSize = vk_shaders::coopmat_fp16_size;
        bindCoopTile(r, d, dev.info.coopmatFP16, coopWGSize, "fp16xfp16+fp32");
      } else {
        d.title   = "Cooperative-matrix fp16xfp16+fp32";
        d.skip    = true;
        d.skipMsg = "No fp16xfp16+fp32 coopmat property! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_BF16
    {
      float A = 1.3f;
      CoopTileRun r;
      vk_compute_desc_t d = {};
      d.resultTag   = "coopmat_bf16";
      d.metricLabel = "coopmat_bf16";
      d.unit        = "tflops";
      d.unitDivider = 1e12;
      d.elemSize    = sizeof(float);
      d.pushData    = &A;
      d.pushSize    = sizeof(A);
      if (dev.info.coopmatBF16.supported) {
        d.spirv     = vk_shaders::coopmat_bf16;
        d.spirvSize = vk_shaders::coopmat_bf16_size;
        bindCoopTile(r, d, dev.info.coopmatBF16, coopWGSize, "bf16xbf16+fp32");
      } else {
        d.title   = "Cooperative-matrix bf16xbf16+fp32";
        d.skip    = true;
        d.skipMsg = "No bf16xbf16+fp32 coopmat property! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E4M3
    {
      float A = 1.3f;
      CoopTileRun r;
      vk_compute_desc_t d = {};
      d.resultTag   = "coopmat_fp8_e4m3";
      d.metricLabel = "coopmat_fp8_e4m3";
      d.unit        = "tflops";
      d.unitDivider = 1e12;
      d.elemSize    = sizeof(float);
      d.pushData    = &A;
      d.pushSize    = sizeof(A);
      // Two gates: the float8 feature must be enabled at device creation
      // (else pipeline creation fails) AND a matching tile must be advertised.
      if (dev.info.fp8Supported && dev.info.coopmatFP8E4M3.supported) {
        d.spirv     = vk_shaders::coopmat_fp8_e4m3;
        d.spirvSize = vk_shaders::coopmat_fp8_e4m3_size;
        bindCoopTile(r, d, dev.info.coopmatFP8E4M3, coopWGSize, "fp8(E4M3)xfp8(E4M3)+fp32");
      } else {
        d.title   = "Cooperative-matrix fp8(E4M3)xfp8(E4M3)+fp32";
        d.skip    = true;
        d.skipMsg = "No fp8-E4M3 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E5M2
    {
      float A = 1.3f;
      CoopTileRun r;
      vk_compute_desc_t d = {};
      d.resultTag   = "coopmat_fp8_e5m2";
      d.metricLabel = "coopmat_fp8_e5m2";
      d.unit        = "tflops";
      d.unitDivider = 1e12;
      d.elemSize    = sizeof(float);
      d.pushData    = &A;
      d.pushSize    = sizeof(A);
      if (dev.info.fp8Supported && dev.info.coopmatFP8E5M2.supported) {
        d.spirv     = vk_shaders::coopmat_fp8_e5m2;
        d.spirvSize = vk_shaders::coopmat_fp8_e5m2_size;
        bindCoopTile(r, d, dev.info.coopmatFP8E5M2, coopWGSize, "fp8(E5M2)xfp8(E5M2)+fp32");
      } else {
        d.title   = "Cooperative-matrix fp8(E5M2)xfp8(E5M2)+fp32";
        d.skip    = true;
        d.skipMsg = "No fp8-E5M2 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
  } // !intPart

#ifdef VK_HAS_COOPMAT_INT8
  if (intPart) {
    int32_t A = 3;
    CoopTileRun r;
    vk_compute_desc_t d = {};
    d.resultTag   = "coopmat_int8";
    d.metricLabel = "coopmat_int8";
    d.unit        = "tops";
    d.unitDivider = 1e12;
    d.elemSize    = sizeof(int32_t);
    d.pushData    = &A;
    d.pushSize    = sizeof(A);
    if (dev.info.coopmatINT8.supported) {
      d.spirv     = vk_shaders::coopmat_int8;
      d.spirvSize = vk_shaders::coopmat_int8_size;
      bindCoopTile(r, d, dev.info.coopmatINT8, coopWGSize, "int8xint8+int32");
    } else {
      d.title   = "Cooperative-matrix int8xint8+int32";
      d.skip    = true;
      d.skipMsg = "No int8xint8+int32 coopmat property! Skipped";
    }
    runComputeKernel(dev, cfg, d);
  } // if (intPart)
#endif
  return 0;
}

#endif // ENABLE_VULKAN
