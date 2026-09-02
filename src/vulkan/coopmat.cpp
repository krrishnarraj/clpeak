#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstddef> // offsetof
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

namespace
{

  // Plain-old-data spec-constant payload (constant_id 0..3 in the shaders).
  // wgSize is the shader's local_size_x, declared as local_size_x_id = 3, so the
  // work-group and the pinned subgroup width always move together.
  struct CoopSpecData
  {
    uint32_t M, N, K, wgSize;
  };

  // Push payload: the value the shader derives its four A and four B fills from,
  // then the number of trips of the inner loop.  float and int32_t are both four
  // bytes at offset 0, so one struct serves every dtype -- only the shader's
  // spelling of the push block differs.
  //
  // The trip count is pushed rather than specialized on purpose.  A compile-time
  // trip count is an invitation for the driver to unroll the whole run, and a
  // fully unrolled run of an emulated tile is what a shader compiler chokes on;
  // it also leaves the loop open to being folded into a closed form, which has
  // inflated these rows before.  Push it and neither is possible.
  struct CoopPush
  {
    union
    {
      float f;
      int32_t i;
    } A;
    int32_t trips;
  };

  // Spec-constant storage for one tile.  Must outlive the runComputeKernel call
  // that consumes specInfo and pushData, so callers declare it in the
  // dispatching scope.
  struct CoopTileRun
  {
    CoopSpecData data;
    VkSpecializationMapEntry entries[4];
    VkSpecializationInfo specInfo;
    CoopPush push;
    std::string note;
  };

  // Bind a selected tile into a desc: build the spec constants, scale the trip
  // count so work-per-WI stays ~COOPMAT_WORK_PER_WI regardless of tile volume,
  // and record the actual MxNxK that runs.
  //
  // The shape goes on the reading's NOTE, not its name.  Different data types
  // land on different shapes on one device (NVIDIA gives the 8-bit types K=32
  // where fp16 gets K=16), so a name carrying the shape would differ between a
  // device that measured the reading and one that skipped it -- the same reading
  // under two ids.  The name stays the data type, which is what identifies it.
  //
  // The caller has already set r.push.A to the fill this dtype wants, and
  // d.metricLabel / d.metricDescription to what this reading is.
  void bindCoopTile(CoopTileRun &r, vk_compute_desc_t &d,
                    const coopmat_tile_t &t, uint32_t wgSize)
  {
    const uint64_t volume = (uint64_t)t.M * t.N * t.K; // MACs per coopMatMulAdd
    uint64_t mmas = ((uint64_t)COOPMAT_WORK_PER_WI * wgSize) / (volume * 2);
    uint64_t trips = mmas / COOPMAT_MMA_PER_TRIP;
    if (trips < 1)
      trips = 1;
    mmas = trips * COOPMAT_MMA_PER_TRIP; // what the shader will actually run

    r.data = {t.M, t.N, t.K, wgSize};
    r.entries[0] = {0, (uint32_t)offsetof(CoopSpecData, M), sizeof(uint32_t)};
    r.entries[1] = {1, (uint32_t)offsetof(CoopSpecData, N), sizeof(uint32_t)};
    r.entries[2] = {2, (uint32_t)offsetof(CoopSpecData, K), sizeof(uint32_t)};
    r.entries[3] = {3, (uint32_t)offsetof(CoopSpecData, wgSize), sizeof(uint32_t)};
    r.specInfo.mapEntryCount = 4;
    r.specInfo.pMapEntries = r.entries;
    r.specInfo.dataSize = sizeof(r.data);
    r.specInfo.pData = &r.data;
    r.push.trips = (int32_t)trips;
    r.note = std::string(d.metricDescription ? d.metricDescription : "") +
             "  Runs at the " + std::to_string(t.M) + "x" + std::to_string(t.N) +
             "x" + std::to_string(t.K) + " tile this driver advertises for it.";

    d.specInfo = &r.specInfo;
    d.metricDescription = r.note.c_str();
    d.wgSize = wgSize;
    d.outElemsPerWG = t.M * t.N;
    d.pushData = &r.push;
    d.pushSize = sizeof(r.push);
    // Reported work per WI = 2*MACs*MulAdds / subgroup-size; exact since M*N is a
    // multiple of the subgroup width for every advertised tile, and the MulAdd
    // count is the trip count the shader was handed times the trip size.
    d.workPerWI = (uint32_t)((volume * 2 * mmas) / wgSize);
    CLPEAK_VLOG("%s: %ux%ux%u at subgroup %u, %llu trips x %u MulAdds, "
                "%u ops/WI\n",
                d.resultTag, t.M, t.N, t.K, wgSize,
                (unsigned long long)trips, COOPMAT_MMA_PER_TRIP, d.workPerWI);
  }

} // namespace

int vkPeak::runCoopMatrix(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // One subgroup per work-group: each subgroup collectively computes one MxN
  // output tile.  The width is a specialization constant bound to both the
  // work-group size and the pinned subgroup size, so they can never disagree --
  // see coopmatSubgroupWidth() for how it is chosen and
  // coopmatRequiredSubgroupSize() for what goes wrong when the driver splits
  // the group into several subgroups instead.
  // Width per tile, not per device: the driver may advertise one dtype's tile
  // only at a narrower subgroup than another's, and a tile run at a width it
  // was never advertised at is a shape no driver promised to compile.
  // coopmat_tile_t carries the width it came from, or 0 when only the
  // width-agnostic KHR query answered and there is no way to know.
  auto tileWG = [&](const coopmat_tile_t &t)
  {
    return coopmatSubgroupWidth(dev.info, t.subgroupSize);
  };
  auto tileSub = [&](const coopmat_tile_t &t)
  {
    return coopmatRequiredSubgroupSize(dev.info, t.subgroupSize);
  };

  // One scope for the whole family -- all data types in one test.
  // The int8 row carries its own unit (ops) so it shares the test.
  auto test = currentDeviceScope->beginTest(
      {"coopmat", "Cooperative matrix", "flops", Category::Unknown,
       "The device's matrix engine -- its tensor cores -- which "
       "multiplies whole small blocks of numbers in one step instead of one "
       "value at a time.  Each reading is a different input format, run at the "
       "block shape the driver advertises for it; which formats the engine "
       "supports, and how much faster the narrow ones go, is most of what "
       "separates one generation of hardware from the next.",
       TestShape::Heterogeneous, "data type"});

#ifdef VK_HAS_COOPMAT_FP32
    {
      CoopTileRun r;
      r.push.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "fp32";
      d.unit = "flops";
      d.metricDescription = "Peak speed of the device's matrix engine (its tensor cores) on "
                            "full 32-bit numbers.  These units multiply whole small blocks "
                            "of numbers in one step instead of one value at a time.";

      d.elemSize = sizeof(float);
      if (dev.info.coopmatFP32.supported)
      {
        d.spirv = vk_shaders::coopmat_fp32;
        d.spirvSize = vk_shaders::coopmat_fp32_size;
        d.requiredSubgroupSize = tileSub(dev.info.coopmatFP32);
        bindCoopTile(r, d, dev.info.coopmatFP32, tileWG(dev.info.coopmatFP32));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp32xfp32+fp32 coopmat property! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP16
    {
      CoopTileRun r;
      r.push.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "fp16";
      d.unit = "flops";
      d.metricDescription = "The matrix engine on 16-bit inputs with a 32-bit running "
                            "total -- the everyday precision of AI inference, and the "
                            "widest-supported row here.  Keeping the total at 32 bits "
                            "costs accuracy nothing and, on consumer graphics cards, "
                            "costs half the speed: see the 16-bit-total row below.";

      d.elemSize = sizeof(float);
      if (dev.info.float16Supported && dev.info.coopmatFP16.supported)
      {
        d.spirv = vk_shaders::coopmat_fp16;
        d.spirvSize = vk_shaders::coopmat_fp16_size;
        d.requiredSubgroupSize = tileSub(dev.info.coopmatFP16);
        bindCoopTile(r, d, dev.info.coopmatFP16, tileWG(dev.info.coopmatFP16));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp16xfp16+fp32 coopmat support (shaderFloat16 or property)! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP16_F16ACC
    {
      CoopTileRun r;
      r.push.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "fp16 f16acc";
      d.unit = "flops";
      d.metricDescription = "The matrix engine on 16-bit inputs with the running total also "
                            "kept at 16 bits.  Consumer graphics cards run this at twice the "
                            "rate of the 32-bit total above, which is why a card's headline "
                            "AI figure is usually this one; server parts run both alike.";

      d.elemSize = sizeof(float);
      if (dev.info.float16Supported && dev.info.coopmatFP16F16.supported)
      {
        d.spirv = vk_shaders::coopmat_fp16_f16acc;
        d.spirvSize = vk_shaders::coopmat_fp16_f16acc_size;
        d.requiredSubgroupSize = tileSub(dev.info.coopmatFP16F16);
        bindCoopTile(r, d, dev.info.coopmatFP16F16, tileWG(dev.info.coopmatFP16F16));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp16xfp16+fp16 coopmat support (shaderFloat16 or property)! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_BF16
    {
      CoopTileRun r;
      r.push.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "bf16";
      d.unit = "flops";
      d.metricDescription = "The matrix engine on bfloat16 -- 16 bits arranged for AI work, "
                            "trading digits of accuracy for the number range of a full "
                            "float, which makes training far more forgiving.";

      d.elemSize = sizeof(float);
      if (dev.info.bfloat16Supported && dev.info.coopmatBF16.supported)
      {
        d.spirv = vk_shaders::coopmat_bf16;
        d.spirvSize = vk_shaders::coopmat_bf16_size;
        d.requiredSubgroupSize = tileSub(dev.info.coopmatBF16);
        bindCoopTile(r, d, dev.info.coopmatBF16, tileWG(dev.info.coopmatBF16));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No bf16xbf16+fp32 coopmat support (shaderBFloat16Type or property)! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E4M3
    {
      CoopTileRun r;
      r.push.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "fp8_e4m3";
      d.unit = "flops";
      d.metricDescription = "The matrix engine on 8-bit numbers, in the variant that spends "
                            "its bits on accuracy rather than range.  Half the data of fp16 "
                            "per value, so the newest hardware runs it at roughly twice the rate.";

      d.elemSize = sizeof(float);
      // Two gates: the float8 feature must be enabled at device creation
      // (else pipeline creation fails) AND a matching tile must be advertised.
      if (dev.info.fp8Supported && dev.info.coopmatFP8E4M3.supported)
      {
        d.spirv = vk_shaders::coopmat_fp8_e4m3;
        d.spirvSize = vk_shaders::coopmat_fp8_e4m3_size;
        d.requiredSubgroupSize = tileSub(dev.info.coopmatFP8E4M3);
        bindCoopTile(r, d, dev.info.coopmatFP8E4M3, tileWG(dev.info.coopmatFP8E4M3));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp8-E4M3 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E5M2
    {
      CoopTileRun r;
      r.push.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "fp8_e5m2";
      d.unit = "flops";
      d.metricDescription = "The same 8-bit matrix path in the other variant, which spends "
                            "its bits on range rather than accuracy -- the one that copes "
                            "with very large and very small values.";

      d.elemSize = sizeof(float);
      if (dev.info.fp8Supported && dev.info.coopmatFP8E5M2.supported)
      {
        d.spirv = vk_shaders::coopmat_fp8_e5m2;
        d.spirvSize = vk_shaders::coopmat_fp8_e5m2_size;
        d.requiredSubgroupSize = tileSub(dev.info.coopmatFP8E5M2);
        bindCoopTile(r, d, dev.info.coopmatFP8E5M2, tileWG(dev.info.coopmatFP8E5M2));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp8-E5M2 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      }
      runComputeKernel(dev, cfg, d);
    }
#endif

#ifdef VK_HAS_COOPMAT_INT8
  // Integer row -- same test, same scope; carries its own unit (ops).

    CoopTileRun r;
    r.push.A.i = 3;
    vk_compute_desc_t d = {};
    d.scope = &test;
    d.resultTag = "coopmat";
    d.metricLabel = "int8";
    // Measured in ops, not flops.  The reading carries that itself, which is
    // what lets it join the floating-point family instead of needing a test of
    // its own; `unit` only heads the test when this reading is the one that
    // opens it, which happens on an integer-only run.
    d.unit = "ops";
    d.metricUnit = "ops";
    d.metricDescription = "8-bit whole numbers with a 32-bit running total -- the "
                          "format quantized neural networks use when they are squeezed "
                          "down to run fast on cheaper hardware.";

    d.elemSize = sizeof(int32_t);
    // Two gates, like fp8: the shader's Int8 capability needs shaderInt8
    // enabled at device creation, and a matching tile must be advertised.
    if (dev.info.int8Supported && dev.info.coopmatINT8.supported)
    {
      d.spirv = vk_shaders::coopmat_int8;
      d.spirvSize = vk_shaders::coopmat_int8_size;
      d.requiredSubgroupSize = tileSub(dev.info.coopmatINT8);
      bindCoopTile(r, d, dev.info.coopmatINT8, tileWG(dev.info.coopmatINT8));
    }
    else
    {
      d.skip = true;
      d.skipMsg = "No int8xint8+int32 coopmat support (shaderInt8 or property)! Skipped";
    }
    runComputeKernel(dev, cfg, d);
#endif
  return 0;
}

#endif // ENABLE_VULKAN
