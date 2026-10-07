#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstddef> // offsetof
#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Cooperative matrix (tensor-core) umbrella.
//
// Runs every dtype combination the driver advertises.  The tile shape
// (M/N/K) is whatever vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
// reported for that dtype -- ranked once in vkPeak::runAll and carried in
// dev.info.coopmat*, best first, the first the driver will build being the
// one that runs -- and is bound into the shader as specialization
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
  // then the number of trips of the inner loop, then whether to fill the tiles
  // element by element.  float and int32_t are both four bytes at offset 0, so
  // one struct serves every dtype -- only the shader's spelling of the push
  // block differs.
  //
  // The trip count is pushed rather than specialized on purpose.  A compile-time
  // trip count is an invitation for the driver to unroll the whole run, and a
  // fully unrolled run of an emulated tile is what a shader compiler chokes on;
  // it also leaves the loop open to being folded into a closed form, which has
  // inflated these rows before.  Push it and neither is possible.
  //
  // `spread` stays 0, so every tile is one value, as the shaders build it.  It
  // is pushed so that no driver can know that: a driver that can see a tile is
  // one value computes its dot products once, which is how llvmpipe read above
  // its CPU's peak (shaders/coopmat_chain.glsl).
  struct CoopPush
  {
    union
    {
      float f;
      int32_t i;
    } A;
    int32_t trips;
    int32_t spread;
  };

  // The most accumulator a lane may carry for the four-accumulator build to be
  // raced at all: 32 registers.  Four of Alchemist's 8x8 int32 tiles come to
  // 32 bytes a lane and four 16x16 fp32 tiles on a 32-wide subgroup to 128;
  // four of the 64x64 tiles an Adreno advertises would be 1 KB, which could
  // only spill -- or take down a compiler already known to fault on less.
  // Counted in output elements, which are the accumulator's or wider, so it
  // errs on the side of not racing.
  const uint32_t kAltAccumulatorBytesPerLane = 128;

  // Each work-group runs this many times the per-WI budget, over this many
  // times fewer work-groups, so a dispatch takes as long as it would at the
  // budget.  What a work-group does before its loop -- building its eight
  // tiles -- it does once, so its cost shrinks with the trips that follow: at
  // the budget alone an RTX 5060's int8 work-group ran 8 trips, and building
  // tiles no compiler can see through (shaders/coopmat_chain.glsl) took 1.6%
  // off that reading; at four times the budget it is within noise.
  const uint32_t kCoopGroupScale = 4;

  // A desc's second build: VK_ALT_SHADER(name) is its (pointer, size), or
  // (nullptr, 0) where glslc did not build one.
  void setAlt(vk_compute_desc_t &d, const uint32_t *spirv, size_t spirvSize)
  {
    d.altSpirv = spirv;
    d.altSpirvSize = spirvSize;
  }

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
  // count so work-per-WI stays ~kCoopGroupScale * COOPMAT_WORK_PER_WI
  // regardless of tile volume, and record the actual MxNxK that runs.
  //
  // The shape goes on the reading's NOTE, not its name.  Different data types
  // land on different shapes on one device (NVIDIA gives the 8-bit types K=32
  // where fp16 gets K=16), so a name carrying the shape would differ between a
  // device that measured the reading and one that skipped it -- the same reading
  // under two ids.  The name stays the data type, which is what identifies it.
  //
  // The caller has already set r.push.A to the fill this dtype wants, and
  // d.metricLabel / d.metricDescription to what this reading is.  `refused`
  // names the tiles the driver refused to build before this one, if any, so
  // a reading taken at a fallback says so.
  //
  // d.altSpirv, the four-accumulator build (shaders/coopmat_chain.glsl), is
  // kept for this tile only where four accumulators fit the lane budget
  // above; it stores four tiles per work-group, so the buffer is sized for
  // four whenever it runs.
  void bindCoopTile(CoopTileRun &r, vk_compute_desc_t &d,
                    const coopmat_tile_t &t, uint32_t wgSize,
                    const std::string &refused)
  {
    if (d.altSpirv &&
        4ull * t.M * t.N * d.elemSize > (uint64_t)kAltAccumulatorBytesPerLane * wgSize)
    {
      d.altSpirv = nullptr;
      d.altSpirvSize = 0;
    }

    const uint64_t volume = (uint64_t)t.M * t.N * t.K; // MACs per coopMatMulAdd
    uint64_t mmas = ((uint64_t)COOPMAT_WORK_PER_WI * kCoopGroupScale * wgSize) /
                    (volume * 2);
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
             "x" + std::to_string(t.K) + " tile this driver advertises for it" +
             (refused.empty() ? std::string(".")
                              : ", after it refused to build " + refused + ".");

    d.specInfo = &r.specInfo;
    d.metricDescription = r.note.c_str();
    d.wgSize = wgSize;
    d.globalDivisor = kCoopGroupScale;
    d.outElemsPerWG = t.M * t.N * (d.altSpirv ? 4 : 1);
    d.pushData = &r.push;
    d.pushSize = sizeof(r.push);
    // Reported work per WI = 2*MACs*MulAdds / subgroup-size; exact since M*N is a
    // multiple of the subgroup width for every advertised tile, and the MulAdd
    // count is the trip count the shader was handed times the trip size.
    d.workPerWI = (uint32_t)((volume * 2 * mmas) / wgSize);
    CLPEAK_VLOG("%s %s: %ux%ux%u at subgroup %u, %llu trips x %u MulAdds, "
                "%u ops/WI\n",
                d.resultTag, d.metricLabel, t.M, t.N, t.K, wgSize,
                (unsigned long long)trips, COOPMAT_MMA_PER_TRIP, d.workPerWI);
  }

#ifdef VK_HAS_COOPMAT_FP16_F16ACC
  // Why the fp16 x fp16 + fp16 row cannot be sent to this driver, or nullptr.
  // A crash gate: the driver takes the process down where it should decline,
  // so asking is itself the fault.  NOTES.md has the entry and what lifting it
  // takes; lifting it is deleting the `if`.
  //
  // Qualcomm's driver dies inside vkCreateComputePipelines on this module -- a
  // null-pointer SIGSEGV in its shader compiler (libllvm-qgl.so), at the
  // 64x64x16 tile an Adreno 840 advertises for it, driver 0842.44.1, compiler
  // E031.50.19.29.  The module is valid.  What the compiler cannot take is the
  // fp16 accumulator reaching OpCooperativeMatrixMulAddKHR through an OpPhi,
  // which is how the loop carries it once glslc -O has run spirv-opt's SSA
  // rewrite.  Left in a Function variable (-O0) the same tile is declined with
  // VK_ERROR_UNKNOWN, so building it unoptimised would buy a refusal, not a
  // reading, and change the module every other driver compiles.  The other
  // rows carry their accumulator through the same OpPhi and are still sent:
  // that driver builds the fp32 one and declines the int8 one itself (it
  // advertises no fp16 + fp32 tile).
  //
  // Keyed on the driver, not the vendor -- Mesa's Turnip drives the same GPUs
  // with a compiler of its own -- and on every release of it, since one has
  // been seen to crash and none is known to be fixed.  And on the row, not
  // the tile: the smaller fp16 + fp16 tiles the 840 also advertises carry the
  // accumulator through the same OpPhi, and a crash is not a refusal the tile
  // fallback could move on from, so none of them is asked either.
  const char *f16AccumulatorFence(const vk_device_info_t &info)
  {
    if (info.driverID == VK_DRIVER_ID_QUALCOMM_PROPRIETARY)
      return "Qualcomm's Vulkan driver crashes the process building this "
             "pipeline -- a segfault in its shader compiler (Adreno 840, driver "
             "0842.44.1) where it should decline -- so this row is not sent to "
             "that driver; the other data types still are";
    return nullptr;
  }
#endif

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
  // width-agnostic KHR query answered and there is no way to know -- in which
  // case runTiles times it at more than one (coopmatSubgroupWidths()).
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

  // Measure one data type at the first of its advertised tiles the driver will
  // build.  They come best first (rankTiles in vk_peak.cpp), and a refusal to
  // build moves on to the next: a driver can advertise a tile and then decline
  // it -- an Adreno 840 refuses int8's 64x64x32 -- yet take a smaller one.
  // Only a refusal moves on.  A dispatch that fails is the row's error, as it
  // would be with one tile, and a desc already marked skip goes straight to
  // the runner, which emits the skip.  `tiles` is non-empty otherwise.
  //
  // A tile whose width the driver did not name is timed at every width
  // coopmatSubgroupWidths() lists and reported at the fastest.  The first is
  // the width it runs at when nothing is raced, so only its refusal moves on
  // to the next tile; a narrower width the driver refuses just drops out.
  auto runTiles = [&](const vk_compute_desc_t &base, const coopmat_tiles_t &tiles,
                      const CoopPush &fill)
  {
    if (base.skip)
    {
      runComputeKernel(dev, cfg, base);
      return;
    }
    std::string refusedTiles;
    for (const coopmat_tile_t &t : tiles)
    {
      const std::vector<uint32_t> widths = coopmatSubgroupWidths(dev.info, t);
      const bool race = widths.size() > 1;
      // Sized once: each run's spec info points into itself.
      std::vector<CoopTileRun> runs(widths.size());
      std::vector<float> readings(widths.size(), 0.0f);
      bool refused = false;
      for (size_t i = 0; i < widths.size() && !refused; i++)
      {
        CoopTileRun &r = runs[i];
        r.push = fill;
        vk_compute_desc_t d = base;
        d.requiredSubgroupSize = i == 0 ? tileSub(t) : widths[i];
        d.pinOnly = i > 0;
        bindCoopTile(r, d, t, widths[i], refusedTiles);
        bool widthRefused = false;
        d.refused = &widthRefused;
        if (race)
          d.reading = &readings[i];
        runComputeKernel(dev, cfg, d);
        if (widthRefused && i == 0)
          refused = true;
        else if (widthRefused)
          CLPEAK_VLOG("%s %s: the driver refused %s pinned to subgroup %u\n",
                      base.resultTag, base.metricLabel, coopmatTileName(t).c_str(),
                      widths[i]);
      }
      if (refused)
      {
        CLPEAK_VLOG("%s %s: the driver refused to build %s\n", base.resultTag,
                    base.metricLabel, coopmatTileName(t).c_str());
        refusedTiles += (refusedTiles.empty() ? "" : ", ") + coopmatTileName(t);
        continue;
      }
      if (!race)
        return;

      // The race's reading, at the fastest width, saying which it was and --
      // where more than one width ran -- what else was timed.
      size_t best = 0;
      std::vector<size_t> ran;
      std::string timed;
      for (size_t i = 0; i < widths.size(); i++)
      {
        if (readings[i] > readings[best])
          best = i;
        if (readings[i] <= 0.0f)
          continue;
        ran.push_back(i);
        char reading[48];
        snprintf(reading, sizeof(reading), "%s%.1f (subgroup %u)",
                 timed.empty() ? "" : ", ", readings[i], widths[i]);
        timed += reading;
      }
      logger::EmitOptions o;
      if (base.metricUnit)
        o.unit = base.metricUnit;
      o.description = runs[best].note;
      if (ran.empty())
      {
        test.skip(base.metricLabel, ResultStatus::Error, "vkQueueSubmit/WaitIdle failed", o);
        return;
      }
      CLPEAK_VLOG("%s %s: %s %s\n", base.resultTag, base.metricLabel, timed.c_str(),
                  base.unit);
      if (ran.size() > 1)
      {
        std::string widthList;
        for (size_t j = 0; j < ran.size(); j++)
          widthList += (j == 0 ? "" : j + 1 == ran.size() ? " and " : ", ") +
                       std::to_string(widths[ran[j]]);
        o.description += "  The driver names no subgroup width for it, so it was timed at " +
                         widthList + " lanes; this is the fastest, at " +
                         std::to_string(widths[best]) + ".";
      }
      test.emit(base.metricLabel, readings[best], o);
      return;
    }
    // Refused at every tile: the row's error, naming what was asked.
    logger::EmitOptions o;
    o.description = base.metricDescription ? base.metricDescription : "";
    if (base.metricUnit)
      o.unit = base.metricUnit;
    test.skip(base.metricLabel, ResultStatus::Error,
              tiles.size() == 1
                  ? "Pipeline creation failed at the one tile the driver advertises "
                    "for it (" + refusedTiles + ")"
                  : "Pipeline creation failed at every tile the driver advertises "
                    "for it (" + refusedTiles + ")",
              o);
  };

#ifdef VK_HAS_COOPMAT_FP32
    {
      CoopPush fill = {};
      fill.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "fp32";
      d.unit = "flops";
      d.metricDescription = "Peak speed of the device's matrix engine (its tensor cores) on "
                            "full 32-bit numbers.  These units multiply whole small blocks "
                            "of numbers in one step instead of one value at a time.";

      d.elemSize = sizeof(float);
      if (!dev.info.coopmatFP32.empty())
      {
        d.spirv = vk_shaders::coopmat_fp32;
        d.spirvSize = vk_shaders::coopmat_fp32_size;
        setAlt(d, VK_ALT_SHADER(coopmat_fp32));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp32xfp32+fp32 coopmat property! Skipped";
      }
      runTiles(d, dev.info.coopmatFP32, fill);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP16
    {
      CoopPush fill = {};
      fill.A.f = 1.3f;
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
      if (dev.info.float16Supported && !dev.info.coopmatFP16.empty())
      {
        d.spirv = vk_shaders::coopmat_fp16;
        d.spirvSize = vk_shaders::coopmat_fp16_size;
        setAlt(d, VK_ALT_SHADER(coopmat_fp16));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp16xfp16+fp32 coopmat support (shaderFloat16 or property)! Skipped";
      }
      runTiles(d, dev.info.coopmatFP16, fill);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP16_F16ACC
    {
      CoopPush fill = {};
      fill.A.f = 1.3f;
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
      if (!dev.info.float16Supported || dev.info.coopmatFP16F16.empty())
      {
        d.skip = true;
        d.skipMsg = "No fp16xfp16+fp16 coopmat support (shaderFloat16 or property)! Skipped";
      }
      else if (const char *fence = f16AccumulatorFence(dev.info))
      {
        d.skip = true;
        d.skipMsg = fence;
      }
      else
      {
        d.spirv = vk_shaders::coopmat_fp16_f16acc;
        d.spirvSize = vk_shaders::coopmat_fp16_f16acc_size;
        setAlt(d, VK_ALT_SHADER(coopmat_fp16_f16acc));
      }
      runTiles(d, dev.info.coopmatFP16F16, fill);
    }
#endif
#ifdef VK_HAS_COOPMAT_BF16
    {
      CoopPush fill = {};
      fill.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "bf16";
      d.unit = "flops";
      d.metricDescription = "The matrix engine on bfloat16 -- 16 bits arranged for AI work, "
                            "trading digits of accuracy for the number range of a full "
                            "float, which makes training far more forgiving.";

      d.elemSize = sizeof(float);
      if (dev.info.bfloat16Supported && !dev.info.coopmatBF16.empty())
      {
        d.spirv = vk_shaders::coopmat_bf16;
        d.spirvSize = vk_shaders::coopmat_bf16_size;
        setAlt(d, VK_ALT_SHADER(coopmat_bf16));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No bf16xbf16+fp32 coopmat support (shaderBFloat16Type or property)! Skipped";
      }
      runTiles(d, dev.info.coopmatBF16, fill);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E4M3
    {
      CoopPush fill = {};
      fill.A.f = 1.3f;
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
      if (dev.info.fp8Supported && !dev.info.coopmatFP8E4M3.empty())
      {
        d.spirv = vk_shaders::coopmat_fp8_e4m3;
        d.spirvSize = vk_shaders::coopmat_fp8_e4m3_size;
        setAlt(d, VK_ALT_SHADER(coopmat_fp8_e4m3));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp8-E4M3 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      }
      runTiles(d, dev.info.coopmatFP8E4M3, fill);
    }
#endif
#ifdef VK_HAS_COOPMAT_FP8_E5M2
    {
      CoopPush fill = {};
      fill.A.f = 1.3f;
      vk_compute_desc_t d = {};
      d.scope = &test;
      d.resultTag = "coopmat";
      d.metricLabel = "fp8_e5m2";
      d.unit = "flops";
      d.metricDescription = "The same 8-bit matrix path in the other variant, which spends "
                            "its bits on range rather than accuracy -- the one that copes "
                            "with very large and very small values.";

      d.elemSize = sizeof(float);
      if (dev.info.fp8Supported && !dev.info.coopmatFP8E5M2.empty())
      {
        d.spirv = vk_shaders::coopmat_fp8_e5m2;
        d.spirvSize = vk_shaders::coopmat_fp8_e5m2_size;
        setAlt(d, VK_ALT_SHADER(coopmat_fp8_e5m2));
      }
      else
      {
        d.skip = true;
        d.skipMsg = "No fp8-E5M2 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
      }
      runTiles(d, dev.info.coopmatFP8E5M2, fill);
    }
#endif

#ifdef VK_HAS_COOPMAT_INT8
  // Integer row -- same test, same scope; carries its own unit (ops).

    CoopPush fill = {};
    fill.A.i = 3;
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
    if (dev.info.int8Supported && !dev.info.coopmatINT8.empty())
    {
      d.spirv = vk_shaders::coopmat_int8;
      d.spirvSize = vk_shaders::coopmat_int8_size;
      setAlt(d, VK_ALT_SHADER(coopmat_int8));
    }
    else
    {
      d.skip = true;
      d.skipMsg = "No int8xint8+int32 coopmat support (shaderInt8 or property)! Skipped";
    }
    runTiles(d, dev.info.coopmatINT8, fill);
#endif
  return 0;
}

#endif // ENABLE_VULKAN
