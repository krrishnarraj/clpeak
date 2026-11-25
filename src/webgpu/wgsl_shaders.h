#ifndef CLPEAK_WEBGPU_WGSL_SHADERS_H
#define CLPEAK_WEBGPU_WGSL_SHADERS_H

#include <common/common.h>
#include <cstdint>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// WGSL compute shader generators for the WebGPU backend.
//
// Each generator mirrors the exact chain shapes, vector widths, and per-WI
// operation / byte counts of clpeak's Vulkan and OpenCL kernels (see the MAD
// chain and image-bandwidth blocks in include/common/common.h and
// src/vulkan/shaders/mad_chain.glsl, dp4a_chain.glsl).
// ---------------------------------------------------------------------------

namespace wgpu_shaders
{

enum class ScalarType { F32, F16, I32 };

static inline std::string wgslType(ScalarType t, uint32_t width)
{
  const char *s = (t == ScalarType::F32) ? "f32" :
                  (t == ScalarType::F16) ? "f16" : "i32";
  if (width == 1) return s;
  return "vec" + std::to_string(width) + "<" + s + ">";
}

static inline std::string wgslLit(ScalarType t, int val)
{
  if (t == ScalarType::F32) return std::to_string(val) + ".0f";
  if (t == ScalarType::F16) return std::to_string(val) + ".0h";
  return std::to_string(val) + "i";
}

static inline std::string wgslSplat(ScalarType t, uint32_t width, const std::string &expr)
{
  if (width == 1) return expr;
  return wgslType(t, width) + "(" + expr + ")";
}

static inline std::string wgslSeedExpr(ScalarType t, uint32_t width)
{
  const std::string ty = wgslType(t, width);
  if (t == ScalarType::F32)
  {
    if (width == 1) return "params.x";
    if (width == 2) return "vec2<f32>(params.x, params.x + 1.0f)";
    return "vec4<f32>(params.x, params.x + 1.0f, params.x + 2.0f, params.x + 3.0f)";
  }
  if (t == ScalarType::F16)
  {
    if (width == 1) return "f16(params.x)";
    if (width == 2) return "vec2<f16>(f16(params.x), f16(params.x + 1.0f))";
    return "vec4<f16>(f16(params.x), f16(params.x + 1.0f), f16(params.x + 2.0f), f16(params.x + 3.0f))";
  }
  // I32
  if (width == 1) return "params.x";
  if (width == 2) return "vec2<i32>(params.x, params.x + 1i)";
  return "vec4<i32>(params.x, params.x + 1i, params.x + 2i, params.x + 3i)";
}

static inline std::string wgslReduce(ScalarType t, uint32_t width, const std::string &var)
{
  std::string expr;
  if (width == 1) expr = var;
  else if (width == 2) expr = "(" + var + ".x + " + var + ".y)";
  else expr = "((" + var + ".x + " + var + ".y) + (" + var + ".z + " + var + ".w))";
  if (t == ScalarType::F16)
    return "f32(" + expr + ")";
  return expr;
}

static inline std::string emitMadOp(ScalarType t, bool useFma,
                                    const std::string &d,
                                    const std::string &m1,
                                    const std::string &m2,
                                    const std::string &ad)
{
  if (useFma && t != ScalarType::I32)
    return "    " + d + " = fma(" + m1 + ", " + m2 + ", " + ad + ");\n";
  return "    " + d + " = (" + m1 + " * " + m2 + ") + " + ad + ";\n";
}

// Emit 16 MAD instructions for one chain shape.
static inline std::string emitMad16(ScalarType t, bool useFma,
                                    bool alt, bool isInt, uint32_t chains)
{
  std::ostringstream ss;
  if (!alt)
  {
    for (int i = 0; i < 16; i++)
      ss << emitMadOp(t, useFma, "x0", "x0", "x0", "c");
    return ss.str();
  }
  if (isInt)
  {
    // Rotating quadratic integer chain: x_k = x_k * x_(k+1) + c (non-foldable).
    const uint32_t c = (chains >= 4) ? 4 : 2;
    const uint32_t reps = 16 / c;
    for (uint32_t r = 0; r < reps; r++)
    {
      for (uint32_t k = 0; k < c; k++)
      {
        std::string xk = "x" + std::to_string(k);
        std::string xn = "x" + std::to_string((k + 1) % c);
        ss << emitMadOp(t, false, xk, xk, xn, "c");
      }
    }
    return ss.str();
  }
  // Floating-point affine chain over `chains` independent accumulators.
  const uint32_t reps = 16 / chains;
  for (uint32_t r = 0; r < reps; r++)
  {
    for (uint32_t k = 0; k < chains; k++)
    {
      std::string xk = "x" + std::to_string(k);
      ss << emitMadOp(t, useFma, xk, "a", xk, "b");
    }
  }
  return ss.str();
}

static inline std::string emitChainDecls(ScalarType t, uint32_t width,
                                         bool alt, bool isInt, uint32_t chains,
                                         const std::string &seed,
                                         const std::string &inv)
{
  std::ostringstream ss;
  const std::string ty = wgslType(t, width);
  if (!alt)
  {
    ss << "  let c: " << ty << " = " << inv << ";\n";
    ss << "  var x0: " << ty << " = " << seed << ";\n";
    return ss.str();
  }
  const uint32_t c = isInt ? ((chains >= 4) ? 4 : 2) : chains;
  if (isInt)
  {
    ss << "  let c: " << ty << " = " << inv << ";\n";
  }
  else
  {
    ss << "  let a: " << ty << " = " << inv << ";\n";
    ss << "  let b: " << ty << " = a + " << wgslSplat(t, width, wgslLit(t, 2)) << ";\n";
  }
  ss << "  let seed0: " << ty << " = " << seed << ";\n";
  for (uint32_t k = 0; k < c; k++)
  {
    ss << "  var x" << k << ": " << ty << " = seed0";
    if (k > 0)
      ss << " + " << wgslSplat(t, width, wgslLit(t, static_cast<int>(k * 4)));
    ss << ";\n";
  }
  return ss.str();
}

static inline std::string emitChainResult(bool alt, bool isInt, uint32_t chains)
{
  if (!alt) return "x0";
  const uint32_t c = isInt ? ((chains >= 4) ? 4 : 2) : chains;
  if (c == 1) return "x0";
  if (c == 2) return "(x0 + x1)";
  if (c == 4) return "((x0 + x1) + (x2 + x3))";
  return "(((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)))";
}

// Single-precision (fp32) or half-precision (fp16) compute shader.
// Per-WI budget: COMPUTE_FP_WORK_PER_WI = 4096 FLOPs across all widths.
static inline std::string makeComputeFpWgsl(ScalarType t, uint32_t width, bool alt)
{
  // Register budget: 4 32-bit registers per lane -> fp32 uses 4/2/1 chains,
  // fp16 packs two per 32-bit register so uses 8/4/2 chains.
  const uint32_t chains = (t == ScalarType::F16) ? (8 / width) : (4 / width);
  // Contracted a*b + c for fp16 (matches Vulkan compute_hp_v*), fma for fp32.
  const bool useFma = (t == ScalarType::F32);
  const uint32_t totalMad16 = 128 / width; // 128 / 64 / 32 -> 4096 FLOPs/WI
  const uint32_t trips = alt ? totalMad16 : (totalMad16 / 8);
  const std::string mad16 = emitMad16(t, useFma, alt, false, chains);

  std::ostringstream ss;
  if (t == ScalarType::F16)
    ss << "enable f16;\n\n";
  ss << "@group(0) @binding(0) var<storage, read_write> outputBuf: array<f32>;\n";
  ss << "@group(0) @binding(1) var<uniform> params: vec4<f32>;\n\n";
  ss << "@compute @workgroup_size(256)\n";
  ss << "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n";
  ss << "        @builtin(local_invocation_id) lid: vec3<u32>) {\n";

  const std::string lidScalar = (t == ScalarType::F16) ? "f16(lid.x)" : "f32(lid.x)";
  ss << emitChainDecls(t, width, alt, false, chains,
                       wgslSeedExpr(t, width),
                       wgslSplat(t, width, lidScalar));

  ss << "  for (var i: u32 = 0u; i < " << trips << "u; i = i + 1u) {\n";
  if (alt)
  {
    ss << mad16;
  }
  else
  {
    // Squaring form: MAD_128 per trip (8 * MAD_16) to amortize loop counter.
    for (int k = 0; k < 8; k++)
      ss << mad16;
  }
  ss << "  }\n";
  ss << "  let r = " << emitChainResult(alt, false, chains) << ";\n";
  ss << "  outputBuf[gid.x] = " << wgslReduce(t, width, "r") << ";\n";
  ss << "}\n";
  return ss.str();
}

// Mixed-precision (fp16 inputs, fp32 accumulators) compute shader.
// Per-WI budget: COMPUTE_FP_WORK_PER_WI = 4096 FLOPs across all widths.
static inline std::string makeComputeMpWgsl(uint32_t width, bool alt)
{
  const uint32_t chains = 4 / width;
  const uint32_t trips = 16 / width; // 16 / 8 / 4 trips * MAD_128 * 2 * width = 4096
  const std::string mad16 = emitMad16(ScalarType::F32, true, alt, false, chains);
  const std::string f32Ty = wgslType(ScalarType::F32, width);
  const std::string f16Ty = wgslType(ScalarType::F16, width);

  std::string seed;
  if (width == 1)
    seed = "f32(f16(params.x))";
  else if (width == 2)
    seed = "vec2<f32>(vec2<f16>(f16(params.x), f16(params.x + 1.0f)))";
  else
    seed = "vec4<f32>(vec4<f16>(f16(params.x), f16(params.x + 1.0f), f16(params.x + 2.0f), f16(params.x + 3.0f)))";

  std::string inv = (width == 1) ? "f32(f16(lid.x))"
                                 : (f32Ty + "(" + f16Ty + "(f16(lid.x)))");

  std::ostringstream ss;
  ss << "enable f16;\n\n";
  ss << "@group(0) @binding(0) var<storage, read_write> outputBuf: array<f32>;\n";
  ss << "@group(0) @binding(1) var<uniform> params: vec4<f32>;\n\n";
  ss << "@compute @workgroup_size(256)\n";
  ss << "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n";
  ss << "        @builtin(local_invocation_id) lid: vec3<u32>) {\n";
  ss << emitChainDecls(ScalarType::F32, width, alt, false, chains, seed, inv);
  ss << "  for (var i: u32 = 0u; i < " << trips << "u; i = i + 1u) {\n";
  for (int k = 0; k < 8; k++)
    ss << mad16;
  const uint32_t activeChains = alt ? chains : 1;
  for (uint32_t k = 0; k < activeChains; k++)
  {
    std::string xk = "x" + std::to_string(k);
    ss << "    " << xk << " = " << f32Ty << "(" << f16Ty << "(" << xk << "));\n";
  }
  ss << "  }\n";
  ss << "  let r = " << emitChainResult(alt, false, chains) << ";\n";
  ss << "  outputBuf[gid.x] = " << wgslReduce(ScalarType::F32, width, "r") << ";\n";
  ss << "}\n";
  return ss.str();
}

// Signed int32 compute shader.
// Per-WI budget: COMPUTE_INT_WORK_PER_WI = 2048 ops across all widths.
static inline std::string makeComputeInt32Wgsl(uint32_t width, bool alt)
{
  const uint32_t chains = 4 / width;
  const uint32_t trips = 64 / width; // 64 / 32 / 16 trips * 16 MADs * 2 * width = 2048
  const std::string mad16 = emitMad16(ScalarType::I32, false, alt, true, chains);

  std::ostringstream ss;
  ss << "@group(0) @binding(0) var<storage, read_write> outputBuf: array<i32>;\n";
  ss << "@group(0) @binding(1) var<uniform> params: vec4<i32>;\n\n";
  ss << "@compute @workgroup_size(256)\n";
  ss << "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n";
  ss << "        @builtin(local_invocation_id) lid: vec3<u32>) {\n";
  ss << emitChainDecls(ScalarType::I32, width, alt, true, chains,
                       wgslSeedExpr(ScalarType::I32, width),
                       wgslSplat(ScalarType::I32, width, "i32(lid.x)"));
  ss << "  for (var i: u32 = 0u; i < " << trips << "u; i = i + 1u) {\n";
  ss << mad16;
  ss << "  }\n";
  ss << "  let r = " << emitChainResult(alt, true, chains) << ";\n";
  ss << "  outputBuf[gid.x] = " << wgslReduce(ScalarType::I32, width, "r") << ";\n";
  ss << "}\n";
  return ss.str();
}

// INT8 packed 4x8 dot-product compute shader (WGSL dot4I8Packed).
// Per-WI budget: COMPUTE_INT8_DP_WORK_PER_WI = 8192 ops (1024 dots * 8 ops).
// numChains = 1, 2, or 4 independent chains; alt = 4-accumulator cycle vs pair.
static inline std::string makeComputeInt8DpWgsl(uint32_t numChains, bool alt)
{
  std::ostringstream ss;
  ss << "@group(0) @binding(0) var<storage, read_write> outputBuf: array<i32>;\n";
  ss << "@group(0) @binding(1) var<uniform> params: vec4<i32>;\n\n";
  ss << "@compute @workgroup_size(256)\n";
  ss << "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n";
  ss << "        @builtin(local_invocation_id) lid: vec3<u32>) {\n";
  ss << "  let lid_i: i32 = i32(lid.x);\n";
  if (alt)
  {
    ss << "  let dpBias: i32 = params.x;\n";
    for (uint32_t k = 0; k < numChains; k++)
    {
      int base = static_cast<int>(k * 16);
      ss << "  var a" << k << ": i32 = lid_i + dpBias + " << base << "i;\n";
      ss << "  var b" << k << ": i32 = lid_i + dpBias + " << (base + 4) << "i;\n";
      ss << "  var c" << k << ": i32 = lid_i + dpBias + " << (base + 8) << "i;\n";
      ss << "  var d" << k << ": i32 = lid_i + dpBias + " << (base + 12) << "i;\n";
    }
  }
  else
  {
    ss << "  let A: u32 = bitcast<u32>(params.x);\n";
    ss << "  let x: u32 = (A & 0xffu) | (((A + 1u) & 0xffu) << 8u) | "
          "(((A + 2u) & 0xffu) << 16u) | (((A + 3u) & 0xffu) << 24u);\n";
    for (uint32_t k = 0; k < numChains; k++)
    {
      int base = static_cast<int>(k * 8);
      ss << "  var a" << k << ": i32 = lid_i + " << base << "i;\n";
      ss << "  var b" << k << ": i32 = lid_i + " << (base + 4) << "i;\n";
    }
  }

  auto emitChain16 = [&](uint32_t k)
  {
    for (int r = 0; r < 4; r++)
    {
      if (alt)
      {
        ss << "    a" << k << " = a" << k << " + dot4I8Packed(bitcast<u32>(b" << k << "), bitcast<u32>(c" << k << "));\n";
        ss << "    b" << k << " = b" << k << " + dot4I8Packed(bitcast<u32>(c" << k << "), bitcast<u32>(d" << k << "));\n";
        ss << "    c" << k << " = c" << k << " + dot4I8Packed(bitcast<u32>(d" << k << "), bitcast<u32>(a" << k << "));\n";
        ss << "    d" << k << " = d" << k << " + dot4I8Packed(bitcast<u32>(a" << k << "), bitcast<u32>(b" << k << "));\n";
      }
      else
      {
        ss << "    a" << k << " = a" << k << " + dot4I8Packed(x, bitcast<u32>(b" << k << "));\n";
        ss << "    b" << k << " = b" << k << " + dot4I8Packed(x, bitcast<u32>(a" << k << "));\n";
        ss << "    a" << k << " = a" << k << " + dot4I8Packed(x, bitcast<u32>(b" << k << "));\n";
        ss << "    b" << k << " = b" << k << " + dot4I8Packed(x, bitcast<u32>(a" << k << "));\n";
      }
    }
  };

  // 16 outer trips * 64 dots per trip = 1024 dots = 8192 ops/WI.
  ss << "  for (var i: u32 = 0u; i < 16u; i = i + 1u) {\n";
  const uint32_t repsPerChain = 4 / numChains;
  for (uint32_t r = 0; r < repsPerChain; r++)
    for (uint32_t k = 0; k < numChains; k++)
      emitChain16(k);
  ss << "  }\n";

  auto chainSum = [&](uint32_t k) -> std::string
  {
    std::string ks = std::to_string(k);
    if (alt)
      return "((a" + ks + " + b" + ks + ") + (c" + ks + " + d" + ks + "))";
    return "(a" + ks + " + b" + ks + ")";
  };

  if (numChains == 1)
    ss << "  outputBuf[gid.x] = " << chainSum(0) << ";\n";
  else if (numChains == 2)
    ss << "  outputBuf[gid.x] = " << chainSum(0) << " + " << chainSum(1) << ";\n";
  else
    ss << "  outputBuf[gid.x] = (" << chainSum(0) << " + " << chainSum(1)
       << ") + (" << chainSum(2) << " + " << chainSum(3) << ");\n";
  ss << "}\n";
  return ss.str();
}

// Global memory bandwidth shader (float / float2 / float4).
static inline std::string makeGlobalBandwidthWgsl(uint32_t width)
{
  const std::string ty = wgslType(ScalarType::F32, width);
  std::ostringstream ss;
  ss << "@group(0) @binding(0) var<storage, read> inputBuf: array<" << ty << ">;\n";
  ss << "@group(0) @binding(1) var<storage, read_write> outputBuf: array<f32>;\n\n";
  ss << "@compute @workgroup_size(256)\n";
  ss << "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n";
  ss << "        @builtin(local_invocation_id) lid: vec3<u32>,\n";
  ss << "        @builtin(workgroup_id) wid: vec3<u32>) {\n";
  ss << "  var offset: u32 = wid.x * 256u * " << FETCH_PER_WI << "u + lid.x;\n";
  ss << "  var sum: " << ty << " = " << wgslSplat(ScalarType::F32, width, "0.0f") << ";\n";
  ss << "  for (var i: u32 = 0u; i < " << FETCH_PER_WI << "u; i = i + 1u) {\n";
  ss << "    sum = sum + inputBuf[offset];\n";
  ss << "    offset = offset + 256u;\n";
  ss << "  }\n";
  ss << "  outputBuf[gid.x] = " << wgslReduce(ScalarType::F32, width, "sum") << ";\n";
  ss << "}\n";
  return ss.str();
}

// Local (workgroup) memory bandwidth shader (float / float2 / float4).
static inline std::string makeLocalBandwidthWgsl(uint32_t width)
{
  const std::string ty = wgslType(ScalarType::F32, width);
  std::string init;
  if (width == 1)
    init = "f32(gid.x)";
  else if (width == 2)
    init = "vec2<f32>(f32(gid.x), f32(gid.x + 1u))";
  else
    init = "vec4<f32>(f32(gid.x), f32(gid.x + 1u), f32(gid.x + 2u), f32(gid.x + 3u))";

  std::ostringstream ss;
  ss << "@group(0) @binding(0) var<storage, read_write> outputBuf: array<f32>;\n";
  ss << "var<workgroup> scratch: array<" << ty << ", 256>;\n\n";
  ss << "@compute @workgroup_size(256)\n";
  ss << "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n";
  ss << "        @builtin(local_invocation_id) lid: vec3<u32>) {\n";
  ss << "  let next: u32 = (lid.x + 1u) & 255u;\n";
  ss << "  var sum: " << ty << " = " << init << ";\n";
  ss << "  for (var i: u32 = 0u; i < " << LMEM_REPS << "u; i = i + 1u) {\n";
  ss << "    scratch[lid.x] = sum;\n";
  ss << "    workgroupBarrier();\n";
  ss << "    sum = scratch[next];\n";
  ss << "    workgroupBarrier();\n";
  ss << "  }\n";
  ss << "  outputBuf[gid.x] = " << wgslReduce(ScalarType::F32, width, "sum") << ";\n";
  ss << "}\n";
  return ss.str();
}

// Image (texture) bandwidth shader.  Uses textureLoad (no sampler) and WGSL
// pipeline override constants so power-of-two divisions fold into shifts/masks.
static inline std::string makeImageBandwidthWgsl()
{
  std::ostringstream ss;
  ss << "override IMG_W: u32 = 4096u;\n"
        "override IMG_H: u32 = 4096u;\n"
        "override TILE_W: u32 = 4096u;\n"
        "override TILE_H: u32 = 1u;\n"
        "override WALK: u32 = 0u;\n\n"
        "@group(0) @binding(0) var img: texture_2d<f32>;\n"
        "@group(0) @binding(1) var<storage, read_write> outputBuf: array<f32>;\n\n"
        "fn blocked(pixel: u32) -> vec2<i32> {\n"
        "  let tilesX: u32 = IMG_W / TILE_W;\n"
        "  let perTile: u32 = TILE_W * TILE_H;\n"
        "  let t: u32 = pixel / perTile;\n"
        "  let it: u32 = pixel - t * perTile;\n"
        "  return vec2<i32>(i32((t % tilesX) * TILE_W + (it % TILE_W)),\n"
        "                   i32((t / tilesX) * TILE_H + (it / TILE_W)));\n"
        "}\n\n"
        "@compute @workgroup_size(256)\n"
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>,\n"
        "        @builtin(num_workgroups) numWg: vec3<u32>) {\n"
        "  let gsize: u32 = numWg.x * 256u;\n"
        "  var sum: vec4<f32> = vec4<f32>(0.0f);\n"
        "  if (WALK == 0u) {\n"
        "    for (var i: u32 = 0u; i < " << IMAGE_FETCH_PER_WI << "u; i = i + 1u) {\n"
        "      sum = sum + textureLoad(img, blocked(gid.x + i * gsize), 0);\n"
        "    }\n"
        "  } else {\n"
        "    for (var i: u32 = 0u; i < " << IMAGE_FETCH_PER_WI << "u; i = i + 1u) {\n"
        "      let pixel: u32 = gid.x + i * gsize;\n"
        "      sum = sum + textureLoad(img, vec2<i32>(i32(pixel / IMG_H), i32(pixel % IMG_H)), 0);\n"
        "    }\n"
        "  }\n"
        "  outputBuf[gid.x] = (sum.x + sum.y) + (sum.z + sum.w);\n"
        "}\n";
  return ss.str();
}

// No-op shader for kernel launch latency.
static inline std::string makeKernelLatencyWgsl()
{
  return R"(
@compute @workgroup_size(1)
fn main() {
}
)";
}

} // namespace wgpu_shaders

#endif // CLPEAK_WEBGPU_WGSL_SHADERS_H
