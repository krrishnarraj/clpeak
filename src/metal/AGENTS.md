# src/metal — Metal Backend Implementation

`MetalPeak` class implementation: device init, per-benchmark runners, and
Metal Shading Language kernels (in `mtl_kernels/`).  Built as `peak_metal`
static library.  Source files are Objective-C++ (`.mm`).

## Quick Lookups

- Looking for the main class (`MetalPeak` ctor, `runAll`, inventory)? → `mtl_peak.mm`
- Looking for device init (`MetalDevice` + device enumeration)? → `mtl_device.mm`
- Looking for shared compute-peak driver (`runComputeKernel` + `mtlRunDispatches`)? → `compute_kernel.mm`
- Looking for Metal library/pipeline caching (`mtlGetLibrary`, `mtlGetPipeline`)? → `mtl_utils.mm`
- Looking for the internal header (ObjC types + pimpls + helpers)? → `mtl_internal.h`
- Looking for FP compute benchmarks? → `compute_float.mm`
- Looking for simdgroup matrix benchmarks? → `simdgroup.mm`
- Looking for MPSGraph GEMM / attention (SDPA) benchmarks? → `mtl_blas.mm`
- Looking for bandwidth benchmarks? → `global_bandwidth.mm`, `local_bandwidth.mm`, `image_bandwidth.mm`
- Looking for kernel latency? → `kernel_latency.mm`
- Looking for .metal kernel sources? → `mtl_kernels/*.metal`
- Looking for kernel embedding logic? → `cmake/EmbedMetalKernels.cmake`

## Key Files

| File | Purpose |
|------|---------|
| `mtl_peak.mm` | `MetalPeak` class: ctor, `applyOptions()`, `runAll()`, `enumerate()`, `printInventory()` |
| `mtl_device.mm` | `MetalDevice` class: ctor, `init()`, `cleanup()` + `copyClpeakMetalDevices()` helper |
| `mtl_utils.mm` | `mtlGetLibrary()`, `mtlGetPipeline()` — Metal library/pipeline caching |
| `compute_kernel.mm` | `MetalPeak::runComputeKernel()` + `mtlRunDispatches()` — shared compute-peak driver and GPU timing |
| `mtl_internal.h` | Internal header: ObjC imports, pimpl definitions, helper declarations — included by all `.mm` files |
| `compute_float.mm` | `runComputeSP`, `runComputeHP`, `runComputeMP` |
| `simdgroup.mm` | `runSimdgroupMatrix` |
| `mtl_blas.mm` | `runMpsGemm` — MPS/MPSGraph matrix multiply; `runMpsAttention` — MPSGraph scaled-dot-product-attention peak (`--mps-attention`, fp16, fixed llama-class shape, needs macOS 15 / iOS 18). Its FLOPs count the two matmuls only, so it reads below raw GEMM peak by design. **Any MPSGraph-based test must gate on `mtl_device_info_t::mpsGraphSupported`** — on the iOS Simulator MPSGraph aborts the process and the throw cannot be caught (rationale in `mtl_device.mm`). **int8 GEMM is impossible on Metal** — MPSGraph matmul is float-only and MPSCNN's UInt8 weights are storage quantization, dequantized before compute |
| `global_bandwidth.mm` | `runGlobalBandwidth` |
| `local_bandwidth.mm` | `runLocalBandwidth` |
| `image_bandwidth.mm` | `runImageBandwidth`; `runTextureSampleRate` (`--texture-sample`, unit `gtexels` with `Category::Bandwidth` passed explicitly) — bilinear filtered-fetch rate from a cache-resident texture, rgba8 + rgba16f. A TMU test, not a bandwidth one; the addressing constraints that keep it at TMU rate are in `mtl_kernels/texture_sample.metal`. M1 Pro reference: rgba8 ~115, rgba16f ~92 GTexels/s |
| `kernel_latency.mm` | `runKernelLatency` |
| `mtl_kernels/` | Metal Shading Language kernels (`.metal`) embedded as C++ string literals |
| `cmake/EmbedMetalKernels.cmake` | `embed_metal_kernels()` — .metal → C++ raw-string arrays |

## Test documentation

See `include/common/AGENTS.md` § Test documentation.  Metal specifics:

- `mtl_compute_desc_t::description` (test) and
  `mtl_compute_variant_t::description` (one reading); `runComputeKernel()`
  forwards both on every path, skips included.  `shape` / `axis` ride the same
  descriptor and are forwarded the same way.
- A data-type family is ONE descriptor with a variant per type, gated
  individually by `mtl_compute_variant_t::skipMsg` — `simdgroup.mm` is the
  worked example: fp16 and bf16 are one test, and an M1 (no bf16) skips that
  one reading instead of losing the test.
- `mtlWidthNote()` (`mtl_internal.h`) is the shared wording for the
  `float`/`float2`/`float4`/… readings.
- The `V` tables in `global_bandwidth.mm` / `local_bandwidth.mm` /
  `image_bandwidth.mm` carry their notes as a struct field.

## Architecture Note

Category files include `mtl_internal.h` which provides ObjC Metal types and
pimpl access. The public header `include/metal/mtl_peak.h` stays pure C++
with only forward declarations — it can be included from non-ObjC TUs.

## Chain shapes

Each float family defines its kernels twice: `compute_*` with the squaring
chain and `compute_*_alt` with the affine one from `mtl_kernels/mad_chain.metal`,
which `EmbedMetalKernels.cmake` prepends to every embedded source (
`newLibraryWithSource` cannot resolve `#include` of a sibling file).
`runComputeKernel` times both and reports the faster; a variant with a null
`altKernelName` races nothing.  Apple GPUs prefer the squaring chain at every
width, so this changes no Apple number -- it is here so every backend behaves
the same.  Why two shapes: the MAD chain block in `include/common/common.h`.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file + update `CMakeLists.txt` + this file.
- If you add a new `.metal` kernel → add to `CLPEAK_MTL_KERNELS` in `CMakeLists.txt`
  and declare its externs in the `mtl_kernels` namespace (`include/metal/mtl_peak.h`).
- If you change `MetalPeak` interface → update `include/metal/mtl_peak.h`.
- If you add a new helper → declare in `mtl_internal.h`, define in the appropriate `.mm` file (`mtl_device.mm`, `mtl_utils.mm`, `compute_kernel.mm`, or a new file).
- If you add Objective-C code → remember ARC is enabled (`-fobjc-arc`).
