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
| `mtl_blas.mm` | `runMpsGemm` — MPS/MPSGraph matrix multiply; `runMpsAttention` — MPSGraph scaled-dot-product-attention peak (`Benchmark::MpsAttention`, `--mps-attention`, fp16, llama-shaped H16/S4096/D128, needs macOS 15 / iOS 18; FLOPs = the two matmuls only, so it reads below raw GEMM peak — M1 Pro ~2.9 TFLOPS vs 3.96 MPS GEMM fp16). int8 GEMM stays impossible on Metal: even MPSCNN's UInt8-weight convolutions dequantize to float before compute (storage quantization only, per MPSCNNConvolution.h) |
| `global_bandwidth.mm` | `runGlobalBandwidth` |
| `local_bandwidth.mm` | `runLocalBandwidth` |
| `image_bandwidth.mm` | `runImageBandwidth`; `runTextureSampleRate` (`Benchmark::TextureSample`, `--texture-sample`, unit `gtexels` with Category::Bandwidth passed explicitly) — bilinear filtered-fetch rate from a cache-resident 1024x1024 texture, rgba8 + rgba16f rows.  TMU test, not bandwidth: coords are forced-fractional so every sample is a real 4-texel blend, and the per-sample address math MUST stay mask/shift (power-of-two dims) — an integer %/÷ throttles below TMU rate (31 → 113 GTexels/s on M1 Pro when fixed).  rgba16f lands ~2/3 of rgba8 (wide-format filtering is half-rate; M1 Pro ~113/~82) |
| `kernel_latency.mm` | `runKernelLatency` |
| `mtl_kernels/` | Metal Shading Language kernels (`.metal`) embedded as C++ string literals |
| `cmake/EmbedMetalKernels.cmake` | `embed_metal_kernels()` — .metal → C++ raw-string arrays |

## Architecture Note

Category files include `mtl_internal.h` which provides ObjC Metal types and
pimpl access. The public header `include/metal/mtl_peak.h` stays pure C++
with only forward declarations — it can be included from non-ObjC TUs.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file + update `CMakeLists.txt` + this file.
- If you add a new `.metal` kernel → add to `CLPEAK_MTL_KERNELS` in `CMakeLists.txt`.
- If you change `MetalPeak` interface → update `include/metal/mtl_peak.h`.
- If you add a new helper → declare in `mtl_internal.h`, define in the appropriate `.mm` file (`mtl_device.mm`, `mtl_utils.mm`, `compute_kernel.mm`, or a new file).
- If you add Objective-C code → remember ARC is enabled (`-fobjc-arc`).
