# src/metal — Metal Backend Implementation

`MetalPeak` class implementation: device init, per-benchmark runners, and
Metal Shading Language kernels (in `mtl_kernels/`).  Built as `peak_metal`
static library.  Source files are Objective-C++ (`.mm`).

## Quick Lookups

- Looking for the main class / device init? → `mtl_peak.mm` (`MetalDevice`, `MetalPeak`, `runAll`, `runComputeKernel`)
- Looking for the internal header (ObjC types + pimpls + helpers)? → `mtl_internal.h`
- Looking for FP compute benchmarks? → `compute_float.mm`
- Looking for int compute benchmarks? → `compute_int.mm`
- Looking for simdgroup matrix benchmarks? → `simdgroup.mm`
- Looking for MPSGraph GEMM benchmarks? → `mtl_blas.mm`
- Looking for bandwidth benchmarks? → `global_bandwidth.mm`, `local_bandwidth.mm`, `image_bandwidth.mm`
- Looking for atomic benchmarks? → `atomic_throughput.mm`
- Looking for kernel latency? → `kernel_latency.mm`
- Looking for .metal kernel sources? → `mtl_kernels/*.metal`
- Looking for kernel embedding logic? → `cmake/EmbedMetalKernels.cmake`

## Key Files

| File | Purpose |
|------|---------|
| `mtl_peak.mm` | `MetalDevice` + `MetalPeak` classes: ctor, `applyOptions()`, `runAll()`, `runComputeKernel()`, helpers (`mtlGetLibrary`, `mtlGetPipeline`, `mtlRunDispatches`), inventory |
| `mtl_internal.h` | Internal header: ObjC imports, pimpl definitions, helper declarations — included by all `.mm` files |
| `compute_float.mm` | `runComputeSP`, `runComputeHP`, `runComputeMP` |
| `compute_int.mm` | `runComputeInt8DP`, `runComputeInt4Packed` |
| `simdgroup.mm` | `runSimdgroupMatrix` + `runSimdgroupMatrixInt` |
| `mtl_blas.mm` | `runMpsGemm` + `runMpsGemmInt` — MPSGraph matrix multiply |
| `global_bandwidth.mm` | `runGlobalBandwidth` |
| `local_bandwidth.mm` | `runLocalBandwidth` |
| `image_bandwidth.mm` | `runImageBandwidth` |
| `atomic_throughput.mm` | `runAtomicThroughput` + `runAtomicThroughputFp` |
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
- If you add a new helper → declare in `mtl_internal.h`, define in `mtl_peak.mm`.
- If you add Objective-C code → remember ARC is enabled (`-fobjc-arc`).
