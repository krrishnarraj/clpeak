# src/metal — Metal Backend Implementation

`MetalPeak` class implementation: device init, per-benchmark runners, and
Metal Shading Language kernels (in `mtl_kernels/`).  Built as `peak_metal`
static library.  Source files are Objective-C++ (`.mm`).

## Quick Lookups

- Looking for the main class? → `mtl_peak.mm`
- Looking for device init / enumeration? → `mtl_peak.mm` (top of file)
- Looking for the unified compute kernel runner? → `mtl_peak.mm` (`runComputeKernel`)
- Looking for MPSGraph benchmarks? → `mtl_blas.mm`
- Looking for .metal kernel sources? → `mtl_kernels/*.metal`
- Looking for kernel embedding logic? → `cmake/EmbedMetalKernels.cmake`

## Key Files

| File | Purpose |
|------|---------|
| `mtl_peak.mm` | `MetalPeak` class: constructor, `applyOptions()`, `runAll()`, all benchmarks |
| `mtl_blas.mm` | `runMpsGemm()` / `runMpsGemmInt()` — MPSGraph matrix multiply |
| `mtl_kernels/` | Metal Shading Language kernels (`.metal`) embedded as C++ string literals |
| `cmake/EmbedMetalKernels.cmake` | `embed_metal_kernels()` — .metal → C++ raw-string arrays |

## When You Change This Directory

- If you add a new benchmark → update `CMakeLists.txt` (kernel list) + this file.
- If you add a new `.metal` kernel → add to `CLPEAK_MTL_KERNELS` in `CMakeLists.txt`.
- If you change `MetalPeak` interface → update `include/metal/mtl_peak.h`.
- If you add Objective-C code → remember ARC is enabled (`-fobjc-arc`).
