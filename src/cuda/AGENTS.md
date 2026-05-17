# src/cuda — CUDA Backend Implementation

`CudaPeak` class implementation: driver init, per-benchmark runners, and
CUDA kernels (in `cuda_kernels/`) compiled via NVRTC at runtime.  Built as
`peak_cuda` static library.

## Quick Lookups

- Looking for the main class? → `cuda_peak.cpp`
- Looking for driver init / device enumeration? → `cuda_peak.cpp` (`initDriver`)
- Looking for the unified compute kernel runner? → `cuda_peak.cpp` (`runComputeKernel`)
- Looking for cuBLAS benchmarks? → `cuda_blas.cpp`
- Looking for .cu kernel sources? → `cuda_kernels/*.cu`
- Looking for kernel embedding logic? → `cmake/EmbedCudaKernels.cmake`

## Key Files

| File | Purpose |
|------|---------|
| `cuda_peak.cpp` | `CudaPeak` class: constructor, `applyOptions()`, `runAll()`, all benchmarks |
| `cuda_blas.cpp` | `runCublas()` — cuBLASLt tensor-core GEMM benchmarks |
| `cuda_kernels/` | CUDA kernel sources (`.cu`) embedded as C++ string literals |
| `cmake/EmbedCudaKernels.cmake` | `embed_cuda_kernels()` — .cu → C++ raw-string arrays |

## When You Change This Directory

- If you add a new benchmark → update `CMakeLists.txt` (kernel list) + this file.
- If you add a new `.cu` kernel → add to `CLPEAK_CUDA_KERNELS` in `CMakeLists.txt`.
- If you change `CudaPeak` interface → update `include/cuda/cuda_peak.h`.
- If you change the NVRTC include path → check `CLPEAK_CUDA_INCLUDE_DIR` compile definition.
