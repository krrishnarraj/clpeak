# src/cuda — CUDA Backend Implementation

`CudaPeak` class implementation: driver init, per-benchmark runners, and
CUDA kernels (in `cuda_kernels/`) compiled via NVRTC at runtime.  Built as
`peak_cuda` static library.

## Quick Lookups

- Looking for the main class / orchestrator? → `cuda_peak.cpp`
- Looking for CudaDevice class (device init, NVRTC compilation, caching)? → `cuda_device.cpp`
- Looking for driver init / device enumeration? → `cuda_peak.cpp` (`initDriver`)
- Looking for the unified compute kernel runner? → `compute_kernel.cpp` (`runComputeKernel`)
- Looking for kernel timing/calibration? → `cuda_peak.cpp` (`runKernel`)
- Looking for FP compute benchmarks? → `compute_float.cpp`
- Looking for int compute benchmarks? → `compute_int.cpp`
- Looking for WMMA/MMA benchmarks? → `wmma.cpp`
- Looking for cuBLAS benchmarks? → `cuda_blas.cpp`
- Looking for bandwidth benchmarks? → `global_bandwidth.cpp`, `local_bandwidth.cpp`, `image_bandwidth.cpp`, `transfer_bandwidth.cpp`
- Looking for atomic benchmarks? → `atomic_throughput.cpp`
- Looking for kernel latency? → `kernel_latency.cpp`
- Looking for .cu kernel sources? → `cuda_kernels/*.cu`
- Looking for kernel embedding logic? → `cmake/EmbedCudaKernels.cmake`

## Key Files

| File | Purpose |
|------|---------|
| `cuda_peak.cpp` | `CudaPeak` class: ctor, `applyOptions()`, `initDriver()`, `runKernel()`, `runAll()`, `enumerate()`, `printInventory()` |
| `cuda_device.cpp` | `CudaDevice` class: `init()`, `cleanup()`, `getKernel()` (NVRTC compilation + module caching) |
| `compute_kernel.cpp` | `CudaPeak::runComputeKernel()` — shared compute-peak driver: buffer allocation, variant dispatch, used by all `runCompute*` wrappers |
| `compute_float.cpp` | `runComputeSP`, `runComputeHP`, `runComputeDP`, `runComputeMP`, `runComputeBF16` |
| `compute_int.cpp` | `runComputeInt32`, `runComputeInt8DP` |
| `wmma.cpp` | `runWmma` — WMMA/MMA tensor-core umbrella |
| `cuda_blas.cpp` | `runCublas` — cuBLASLt tensor-core GEMM benchmarks |
| `global_bandwidth.cpp` | `runGlobalBandwidth` |
| `local_bandwidth.cpp` | `runLocalBandwidth` |
| `image_bandwidth.cpp` | `runImageBandwidth` |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` |
| `atomic_throughput.cpp` | `runAtomicThroughput` |
| `kernel_latency.cpp` | `runKernelLatency` |
| `cuda_kernels/` | CUDA kernel sources (`.cu`) embedded as C++ string literals |
| `cmake/EmbedCudaKernels.cmake` | `embed_cuda_kernels()` — .cu → C++ raw-string arrays |

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file + update `CMakeLists.txt` + this file.
- If you add a new `.cu` kernel → add to `CLPEAK_CUDA_KERNELS` in `CMakeLists.txt`.
- If you change `CudaPeak` interface → update `include/cuda/cuda_peak.h`.
- If you change the NVRTC include path → check `CLPEAK_CUDA_INCLUDE_DIR` compile definition.
