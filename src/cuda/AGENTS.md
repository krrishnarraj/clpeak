# src/cuda — CUDA Backend Implementation

`CudaPeak` class implementation: driver init, per-benchmark runners, and
CUDA kernels (in `cuda_kernels/`) compiled **ahead-of-time** by nvcc to
multi-arch fatbins at build time and loaded via the CUDA driver at run time.
The shipped binary needs only the NVIDIA driver — no NVRTC, no toolkit headers.
Built as `peak_cuda` static library.

## Quick Lookups

- Looking for the main class / orchestrator? → `cuda_peak.cpp`
- Looking for CudaDevice class (device init, fatbin module load, caching)? → `cuda_device.cpp`
- Looking for driver init / device enumeration? → `cuda_peak.cpp` (`initDriver`)
- Looking for the unified compute kernel runner? → `compute_kernel.cpp` (`runComputeKernel`)
- Looking for kernel timing/calibration? → `cuda_peak.cpp` (`runKernel`)
- Looking for FP compute benchmarks? → `compute_float.cpp`
- Looking for int compute benchmarks? → `compute_int.cpp`
- Looking for WMMA/MMA benchmarks? → `wmma.cpp`
- Looking for cuBLAS benchmarks? → `cuda_blas.cpp`
- Looking for bandwidth benchmarks? → `global_bandwidth.cpp`, `local_bandwidth.cpp`, `image_bandwidth.cpp`, `transfer_bandwidth.cpp`

- Looking for kernel latency? → `kernel_latency.cpp`
- Looking for .cu kernel sources? → `cuda_kernels/*.cu`
- Looking for AOT compile + embedding logic? → `cmake/EmbedCudaKernels.cmake` (+ `cmake/EmbedBin.cmake`)

## Key Files

| File | Purpose |
|------|---------|
| `cuda_peak.cpp` | `CudaPeak` class: ctor, `applyOptions()`, `initDriver()`, `runKernel()`, `runAll()`, `enumerate()`, `printInventory()` |
| `cuda_device.cpp` | `CudaDevice` class: `init()`, `cleanup()`, `getKernel()` (fatbin `cuModuleLoadData` + module caching) |
| `compute_kernel.cpp` | `CudaPeak::runComputeKernel()` — shared compute-peak driver: buffer allocation, variant dispatch, used by all `runCompute*` wrappers |
| `compute_float.cpp` | `runComputeSP`, `runComputeHP`, `runComputeDP`, `runComputeMP`, `runComputeBF16` |
| `compute_int.cpp` | `runComputeInt32`, `runComputeInt8DP` |
| `wmma.cpp` | `runWmma` — WMMA/MMA tensor-core umbrella |
| `cuda_blas.cpp` | `runCublas` — cuBLASLt tensor-core GEMM benchmarks |
| `global_bandwidth.cpp` | `runGlobalBandwidth` |
| `local_bandwidth.cpp` | `runLocalBandwidth` |
| `image_bandwidth.cpp` | `runImageBandwidth` |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` |

| `kernel_latency.cpp` | `runKernelLatency` |
| `cuda_kernels/` | CUDA kernel sources (`.cu`), AOT-compiled to fatbins and embedded as byte arrays |
| `cmake/EmbedCudaKernels.cmake` | `embed_cuda_kernels()` — nvcc `-fatbin` per arch group + byte embed |
| `cmake/EmbedBin.cmake` | build-time `-P` script: binary → C++ `Blob` byte array |

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file + update `CMakeLists.txt` + this file.
- If you add a new `.cu` kernel → add it to the appropriate `embed_cuda_kernels()` arch group in `CMakeLists.txt`, and declare its `Blob` extern in `include/cuda/cuda_peak.h`.
- If a kernel uses an instruction valid only on certain compute capabilities → put it in (or create) the matching `MIN_ARCH`/`MAX_ARCH` group so nvcc never targets an unsupported arch (ptxas would fail the build).
- If you change `CudaPeak` interface → update `include/cuda/cuda_peak.h`.
