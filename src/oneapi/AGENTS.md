# src/oneapi — oneAPI/SYCL Backend Implementation

`OneapiPeak` class implementation: SYCL queue/device init, per-benchmark
runners, and SYCL kernels expressed inline as C++ lambdas. Built as
`peak_oneapi` static library. Compiles only with `icpx` or
`clang++ -fsycl`; the IntelSYCL CMake package is required.

Unlike the CUDA/ROCm/OpenCL backends there is no runtime kernel compilation
and no kernel-source embedding step — the DPC++ compiler emits SPIR-V at
build time and the SYCL runtime JITs it on first launch.

## Quick Lookups

- Looking for the main class / orchestrator? → `oneapi_peak.cpp`
- Looking for OneapiDevice class (SYCL device/queue init, info)? → `oneapi_device.cpp`
- Looking for SYCL device enumeration? → `oneapi_peak.cpp` (`enumerateDevices` — prefers GPUs, falls back to CPU/accelerator when no GPU is visible)
- Looking for the shared compute helpers (block sizing, gflops math)? → `compute_kernel.cpp`
- Looking for kernel timing? → `oneapi_peak.cpp` (`OneapiPeak::runKernel`)
- Looking for FP compute benchmarks? → `compute_float.cpp`
- Looking for int compute benchmarks? → `compute_int.cpp`
- Looking for joint_matrix (XMX) benchmarks? → `joint_matrix.cpp`
- Looking for oneMKL GEMM benchmark? → `onemkl.cpp`
- Looking for bandwidth benchmarks? → `global_bandwidth.cpp`, `local_bandwidth.cpp`, `image_bandwidth.cpp`, `transfer_bandwidth.cpp`
- Looking for atomic benchmarks? → `atomic_throughput.cpp`
- Looking for kernel latency? → `kernel_latency.cpp`

## Key Files

| File | Purpose |
|------|---------|
| `oneapi_peak.cpp` | `OneapiPeak`: ctor, `applyOptions()`, `runAll()`, `runKernel()`, `enumerate()`, `printInventory()`, `enumerateGpus()` |
| `oneapi_device.cpp` | `OneapiDevice::init()` — sets up `sycl::queue`, populates `oneapi_device_info_t` (vendor, CUs, sub-group sizes, fp16/fp64/bf16/XMX flags) |
| `compute_kernel.cpp` | Shared helpers (`pickComputeBlocks`, `computeGflops`) reused by `compute_float.cpp` / `compute_int.cpp` |
| `compute_float.cpp` | `runComputeSP`, `runComputeHP`, `runComputeDP`, `runComputeMP`, `runComputeBF16` |
| `compute_int.cpp` | `runComputeInt32`, `runComputeInt8DP`, `runComputeInt4Packed` |
| `joint_matrix.cpp` | `runJointMatrix` — XMX matrix engine via `sycl::ext::oneapi::matrix` (gated by `CLPEAK_ONEAPI_HAS_JOINT_MATRIX`) |
| `onemkl.cpp` | `runOnemkl` — oneMKL GEMM peak fp32/fp64/fp16 (gated by `CLPEAK_ONEAPI_HAS_ONEMKL`) |
| `global_bandwidth.cpp` | `runGlobalBandwidth` (float/float2/float4) |
| `local_bandwidth.cpp` | `runLocalBandwidth` (float/float2/float4 via `local_accessor`) |
| `image_bandwidth.cpp` | `runImageBandwidth` (float4 via `sycl::image<2>`) |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` (H2D / D2H via `queue.memcpy` on USM-host pinned memory) |
| `atomic_throughput.cpp` | `runAtomicThroughput` (global + SLM via `sycl::atomic_ref`) |
| `kernel_latency.cpp` | `runKernelLatency` (empty kernel submit + `queue.wait_and_throw()`) |

## Build Gates

- `CLPEAK_ENABLE_ONEAPI` — top-level CMake option (default ON). Backend silently no-ops if `IntelSYCL` package is not found.
- `CLPEAK_ONEAPI_HAS_ONEMKL` — defined when `MKL::MKL_SYCL` target was found. `onemkl.cpp` records skip rows otherwise.
- `CLPEAK_ONEAPI_HAS_JOINT_MATRIX` — defined when `<sycl/ext/oneapi/matrix/matrix.hpp>` is available. `joint_matrix.cpp` records skip rows otherwise. The benchmark additionally skips at runtime on devices without XMX (detected via vendor/name heuristic in `oneapi_device.cpp`).

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate file + update `CMakeLists.txt` + this file.
- If you add a new device-capability gate → populate it in `oneapi_device.cpp::init()` and document it under `oneapi_device_info_t`.
- If you change the `OneapiPeak` interface → update `include/oneapi/oneapi_peak.h`.
