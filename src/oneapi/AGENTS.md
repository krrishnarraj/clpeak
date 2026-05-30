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
| `oneapi_peak.cpp` | `OneapiPeak`: ctor, `applyOptions()`, `runAll()`, `runKernel()`, `enumerate()`, `printInventory()`, `enumerateDevices()` |
| `oneapi_device.cpp` | `OneapiDevice::init()` — sets up `sycl::queue`, populates `oneapi_device_info_t` (vendor, CUs, sub-group sizes, fp16/fp64/bf16/XMX flags) |
| `compute_kernel.cpp` | Shared helpers (`pickComputeBlocks`, `computeGflops`) reused by `compute_float.cpp` / `compute_int.cpp` |
| `compute_float.cpp` | `runComputeSP`/`HP`/`DP` (vector-width sweep `{1,2,4,8,16}` via `sycl::vec<T,W>`+`fma`, e.g. `float/float2/.../float16`), `runComputeMP`/`runComputeBF16` (scalar) |
| `compute_int.cpp` | `runComputeInt32` (width sweep `int/int2/.../int16`), `runComputeInt8DP` (DP4a-style `int8_dp/dp2/dp4/dp8` ILP-chain variants with accumulator feedback — see note) |
| `joint_matrix.cpp` | `runJointMatrix` — XMX matrix engine via `sycl::ext::oneapi::matrix` (gated by `CLPEAK_ONEAPI_HAS_JOINT_MATRIX`). FP category emits `joint_matrix_bf16`/`_fp16` (8x16x16) + `_tf32` (8x16x8); int category emits `joint_matrix_int8` (8x16x32) |
| `onemkl.cpp` | `runOnemkl` — oneMKL GEMM peak fp32/fp64/fp16 (gated by `CLPEAK_ONEAPI_HAS_ONEMKL`) |
| `global_bandwidth.cpp` | `runGlobalBandwidth` (float/float2/float4) |
| `local_bandwidth.cpp` | `runLocalBandwidth` (float/float2/float4 via `local_accessor`) |
| `image_bandwidth.cpp` | `runImageBandwidth` (float4 via `sycl::image<2>`) |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` (H2D / D2H via `queue.memcpy` on USM-host pinned memory) |
| `atomic_throughput.cpp` | `runAtomicThroughput` (global + SLM via `sycl::atomic_ref`) |
| `kernel_latency.cpp` | `runKernelLatency` (empty kernel submit + `queue.wait_and_throw()`) |


## Build

- oneAPI requires the Intel oneAPI Base Toolkit (sources `setvars.sh` before invoking cmake).
- oneAPI requires -DCMAKE_CXX_COMPILER=icpx to be set in cmake step
- Cmake step: `cmake -S . -B build -DCLPEAK_ENABLE_ONEAPI=ON -DCMAKE_CXX_COMPILER=icpx`

## Build Gates

- `CLPEAK_ENABLE_ONEAPI` — top-level CMake option (default ON). Backend silently no-ops if `IntelSYCL` package is not found.
- `CLPEAK_ONEAPI_HAS_ONEMKL` — defined when `MKL::MKL_SYCL` target was found. `onemkl.cpp` records skip rows otherwise.
- `CLPEAK_ONEAPI_HAS_JOINT_MATRIX` — defined when `<sycl/ext/oneapi/matrix/matrix.hpp>` is available. `joint_matrix.cpp` records skip rows otherwise. The benchmark additionally skips at runtime on devices without XMX (detected via vendor/name heuristic in `oneapi_device.cpp`).

## Gotchas

- **Compute kernels must carry a real data-dependency chain** or the SYCL
  compiler hoists loop-invariant work out and reports a fabricated peak. The
  FP/INT MAD kernels alternate `x = fma(y,x,y); y = fma(x,y,x);` and the INT8
  DP kernel feeds the accumulator back via `y ^= a`. A symptom of getting this
  wrong was the INT8 test reporting ~768 TOPS on a Xeon CPU (physically
  impossible). Keep every new compute kernel's output dependent on the loop.
- **Vector-width sweeps keep ops/WI constant** by running `baseIters/W` outer
  iterations for width `W`, so the same work-constant (`COMPUTE_FP_WORK_PER_WI`
  etc.) is reported for every width and the numbers stay comparable.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate file + update `CMakeLists.txt` + this file.
- If you add a new device-capability gate → populate it in `oneapi_device.cpp::init()` and document it under `oneapi_device_info_t`.
- If you change the `OneapiPeak` interface → update `include/oneapi/oneapi_peak.h`.
