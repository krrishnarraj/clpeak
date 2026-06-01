# src/rocm — ROCm Backend Implementation

`RocmPeak` class implementation: HIP runtime init, per-benchmark runners, and
HIP kernels (in `rocm_kernels/`) compiled via HIPRTC at runtime. Built as
`peak_rocm` static library.

## Quick Lookups

- Looking for the main class / orchestrator? → `rocm_peak.cpp`
- Looking for RocmDevice class (device init, HIPRTC compilation, caching)? → `rocm_device.cpp`
- Looking for HIP runtime init / device enumeration? → `rocm_peak.cpp` (`initRuntime`)
- Looking for the unified compute kernel runner? → `compute_kernel.cpp` (`runComputeKernel`)
- Looking for kernel timing/calibration? → `rocm_peak.cpp` (`runKernel`)
- Looking for FP compute benchmarks? → `compute_float.cpp`
- Looking for int compute benchmarks? → `compute_int.cpp`
- Looking for rocWMMA matrix benchmarks? → `rocwmma.cpp`
- Looking for raw MFMA matrix-core peak benchmarks (incl. scaled-MFMA mxfp4)? → `mfma.cpp` + `rocm_kernels/mfma_*.hip`
- Looking for 2:4 structured-sparse MFMA peak benchmarks? → `sparse_mfma.cpp` + `rocm_kernels/smfmac_*.hip`
- Looking for rocBLAS GEMM benchmarks? → `rocblas.cpp`
- Looking for hipBLASLt FP8 GEMM benchmarks? → `hipblaslt_gemm.cpp`
- Looking for bandwidth benchmarks? → `global_bandwidth.cpp`, `local_bandwidth.cpp`, `image_bandwidth.cpp`, `transfer_bandwidth.cpp`
- Looking for atomic benchmarks? → `atomic_throughput.cpp`
- Looking for kernel latency? → `kernel_latency.cpp`
- Looking for .hip kernel sources? → `rocm_kernels/*.hip`
- Looking for kernel embedding logic? → `cmake/EmbedRocmKernels.cmake`

## Key Files

| File | Purpose |
|------|---------|
| `rocm_peak.cpp` | `RocmPeak` class: ctor, `applyOptions()`, `initRuntime()`, `runKernel()`, `runAll()`, `enumerate()`, `printInventory()` |
| `rocm_device.cpp` | `RocmDevice` class: `init()`, `cleanup()`, `getKernel()` (HIPRTC compilation + module caching) |
| `compute_kernel.cpp` | `RocmPeak::runComputeKernel()` — shared compute-peak driver: buffer allocation, variant dispatch, used by all `runCompute*` wrappers |
| `compute_float.cpp` | `runComputeSP`, `runComputeHP`, `runComputeDP`, `runComputeMP`, `runComputeBF16` |
| `compute_int.cpp` | `runComputeInt32`, `runComputeInt8DP` |
| `rocwmma.cpp` | `runRocwmma` — raw rocWMMA matrix-engine benchmark |
| `mfma.cpp` | `runMfma` — raw MFMA matrix-core peak (fp16/bf16/int8/fp8/mxfp4) via `__builtin_amdgcn_mfma_*` |
| `sparse_mfma.cpp` | `runSparseMfma` — 2:4 structured-sparse MFMA peak (fp16/bf16/int8/fp8) via `__builtin_amdgcn_smfmac_*` |
| `rocblas.cpp` | `runRocblas` — rocBLAS GEMM peak; FP category fp32/fp64/fp16/bf16 (tflops), INT category int8 (tops) |
| `hipblaslt_gemm.cpp` | `runHipblasLt` — hipBLASLt GEMM peak: fp8 e4m3/e5m2 fnuz + mxfp4 (block-scaled, gated by `CLPEAK_HIPBLASLT_HAS_FP4`) |
| `global_bandwidth.cpp` | `runGlobalBandwidth` |
| `local_bandwidth.cpp` | `runLocalBandwidth` |
| `image_bandwidth.cpp` | `runImageBandwidth` via HIP texture object |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` |
| `atomic_throughput.cpp` | `runAtomicThroughput` |
| `kernel_latency.cpp` | `runKernelLatency` |
| `rocm_kernels/` | HIP kernel sources (`.hip`) embedded as C++ string literals |
| `cmake/EmbedRocmKernels.cmake` | `embed_rocm_kernels()` — .hip → C++ raw-string arrays |

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file + update `CMakeLists.txt` + this file.
- If you add a new `.hip` kernel → add to `CLPEAK_ROCM_KERNELS` in `CMakeLists.txt`.
- If you change `RocmPeak` interface → update `include/rocm/rocm_peak.h`.
- If you change the HIPRTC include path → check `CLPEAK_ROCM_INCLUDE_DIR` compile definition.
