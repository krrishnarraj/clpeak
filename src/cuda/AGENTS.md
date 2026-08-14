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

## Test documentation

Every test here carries a plain-language `description` for non-expert readers
(rendered by the GUI's info glyph and the CLI's `--describe`; see
`include/common/AGENTS.md` § Test documentation for the two levels and the
style rules).  CUDA-specific plumbing:

- `cuda_compute_desc_t::description` → the test-level text, and
  `cuda_compute_variant_t::description` → the note for that one reading.
  `runComputeKernel()` forwards both on every path, skips included, so the many
  Unsupported rows (every dtype above the device's compute capability) still
  explain what they would have measured.
- `cudaWidthNote()` (`cuda_peak.h`) covers the readings that really are vector
  widths: `half`/`half2` and the `float`/`float2`/`float4` bandwidth sweeps.
  **The `int8_dp`/`dp2`/`dp4`/`dp8` rows are NOT widths** — they are one, two,
  four and eight *independent chains* (see `compute_int8_dp.cu`), so they carry
  their own notes.  Same trap as Vulkan's int8-dot rows.
- The WMMA rows are one reading each, named after the test, so they document
  the test only.
- `cuda_blas.cpp` threads a `note` parameter through `runVariantAB` /
  `runVariant` / `runVariantFp4` next to `label`, so each dtype's explanation
  reaches all six skip paths and the emit.

**Syntax-checking without a CUDA toolkit.** The driver/cuBLASLt surface this
backend touches is small and almost entirely opaque handles, so a stub
`cuda.h` + `cublasLt.h` (a few dozen typedefs, enums and prototypes) is enough
to run `clang++ -fsyntax-only -std=c++17 -DENABLE_CUDA -I<stubs> -Iinclude
src/cuda/*.cpp` on a machine with no NVIDIA anything.  That catches the whole
class of errors an edit to these files usually introduces — wrong struct field,
wrong argument count, a brace-init missing a field.  Check the FP4 path both
ways (`-DCLPEAK_CUBLASLT_HAS_FP4=1` and without); the `#else` branch has its own
call sites.  It does NOT substitute for a run on real hardware.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file + update `CMakeLists.txt` + this file.
- If you add a new `.cu` kernel → add it to the appropriate `embed_cuda_kernels()` arch group in `CMakeLists.txt`, and declare its `Blob` extern in `include/cuda/cuda_peak.h`.
- If a kernel uses an instruction valid only on certain compute capabilities → put it in (or create) the matching `MIN_ARCH`/`MAX_ARCH` group so nvcc never targets an unsupported arch (ptxas would fail the build).
- If you change `CudaPeak` interface → update `include/cuda/cuda_peak.h`.
