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
| `cuda_peak.cpp` | `CudaPeak` class: ctor, `initDriver()`, `runKernel()`, `runAll()`, `enumerate()`; `cudaIdentify()`, who each device is (*Identity*) |
| `cuda_device.cpp` | `CudaDevice` class: `init()`, `cleanup()`, `getKernel()` (fatbin `cuModuleLoadData` + module caching; returns a `CudaKernel`, whose `status`/`reason` a runner skips with when it did not load: a stub the build never compiled, no code for this sm, or the driver's error) |
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
| `cmake/nvccw.bat.in` | Windows nvcc wrapper: strips clang resource dirs from `INCLUDE` so nvcc-spawned `cl.exe` survives the ClangCL build env |

## Identity

Each device reports who it is with its `beginDevice()` (`cudaIdentify()` in
`cuda_peak.cpp`, read by `include/common/hardware_link.h`): its UUID
(`cuDeviceGetUuid`), its PCI address (the domain, bus and device
attributes), on Windows its adapter LUID, and as facts its vendor (NVIDIA),
`sm_` arch, SM count, clock, whether it is integrated and, if not, its
memory.  NVIDIA's Vulkan and OpenCL drivers report the same UUID, which is
what links the three; ONNX Runtime's CUDA plugin links by the PCI address.
The helper needs only `cuInit`, so `enumerate()` can call it too.

The helper is called from `enumerate()` too, with what it already holds, so
a listing's device has the type, facts and evidence a run's has
(`InventoryDevice`, `include/common/inventory.h`) and `--list-devices`
links it as a run would.

## Test documentation

See `include/common/AGENTS.md` § Test documentation.  CUDA specifics:

- `cuda_compute_desc_t::description` (test) and
  `cuda_compute_variant_t::description` (one reading); `runComputeKernel()`
  forwards both on every path, skips included.
- `cudaWidthNote()` (`cuda_peak.h`) covers `half` and the
  `float`/`float2`/`float4` bandwidth sweeps; packed `half2` carries its own
  note.
- **`int8_dp`/`dp2`/`dp4`/`dp8` are NOT widths** — one, two, four and eight
  *independent chains* (see `compute_int8_dp.cu`).  They carry their own notes,
  and their axis is "chains in flight".
- **Every tensor-core row is one reading of ONE test** (`wmma`), not a test
  each.  `runWmma` opens one scope and every block writes
  into it via `cuda_compute_desc_t::scope`; the block's prose goes on
  `metricDescription`
  and the exact instruction ends it in parentheses (`(mma.sync m16n8k32)`),
  since the metric label is the data type alone.  The integer rows carry `metricUnit = "ops"` — that override is what makes a
  `wmma_int` twin unnecessary.
- `cublas_gemm` is likewise one test; `blasOpts()` attaches
  the note and integer rows carry their own unit.
- `cuda_blas.cpp` threads a `note` next to `label` through `runVariantAB` /
  `runVariant` / `runVariantFp4`.

## Reading dims

See `include/common/AGENTS.md` § `dims`.  CUDA specifics:

- `cuda_compute_desc_t::dims` and `::revision` (the test it opens),
  `::metricDims` (the single-variant path's reading) and
  `cuda_compute_variant_t::dims` (one reading); `runComputeKernel()`
  forwards them on every path.
- Vector tests carry their data type on the test and `width` per reading;
  `bfloat16_compute` is `{dtype: bf16, acc: fp32}`, and the int8-dot rows are
  `chains`.
- `wmma`: each block's `metricDims` say `dtype`, `acc` where it is fp16 (the
  f16acc rows), `sparsity`, and `tile`, the instruction's M x N x K
  (mma.sync m16n8k32 is "16x8x32").  `fp4_e2m1` is plain E2M1 with no block
  scale, `mxf4_e2m1` and `nvf4_e2m1` are `mxfp4` and `nvfp4`, and `int8 k32`
  is int8 at another `tile`.
- `cublas_gemm` is `impl` cublaslt, and each row's dims sit beside its note
  at the `runVariant` / `runVariantAB` / `runVariantFp4` call.  fp16 is
  `{acc: fp16}` (CUBLAS_COMPUTE_16F); `fp8_e5m2` is E5M2 x E4M3, so
  `{dtype: fp8_e5m2, weights: fp8_e4m3}`.
- Bandwidth: global `{op: read}` and `width`, local `width`, image
  `pixel_format: rgba32f`, transfer `{memory: pinned}` and `direction`.
  Latency: `op`.
- `revision` is 2 where a test's method changed, in a release after the test
  first shipped (its history before a 3.0 rename included), in a way that
  moved its numbers; the reason is beside the line that sets it.  1
  otherwise.  The Vulkan and OpenCL backends count the same way.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file + update `CMakeLists.txt` + this file.
- If you add a new `.cu` kernel → add it to the appropriate `embed_cuda_kernels()` arch group in `CMakeLists.txt`, and declare its `Blob` extern in `include/cuda/cuda_peak.h`.
- If a kernel uses an instruction valid only on certain compute capabilities → put it in (or create) the matching `MIN_ARCH`/`MAX_ARCH` group so nvcc never targets an unsupported arch (ptxas would fail the build).
- If you change `CudaPeak` interface → update `include/cuda/cuda_peak.h`.
