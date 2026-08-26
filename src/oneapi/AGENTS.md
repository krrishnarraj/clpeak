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
- Looking for kernel latency? → `kernel_latency.cpp`

## Key Files

| File | Purpose |
|------|---------|
| `oneapi_peak.cpp` | `OneapiPeak`: ctor, `applyOptions()`, `runAll()`, `runKernel()`, `enumerate()`, `printInventory()`, `enumerateDevices()` |
| `oneapi_device.cpp` | `OneapiDevice::init()` — sets up `sycl::queue`, populates `oneapi_device_info_t` (vendor, CUs, sub-group sizes, fp16/fp64/bf16/XMX flags) |
| `compute_kernel.cpp` | Shared helpers (`pickComputeBlocks`, `computeGflops`) reused by `compute_float.cpp` / `compute_int.cpp` |
| `compute_float.cpp` | `runComputeSP`/`HP`/`DP` (vector-width sweep `{1,2,4,8,16}` via `sycl::vec<T,W>`+`fma`, e.g. `float/float2/.../float16`), `runComputeMP`/`runComputeBF16` (scalar) |
| `compute_int.cpp` | `runComputeInt32` (width sweep `int/int2/.../int16`), `runComputeInt8DP` (DP4a-style `int8_dp/dp2/dp4/dp8` ILP-chain variants with accumulator feedback — see note) |
| `joint_matrix.cpp` | `runJointMatrix` — XMX matrix engine via `sycl::ext::oneapi::matrix` (gated by `CLPEAK_ONEAPI_HAS_JOINT_MATRIX`). The row list is **derived from the device's `matrix_combinations` table**, not hardcoded: one row per advertised (A, B, accumulator) type triple and tile shape, split into the FP and int categories by accumulator type. `CLPEAK_JM_SHAPES` is the compiled (M,N) set — joint_matrix needs the tile at compile time — and an advertised shape outside it records an `Unsupported` row naming the shape. Row names come from `jmBaseName()`: bare dtype (`joint_matrix_bf16`, `_fp16`, `_tf32`, `_int8`) for the first tile of each dtype, `_MxNxK` suffix for any further tile of the same dtype |
| `onemkl.cpp` | `runOnemkl` — oneMKL GEMM peak; FP category fp32/fp64/fp16/bf16 (tflops), INT category int8 via `gemm_bias` (tops). Gated by `CLPEAK_ONEAPI_HAS_ONEMKL`. Each dtype runs in its **own private context + queue + buffers** so one that faults the driver (fp64 → sticky `CL_OUT_OF_RESOURCES`) can't poison the others or the shared `dev.stream`; each reports its own pass/fail |
| `global_bandwidth.cpp` | `runGlobalBandwidth` (float/float2/float4) |
| `local_bandwidth.cpp` | `runLocalBandwidth` (float/float2/float4 via `local_accessor`) |
| `image_bandwidth.cpp` | `runImageBandwidth` (float4 via `sycl::image<2>`) |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` (H2D / D2H via `queue.memcpy` on USM-host pinned memory) |
| `kernel_latency.cpp` | `runKernelLatency` (empty kernel submit + `queue.wait_and_throw()`) |

## Build

Needs the Intel oneAPI Base Toolkit; source `setvars.sh` first, then:

```console
cmake -S . -B build -DCLPEAK_ENABLE_ONEAPI=ON -DCMAKE_CXX_COMPILER=icpx
```

Gates:

- `CLPEAK_ENABLE_ONEAPI` — top-level CMake option (default ON). Backend silently no-ops if `IntelSYCL` package is not found.
- `CLPEAK_ONEAPI_HAS_ONEMKL` — defined when `MKL::MKL_SYCL` target was found. `onemkl.cpp` records skip rows otherwise.
- `CLPEAK_ONEAPI_HAS_JOINT_MATRIX` — defined when `<sycl/ext/oneapi/matrix/matrix.hpp>` is available. `joint_matrix.cpp` records skip rows otherwise. The benchmark additionally skips at runtime on devices without XMX (detected via vendor/name heuristic in `oneapi_device.cpp`).

## Gotchas

- **Compute kernels must carry a real data-dependency chain** or the SYCL
  compiler hoists loop-invariant work out and reports a fabricated peak. The
  FP/INT MAD kernels alternate `x = fma(y,x,y); y = fma(x,y,x);` and the INT8
  DP kernel runs two accumulators feeding each other (`p = dp4(x,q,p);
  q = dp4(x,p,q);`). A symptom of getting this wrong was the INT8 test
  reporting ~768 TOPS on a Xeon CPU (physically impossible). Keep every new
  compute kernel's output dependent on the loop — but the dependency may not
  cost an extra instruction: the INT8 DP kernel used to rewrite an operand with
  `y ^= a`, a second dependent integer op per dot that the op budget credits
  none of, so the reading came out deflated instead. Full rationale above
  `runInt8DpVariant` in `compute_int.cpp`.
- **Vector-width sweeps keep ops/WI constant** by running `baseIters/W` outer
  iterations for width `W`, so the same work-constant (`COMPUTE_FP_WORK_PER_WI`
  etc.) is reported for every width and the numbers stay comparable.
- **No `double` inside fp32/fp16 kernels.** A stray `double` (even just for
  computing per-lane seeds) pulls in the `fp64` aspect, so the kernel fails to
  *launch* on devices without fp64 (Intel Arc) with "Required aspect fp64 is not
  supported". The scalar W=1 case constant-folds the double away and survives,
  so the symptom is "scalar works, every vector width fails" — compute seeds in
  the kernel's own element type.
- **Intel XMX needs the `joint_matrix` B operand in `layout::ext_intel_packed`**
  (VNNI). A `row_major` B is rejected at launch on Xe-HPG (Arc/DG2) as an
  unsupported combination. Launch only tile shapes the device's
  `matrix_combinations` table advertises.
- **A work-group is not one sub-group on Intel**, so per-sub-group ops
  accounting cannot assume it is. `reqd_sub_group_size` cannot pin it (IGC
  internal compiler error on DG2), and IGC may compile a 32-wide work-group as
  four SIMD8 sub-groups, each running the whole chain. `joint_matrix.cpp`
  measures the real count in-kernel and reports it back — read
  `JM_SG_COUNT_NOTE` there before writing another sub-group-collective test.

## Test documentation

See `include/common/AGENTS.md` § Test documentation.  oneAPI specifics:

- No descriptor struct: the description is the 5th field of the braced
  `TestSpec` at each `beginTest()` call site.
- The width and chain notes come off the **template parameter**, not a call
  site: `runFpWidth<…,W>` / `runIntWidth<…,W>` call `oneapiWidthNote(W)` and
  `runInt8DpVariant<NCH>` calls `oneapiChainNote(NCH)`, each once inside the
  function.  Both helpers live in `oneapi_peak.h`.
- **`int8_dp`/`dp2`/`dp4`/`dp8` are chains, not widths** — hence the separate
  `oneapiChainNote()`.
- `joint_matrix.cpp`: `jmNote()` composes each row's note from its dtype pair
  and tile shape, so a row documents the shape it actually ran.
- `onemkl.cpp` threads a `note` next to `label` through `measure()`.

## Chain shapes

`runFpWidth` and `runIntWidth` each submit two kernels -- the squaring chain
and a second shape -- and report the faster.  Float families use an affine
chain, integer families a rotating one, because an integer affine recurrence
folds legally and one compiler in the fleet folds it.  This backend runs on
Intel GPUs, where the squaring chain alone reports half rate on Alchemist.
Each shape needs its own SYCL kernel-name type.  Why: the MAD chain block in
`include/common/common.h`.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate file + update `CMakeLists.txt` + this file.
- If you add a new device-capability gate → populate it in `oneapi_device.cpp::init()` and document it under `oneapi_device_info_t`.
- If you change the `OneapiPeak` interface → update `include/oneapi/oneapi_peak.h`.
