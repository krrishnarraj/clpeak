# src/opencl — OpenCL Backend Implementation

`clPeak` class implementation: device init, per-benchmark runners, and
OpenCL C kernels (in `kernels/`).  Built as `peak_opencl` static library.

## Quick Lookups

- Looking for the main class / orchestrator? → `cl_peak.cpp`
- Looking for device init / platform enumeration? → `cl_peak.cpp` (top of file)
- Looking for kernel string definitions? → `cl_kernels.cpp`
- Looking for the unified compute test helper? → `compute_test.cpp` (`runComputeTest`)
- Looking for OpenCL utility types? → `cl_common.cpp` + `include/opencl/cl_common.h`
- Looking for .cl kernel sources? → `kernels/*.cl`
- Looking for why a compute family races several chain shapes? → `kernels/mad_chain.cl`
- Looking for the CMake build logic? → `CMakeLists.txt`

## Key Files

| File | Purpose |
|------|---------|
| `cl_peak.cpp` | `clPeak` class: constructor, `runAll()` (devices numbered consecutively across platforms), `run_kernel()`, `enumerate()` |
| `cl_kernels.cpp` | Kernel source strings (stringified .cl includes) + accessor functions; `clGetMainKernels(selected)` assembles the main program from only the families the selected tests use |
| `compute_test.cpp` | `runComputeTest()` — shared compute-peak driver for float/int/char/short/etc. |
| `cl_common.cpp` | `device_info_t` struct, device capability queries |
| `cl_utils.cpp` | OpenCL-only helpers (`roundToMultipleOf`, `trimString`) — `include/opencl/cl_utils.h` |
| `global_bandwidth.cpp` | `runGlobalBandwidthTest()` — global memory bandwidth |
| `local_bandwidth.cpp` | `runLocalBandwidthTest()` — local memory bandwidth |
| `image_bandwidth.cpp` | `runImageBandwidthTest()` — image object bandwidth |
| `transfer_bandwidth.cpp` | `runTransferBandwidthTest()` — host↔device transfer |
| `kernel_latency.cpp` | `runKernelLatency()` — single-dispatch kernel latency |
| `kernels/` | OpenCL C kernel sources (`.cl` files) |
| `kernels/mad_chain.cl` | The alternate MAD chains (`AF*` affine, `RT*` rotating) every `compute_*_alt_v*` kernel expands.  Included first in `cl_kernels.cpp` so the macros exist before use |
| `cmake/` | `BuildSdk.cmake` — SDK fallback finder |

## Test documentation

See `include/common/AGENTS.md` § Test documentation.  OpenCL specifics:

- `runComputeTest()` takes the test-level text as a `description` parameter
  (after `unit`); the per-width readings are documented inside the helper,
  which also classifies every test it drives as a homogeneous "vector width"
  sweep — that is all it ever runs.
- `clWidthNote()` (`cl_peak.h`) is the shared wording for the
  `float`/`float2`/…/`float16` readings, used by the compute helper and the
  global/local bandwidth tests.
- `transfer_bandwidth` reports the same two readings as the CUDA/ROCm/oneAPI
  backends -- `h2d_pinned` / `d2h_pinned` -- so the four are directly
  comparable on one machine.

## Chain shapes

Every compute family defines its kernels at least twice: `compute_<fam>_v<W>`
with the squaring chain, and `compute_<fam>_alt_v<W>` with a second shape from
`kernels/mad_chain.cl`.  Float families use the affine `AF*` macros, written
once per width as `<FAM>_ALT_V<W>(NAME, ATTR)`; where `runAll` passes
`-DCLPEAK_ALT_SG16` (a device offering sub-group 16 through
`cl_intel_required_subgroup_size`) the same macro also instantiates
`compute_<fam>_alt_sg16_v<W>` pinned to 16.  Integer families use the rotating
`RT*` ones -- an integer affine recurrence folds legally and Apple's compiler
folds it.  `runComputeTest` times every kernel a width has and reports the
fastest; a family with no `_alt` kernel simply races nothing, which is how
`compute_int8_dp` currently behaves.  All shapes must spell the same number of
chain instructions per work-item so the readings stay comparable.  Per loop
trip, the fp32/fp16 squaring kernels run 128 (`MAD_128`, but 16 at width 16),
mixed precision runs 128 in both shapes (`AF*_128`), and everything else 16.
Full rationale: `mad_chain.cl` and the MAD chain block in
`include/common/common.h`.

## When You Change This Directory

- If you add a new benchmark `.cpp` → update `CMakeLists.txt` + this file.
- If you add a new `.cl` kernel → update `cl_kernels.cpp` + the kernels table above.
- If you change `clPeak` interface → update `include/opencl/cl_peak.h`.
- If you change the SDK detection → test on macOS (framework) and Linux (ICD).
