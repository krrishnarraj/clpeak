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
- Looking for the CMake build logic? → `CMakeLists.txt`

## Key Files

| File | Purpose |
|------|---------|
| `cl_peak.cpp` | `clPeak` class: constructor, `applyOptions()`, `runAll()`, `run_kernel()`, `enumerate()`, `printInventory()` |
| `cl_kernels.cpp` | Kernel source strings (stringified .cl includes) + accessor functions |
| `compute_test.cpp` | `runComputeTest()` — shared compute-peak driver for float/int/char/short/etc. |
| `cl_common.cpp` | `device_info_t` struct, device capability queries |
| `global_bandwidth.cpp` | `runGlobalBandwidthTest()` — global memory bandwidth |
| `local_bandwidth.cpp` | `runLocalBandwidthTest()` — local memory bandwidth |
| `image_bandwidth.cpp` | `runImageBandwidthTest()` — image object bandwidth |
| `transfer_bandwidth.cpp` | `runTransferBandwidthTest()` — host↔device transfer |
| `kernel_latency.cpp` | `runKernelLatency()` — single-dispatch kernel latency |
| `kernels/` | OpenCL C kernel sources (`.cl` files) |
| `cmake/` | `BuildSdk.cmake` — SDK fallback finder |

## Test documentation

Every test here carries a plain-language `description` for non-expert readers
(rendered by the GUI's info glyph and the CLI's `--describe`; see
`include/common/AGENTS.md` § Test documentation for the two levels and the
style rules).  OpenCL-specific plumbing:

- `runComputeTest()` takes the test-level text as a `description` parameter
  (after `unit`), so each call site in `cl_peak.cpp` carries its own; the
  per-width readings are documented inside the helper.
- `clWidthNote()` (`cl_peak.h`) is the shared wording for the
  `float`/`float2`/…/`float16` readings, used by the compute helper and the
  global/local bandwidth tests.  The bulk-skip paths pass it too, so an errored
  row still says what it would have measured.
- **`transfer_bandwidth`'s zero-copy convention is now explained to the user**:
  the test description says a route that moves nothing reads as zero, and the
  `enqueuemapbuffer` / `enqueueunmap` notes repeat it.  If you change
  `ZERO_COPY_MULTIPLIER` or the reporting rule, update those strings — a bare
  `0.00` with no explanation is what they exist to prevent.

## When You Change This Directory

- If you add a new benchmark `.cpp` → update `CMakeLists.txt` + this file.
- If you add a new `.cl` kernel → update `cl_kernels.cpp` + the kernels table above.
- If you change `clPeak` interface → update `include/opencl/cl_peak.h`.
- If you change the SDK detection → test on macOS (framework) and Linux (ICD).
