# src/vulkan — Vulkan Backend Implementation

`vkPeak` class implementation: instance/device init, per-benchmark runners,
and GLSL compute shaders (in `shaders/`).  Built as `peak_vulkan` static library.

## Quick Lookups

- Looking for the main class / orchestrator? → `vk_peak.cpp`
- Looking for VulkanDevice class (logical device, buffers, pipelines)? → `vulkan_device.cpp`
- Looking for instance init? → `vk_peak.cpp` (`initInstance`)
- Looking for the unified compute kernel runner? → `compute_kernel.cpp` (`runComputeKernel`)
- Looking for kernel timing/calibration? → `vk_peak.cpp` (`runKernel`)
- Looking for GLSL shader sources? → `shaders/*.comp`
- Looking for shader compilation logic? → `cmake/CompileShaders.cmake`
- Looking for FP compute benchmarks? → `compute_float.cpp`
- Looking for int compute benchmarks? → `compute_int.cpp`
- Looking for cooperative matrix benchmarks? → `coopmat.cpp`
- Looking for bandwidth benchmarks? → `global_bandwidth.cpp`, `local_bandwidth.cpp`, `image_bandwidth.cpp`, `transfer_bandwidth.cpp`
- Looking for kernel latency? → `kernel_latency.cpp`

## Key Files

| File | Purpose |
|------|---------|
| `vk_peak.cpp` | `vkPeak` class: ctor, `applyOptions()`, `initInstance()`, `cleanup()`, `runKernel()`, `runAll()`, `enumerate()`, `printInventory()` |
| `vulkan_device.cpp` | `VulkanDevice` class: `init()` (4-step: basic info → CU count → optional features → logical device), `cleanup()`, `createBuffer()`, `createComputePipeline()`, `submitAndWait()`, `zeroBuffer()` |
| `compute_kernel.cpp` | `vkPeak::runComputeKernel()` — shared compute-peak driver: buffer/descriptor/pipeline scaffolding used by all `runCompute*` wrappers |
| `compute_float.cpp` | `runComputeSP`, `runComputeHP`, `runComputeDP`, `runComputeMP`, `runComputeBF16` |
| `compute_int.cpp` | `runComputeInt32`, `runComputeInt8DP` |
| `coopmat.cpp` | `runCoopMatrix` — cooperative matrix (tensor-core) umbrella |
| `global_bandwidth.cpp` | `runGlobalBandwidth` |
| `local_bandwidth.cpp` | `runLocalBandwidth` |
| `image_bandwidth.cpp` | `runImageBandwidth` |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` |
| `kernel_latency.cpp` | `runKernelLatency` |
| `shaders/` | GLSL compute shaders (`.comp`) compiled to SPIR-V at build time |
| `shaders/mad_chain.glsl` | The two MAD-chain shapes every compute-peak shader races (`CHAIN_DECL` / `MAD_16` / `MAD_128` / `CHAIN_MAP` / `CHAIN_RESULT`) |
| `shaders/coopmat_chain.glsl` | The MulAdd run every coopmat shader runs (`CM_TA`/`CM_TB`/`CM_TC`, `CM_DECLARE`, `CM_MMA_TRIP`) — and why it is shaped that way |
| `cmake/CompileShaders.cmake` | `compile_shaders()` — glslc → SPIR-V → embedded C++ arrays |

## Test documentation

See `include/common/AGENTS.md` § Test documentation.  Vulkan specifics:

- `vk_compute_desc_t::description` (test) and
  `vk_compute_variant_t::description` (one reading); `runComputeKernel()`
  forwards both on every path, skips included.
- `vkWidthNote()` (`vk_peak.h`) is the shared wording for the
  `float`/`float2`/`float4` readings.
- **`int8_dp`/`int8_dp2`/`int8_dp4` are NOT a width sweep** — one, two and four
  *independent dot-product chains*.  They carry their own notes; do not unify
  them onto `vkWidthNote()`, and their axis is "chains in flight", not "vector
  width".
- **Every coopmat data type is one reading of ONE test** (`coopmat`), not a
  test each.  `runCoopMatrix` opens one scope and every
  `#ifdef` block writes into it via `vk_compute_desc_t::scope` — a desc that
  does still sets `resultTag`, which the runner's --verbose lines report
  themselves under, but leaves the other header fields null.  The reading is
  named by its data type alone;
  the driver-advertised tile goes in its NOTE, appended by `bindCoopTile`,
  because a shape in the name would differ between a device that measured the
  reading and one that skipped it — the same reading under two ids.  The prose
  goes on `metricDescription` — the test's own description covers the family.
  int8 carries `metricUnit = "tops"`, which is what
  lets it share the test instead of needing a `coopmat_int8` twin.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file (or create a new one) + update `CMakeLists.txt` + this file.
- If you add a new `.comp` shader → add to `CLPEAK_VK_SHADERS` in `CMakeLists.txt`
  and declare its extern in the `vk_shaders` namespace (`include/vulkan/vk_peak.h`).
- **Never change a compute-peak shader's inner loop without reading the compiled
  SPIR-V back.**  `tool/shader_ops.py <shader.comp> [--tile MxNxK]` freezes the
  specialization constants to a real device's tile and prints what the loop
  actually issues: the work ops, anything issued beside them that the op budget
  does not credit, how many distinct operand pairs the run uses, and whether the
  chain is dependent.  It exits non-zero on the two failure modes that have
  shipped wrong numbers from here — an uncounted op in the loop, and a run of
  identical work ops a compiler can strength-reduce.  Every backend has a dtype
  somebody cannot test on; this is the check that needs no hardware.
- Shaders compile to **SPIR-V 1.5** (`--target-env=vulkan1.2`), not 1.6.  At 1.6
  a specialized work-group size becomes `OpExecutionModeId LocalSizeId`, gated on
  maintenance4; at 1.5 the same GLSL becomes the classic `LocalSize` mode plus a
  `gl_WorkGroupSize` spec constant, which every driver has handled since Vulkan
  1.0.  Nothing here needs 1.6.
- Coopmat shaders take **M/N/K and the work-group width as specialization
  constants (ids 0–3) and the trip count as a push constant**.  The trip count
  is pushed on purpose: a compile-time bound lets the driver unroll the whole
  run, and a fully unrolled run is both what a shader compiler chokes on and what
  it can fold into a closed form.  `COOPMAT_MMA_PER_TRIP` in
  `include/common/common.h` must stay equal to `CM_MMA_PER_TRIP` in
  `shaders/coopmat_chain.glsl`.
- A shader that `#include`s `shaders/mad_chain.glsl` is compiled **twice**,
  the second time with `-DMAD_CHAIN_AFFINE`, and embedded as `<name>` and
  `<name>_alt`.  `CompileShaders.cmake` detects this from the source, so
  adopting the shared chain is the only step; then declare the `_alt` extern
  and its `VK_ALT_<name>` block in `vk_peak.h` and pass `VK_ALT_SHADER(<name>)`
  as the variant's last field.  `runComputeKernel` times both and emits the
  faster; `--verbose` prints both readings.  Why two shapes: the MAD chain
  block in `include/common/common.h`.
- If you change `vkPeak` interface → update `include/vulkan/vk_peak.h`.
- If you change `VulkanDevice` → update `vulkan_device.cpp` + `include/vulkan/vk_peak.h`.
- If you change `CompileShaders.cmake` → test that `glslc` is found or gracefully skipped.
