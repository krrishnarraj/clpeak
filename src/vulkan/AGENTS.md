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
  them onto `vkWidthNote()`.
- Coopmat tests are single-reading and their metric name restates the title, so
  they document the test only (`nullptr` on the single-variant path).

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file (or create a new one) + update `CMakeLists.txt` + this file.
- If you add a new `.comp` shader → add to `CLPEAK_VK_SHADERS` in `CMakeLists.txt`
  and declare its extern in the `vk_shaders` namespace (`include/vulkan/vk_peak.h`).
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
