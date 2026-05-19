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
- Looking for atomic benchmarks? → `atomic_throughput.cpp`
- Looking for kernel latency? → `kernel_latency.cpp`

## Key Files

| File | Purpose |
|------|---------|
| `vk_peak.cpp` | `vkPeak` class: ctor, `applyOptions()`, `initInstance()`, `cleanup()`, `runKernel()`, `runAll()`, `enumerate()`, `printInventory()` |
| `vulkan_device.cpp` | `VulkanDevice` class: `init()` (4-step: basic info → CU count → optional features → logical device), `cleanup()`, `createBuffer()`, `createComputePipeline()`, `submitAndWait()`, `zeroBuffer()` |
| `compute_kernel.cpp` | `vkPeak::runComputeKernel()` — shared compute-peak driver: buffer/descriptor/pipeline scaffolding used by all `runCompute*` wrappers |
| `compute_float.cpp` | `runComputeSP`, `runComputeHP`, `runComputeDP`, `runComputeMP`, `runComputeBF16` |
| `compute_int.cpp` | `runComputeInt32`, `runComputeInt8DP`, `runComputeInt4Packed` |
| `coopmat.cpp` | `runCoopMatrix` — cooperative matrix (tensor-core) umbrella |
| `global_bandwidth.cpp` | `runGlobalBandwidth` |
| `local_bandwidth.cpp` | `runLocalBandwidth` |
| `image_bandwidth.cpp` | `runImageBandwidth` |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` |
| `atomic_throughput.cpp` | `runAtomicThroughput` + `runAtomicThroughputFp` |
| `kernel_latency.cpp` | `runKernelLatency` |
| `shaders/` | GLSL compute shaders (`.comp`) compiled to SPIR-V at build time |
| `cmake/CompileShaders.cmake` | `compile_shaders()` — glslc → SPIR-V → embedded C++ arrays |

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate category file (or create a new one) + update `CMakeLists.txt` + this file.
- If you add a new `.comp` shader → add to `CLPEAK_VK_SHADERS` in `CMakeLists.txt`.
- If you change `vkPeak` interface → update `include/vulkan/vk_peak.h`.
- If you change `VulkanDevice` → update `vulkan_device.cpp` + `include/vulkan/vk_peak.h`.
- If you change `CompileShaders.cmake` → test that `glslc` is found or gracefully skipped.
