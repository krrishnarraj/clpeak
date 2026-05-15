# src/vulkan — Vulkan Backend Implementation

`vkPeak` class implementation: instance/device init, per-benchmark runners,
and GLSL compute shaders (in `shaders/`).  Built as `peak_vulkan` static library.

## Quick Lookups

- Looking for the main class? → `vk_peak.cpp`
- Looking for instance/device init? → `vk_peak.cpp` (`initInstance`, `VulkanDevice::init`)
- Looking for the unified compute kernel runner? → `vk_peak.cpp` (`runComputeKernel`)
- Looking for GLSL shader sources? → `shaders/*.comp`
- Looking for shader compilation logic? → `cmake/CompileShaders.cmake`

## Key Files

| File | Purpose |
|------|---------|
| `vk_peak.cpp` | `vkPeak` class: constructor, `applyOptions()`, `runAll()`, all benchmarks |
| `shaders/` | GLSL compute shaders (`.comp`) compiled to SPIR-V at build time |
| `cmake/CompileShaders.cmake` | `compile_shaders()` — glslc → SPIR-V → embedded C++ arrays |

## When You Change This Directory

- If you add a new benchmark → update `CMakeLists.txt` (shader list) + this file.
- If you add a new `.comp` shader → add to `CLPEAK_VK_SHADERS` in `CMakeLists.txt`.
- If you change `vkPeak` interface → update `include/vulkan/vk_peak.h`.
- If you change `CompileShaders.cmake` → test that `glslc` is found or gracefully skipped.
