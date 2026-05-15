# include/vulkan — Vulkan Backend Header

Public header for the Vulkan backend: the `vkPeak` class, device wrapper, and
embedded SPIR-V shader externs.

## Quick Lookups

- Looking for the vkPeak class declaration? → `vk_peak.h`
- Looking for Vulkan device info? → `vk_device_info_t` in `vk_peak.h`
- Looking for embedded SPIR-V symbols? → `vk_shaders` namespace in `vk_peak.h`
- Looking for compute descriptor structs? → `vk_compute_desc_t` in `vk_peak.h`

## Key Files

| File | Purpose |
|------|---------|
| `vk_peak.h` | `vkPeak`, `VulkanDevice`, `vk_device_info_t`, `vk_compute_desc_t`, SPIR-V externs |

## When You Change This Directory

- If you add a new benchmark method → update `src/vulkan/AGENTS.md`.
- If you add a new SPIR-V shader → add its extern to `vk_shaders` namespace.
- If you change `vk_device_info_t` → check `vk_peak.cpp` init logic.
