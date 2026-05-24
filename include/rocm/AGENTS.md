# include/rocm — ROCm Backend Header

Public header for the ROCm/HIP backend: the `RocmPeak` class, device wrapper,
and embedded HIP kernel source externs.

## Quick Lookups

- Looking for the RocmPeak class declaration? → `rocm_peak.h`
- Looking for ROCm device info? → `rocm_device_info_t` in `rocm_peak.h`
- Looking for embedded HIP kernel sources? → `rocm_kernels` namespace in `rocm_peak.h`
- Looking for compute descriptor structs? → `rocm_compute_desc_t` in `rocm_peak.h`

## Key Files

| File | Purpose |
|------|---------|
| `rocm_peak.h` | `RocmPeak`, `RocmDevice`, `rocm_device_info_t`, `rocm_compute_desc_t`, kernel externs |

## When You Change This Directory

- If you add a new benchmark method → update `src/rocm/AGENTS.md`.
- If you add a new `.hip` kernel → add its extern to `rocm_kernels` namespace.
- If you change `rocm_device_info_t` → check `rocm_peak.cpp` / `rocm_device.cpp` init logic.
