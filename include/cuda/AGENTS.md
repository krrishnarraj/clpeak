# include/cuda — CUDA Backend Header

Public header for the CUDA backend: the `CudaPeak` class, device wrapper,
and embedded CUDA kernel source externs.

## Quick Lookups

- Looking for the CudaPeak class declaration? → `cuda_peak.h`
- Looking for CUDA device info? → `cuda_device_info_t` in `cuda_peak.h`
- Looking for embedded kernel sources? → `cuda_kernels` namespace in `cuda_peak.h`
- Looking for compute descriptor structs? → `cuda_compute_desc_t` in `cuda_peak.h`

## Key Files

| File | Purpose |
|------|---------|
| `cuda_peak.h` | `CudaPeak`, `CudaDevice`, `cuda_device_info_t`, `cuda_compute_desc_t`, kernel externs |

## When You Change This Directory

- If you add a new benchmark method → update `src/cuda/AGENTS.md`.
- If you add a new .cu kernel → add its extern to `cuda_kernels` namespace.
- If you change `cuda_device_info_t` → check `cuda_peak.cpp` init logic.
