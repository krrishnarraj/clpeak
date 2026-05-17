# include/opencl — OpenCL Backend Headers

Public headers for the OpenCL backend: the `clPeak` class and OpenCL utility types.

## Quick Lookups

- Looking for the clPeak class declaration? → `cl_peak.h`
- Looking for OpenCL device info struct? → `cl_common.h`

## Key Files

| File | Purpose |
|------|---------|
| `cl_peak.h` | `clPeak` class — extends `Peak`, declares all per-benchmark methods + `enumerate()` |
| `cl_common.h` | `device_info_t`, `device_extra_info_t`, kernel source externs |

## When You Change This Directory

- If you add a new benchmark method to `clPeak` → update `src/opencl/AGENTS.md`.
- If you change `device_info_t` → check all OpenCL benchmark `.cpp` files.
