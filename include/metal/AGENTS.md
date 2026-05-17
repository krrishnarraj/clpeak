# include/metal — Metal Backend Header

Public header for the Metal backend: the `MetalPeak` class, device wrapper,
and embedded Metal kernel source externs.

## Quick Lookups

- Looking for the MetalPeak class declaration? → `mtl_peak.h`
- Looking for Metal device info? → `mtl_device_info_t` in `mtl_peak.h`
- Looking for embedded kernel sources? → `mtl_kernels` namespace in `mtl_peak.h`
- Looking for compute descriptor structs? → `mtl_compute_desc_t` in `mtl_peak.h`

## Key Files

| File | Purpose |
|------|---------|
| `mtl_peak.h` | `MetalPeak`, `MetalDevice`, `mtl_device_info_t`, `mtl_compute_desc_t`, kernel externs |

## When You Change This Directory

- If you add a new benchmark method → update `src/metal/AGENTS.md`.
- If you add a new .metal kernel → add its extern to `mtl_kernels` namespace.
- If you change `mtl_device_info_t` → check `mtl_peak.mm` init logic.
