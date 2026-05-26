# include/oneapi — oneAPI/SYCL Backend Header

Public header for the oneAPI backend: the `OneapiPeak` class and the
`OneapiDevice` wrapper. SYCL kernels live alongside their host code in
`src/oneapi/*.cpp` as C++ lambdas, so this header has no kernel-source
externs (unlike the ROCm/CUDA equivalents).

## Quick Lookups

- Looking for the OneapiPeak class declaration? → `oneapi_peak.h`
- Looking for oneAPI device info? → `oneapi_device_info_t` in `oneapi_peak.h`
- Looking for the runKernel submitter callback type? → `OneapiPeak::KernelSubmitter` in `oneapi_peak.h`

## Key Files

| File | Purpose |
|------|---------|
| `oneapi_peak.h` | `OneapiPeak`, `OneapiDevice`, `oneapi_device_info_t` |

## When You Change This Directory

- If you add a new benchmark method → update `src/oneapi/AGENTS.md`.
- If you change `oneapi_device_info_t` → check `src/oneapi/oneapi_device.cpp` init logic.
- If you add a new SYCL feature gate (e.g. `CLPEAK_ONEAPI_HAS_X`) → wire it in
  `src/oneapi/CMakeLists.txt` and document it here.
