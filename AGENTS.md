# clpeak — "compute latency peak"

Cross-API compute benchmark tool. Measures compute, bandwidth, and latency
across OpenCL, Vulkan, CUDA, ROCm/HIP, Metal, and oneAPI/SYCL GPU backends —
plus a native CPU backend — from a single binary.

## Architecture

```
Peak (src/common/peak.cpp, include/common/peak.h)   ← abstract base
├── CpuPeak    → src/cpu/                            ← native CPU backend (plain C++ / std::thread; runs first)
├── clPeak     → src/opencl/                         ← OpenCL backend
├── vkPeak     → src/vulkan/                         ← Vulkan backend
├── CudaPeak   → src/cuda/                           ← CUDA backend
├── RocmPeak   → src/rocm/                           ← ROCm/HIP backend
├── MetalPeak  → src/metal/                          ← Metal backend
└── OneapiPeak → src/oneapi/                         ← oneAPI/SYCL backend (Intel GPUs)
```

Shared code lives in `src/common/` and `include/common/`. Each backend has its
own `CMakeLists.txt` that builds a static library (`peak_opencl`, etc.).
The CLI entry point is `src/cli/main.cpp` with its own `logger.cpp`.

## Directory Map

| Path | Purpose |
|------|---------|
| `include/common/` | All neutral headers — `peak.h`, `benchmark_enums.h`, `logger.h` (base), etc. |
| `include/cli/` | CLI-specific headers — `logger_cli.h` |
| `include/opencl/` | OpenCL backend headers — `cl_peak.h`, `cl_common.h` |
| `include/vulkan/` | Vulkan backend header — `vk_peak.h` |
| `include/cuda/` | CUDA backend header — `cuda_peak.h` |
| `include/rocm/` | ROCm/HIP backend header — `rocm_peak.h` |
| `include/metal/` | Metal backend header — `mtl_peak.h` |
| `include/oneapi/` | oneAPI/SYCL backend header — `oneapi_peak.h` |
| `include/cpu/` | Native CPU backend header — `cpu_peak.h` |
| `src/common/` | `Peak` base, gating, result store, calibration, inventory (no logger) |
| `src/opencl/` | OpenCL backend: `clPeak` class + per-benchmark `.cpp` + `.cl` kernels |
| `src/vulkan/` | Vulkan backend: `vkPeak` class + SPIR-V shaders |
| `src/cuda/` | CUDA backend: `CudaPeak` class + `.cu` kernels (NVRTC-compiled at runtime) |
| `src/rocm/` | ROCm/HIP backend: `RocmPeak` class + `.hip` kernels (HIPRTC-compiled at runtime) |
| `src/metal/` | Metal backend: `MetalPeak` class (ObjC++) + `.metal` kernels |
| `src/oneapi/` | oneAPI/SYCL backend: `OneapiPeak` class + SYCL kernels (inline lambdas, AOT/JIT via DPC++) |
| `src/cpu/` | Native CPU backend: `CpuPeak` class + `std::thread` pool + per-ISA SIMD kernels (`-march`/`-mcpu=native`); cache/DRAM bandwidth + memory latency |
| `src/cli/` | Desktop CLI: `main.cpp`, `logger.cpp` (stdout output) |
| `src/common/cmake/` | Version handling (`version.cmake`, `GenVersion.cmake`, `version.h.in`) |
| `android/` | Android app (Vulkan, OpenCL, CPU) with JNI native module, its own `logger_android.cpp` |
| `ios/` | iOS SwiftUI app with Vulkan-over-MoltenVK, Metal, and CPU backends |

## Build

- Desktop: `cmake -B build && cmake --build build`
- Each backend: `-DCLPEAK_ENABLE_VULKAN=OFF`, etc.

## Quick Lookups

- **Adding a new benchmark?** → See the backend's `AGENTS.md` + `include/common/benchmark_enums.h`
- **Adding a new backend?** → See `src/common/AGENTS.md` for the `Peak` interface
- **Fixing a Metal test?** → See `src/metal/AGENTS.md`
- **Understanding result output?** → See `include/common/result_store.h` + `src/common/AGENTS.md`
- **Understanding CLI options?** → See `include/common/options.h`

## AGENTS.md System

This tree uses `AGENTS.md` files so AI agents can understand structure quickly.
The system is self-maintaining: the rules below are the canonical definition.

### When to Update AGENTS.md

**DO update when you:**
- Add, remove, rename, or move files/directories
- Change class hierarchies, interfaces, or module boundaries
- Add a new backend, benchmark category, or major feature
- Change build system structure (cmake files, dependencies)

**Do NOT update when you:**
- Fix bugs within existing functions
- Change tuning constants, thresholds, kernel loop counts
- Make cosmetic changes (formatting, comments, variable names)

### When Creating a New AGENTS.md

Use this template:
```markdown
# <Directory Name>
One-line purpose.

## Quick Lookups
- Looking for X? → see `<path>`

## Key Files
| File | Purpose |
|------|---------|

## When You Change This Directory
- If you ... → update `<path/to/other/AGENTS.md>`
```

### Hierarchy Rule
If a directory has subdirectories with their own AGENTS.md, the parent only
summarizes — details live in the child. No duplication across levels.
