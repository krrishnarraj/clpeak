# clpeak — "compute latency peak"

Cross-API GPU benchmark tool. Measures compute, bandwidth, and latency
across OpenCL, Vulkan, CUDA, and Metal backends from a single binary.

## Architecture

```
Peak (src/common/peak.cpp, include/peak.h)   ← abstract base
├── clPeak    → src/opencl/                  ← OpenCL backend
├── vkPeak    → src/vulkan/                  ← Vulkan backend
├── CudaPeak  → src/cuda/                    ← CUDA backend
└── MetalPeak → src/metal/                   ← Metal backend
```

Shared code lives in `src/common/`. CLI entry point is `src/cli/main.cpp`.

## Directory Map

| Path | Purpose |
|------|---------|
| `include/` | All public headers — neutral headers flat, backend headers in `include/<backend>/` |
| `include/benchmark_enums.h` | `Benchmark`, `Category`, `DeviceType` enums — shared by all backends |
| `include/peak.h` | `Peak` abstract base class |
| `include/common.h` | OS macros, `Timer`, utility functions |
| `include/backend_gating.h` | `BackendGating` — test/category enable/disable |
| `src/common/` | `Peak` base, logger, gating, result store, calibration, inventory |
| `src/opencl/` | OpenCL backend: `clPeak` class + per-benchmark `.cpp` + `.cl` kernels |
| `src/vulkan/` | Vulkan backend: `vkPeak` class + SPIR-V shaders |
| `src/cuda/` | CUDA backend: `CudaPeak` class + `.cu` kernels (NVRTC-compiled at runtime) |
| `src/metal/` | Metal backend: `MetalPeak` class (ObjC++) + `.metal` kernels |
| `src/cli/` | Desktop CLI entry point (`main.cpp`) |
| `android/` | Android app with JNI native module |
| `cmake/` | Shared cmake: version handling (`version.cmake`, `GenVersion.cmake`) |

## Build

- Desktop: `cmake -B build && cmake --build build`
- Each backend: `-DCLPEAK_ENABLE_OPENCL=OFF`, etc.
- Android: open `android/` in Android Studio

## Quick Lookups

- **Adding a new benchmark?** → See the backend's `AGENTS.md` + `include/benchmark_enums.h`
- **Adding a new backend?** → See `src/common/AGENTS.md` for the `Peak` interface
- **Fixing a Metal test?** → See `src/metal/AGENTS.md`
- **Understanding result output?** → See `include/result_store.h` + `src/common/AGENTS.md`
- **Understanding CLI options?** → See `include/options.h`

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
