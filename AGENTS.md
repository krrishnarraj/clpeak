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
The CLI entry point is `src/cli/main.cpp`. The Flutter GUI (`app/`) drives the
same backends through the `clpeak_ffi` C-ABI bridge (`src/ffi/`).

## Directory Map

| Path | Purpose |
|------|---------|
| `include/common/` | All neutral headers — `peak.h`, `benchmark_enums.h`, `logger.h` (base), `logger_text.h` (shared text logger), etc. |
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
| `src/cuda/` | CUDA backend: `CudaPeak` class + `.cu` kernels (AOT-compiled to fatbins at build time, embedded in the binary) |
| `src/rocm/` | ROCm/HIP backend: `RocmPeak` class + `.hip` kernels (AOT-compiled with hipcc --genco at build time, embedded in the binary) |
| `src/metal/` | Metal backend: `MetalPeak` class (ObjC++) + `.metal` kernels |
| `src/oneapi/` | oneAPI/SYCL backend: `OneapiPeak` class + SYCL kernels (inline lambdas, AOT/JIT via DPC++) |
| `src/cpu/` | Native CPU backend: `CpuPeak` class + `std::thread` pool + per-ISA SIMD kernels (one feature TU per ISA, runtime-dispatched); cache/DRAM bandwidth + memory latency |
| `src/cli/` | Desktop CLI: `main.cpp` |
| `src/ffi/` | `clpeak_ffi` C-ABI bridge for the GUI (event-stream logger, launch/cancel, catalog); `clpeak-gui` CMake target; Android/iOS build superprojects |
| `app/` | Flutter GUI — one codebase for Android, iOS, macOS, Linux, Windows (Dart FFI over `src/ffi`) |
| `third_party/` | Vendored submodules: `libopencl-stub`, `Vulkan-Headers` (Android build) |
| `tool/` | Helper scripts (`build_ios_native.sh` — stages the iOS xcframework; `make_dmg.sh` — macOS GUI disk image) |
| `src/common/cmake/` | Version handling (`version.cmake`, `version.h.in`) — git-describe once at configure time |
| `results/` | Saved reference runs (`--xml-file` output) per vendor — the baselines a suspicious number gets checked against |
| `snap/` | Snap packaging (`snapcraft.yaml`, classic confinement) |
| `packaging/flatpak/` | Flathub packaging — manifest + AppStream MetaInfo (Vulkan+OpenCL+CPU only) |
| `packaging/homebrew/` | Homebrew formula (`clpeak.rb`) for macOS + Linuxbrew, targeting homebrew-core |
| `docs/` | GitHub Pages site (Jekyll, built natively by Pages from this folder — no plugins). Also holds the app screenshots the README links to, in `docs/assets/img/` |

## Build

- Desktop: `cmake -B build && cmake --build build`
- Each backend: `-DCLPEAK_ENABLE_VULKAN=OFF`, etc.
- GUI: built automatically as `clpeak-gui` when the Flutter SDK is detected
  (disable with `-DCLPEAK_ENABLE_GUI=OFF`); bundle lands in `build/clpeak-gui/`.
  Flutter's desktop SDK is x64-only on Linux/Windows, so those arm64 CI jobs
  build CLI-only (`gui: false` in the workflow matrix). Mobile builds: see
  `app/AGENTS.md`.
- All backend static libs are built PIC (`CMAKE_POSITION_INDEPENDENT_CODE`):
  they link into both `clpeak` and the `clpeak_ffi` shared library.
- Packaging: `cpack -G ZIP` ships CLI + GUI in one archive — `bin/clpeak`,
  `bin/clpeak-gui` (wrapper) and the Flutter bundle under `gui/`; macOS puts
  `clpeak-gui.app` at the archive root instead. macOS also has
  `--target clpeak-gui-dmg` (`tool/make_dmg.sh`) for the drag-to-Applications
  disk image shipped next to the zip.

## Quick Lookups

- **Adding a new benchmark?** → the backend's `AGENTS.md` + `include/common/benchmark_enums.h`
- **Adding a new backend?** → `src/common/AGENTS.md` for the `Peak` interface
- **Explaining what a test measures?** → `include/common/AGENTS.md` § Test documentation
- **Result output format?** → `include/common/result_store.h` + `src/common/AGENTS.md`
- **CLI options?** → `include/common/options.h`
- **Is this number plausible?** → the saved runs in `results/<vendor>/`

## AGENTS.md System

These files are a **map**, not a knowledge base. Three rules keep them useful:

- **No duplication across levels.** A parent summarizes; details live in the
  child's `AGENTS.md`. A directory whose whole content is already stated by its
  parent doesn't need a file at all.
- **No duplication with the code.** Why a kernel is shaped a certain way, what a
  compiler does to it, and what was measured belong in a comment next to that
  code, where it is read at the moment it matters. `AGENTS.md` states the rules
  that span files, and points at the code for the rest.
- **Current facts only.** Not investigation history, not before/after tuning
  deltas, not corrections of earlier notes — git holds those. If a note is only
  true of a past version of the code, delete it.

Update one when you add/remove/move files, change an interface or module
boundary, add a backend or benchmark category, or change build structure. Don't
update for bug fixes, tuning constants, or cosmetic changes.
