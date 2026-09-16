# clpeak — "compute latency peak"

Cross-API compute benchmark tool. Measures compute, bandwidth, and latency
across OpenCL, Vulkan, CUDA, ROCm/HIP, Metal, and oneAPI/SYCL GPU backends —
plus a native CPU backend, an ONNX Runtime backend that reaches NPUs, a
Core ML backend for Apple's Neural Engine, and a LiteRT backend for Android's
NPUs (Qualcomm, MediaTek, Google Tensor, Samsung) and its GPU/CPU runtime —
from a single binary.

## Architecture

```
Peak (src/common/peak.cpp, include/common/peak.h)   ← abstract base
├── CpuPeak    → src/cpu/                            ← native CPU backend (plain C++ / std::thread; runs first)
├── clPeak     → src/opencl/                         ← OpenCL backend
├── vkPeak     → src/vulkan/                         ← Vulkan backend
├── CudaPeak   → src/cuda/                           ← CUDA backend
├── RocmPeak   → src/rocm/                           ← ROCm/HIP backend
├── MetalPeak  → src/metal/                          ← Metal backend
├── OneapiPeak → src/oneapi/                         ← oneAPI/SYCL backend (Intel GPUs)
├── OnnxPeak   → src/onnx/                           ← ONNX Runtime backend (NPUs via execution providers)
├── CoreMLPeak → src/coreml/                         ← Core ML backend (Apple Neural Engine / GPU / CPU; Apple only)
└── LitertPeak → src/litert/                         ← LiteRT backend (NPU / GPU / CPU accelerators; Android's native AI runtime)
```

Shared code lives in `src/common/` and `include/common/`. Each backend has its
own `CMakeLists.txt` that builds a static library (`peak_opencl`, etc.).
The CLI entry point is `src/cli/main.cpp`. The Flutter GUI (`app/`) drives the
same backends through the `clpeak_ffi` C-ABI bridge (`src/ffi/`). Both iterate
the one backend registry (`src/registry/`), which is sorted by the `Backend`
enum -- the single order that `--help`, `--list-devices`, the GUI catalog, a
run and the result document all share.

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
| `include/onnx/` | ONNX Runtime backend header — `onnx_peak.h` |
| `include/coreml/` | Core ML backend header — `coreml_peak.h` |
| `include/litert/` | LiteRT backend header — `litert_peak.h` |
| `src/common/` | `Peak` base, gating, result store, calibration, inventory (no logger) |
| `src/opencl/` | OpenCL backend: `clPeak` class + per-benchmark `.cpp` + `.cl` kernels |
| `src/vulkan/` | Vulkan backend: `vkPeak` class + SPIR-V shaders |
| `src/cuda/` | CUDA backend: `CudaPeak` class + `.cu` kernels (AOT-compiled to fatbins at build time, embedded in the binary) |
| `src/rocm/` | ROCm/HIP backend: `RocmPeak` class + `.hip` kernels (AOT-compiled with hipcc --genco at build time, embedded in the binary) |
| `src/metal/` | Metal backend: `MetalPeak` class (ObjC++) + `.metal` kernels |
| `src/oneapi/` | oneAPI/SYCL backend: `OneapiPeak` class + SYCL kernels (inline lambdas, AOT/JIT via DPC++) |
| `src/cpu/` | Native CPU backend: `CpuPeak` class + `std::thread` pool + per-ISA SIMD kernels (one feature TU per ISA, runtime-dispatched); cache/DRAM bandwidth + memory latency |
| `src/onnx/` | ONNX Runtime backend: `OnnxPeak` class + per-benchmark `.cpp`. Each execution provider (QNN / OpenVINO / VitisAI / CoreML / NNAPI / GPU / CPU) is one device; the runtime is dlopen'd and models are emitted as protobuf bytes in memory |
| `src/coreml/` | Core ML backend (ObjC++ session + plain C++ tests): `CoreMLPeak` class + per-benchmark `.cpp`. Each Core ML compute device (Neural Engine / GPU / CPU) is one device; ML Program models are emitted as protobuf bytes + a weight blob and compiled at run time, and the compute plan proves per operation where they ran |
| `src/litert/` | LiteRT backend: `LitertPeak` class + per-benchmark `.cpp`. Each accelerator (NPU via a vendor dispatch library / GPU / CPU) is one device; `.tflite` models are emitted as FlatBuffer bytes in memory, the runtime is dlopen'd, and `IsFullyAccelerated` plus the profiler prove what ran where |
| `src/registry/` | `backend_registry.cpp` — the one list of backends in this build (`backendRegistry()`, `include/common/backend_registry.h`), sorted by the `Backend` enum; compiled into both `clpeak` and `clpeak_ffi` by `src/common/cmake/backends.cmake`, which also links the backend libraries and sets `ENABLE_*` for both |
| `src/cli/` | Desktop CLI: `main.cpp` |
| `src/ffi/` | `clpeak_ffi` C-ABI bridge for the GUI (event-stream logger, launch/cancel, catalog); `clpeak-gui` CMake target; Android/iOS build superprojects |
| `app/` | Flutter GUI — one codebase for Android, iOS, macOS, Linux, Windows (Dart FFI over `src/ffi`) |
| `third_party/` | Vendored submodules: `libopencl-stub`, `Vulkan-Headers` (Android build); vendored headers: `onnxruntime/` and `litert/` (C APIs — no library needed to build) |
| `tool/` | Helper scripts (`build_ios_native.sh` — stages the iOS xcframework; `make_dmg.sh` — macOS GUI disk image; `update_onnx_headers.sh` / `update_litert_headers.sh` — refresh the vendored runtime headers; `fetch_litert_npu.sh` — stage LiteRT's NPU dispatch shims for the Android app) |
| `src/common/cmake/` | Version handling (`version.cmake`, `version.h.in`) — git-describe once at configure time |
| `results/` | Saved reference runs (`-o` output, `.clpeak.json`) per vendor — the baselines a suspicious number gets checked against |
| `snap/` | Snap packaging (`snapcraft.yaml`, classic confinement) |
| `packaging/flatpak/` | Flathub packaging — manifest + AppStream MetaInfo (Vulkan+OpenCL+CPU only) |
| `packaging/homebrew/` | Homebrew formula (`clpeak.rb`) for macOS + Linuxbrew, targeting homebrew-core |
| `docs/` | GitHub Pages site (Jekyll, built natively by Pages from this folder — no plugins). `format-v3.md` is the published result-format schema. Also holds the app screenshots the README links to, in `docs/assets/img/` |

## Build

- Desktop: `cmake -B build && cmake --build build`
- Each backend: `-DCLPEAK_ENABLE_VULKAN=OFF`, etc.
- GUI: built automatically as `clpeak-gui` when the Flutter SDK is detected
  (disable with `-DCLPEAK_ENABLE_GUI=OFF`); bundle lands in `build/clpeak-gui/`.
  Mobile builds: see `app/AGENTS.md`.
- All backend static libs are built PIC (`CMAKE_POSITION_INDEPENDENT_CODE`):
  they link into both `clpeak` and the `clpeak_ffi` shared library.
- Packaging: `cpack -G ZIP` ships CLI + GUI in one archive — `bin/clpeak`,
  `bin/clpeak-gui` (wrapper) and the Flutter bundle under `gui/`; macOS puts
  `clpeak-gui.app` at the archive root instead. macOS also has
  `--target clpeak-gui-dmg` (`tool/make_dmg.sh`) for the drag-to-Applications
  disk image shipped next to the zip.

## Quick Lookups

- **Adding a new benchmark?** → the backend's `AGENTS.md` + `include/common/benchmark_enums.h` (a `Benchmark` names what is measured, never which backend; reuse one when the measurement exists elsewhere)
- **Adding a new backend?** → `src/common/AGENTS.md` for the `Peak` interface; then a `Backend` enum value (`include/common/benchmark_enums.h`), a row in the backend table (`src/common/options.cpp`), an `entry<YourPeak>()` in `src/registry/backend_registry.cpp`, and its name in the `foreach` of `src/common/cmake/backends.cmake` -- the CLI, the GUI bridge, the help, the listing and `--device` need nothing else
- **Classifying or explaining a test?** → `include/common/AGENTS.md`
  § What a backend authors at `beginTest()` (shape, axis, variant, prose)
- **Result output format?** → `docs/format-v3.md` (the schema) + `include/common/run_document.h`
- **Emitting a diagnostic, or reading one back?** → `CLPEAK_LOG` / `CLPEAK_VLOG` in `include/common/common.h`; every line lands on the document's `log` via `RunLog` (`include/common/run_log.h`) — `--verbose` is what makes a dump debuggable without the machine
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
