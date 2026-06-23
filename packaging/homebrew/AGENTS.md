# packaging/homebrew
Homebrew formula for clpeak (macOS + Linuxbrew), targeting Homebrew/homebrew-core.

## Key Files
| File | Purpose |
|------|---------|
| `clpeak.rb` | The formula. Builds via CMake; enables Metal/OpenCL/Vulkan(MoltenVK) on macOS and Vulkan/OpenCL/CPU on Linux. CUDA/ROCm/oneAPI are disabled (no vendor toolkits in Homebrew). |

## Notes
- Source is a **git URL pinned to `tag` + `revision`** (not a tarball) so
  `git describe` in `src/common/cmake/version.cmake` reports the real version.
- Vulkan shaders need `glslc` at build time → `shaderc` build dependency.
- Build uses **clang** for the CPU-backend codegen (GCC<=14 halves fp32/fp64).
  macOS already uses AppleClang; Linux depends on `llvm` and forces clang via
  `-DCMAKE_C[XX]_COMPILER` (Homebrew sets CC/CXX, which would otherwise suppress
  clpeak's own clang auto-detection in CMakeLists.txt).
- `bin.install "build/clpeak"` is used instead of the in-tree install rule,
  which keeps a flat layout for the release zips.

## When You Change This Directory
- On each release → bump `tag` + `revision` in `clpeak.rb` (the homebrew-core
  autobump bot does this once the formula is accepted). Keep in sync with the
  Flatpak manifest pin in `../flatpak/`.
