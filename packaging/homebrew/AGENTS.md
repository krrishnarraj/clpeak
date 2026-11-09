# packaging/homebrew
Homebrew formula for XpuPeak (macOS + Linuxbrew), written for
Homebrew/homebrew-core and never published yet.

## Key Files
| File | Purpose |
|------|---------|
| `xpupeak.rb` | The formula, `class Xpupeak` (Homebrew derives the class from the file name). Builds the CLI via CMake; enables Metal/OpenCL/Vulkan(MoltenVK) on macOS and Vulkan/OpenCL/CPU on Linux. CUDA/ROCm/oneAPI and the desktop app are disabled (no vendor toolkits or Flutter SDK in Homebrew). Installs `xpupeak` and the `clpeak` link; the test runs both. |

## Local Build

```console
brew tap-new --no-git local/xpupeak
cp packaging/homebrew/xpupeak.rb "$(brew --repository local/xpupeak)/Formula/"
brew install --build-from-source --verbose -y local/xpupeak/xpupeak
brew test local/xpupeak/xpupeak
brew audit --strict --online local/xpupeak/xpupeak
brew uninstall xpupeak
brew untap local/xpupeak
```

Until the repo is renamed and `4.0.0` is tagged, the formula's URLs don't
resolve: point `url` at a local clone with a `4.0.0` tag to build it.

## Notes
- Source is a **git URL pinned to `tag` + `revision`** (not a tarball) so
  `git describe` in `src/common/cmake/version.cmake` reports the real version.
- Vulkan shaders need `glslc` at build time → `shaderc` build dependency.
- Build uses **clang** for the CPU-backend codegen (GCC<=14 halves fp32/fp64).
  macOS already uses AppleClang; Linux depends on `llvm` and forces clang via
  `-DCMAKE_C[XX]_COMPILER` (Homebrew sets CC/CXX, which would otherwise suppress
  the build's own clang auto-detection in CMakeLists.txt).
- `cmake --install` lays out the keg: `bin/xpupeak`, `bin/clpeak` (a link the
  root install rules make) and `share/xpupeak/LICENSE`.

## When You Change This Directory
- On each release → bump `tag` + `revision` in `xpupeak.rb` (the homebrew-core
  autobump bot does this once the formula is accepted). Keep in sync with the
  Flatpak manifest pin in `../flatpak/`.
