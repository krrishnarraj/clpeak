# src/ffi — C-ABI Bridge for the Flutter GUI

The single native bridge every clpeak GUI platform (macOS/Linux/Windows
desktop, Android, iOS) consumes via Dart FFI. Builds the `clpeak_ffi`
shared library / Apple framework.

## Quick Lookups

- The C ABI? → `clpeak_ffi.h` (catalog JSON, blocking `clpeak_launch` with a
  streaming event callback, `clpeak_request_cancel`). There is no saved-result
  loader: result files are JSON in the shape the GUI renders, so it reads them
  with `dart:convert` and history survives a native library that won't load
- Event JSON schema? → documented in `clpeak_ffi.h`; produced in
  `logger_ffi.cpp` (1:1 mirror of `LogEvent` in `include/common/logger.h`);
  decoded in `app/lib/src/ffi/clpeak_events.dart`
- Test documentation (`desc` / `minfo`)? → authored natively, the test's at its
  `beginTest()` and each reading's at its own `emit()`
  (`include/common/AGENTS.md`).  The test's arrives once on `test_begin`, with
  the rest of the resolved header (`shape`, `axis`, `direction`, `unit`,
  `scale`); each reading's rides the reading
- Backend run loop? → `clpeak_ffi.cpp` (ports `src/cli/main.cpp`: same order,
  `RunDocument::append` merge, centralized `-o` save — which also stamps
  `cancelled` so a partial run does not read as a complete one)
- Desktop build + `clpeak-gui` target? → `CMakeLists.txt` (gated on
  `CLPEAK_ENABLE_GUI` + detected Flutter SDK; assembles the final bundle at
  `<build>/clpeak-gui/` so Flutter-generated runner projects stay untouched)
- Release layout of the GUI? → the `install()` block at the end of
  `CMakeLists.txt`: bundle → `gui/`, generated launcher → `bin/clpeak-gui`;
  macOS ditto's `clpeak-gui.app` to the package root (keeps framework symlinks
  + ad-hoc signature). Windows staging dir is resolved at build time by
  `cmake/stage_windows_bundle.cmake` (arm64/x64 arch dir)
- Android build? → `android/CMakeLists.txt` (standalone superproject used by
  `app/android/app/build.gradle.kts` externalNativeBuild; OpenCL stub +
  Vulkan headers from `third_party/`)
- iOS build? → `ios/CMakeLists.txt` + `tool/build_ios_native.sh` (device +
  simulator frameworks → `app/ios/clpeak_native/clpeak_ffi.xcframework`;
  Vulkan/MoltenVK env-gated on the LunarG iOS SDK)

## Key Files

| File | Purpose |
|------|---------|
| `clpeak_ffi.h` | `extern "C"` surface + event schema + `CLPEAK_RUN_*` codes |
| `clpeak_ffi.cpp` | launch loop, catalog, cancel, run-document assembly + save |
| `logger_ffi.{h,cpp}` | `LoggerFfi : logger` — `LogEvent` → malloc'd JSON → callback (ownership transfers to the callee) |
| `CMakeLists.txt` | `clpeak_ffi` SHARED target + `clpeak-gui` bundle-assembly target + GUI install/package rules |
| `cmake/stage_windows_bundle.cmake` | Build-time copy of Flutter's `build/windows/<arch>/runner/Release` into the staging dir |
| `android/CMakeLists.txt` | Android superproject (OpenCL stub + NDK Vulkan + CPU) |
| `ios/CMakeLists.txt` | iOS superproject (Metal + CPU + optional MoltenVK Vulkan) |

## Traps

- **Windows: never sequence a command after `flutter` in one custom target.**
  `FLUTTER_EXECUTABLE` is `flutter.bat`, and the VS/Ninja generators pack every
  `COMMAND` of a target into a single batch script; cmd.exe transfers control
  permanently when a `.bat` calls a `.bat` without `call`, so later commands are
  skipped *with the build still green*. The flutter call therefore lives in its
  own `clpeak-gui-flutter` target.
- The backends are static libs that also link into `clpeak_ffi` (a `.so`), so
  everything they contain must be PIC — including the vendored OpenCL ICD
  loader (`src/opencl/cmake/BuildSdk.cmake` passes
  `-DCMAKE_POSITION_INDEPENDENT_CODE=ON` into that nested build).

## Contracts

- Event strings are malloc'd and OWNED BY THE CALLEE (Dart frees via
  `clpeak_free_string`) — required for `NativeCallable.listener`, which
  decodes after the native call returns.
- One launch at a time (`CLPEAK_RUN_BUSY`); the final `done` event is the
  consumer's drain barrier.
- argv follows the CLI grammar; parsing uses `parseCliOptionsNoExit` so a bad
  flag can never kill the host process.

## When You Change This Directory

- If you change the C ABI or event schema → update `clpeak_ffi.h` docs,
  `app/lib/src/ffi/clpeak_bindings.dart` + `clpeak_events.dart`, and this file.
- If backend wiring changes in `src/cli/main.cpp` → mirror it in
  `clpeak_ffi.cpp`.
