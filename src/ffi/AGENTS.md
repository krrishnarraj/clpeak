# src/ffi — C-ABI Bridge for the Flutter GUI

The single native bridge every clpeak GUI platform (macOS/Linux/Windows
desktop, Android, iOS) consumes via Dart FFI. Builds the `clpeak_ffi`
shared library / Apple framework.

## Quick Lookups

- The C ABI? → `clpeak_ffi.h` (catalog JSON, blocking `clpeak_launch` with a
  streaming event callback, `clpeak_request_cancel`, the runtime setters:
  `clpeak_set_onnx_library` / `_ep_libraries` / `_winml`, `clpeak_set_litert_*`,
  each with a `clpeak_copy_*_status_json`). There is no saved-result
  loader: result files are JSON in the shape the GUI renders, so it reads them
  with `dart:convert` and history survives a native library that won't load
- Event JSON schema? → documented in `clpeak_ffi.h`; produced in
  `logger_ffi.cpp` (1:1 mirror of `LogEvent` in `include/common/logger.h`);
  decoded in `app/lib/src/ffi/clpeak_events.dart`
- Test documentation (`desc` / `minfo`)? → authored natively, the test's at its
  `beginTest()` and each reading's at its own `emit()`
  (`include/common/AGENTS.md`).  The test's arrives once on `test_begin`, with
  the rest of the resolved header (`shape`, `axis`, `direction`, `unit`);
  each reading's rides the reading
- Where the desktop app runs things? → not in its own process: every catalog
  and run is a `clpeak-engine` process (`engine_host.cpp`, a loader for
  `clpeak_engine_main` in `engine.cpp`; protocol in `clpeak_ffi.h`, Dart side
  `app/lib/src/ffi/clpeak_engine.dart`).  A vendor runtime beside the GUI
  toolkit and GL driver can break on what they load -- `engine.cpp` has the
  case.  Mobile calls the C ABI in-process
- Backend run loop? → `clpeak_ffi.cpp` (ports `src/cli/main.cpp`: both walk
  the shared `backendRegistry()` so the catalog, the run and the CLI agree on
  which backends exist and in what order; `RunDocument::append` merge,
  centralized `-o` save — which also stamps `cancelled` so a partial run does
  not read as a complete one). The `RunLog` is created before argv is parsed,
  so a rejected argument is a `log` event like any other diagnostic; with `-o`
  it streams the `<output>.log` sidecar the app adopts after a native crash
- Diagnostics (`log` events, `--verbose`)? → `clpeak_ffi.h` documents the
  event; the entry is the document's `LogEntry` verbatim. Vendor-relayed
  entries (ONNX Runtime's logger, a Vulkan messenger, a console capture) can
  arrive on another thread, which `NativeCallable.listener` is built for
- Desktop build + `clpeak-gui` target? → `CMakeLists.txt` (gated on
  `CLPEAK_ENABLE_GUI` + detected Flutter SDK; assembles the final bundle at
  `<build>/clpeak-gui/` so Flutter-generated runner projects stay untouched,
  `clpeak-engine` beside the runner -- signed before the app is sealed on
  macOS -- and at the build root beside `clpeak_ffi` for the dev loop)
- Release layout of the GUI? → the `install()` block at the end of
  `CMakeLists.txt`: bundle → `gui/`, generated launcher → `bin/clpeak-gui`;
  macOS ditto's `clpeak-gui.app` to the package root (keeps framework symlinks
  + ad-hoc signature). Windows staging dir is resolved at build time by
  `cmake/stage_windows_bundle.cmake` (arm64/x64 arch dir)
- Android build? → `android/CMakeLists.txt` (standalone superproject used by
  `app/android/app/build.gradle.kts` externalNativeBuild; OpenCL stub +
  Vulkan headers from `third_party/`)
- iOS build? → `ios/CMakeLists.txt` + `tools/build_ios_native.sh` (device +
  simulator frameworks → `app/ios/clpeak_native/clpeak_ffi.xcframework`;
  Vulkan/MoltenVK env-gated on the LunarG iOS SDK; ONNX Runtime linked in
  from the static pod; LiteRT's dylibs fetched and staged for the Runner to
  embed and dlopen)

## Key Files

| File | Purpose |
|------|---------|
| `clpeak_ffi.h` | `extern "C"` surface + event schema + `CLPEAK_RUN_*` codes |
| `clpeak_ffi.cpp` | launch loop, catalog, cancel, run-document assembly + save, the run's `RunLog` + sidecar, the `--verbose` inventory |
| `launch.h` | `clpeakLaunch()`: `clpeak_launch()` with a choice of whether to clear a pending cancel (the engine does not) |
| `engine.cpp` | `clpeak_engine_main`: the desktop engine process's catalog / launch modes, events one JSON line per stdout line, cancel on stdin |
| `engine_host.cpp` | `clpeak-engine` executable: dlopens the library path the app passes and calls `clpeak_engine_main`; beside the runner in every desktop bundle |
| `logger_ffi.{h,cpp}` | `LoggerFfi : logger` — `LogEvent` → malloc'd JSON → callback (ownership transfers to the callee) |
| `CMakeLists.txt` | `clpeak_ffi` SHARED target + `clpeak-gui` bundle-assembly target + GUI install/package rules |
| `cmake/stage_windows_bundle.cmake` | Build-time copy of Flutter's `build/windows/<arch>/runner/Release` into the staging dir |
| `android/CMakeLists.txt` | Android superproject (OpenCL stub + NDK Vulkan + CPU) |
| `ios/CMakeLists.txt` | iOS superproject (Metal + CPU + Core ML + ONNX Runtime + LiteRT + optional MoltenVK Vulkan) |

## Traps

- **Windows: `clpeak-engine` is a GUI-subsystem binary** (`WIN32_EXECUTABLE`,
  `wWinMain`).  A console one, started from the app, opens a console window
  for every catalog and run; the pipes the app hands over still work.

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
- The catalog JSON keys are snake_case (`compute_units`, `global_mem_bytes`)
  — the same serializer writes the run document's `inventory`, so the two
  cannot drift.
- argv follows the CLI grammar; parsing uses `parseCliOptionsNoExit` so a bad
  flag can never kill the host process.
- The engine process takes the runtime setup as `--set-*` options applied
  through the `clpeak_set_*` calls, never as run flags: the run's argv is
  recorded in the saved document, and a library path does not belong there.
- The engine's stdout carries events and nothing else -- it moves fd 1 onto
  stderr before loading anything -- and its events descriptor is kept from
  child processes, so the app sees EOF when the engine exits.
- The engine ends with `_exit` / `TerminateProcess` once the mode is done:
  results are saved by then, and runtime teardown is where ORT and LiteRT
  crash.  stdin closing is a cancel (the app has gone).

## When You Change This Directory

- If you change the C ABI or event schema → update `clpeak_ffi.h` docs,
  `app/lib/src/ffi/clpeak_bindings.dart` + `clpeak_events.dart`, and this file.
- If the run loop changes in `src/cli/main.cpp` → mirror it in
  `clpeak_ffi.cpp`.  A new backend needs nothing here: the registry and
  `src/common/cmake/backends.cmake` bring it into both binaries.
