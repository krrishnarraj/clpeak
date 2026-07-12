# src/ffi — C-ABI Bridge for the Flutter GUI

The single native bridge every clpeak GUI platform (macOS/Linux/Windows
desktop, Android, iOS) consumes via Dart FFI. Builds the `clpeak_ffi`
shared library / Apple framework.

## Quick Lookups

- The C ABI? → `clpeak_ffi.h` (catalog JSON, blocking `clpeak_launch` with a
  streaming event callback, `clpeak_request_cancel`, saved-result loader)
- Event JSON schema? → documented in `clpeak_ffi.h`; produced in
  `logger_ffi.cpp` (1:1 mirror of `LogEvent` in `include/common/logger.h`);
  decoded in `app/lib/src/ffi/clpeak_events.dart`
- Backend run loop? → `clpeak_ffi.cpp` (ports `src/cli/main.cpp`: same order,
  result merge, centralized `--xml-file` save)
- Desktop build + `clpeak-gui` target? → `CMakeLists.txt` (gated on
  `CLPEAK_ENABLE_GUI` + detected Flutter SDK; assembles the final bundle at
  `<build>/clpeak-gui/` so Flutter-generated runner projects stay untouched)
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
| `clpeak_ffi.cpp` | launch loop, catalog, cancel, `clpeak_load_result_file_json` |
| `logger_ffi.{h,cpp}` | `LoggerFfi : logger` — `LogEvent` → malloc'd JSON → callback (ownership transfers to the callee) |
| `CMakeLists.txt` | `clpeak_ffi` SHARED target + `clpeak-gui` bundle-assembly target |
| `android/CMakeLists.txt` | Android superproject (OpenCL stub + NDK Vulkan + CPU) |
| `ios/CMakeLists.txt` | iOS superproject (Metal + CPU + optional MoltenVK Vulkan) |

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
