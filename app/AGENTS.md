# app — Flutter GUI (Android, iOS, macOS, Linux, Windows)

One Flutter app for every GUI platform, driving the native backends through
the `src/ffi` C ABI (Dart FFI — no JNI, no platform channels for the bridge).

## Building / running

- Desktop (canonical): `cmake -B build && cmake --build build --target clpeak-gui`
  → complete bundle at `build/clpeak-gui/` (macOS: `.app` with
  `clpeak_ffi.framework` embedded + re-signed; Linux: bundle with
  `lib/libclpeak_ffi.so`; Windows: flat dir with `clpeak_ffi.dll`).
  GUI is skipped when the Flutter SDK isn't detectable or
  `-DCLPEAK_ENABLE_GUI=OFF`.
- Desktop dev loop: build `clpeak_ffi` once, then
  `CLPEAK_FFI_PATH=<build>/clpeak_ffi.framework/clpeak_ffi flutter run -d macos`
  (a plain `flutter build macos` does NOT embed the framework — the
  clpeak-gui target owns final assembly).
- Android: `flutter build apk --release` (Gradle drives
  `src/ffi/android/CMakeLists.txt`; needs `git submodule update --init`).
- iOS: `tool/build_ios_native.sh` first (stages
  `ios/clpeak_native/clpeak_ffi.xcframework` + optional Vulkan pieces), then
  `flutter build ios` / `flutter run`.
- Tests: `flutter test` (pure Dart) or
  `CLPEAK_FFI_PATH=… flutter test` to include the native-bridge tests.

## Quick Lookups

- Native bindings / event decoding? → `lib/src/ffi/` (`clpeak_bindings.dart`,
  `clpeak_events.dart`; threading contract in `clpeak_runner.dart` —
  `NativeCallable.listener` + `Isolate.run`, `done` event = drain barrier)
- Argv construction (device/category/time flags)? → `lib/src/model/run_config.dart`
  (never emits per-test flags — the UI is data-driven so test churn in the
  core needs no app changes)
- Run grouping / formatting? → `lib/src/model/run_document.dart`
- History persistence? → `lib/src/services/run_history_store.dart`
  (`<documents>/clpeak/runs/<id>.xml` written natively via `--xml-file`,
  `index.json` sidecar; viewing goes XML → native loader → JSON)
- Run lifecycle state? → `lib/src/services/benchmark_service.dart`
- Screens? → `lib/src/ui/` (dashboard, run_config, live_run, results,
  history, about; adaptive shell in `app.dart`)

## Hand-edited generated files

`flutter create` regeneration can clobber these — re-apply if you recreate
the platform dirs:

- `macos/Runner/{DebugProfile,Release}.entitlements` — App Sandbox disabled
  (device probing, dlopen, real ~/Documents)
- `ios/Runner.xcodeproj/project.pbxproj` — bundle id `kr.clpeak.ios` +
  "Embed clpeak native frameworks" script phase (consumes
  `ios/clpeak_native/`, staged by `tool/build_ios_native.sh`)
- `android/app/build.gradle.kts` — `kr.clpeak`, minSdk 33, abiFilters,
  `externalNativeBuild` → `src/ffi/android/CMakeLists.txt`
- `android/app/src/main/AndroidManifest.xml` — `uses-native-library
  libOpenCL.so`

## When You Change This Directory

- If the event schema or C ABI changes → update `lib/src/ffi/` and
  `src/ffi/AGENTS.md`.
- If you add a CLI-flag mapping → keep `run_config.dart` in sync with
  `src/common/options.cpp`.
- versionCode continues the retired native app's sequence (pubspec
  `version: x.y.z+N`).
