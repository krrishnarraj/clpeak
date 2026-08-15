# app — Flutter GUI (Android, iOS, macOS, Linux, Windows)

One Flutter app for every GUI platform, driving the native backends through
the `src/ffi` C ABI (Dart FFI — no JNI, no platform channels for the bridge).

## Building / running

- Desktop (canonical): `cmake -B build && cmake --build build --target clpeak-gui`
  → complete bundle at `build/clpeak-gui/` (macOS: `clpeak-gui.app` with
  `clpeak_ffi.framework` embedded + re-signed; Linux: bundle with
  `lib/libclpeak_ffi.so`; Windows: flat dir with `clpeak_ffi.dll`).
  GUI is skipped when the Flutter SDK isn't detectable or
  `-DCLPEAK_ENABLE_GUI=OFF`; `-DCLPEAK_REQUIRE_GUI=ON` makes a missing SDK a
  configure error (release CI).
- The runner executable/bundle is named **clpeak-gui**, never `clpeak` — the
  release zip puts it next to the CLI binary of that name. macOS keeps the
  user-visible name "clpeak" via `CFBundleName`/`CFBundleDisplayName`.
- macOS disk image: `cmake --build build --target clpeak-gui-dmg`
  (`tool/make_dmg.sh`; ad-hoc signed, so a downloaded copy is quarantined).
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
- "What does this test measure?" → an info glyph beside the name, at both
  levels (test title and each reading's label in the expanded breakdown), one
  explanation per dialog: `_InfoGlyph` → `_showInfoDialog()` in
  `lib/src/ui/results/results_body.dart`.  Text comes off the result rows
  (`ResultEntry.description` / `.metricDescription`), **not** off the
  `test_begin` event — that would create a `TestResult` before its first
  measurement, which `CategoryGroup` reads as fully-skipped and files under
  "not supported".
- History persistence? → `lib/src/services/run_history_store.dart`
  (`<base>/runs/<id>.xml` written natively via `--xml-file`, `index.json`
  sidecar; viewing goes XML → native loader → JSON).  `<base>` is
  `$HOME/.clpeak` on desktop — never `~/Documents`, which costs a macOS TCC
  consent prompt — and `<app documents>/clpeak` on Android/iOS, where that
  directory is inside the sandbox and is what the Files app shows.  Device
  properties reach a reopened run through the file's `devices` block, not the
  event stream — see `RunDocument.fromEntriesJson`.
- Run lifecycle state? → `lib/src/services/benchmark_service.dart`
- Screens? → `lib/src/ui/` (dashboard, run_config, live_run, results,
  history, about; adaptive shell in `app.dart`)
- Colours / type / geometry? → `lib/src/theme/clpeak_theme.dart` (`CP.of(context)`
  tokens + category tints; the `ThemeData` there is glue only)
- Buttons, panels, chips, switches, table rows? → `lib/src/ui/common/kit.dart`

## Design language

The GUI is an *instrument console*, not a stock Material app: monochrome
chrome with the category tints as the only colour, monospace for anything
technical, hairline tables instead of cards, square-ish corners, zero
elevation, and inverted (solid block) primary actions.  `ColorScheme.fromSeed`
is deliberately not used — both palettes are fixed in `CP`.

Build screens from `ui/common/kit.dart` (`CPanel`, `CSection`, `CRow`,
`CButton`, `CChip`, `CSwitch`, `CCheckbox`, `CTag`, `CValue`, `CMeter`,
`CHeader`, `CDialog`, …), not from `Card` / `Chip` / `Switch` / `AppBar` /
`NavigationRail` / `ListTile`.  The kit is built on raw `GestureDetector` +
`MouseRegion`, so nothing splashes — which is also why the theme sets
`NoSplash.splashFactory` (see the animation trap below).

## Traps

- **The live-run screen must not animate, and must not rebuild per event.**
  The GUI process holds a graphics context on the same GPU it benchmarks
  (`C+G` in nvidia-smi), so every presented frame is GPU work competing with
  the running kernel — an indeterminate progress indicator pins the app at
  60 fps for the whole run and costs **10-15% of the GPU score**. Hence:
  static indicators only, `BenchmarkService` coalesces events onto a slow
  tick, the elapsed clock is its own 1 Hz leaf widget, and `ResultsBody`
  builds rows lazily. Cutting frame rate, not CPU work, is what recovers the
  score. Rationale is at each site (`live_run_screen.dart`,
  `benchmark_service.dart`).

## Hand-edited generated files

`flutter create` regeneration can clobber these — re-apply if you recreate
the platform dirs:

- `linux/CMakeLists.txt`, `windows/CMakeLists.txt` — `BINARY_NAME clpeak-gui`
  (+ `windows/runner/Runner.rc` InternalName/OriginalFilename);
  `macos/Runner/Configs/AppInfo.xcconfig` — `PRODUCT_NAME = clpeak-gui`;
  `macos/Runner/Info.plist` — `CFBundleName`/`CFBundleDisplayName` pinned to the
  literal "clpeak" (not `$(PRODUCT_NAME)`), so the app is `clpeak-gui.app` on
  disk but still reads "clpeak" in the menu bar and Dock
- `macos/Runner/{DebugProfile,Release}.entitlements` — App Sandbox disabled
  (device probing, dlopen, real ~/.clpeak)
- `macos/Runner/MainFlutterWindow.swift`, `linux/runner/my_application.cc`,
  `windows/runner/main.cpp` — 1280x860 default window size (macOS also sets a
  900x640 content minimum and centers)
- `linux/runner/my_application.cc` + `linux/CMakeLists.txt` — window icon:
  GTK has no `.rc`-style resource embedding, so the runner loads
  `data/clpeak_icon.png` from the (relocatable) bundle at startup and the
  runner CMake installs it there. X11 only — Wayland ignores window icons and
  matches an installed `.desktop` file by application ID instead.
- `ios/Runner.xcodeproj/project.pbxproj` — bundle id `kr.clpeak.ios` +
  "Embed clpeak native frameworks" script phase (consumes
  `ios/clpeak_native/`, staged by `tool/build_ios_native.sh`)
- `android/app/build.gradle.kts` — `kr.clpeak`, minSdk 33, abiFilters,
  `externalNativeBuild` → `src/ffi/android/CMakeLists.txt`
- `android/app/src/main/AndroidManifest.xml` — `uses-native-library
  libOpenCL.so`
- every platform's app icon — all generated by
  `tool/icons/generate_icons.py` (needs Pillow) from the original clpeak
  wordmark in `tool/icons/clpeak_master_1024.png` (both repo-root `tool/`,
  alongside `build_ios_native.sh`). Never hand-edit an icon
  PNG; change the script and re-run it. It also writes the iOS
  `AppIcon.appiconset/Contents.json` (single 1024 universal + dark/tinted)
  and Android's `mipmap-anydpi-v26/ic_launcher.xml` +
  `values/ic_launcher_background.xml`.

## When You Change This Directory

- If the event schema or C ABI changes → update `lib/src/ffi/` and
  `src/ffi/AGENTS.md`.
- If you add a CLI-flag mapping → keep `run_config.dart` in sync with
  `src/common/options.cpp`.
- versionCode continues the retired native app's sequence (pubspec
  `version: x.y.z+N`).
