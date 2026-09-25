# app — Flutter GUI (Android, iOS, macOS, Linux, Windows)

One Flutter app for every GUI platform, driving the native backends through
the `src/ffi` C ABI (Dart FFI — no JNI, no platform channels for the bridge).

## Building / running

- Desktop (canonical): `cmake -B build && cmake --build build --target clpeak-gui`
  → complete bundle at `build/clpeak-gui/` (macOS: `clpeak-gui.app` with
  `clpeak_ffi.framework` embedded + re-signed; Linux: bundle with
  `lib/libclpeak_ffi.so`; Windows: flat dir with `clpeak_ffi.dll`).
  GUI is skipped when the Flutter SDK isn't detectable or `-DCLPEAK_ENABLE_GUI=OFF`.
- The runner executable/bundle is named **clpeak-gui**, never `clpeak` — the
  release zip puts it next to the CLI binary of that name. macOS keeps the
  user-visible name "clpeak" via `CFBundleName`/`CFBundleDisplayName`.
- macOS disk image: `cmake --build build --target clpeak-gui-dmg`
  (`tools/make_dmg.sh`; ad-hoc signed, so a downloaded copy is quarantined).
- Desktop dev loop: build `clpeak_ffi` once, then
  `CLPEAK_FFI_PATH=<build>/clpeak_ffi.framework/clpeak_ffi flutter run -d macos`
  (a plain `flutter build macos` does NOT embed the framework — the
  clpeak-gui target owns final assembly).
- Android: `flutter build apk --release` / `flutter build appbundle --release`
  (Gradle drives `src/ffi/android/CMakeLists.txt`; needs
  `git submodule update --init`). The bundle includes ONNX Runtime and
  LiteRT (`libLiteRt.so` + its OpenCL GPU accelerator, 8.6 MB) for
  **arm64-v8a** (devices) and **x86_64** (emulator / Chromebooks) —
  `armeabi-v7a`/`x86` are excluded as legacy 32-bit ABIs; see the packaging
  block in `android/app/build.gradle.kts`. With AAB Play serves a split APK
  per ABI, so per-device size stays bounded (fat APK would be 86 MB vs
  107 MB for every slice).  LiteRT's NPU dispatch shims are not on Maven:
  `tools/fetch_litert_npu.sh qualcomm|google_tensor` stages one vendor's under
  `android/app/src/main/jniLibs/` (git-ignored) before a build that should
  reach that NPU -- one vendor, because LiteRT loads the first shim it lists.
  Staging any switches the build to extracting native libraries at install
  (LiteRT finds the shim by listing a directory, which an APK's internal
  `lib/` is not).  Qualcomm's own runtime is a further 67 MB opt-in,
  `clpeakQnn=true` in `android/gradle.properties` (Maven Central's
  `com.qualcomm.qti:qnn-runtime`, every Hexagon generation); MediaTek's and
  Google Tensor's are system libraries on the device.  See
  `src/litert/AGENTS.md`, Packaging.
- iOS: `tools/build_ios_native.sh` first (stages
  `ios/clpeak_native/clpeak_ffi.xcframework` + optional Vulkan pieces), then
  `flutter build ios` / `flutter run`.  That script also fetches the ONNX
  Runtime pod archive (~61 MB, cached under `build-ios/`) and links it in;
  `--no-onnx` skips it, `CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK` points at your
  own build.  It fetches LiteRT too -- Google's iOS dylibs of the runtime
  and its Metal accelerator, device and simulator slices, ~31 MB, from its
  litert bucket -- and stages them under `ios/clpeak_native/embed-{device,
  simulator}/`, from where the Runner's embed phase copies and signs them
  into `Frameworks/` to be dlopen'd; `--no-litert` leaves the backend out,
  `CLPEAK_IOS_LITERT_DIR` points at your own slices.  See
  `src/litert/AGENTS.md`, Packaging.
- Tests: `flutter test` (pure Dart) or
  `CLPEAK_FFI_PATH=… flutter test` to include the native-bridge tests.

## Quick Lookups

- Native bindings / event decoding? → `lib/src/ffi/` (`clpeak_bindings.dart`,
  `clpeak_events.dart`; threading contract in `clpeak_runner.dart` —
  `NativeCallable.listener` + `Isolate.run`, `done` event = drain barrier)
- Argv construction (device/category/time flags)? → `lib/src/model/run_config.dart`
  (never emits per-test flags — the UI is data-driven so test churn in the
  core needs no app changes). `--verbose` is not run configuration: it is the
  app setting `SettingsService.verbose`, read at the moment of launch and
  passed as `BenchmarkService.start(verbose:)` by every launch site
- Diagnostics? → the document's `log` (`LogEntry` in
  `lib/src/model/run_document.dart`, one stream for the whole run, built from
  `log` events live and read straight off the file in history).
  `_DiagnosticsSection` at the foot of `results_body.dart` is counts-only
  (total lines + a pointer at the exported file): rendering per-line rows
  made the GUI sluggish on phones during heavy runs, so the file a user
  exports is where the contents are read.  Settings → "Verbose diagnostics"
  (`settings_screen.dart`) turns debug-level recording on for every run and
  links to the issue tracker, which is how a problem on a phone reaches a
  maintainer
- A run the app died in? → `RunHistoryStore.listCrashLogs()`: the native side
  streams `<id>.clpeak.log` while a run is in flight and removes it once the
  document is written, so one left behind is a crashed run's only record.
  History lists it under "Runs that did not finish" (`_CrashLogTile`) with
  export and delete; the in-flight run's own sidecar is excluded by
  `BenchmarkService.inFlightRunId`
- Run grouping / formatting? → `lib/src/model/run_document.dart`
- "What does this test measure?" → an info glyph beside the name, at both
  levels (test title and each reading's label in the expanded breakdown), one
  explanation per dialog: `_InfoGlyph` → `_showInfoDialog()` in
  `lib/src/ui/results/results_body.dart`.  Names wrap rather than ellipsize
  (a device or test name is identity — its tail is what distinguishes two of
  them), so every name goes through `_NameWithInfo`, which rides the glyph in
  the text as a `WidgetSpan` after the last word — a reserved column at the
  name's edge was tried and dropped, since on a wide desktop window it left a
  visible gap between a short name and its glyph.  In tests that makes a
  documented name's plain text `name + U+FFFC`: match it with
  `find.textContaining`, not `find.text`.  A test's own text arrives on
  `test_begin`, inside the `TestHeader` a `TestResult` is built from; each
  reading's rides the reading.
- Collapsed number, or a table of readings? → `TestResult.shape` /
  `.collapsible` (`lib/src/model/run_document.dart`).  A **homogeneous** test
  collapses to its best reading — its readings are variants of one measurement,
  so one number is the answer.  A **heterogeneous** one starts expanded — its
  readings measure different things, and the largest is not the test's result
  but merely its largest number, so it renders as a mini-table headed by its
  `axis` ("DATA TYPE"), expanded by default but collapsible to that header.
  A single-reading test collapses whatever its shape, since a one-row table
  says nothing the row does not.  `shape` is authored natively and cannot be
  inferred — see `docs/format-v3.md`.
- What do the meters mean? → `TestResult.barFraction`: a reading's size against
  the largest reading in its test, in every test, whichever direction is
  better.  The meter is a picture of the number printed beside it and nothing
  more.  Scaling it by which reading is *best* was tried and reverted — on a
  latency test it drew the shortest time as the longest bar, and a bar next to
  a number is read as that number's size, so it looked wrong however it was
  captioned.  `direction` still decides what it should: which reading a
  homogeneous test collapses to, and whether a `--compare` delta reads better
  or worse.
- Readings that could not be taken? → `DeviceRun.unavailable`, rendered by
  `_UnavailableSection` at the foot of the page.  It collects whole
  unsupported tests **and** the individual readings missing from tests that
  otherwise ran, so the tables above hold nothing but measurements.
- History persistence? → `lib/src/services/run_history_store.dart`
  (`<base>/runs/<id>.clpeak.json` written natively via `-o`, `index.json`
  sidecar).  Viewing is a plain `jsonDecode` — the saved document is already
  the shape the UI renders, so there is no native loader in the path and
  history stays readable when the native library cannot be loaded at all.
  `<base>` is `$HOME/.clpeak` on desktop — never `~/Documents`, which costs a
  macOS TCC consent prompt — and `<app documents>/clpeak` on Android/iOS,
  where that directory is inside the sandbox and is what the Files app shows.
  Files from an older `format_version` are skipped, not half-parsed.
- Run lifecycle state? → `lib/src/services/benchmark_service.dart`
- Which ONNX Runtime is loaded, and how a user changes it? →
  `lib/src/ui/settings/settings_screen.dart` + `SettingsService.onnxLibraryPath`.
  The path is applied in `main()` **before** `BenchmarkService` is built,
  because that constructor enumerates and enumeration is what loads the
  runtime — applied any later it would be a launch too late, which is why
  `main()` is async and `SettingsService.load()` reads prefs up front.
  The runtime setup -- library, Windows ML switch and folder -- is set
  once per launch: the first runtime that loads fixes it
  (`src/onnx/AGENTS.md`, "One runtime setup per process"), and since the
  saved one loads at startup, a change is normally saved for the next
  launch.  It always reaches `clpeak_set_onnx_library()` /
  `clpeak_set_onnx_winml()`: while no runtime has loaded it takes effect and
  `BenchmarkService` re-enumerates; otherwise the native side keeps it and
  reports it as `OnnxStatus.pendingRuntime`, and the panel shows the active
  setup and, dimmed beneath it, the inactive "Next launch" one with the
  note that it takes effect then.
  There is no in-app relaunch: the old process's exit, with everything the
  backends had loaded, was slow and visible.  LiteRT's library works the
  same way (`LitertStatus.pendingPath`).  On iOS the picker is
  replaced by "Built into the app": ONNX Runtime is statically linked there
  (Apple's pod is a static framework and iOS will not dlopen another), which
  `OnnxStatus.linkedIn` reports.
- Plugin execution providers and Windows ML? → the same screen, desktop
  only: `SettingsService.onnxEpLibraries` (name + path per library, the
  registration name guessed from the file name and confirmed in a dialog)
  and `onnxWinml` / `onnxWinmlPath` (Windows, in its own panel directly
  under the runtime it belongs with), applied in `main()` beside the
  library path.  The plugin list stays live -- a change goes through
  `BenchmarkService.setOnnxEpLibraries` and a re-enumeration -- while
  Windows ML is part of the runtime setup above.  `OnnxStatus.epLibraries`
  says how each registered on the last enumeration (empty = pending), and
  the panel shows it per row.  Enabling Windows ML makes the first
  enumeration of the launch install the Store providers, so it can take
  minutes the first time.
- Phone screen sleeping mid-run? → `lib/src/services/screen_wake.dart`
  (`wakelock_plus`, held from `BenchmarkService.start()` to `_finalize()`;
  Android/iOS only — a sleeping display stops the frames the run was budgeted
  for, which moves the scores of everything measured after it).  A desktop
  machine sleeping mid-run is the native launch's to prevent, as in the CLI:
  `include/common/keep_awake.h`
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
  `ios/clpeak_native/`, staged by `tools/build_ios_native.sh`)
- `android/app/build.gradle.kts` — `kr.clpeak`, minSdk 33, abiFilters,
  `externalNativeBuild` → `src/ffi/android/CMakeLists.txt`
- `android/app/src/main/AndroidManifest.xml` — `uses-native-library
  libOpenCL.so`
- every platform's app icon — all generated by
  `tools/icons/generate_icons.py` (needs Pillow) from the original clpeak
  wordmark in `tools/icons/clpeak_master_1024.png` (both repo-root `tools/`,
  alongside `build_ios_native.sh`). Never hand-edit an icon
  PNG; change the script and re-run it. It also writes the iOS
  `AppIcon.appiconset/Contents.json` (single 1024 universal + dark/tinted)
  and Android's `mipmap-anydpi-v26/ic_launcher.xml` +
  `values/ic_launcher_background.xml`.

## When You Change This Directory

- If the event schema or C ABI changes → update `lib/src/ffi/` and
  `src/ffi/AGENTS.md`.
- If `LogEntry` or the sidecar header changes natively → mirror it in
  `lib/src/model/run_document.dart` and `CrashLog.read()` in
  `lib/src/services/run_history_store.dart`.
- If you add a CLI-flag mapping → keep `run_config.dart` in sync with
  `src/common/options.cpp`.
- versionCode continues the retired native app's sequence (pubspec
  `version: x.y.z+N`).
