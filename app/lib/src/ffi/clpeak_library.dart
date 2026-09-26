import 'dart:ffi';
import 'dart:io';

import 'package:path/path.dart' as p;

/// Resolves and opens the clpeak_ffi native library on every platform.
///
/// Release layouts:
///  - macOS:   Contents/Frameworks/clpeak_ffi.framework (embedded via the
///             local clpeak_native pod)
///  - Linux:   bundle/lib/libclpeak_ffi.so (installed by the runner CMake)
///  - Windows: clpeak_ffi.dll next to the runner executable
///  - Android: libclpeak_ffi.so packed in the APK (built by Gradle/NDK)
///  - iOS:     Frameworks/clpeak_ffi.framework (vendored xcframework pod)
///
/// Development override: set CLPEAK_FFI_PATH to the full path of the built
/// library (e.g. .../build-gui/clpeak_ffi.framework/clpeak_ffi) to run the
/// app before the native artifact is staged into the runner.
DynamicLibrary openClpeakLibrary() {
  final override = Platform.environment['CLPEAK_FFI_PATH'];
  if (override != null && override.isNotEmpty) {
    return DynamicLibrary.open(override);
  }

  if (Platform.isAndroid) {
    return DynamicLibrary.open('libclpeak_ffi.so');
  }
  if (Platform.isIOS || Platform.isMacOS) {
    try {
      return DynamicLibrary.open('clpeak_ffi.framework/clpeak_ffi');
    } on ArgumentError {
      // Fall back to an explicit path next to the executable
      // (<app>/Contents/MacOS/clpeak → <app>/Contents/Frameworks/...).
      final exeDir = p.dirname(Platform.resolvedExecutable);
      return DynamicLibrary.open(p.join(
          exeDir, '..', 'Frameworks', 'clpeak_ffi.framework', 'clpeak_ffi'));
    }
  }
  if (Platform.isLinux) {
    final exeDir = p.dirname(Platform.resolvedExecutable);
    return DynamicLibrary.open(p.join(exeDir, 'lib', 'libclpeak_ffi.so'));
  }
  if (Platform.isWindows) {
    return DynamicLibrary.open('clpeak_ffi.dll');
  }
  throw UnsupportedError('Unsupported platform for clpeak_ffi');
}

/// The clpeak_ffi library as a file, for the desktop engine host, which loads
/// it by path: CLPEAK_FFI_PATH, else where the release layout above puts it.
/// Null on Android and iOS (loaded from the app package) and when the file is
/// not there.
String? clpeakLibraryPath() {
  final override = Platform.environment['CLPEAK_FFI_PATH'];
  if (override != null && override.isNotEmpty) return override;
  final exeDir = p.dirname(Platform.resolvedExecutable);
  final String path;
  if (Platform.isMacOS) {
    path = p.normalize(p.join(
        exeDir, '..', 'Frameworks', 'clpeak_ffi.framework', 'clpeak_ffi'));
  } else if (Platform.isLinux) {
    path = p.join(exeDir, 'lib', 'libclpeak_ffi.so');
  } else if (Platform.isWindows) {
    path = p.join(exeDir, 'clpeak_ffi.dll');
  } else {
    return null;
  }
  return File(path).existsSync() ? path : null;
}

/// The desktop engine host, `clpeak-engine` (src/ffi/engine_host.cpp).  The
/// clpeak-gui target puts it beside the runner in every bundle, and the build
/// leaves it at the build root, beside the library CLPEAK_FFI_PATH names in
/// the dev loop -- so that one is looked for first, matching the library.
/// CLPEAK_ENGINE_PATH names it outright.  Null when there is none.
String? clpeakEngineHostPath() {
  final explicit = Platform.environment['CLPEAK_ENGINE_PATH'];
  if (explicit != null && explicit.isNotEmpty) {
    return File(explicit).existsSync() ? explicit : null;
  }
  final name = Platform.isWindows ? 'clpeak-engine.exe' : 'clpeak-engine';
  final candidates = <String>[];
  final ffi = Platform.environment['CLPEAK_FFI_PATH'];
  if (ffi != null && ffi.isNotEmpty) {
    // <build>/libclpeak_ffi.so, or <build>/clpeak_ffi.framework/clpeak_ffi.
    final dir = p.dirname(ffi);
    candidates
      ..add(p.join(dir, name))
      ..add(p.join(p.dirname(dir), name));
  }
  candidates.add(p.join(p.dirname(Platform.resolvedExecutable), name));
  for (final c in candidates) {
    if (File(c).existsSync()) return c;
  }
  return null;
}
