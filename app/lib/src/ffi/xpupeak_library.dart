import 'dart:ffi';
import 'dart:io';

import 'package:path/path.dart' as p;

/// Resolves and opens the xpupeak_ffi native library on every platform.
///
/// Release layouts:
///  - macOS:   Contents/Frameworks/xpupeak_ffi.framework (embedded via the
///             local xpupeak_native pod)
///  - Linux:   bundle/lib/libxpupeak_ffi.so (installed by the runner CMake)
///  - Windows: xpupeak_ffi.dll next to the runner executable
///  - Android: libxpupeak_ffi.so packed in the APK (built by Gradle/NDK)
///  - iOS:     Frameworks/xpupeak_ffi.framework (vendored xcframework pod)
///
/// Development override: set XPUPEAK_FFI_PATH to the full path of the built
/// library (e.g. .../build-gui/xpupeak_ffi.framework/xpupeak_ffi) to run the
/// app before the native artifact is staged into the runner.
DynamicLibrary openXpuPeakLibrary() {
  final override = Platform.environment['XPUPEAK_FFI_PATH'];
  if (override != null && override.isNotEmpty) {
    return DynamicLibrary.open(override);
  }

  if (Platform.isAndroid) {
    return DynamicLibrary.open('libxpupeak_ffi.so');
  }
  if (Platform.isIOS || Platform.isMacOS) {
    try {
      return DynamicLibrary.open('xpupeak_ffi.framework/xpupeak_ffi');
    } on ArgumentError {
      // Fall back to an explicit path next to the executable
      // (<app>/Contents/MacOS/xpupeak → <app>/Contents/Frameworks/...).
      final exeDir = p.dirname(Platform.resolvedExecutable);
      return DynamicLibrary.open(p.join(
          exeDir, '..', 'Frameworks', 'xpupeak_ffi.framework', 'xpupeak_ffi'));
    }
  }
  if (Platform.isLinux) {
    final exeDir = p.dirname(Platform.resolvedExecutable);
    return DynamicLibrary.open(p.join(exeDir, 'lib', 'libxpupeak_ffi.so'));
  }
  if (Platform.isWindows) {
    return DynamicLibrary.open('xpupeak_ffi.dll');
  }
  throw UnsupportedError('Unsupported platform for xpupeak_ffi');
}

/// The xpupeak_ffi library as a file, for the desktop engine host, which loads
/// it by path: XPUPEAK_FFI_PATH, else where the release layout above puts it.
/// Null on Android and iOS (loaded from the app package) and when the file is
/// not there.
String? xpuPeakLibraryPath() {
  final override = Platform.environment['XPUPEAK_FFI_PATH'];
  if (override != null && override.isNotEmpty) return override;
  final exeDir = p.dirname(Platform.resolvedExecutable);
  final String path;
  if (Platform.isMacOS) {
    path = p.normalize(p.join(
        exeDir, '..', 'Frameworks', 'xpupeak_ffi.framework', 'xpupeak_ffi'));
  } else if (Platform.isLinux) {
    path = p.join(exeDir, 'lib', 'libxpupeak_ffi.so');
  } else if (Platform.isWindows) {
    path = p.join(exeDir, 'xpupeak_ffi.dll');
  } else {
    return null;
  }
  return File(path).existsSync() ? path : null;
}

/// The desktop engine host, `xpupeak-engine` (src/ffi/engine_host.cpp).  The
/// xpupeak-gui target puts it beside the runner in every bundle, and the build
/// leaves it at the build root, beside the library XPUPEAK_FFI_PATH names in
/// the dev loop -- so that one is looked for first, matching the library.
/// XPUPEAK_ENGINE_PATH names it outright.  Null when there is none.
String? xpuPeakEngineHostPath() {
  final explicit = Platform.environment['XPUPEAK_ENGINE_PATH'];
  if (explicit != null && explicit.isNotEmpty) {
    return File(explicit).existsSync() ? explicit : null;
  }
  final name = Platform.isWindows ? 'xpupeak-engine.exe' : 'xpupeak-engine';
  final candidates = <String>[];
  final ffi = Platform.environment['XPUPEAK_FFI_PATH'];
  if (ffi != null && ffi.isNotEmpty) {
    // <build>/libxpupeak_ffi.so, or <build>/xpupeak_ffi.framework/xpupeak_ffi.
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
