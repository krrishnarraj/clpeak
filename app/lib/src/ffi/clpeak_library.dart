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
