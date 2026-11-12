import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// What only the app can read of the state a run starts in, for the result
/// document's `conditions` (docs/format-v4.md, Conditions).
///
/// The native side reads the rest itself, but on a phone some of it is the
/// platform's to tell an app alone: Android's SELinux hides the power
/// supplies from native code, and its battery saver has no native API;
/// iOS's power source is UIKit's.  So just before each launch
/// [BenchmarkService] reads them here and hands them to the native side
/// (xpupeak_set_host_facts) -- every launch, so that none records the state
/// another started in.  On desktop the engine process reads them itself.
///
/// Android and iOS only, and best effort: what the platform cannot answer
/// is left out, and the run records what the native side read instead.  The
/// platform halves are `android/app/src/main/kotlin/kr/clpeak/MainActivity.kt`
/// and `ios/Runner/AppDelegate.swift`.
class HostFacts {
  const HostFacts._();

  static const _channel = MethodChannel('kr.XpuPeak/system');

  /// Whether the platform has any to tell: Android and iOS.
  static bool get supported => Platform.isAndroid || Platform.isIOS;

  /// The facts, in the document's words: `power_source` ("ac" or
  /// "battery") on both, and on Android `low_power_mode`, battery saver --
  /// iOS's Low Power Mode the native side reads there.  Empty when the
  /// platform answered nothing.
  static Future<Map<String, Object>> read() async {
    if (!supported) return const {};
    try {
      return await _channel.invokeMapMethod<String, Object>('hostFacts') ??
          const {};
    } catch (e) {
      debugPrint('xpupeak: host facts: $e');
      return const {};
    }
  }
}
