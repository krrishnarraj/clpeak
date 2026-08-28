import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

/// Holds the phone's screen on for the length of a run.
///
/// A full run is minutes of native compute with no touch input, which is
/// exactly what a handset's idle timer waits for.  Letting it fire costs more
/// than the live progress: the engine stops presenting frames once the display
/// sleeps, so the GPU contention the live-run screen is carefully budgeted for
/// (see the trap in app/AGENTS.md) disappears partway through and the tests
/// after the blackout are measured under different conditions than the ones
/// before it.  A run should be uniform end to end, watched or not.
///
/// Mobile only.  Desktop has no idle timer to fight while a window is up, and
/// a benchmark app has no business overriding a machine's own sleep policy.
///
/// Every call is best-effort.  The lock is a convenience, not part of the
/// measurement, so a platform that has no implementation or an OS that
/// refuses must never take a run down with it.
class ScreenWake {
  const ScreenWake._();

  static bool get _applies => Platform.isAndroid || Platform.isIOS;

  /// Keep the screen awake.  Safe to call when already held.
  static Future<void> acquire() => _set(true);

  /// Let the idle timer resume.  Safe to call when not held.
  static Future<void> release() => _set(false);

  static Future<void> _set(bool on) async {
    if (!_applies) return;
    try {
      await WakelockPlus.toggle(enable: on);
    } catch (e) {
      debugPrint('clpeak: screen wakelock ${on ? 'acquire' : 'release'}: $e');
    }
  }
}
