import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// The notification a run posts when the app leaves the screen mid-run.
///
/// Off screen a run holds at its next boundary between readings (include/
/// common/screen_hold.h), and a phone soon freezes the app and may end it,
/// so a run can sit there for as long as it takes the person to remember
/// it.  The notification says it is waiting, and tapping it brings the app
/// back, which lets the run go on -- or, when the phone ended the app
/// meanwhile, opens it on the offer to continue the run.  It is posted the
/// moment the app is hidden, while the process still runs: a Galaxy Z Fold8
/// froze clpeak 45 to 70 s after it left.
///
/// Android and iOS only, and best effort: without the permission nothing is
/// shown, and the run behaves the same.  The platform halves are
/// `android/app/src/main/kotlin/kr/clpeak/MainActivity.kt` and
/// `ios/Runner/AppDelegate.swift`.
class RunNotice {
  const RunNotice._();

  static const _channel = MethodChannel('kr.clpeak/notify');
  static const _askedKey = 'runNoticeAsked';

  static bool get _supported => Platform.isAndroid || Platform.isIOS;

  /// Ask for the permission to post, at the first run and never again: it
  /// is the one moment the request makes sense to the person, and the app
  /// is on screen, which a request needs.  Android stops showing the
  /// dialog after two refusals anyway.
  static Future<void> askOnce() async {
    if (!_supported) return;
    try {
      final prefs = await SharedPreferences.getInstance();
      if (prefs.getBool(_askedKey) ?? false) return;
      await prefs.setBool(_askedKey, true);
      await _channel.invokeMethod<bool>('ask');
    } catch (e) {
      debugPrint('clpeak: notification permission: $e');
    }
  }

  /// Post, or replace, the run's notification.
  static Future<void> show({required String title, required String body}) =>
      _invoke('show', {'title': title, 'body': body});

  /// Take it down: back on screen, the run ended, a stale one at launch.
  static Future<void> clear() => _invoke('clear');

  static Future<void> _invoke(String method, [Object? args]) async {
    if (!_supported) return;
    try {
      await _channel.invokeMethod<void>(method, args);
    } catch (e) {
      debugPrint('clpeak: run notification: $e');
    }
  }
}
