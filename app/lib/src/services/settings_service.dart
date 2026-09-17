import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../ffi/clpeak_bindings.dart' show OnnxEpLibrary;

/// App settings: the theme mode, which ONNX Runtime and LiteRT libraries the
/// two backends load, and whether runs record verbose diagnostics.
///
/// Loaded synchronously from an already-read [SharedPreferences] so main()
/// can apply the ONNX library before the first enumeration — enumeration is
/// what loads the runtime, so a path applied afterwards would be a restart
/// late.
class SettingsService extends ChangeNotifier {
  SettingsService._(this._prefs)
      : _themeMode = ThemeMode.values.firstWhere(
            (m) => m.name == _prefs.getString(_themeKey),
            orElse: () => ThemeMode.system),
        _onnxLibraryPath = _prefs.getString(_onnxLibKey) ?? '',
        _onnxEpLibraries = _decodeEpLibraries(_prefs.getString(_onnxEpLibsKey)),
        _onnxWinml = _prefs.getBool(_onnxWinmlKey) ?? false,
        _onnxWinmlPath = _prefs.getString(_onnxWinmlPathKey) ?? '',
        _litertLibraryPath = _prefs.getString(_litertLibKey) ?? '',
        _verbose = _prefs.getBool(_verboseKey) ?? false;

  static Future<SettingsService> load() async =>
      SettingsService._(await SharedPreferences.getInstance());

  static const _themeKey = 'themeMode';
  static const _onnxLibKey = 'onnxLibraryPath';
  static const _onnxEpLibsKey = 'onnxEpLibraries';
  static const _onnxWinmlKey = 'onnxWinml';
  static const _onnxWinmlPathKey = 'onnxWinmlPath';
  static const _litertLibKey = 'litertLibraryPath';
  static const _verboseKey = 'verbose';

  final SharedPreferences _prefs;

  ThemeMode _themeMode;
  ThemeMode get themeMode => _themeMode;

  /// Absolute path to an onnxruntime shared library, or empty to let the
  /// backend search its conventional names.  Ignored where ONNX Runtime is
  /// linked into the app (iOS).
  String _onnxLibraryPath;
  String get onnxLibraryPath => _onnxLibraryPath;

  /// Plugin execution-provider libraries the ONNX backend registers on its
  /// runtime (ONNX Runtime 1.22+): Qualcomm's QNN plugin, a vendor's
  /// provider fetched by hand.  Each carries the registration name the
  /// provider expects and the library's path.
  List<OnnxEpLibrary> _onnxEpLibraries;
  List<OnnxEpLibrary> get onnxEpLibraries => List.unmodifiable(_onnxEpLibraries);

  /// The saved libraries plus the platform's implicit ones: on Android the
  /// Qualcomm QNN plugin the APK may carry (`clpeakQnn=true` packages it
  /// beside the QNN runtime), registered by its bare soname and, absent,
  /// failing into the verbose log only.  This is what the native side gets.
  List<OnnxEpLibrary> get effectiveOnnxEpLibraries => [
        ..._onnxEpLibraries,
        if (Platform.isAndroid)
          const OnnxEpLibrary(
              name: 'QNNExecutionProvider',
              path: 'libonnxruntime_providers_qnn.so',
              implicit: true),
      ];

  /// Windows ML's execution-provider catalog: on Windows 11 24H2+ the
  /// vendor providers come from the Microsoft Store through
  /// Microsoft.Windows.AI.MachineLearning.dll, which [onnxWinmlPath] names
  /// (the file or its directory; empty to search beside the runtime and the
  /// app).  A switch rather than a default because a missing provider is
  /// downloaded.
  bool _onnxWinml;
  bool get onnxWinml => _onnxWinml;
  String _onnxWinmlPath;
  String get onnxWinmlPath => _onnxWinmlPath;

  /// The same for LiteRT: a libLiteRt to load ahead of the packaged /
  /// conventional one, or empty.
  String _litertLibraryPath;
  String get litertLibraryPath => _litertLibraryPath;

  /// Pass `--verbose` to every run: the saved document (and so the exported
  /// file) then carries the backends' debug output, the device inventory and
  /// whatever the runtimes had to say — what a maintainer needs to analyse a
  /// problem on a device they cannot reach.  Off by default: it makes the
  /// file larger, and the extra lines are for reading with a bug report.
  bool _verbose;
  bool get verbose => _verbose;

  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    notifyListeners();
    await _prefs.setString(_themeKey, mode.name);
  }

  Future<void> setVerbose(bool on) async {
    if (on == _verbose) return;
    _verbose = on;
    notifyListeners();
    await _prefs.setBool(_verboseKey, on);
  }

  Future<void> setOnnxLibraryPath(String path) async {
    if (path == _onnxLibraryPath) return;
    _onnxLibraryPath = path;
    notifyListeners();
    if (path.isEmpty) {
      await _prefs.remove(_onnxLibKey);
    } else {
      await _prefs.setString(_onnxLibKey, path);
    }
  }

  Future<void> setOnnxEpLibraries(List<OnnxEpLibrary> libs) async {
    _onnxEpLibraries = List.of(libs);
    notifyListeners();
    if (libs.isEmpty) {
      await _prefs.remove(_onnxEpLibsKey);
    } else {
      await _prefs.setString(
          _onnxEpLibsKey, jsonEncode([for (final l in libs) l.toJson()]));
    }
  }

  Future<void> setOnnxWinml({required bool enabled, required String path}) async {
    if (enabled == _onnxWinml && path == _onnxWinmlPath) return;
    _onnxWinml = enabled;
    _onnxWinmlPath = path;
    notifyListeners();
    await _prefs.setBool(_onnxWinmlKey, enabled);
    if (path.isEmpty) {
      await _prefs.remove(_onnxWinmlPathKey);
    } else {
      await _prefs.setString(_onnxWinmlPathKey, path);
    }
  }

  static List<OnnxEpLibrary> _decodeEpLibraries(String? json) {
    if (json == null || json.isEmpty) return const [];
    try {
      final list = jsonDecode(json) as List<dynamic>;
      return [
        for (final e in list) OnnxEpLibrary.fromJson(e as Map<String, dynamic>)
      ];
    } catch (_) {
      // A preference this build cannot read is dropped, not fatal.
      return const [];
    }
  }

  Future<void> setLitertLibraryPath(String path) async {
    if (path == _litertLibraryPath) return;
    _litertLibraryPath = path;
    notifyListeners();
    if (path.isEmpty) {
      await _prefs.remove(_litertLibKey);
    } else {
      await _prefs.setString(_litertLibKey, path);
    }
  }
}
