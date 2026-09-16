import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

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
        _litertLibraryPath = _prefs.getString(_litertLibKey) ?? '',
        _verbose = _prefs.getBool(_verboseKey) ?? false;

  static Future<SettingsService> load() async =>
      SettingsService._(await SharedPreferences.getInstance());

  static const _themeKey = 'themeMode';
  static const _onnxLibKey = 'onnxLibraryPath';
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
