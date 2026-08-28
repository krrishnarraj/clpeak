import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// App settings: the theme mode, and which ONNX Runtime the ONNX backend
/// loads.
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
        _onnxLibraryPath = _prefs.getString(_onnxLibKey) ?? '';

  static Future<SettingsService> load() async =>
      SettingsService._(await SharedPreferences.getInstance());

  static const _themeKey = 'themeMode';
  static const _onnxLibKey = 'onnxLibraryPath';

  final SharedPreferences _prefs;

  ThemeMode _themeMode;
  ThemeMode get themeMode => _themeMode;

  /// Absolute path to an onnxruntime shared library, or empty to let the
  /// backend search its conventional names.  Ignored where ONNX Runtime is
  /// linked into the app (iOS).
  String _onnxLibraryPath;
  String get onnxLibraryPath => _onnxLibraryPath;

  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    notifyListeners();
    await _prefs.setString(_themeKey, mode.name);
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
}
