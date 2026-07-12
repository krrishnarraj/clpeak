import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// App settings — currently just the theme mode (dark-first default).
class SettingsService extends ChangeNotifier {
  SettingsService() {
    _load();
  }

  static const _themeKey = 'themeMode';

  ThemeMode _themeMode = ThemeMode.dark;
  ThemeMode get themeMode => _themeMode;

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    final v = prefs.getString(_themeKey);
    if (v != null) {
      _themeMode = ThemeMode.values.firstWhere((m) => m.name == v,
          orElse: () => ThemeMode.dark);
      notifyListeners();
    }
  }

  Future<void> setThemeMode(ThemeMode mode) async {
    _themeMode = mode;
    notifyListeners();
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_themeKey, mode.name);
  }
}
