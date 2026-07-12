import 'package:flutter/material.dart';

import '../model/result_entry.dart';

/// "Pro instrument" theme: deep neutral surfaces, one vivid accent, and a
/// stable tint per benchmark category.
class ClpeakTheme {
  static const seed = Color(0xFF34E0C2); // instrument teal

  static ThemeData dark() => _base(Brightness.dark);
  static ThemeData light() => _base(Brightness.light);

  static ThemeData _base(Brightness brightness) {
    final scheme = ColorScheme.fromSeed(
      seedColor: seed,
      brightness: brightness,
      dynamicSchemeVariant: DynamicSchemeVariant.fidelity,
    );
    final dark = brightness == Brightness.dark;
    return ThemeData(
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor:
          dark ? const Color(0xFF0D1114) : const Color(0xFFF6F8F9),
      cardTheme: CardThemeData(
        elevation: 0,
        color: dark ? const Color(0xFF161C21) : Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(
            color: dark ? const Color(0xFF232B32) : const Color(0xFFE3E8EB),
          ),
        ),
        margin: EdgeInsets.zero,
      ),
      dividerTheme: DividerThemeData(
        color: dark ? const Color(0xFF232B32) : const Color(0xFFE3E8EB),
        space: 1,
      ),
      navigationBarTheme: const NavigationBarThemeData(height: 68),
      snackBarTheme: const SnackBarThemeData(behavior: SnackBarBehavior.floating),
    );
  }

  /// Stable per-category tints (used for chips, bars, and section accents).
  static Color categoryColor(BenchCategory c, {Brightness? brightness}) =>
      switch (c) {
        BenchCategory.fpCompute => const Color(0xFF4FA8FF), // blue
        BenchCategory.intCompute => const Color(0xFFB08CFF), // violet
        BenchCategory.crypto => const Color(0xFFFF9D5C), // orange
        BenchCategory.string => const Color(0xFF34E0C2), // teal
        BenchCategory.bandwidth => const Color(0xFF6FDD75), // green
        BenchCategory.latency => const Color(0xFFFFD35C), // amber
        BenchCategory.unknown => const Color(0xFF9AA7B0), // neutral
      };

  static IconData categoryIcon(BenchCategory c) => switch (c) {
        BenchCategory.fpCompute => Icons.speed,
        BenchCategory.intCompute => Icons.tag,
        BenchCategory.crypto => Icons.lock_outline,
        BenchCategory.string => Icons.text_fields,
        BenchCategory.bandwidth => Icons.swap_vert,
        BenchCategory.latency => Icons.timer_outlined,
        BenchCategory.unknown => Icons.help_outline,
      };

  static IconData backendIcon(String backend) => switch (backend) {
        'CPU' => Icons.memory,
        'Metal' || 'Vulkan' || 'OpenCL' || 'CUDA' || 'ROCm' || 'oneAPI' =>
          Icons.developer_board,
        _ => Icons.device_unknown,
      };
}
