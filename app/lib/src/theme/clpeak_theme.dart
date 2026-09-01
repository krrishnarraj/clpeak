import 'package:flutter/material.dart';

import '../model/result_model.dart';

/// clpeak's design language — an *instrument console*, not a Material app.
///
/// The rules, in short:
///   * **Monochrome chrome, colour only in the data.**  Every frame, rule,
///     label and button is greyscale; the category tints are the only hues on
///     screen, so a number's colour always means something.
///   * **Monospace for anything technical** (labels, specs, numbers) and a
///     proportional face only for prose.  Readings never re-flow as digits
///     change.
///   * **Hairlines and tables, not cards and elevation.**  1px rules, square
///     corners, zero shadows.
///   * **Inverted primary actions** — a solid block of the text colour — in
///     place of tinted Material buttons.
///
/// Nothing here uses `ColorScheme.fromSeed`: the palettes are fixed so the two
/// brightnesses stay predictable instead of being derived from a seed.
class CP {
  const CP._({
    required this.bg,
    required this.panel,
    required this.hover,
    required this.line,
    required this.text,
    required this.dim,
    required this.faint,
    required this.inverse,
    required this.onInverse,
    required this.danger,
    required this.isDark,
  });

  /// Page ground.
  final Color bg;

  /// Panel / table fill, one step off the ground.
  final Color panel;

  /// Hover + pressed wash, and the fill of inset tracks.
  final Color hover;

  /// Hairline rules and panel borders.
  final Color line;

  /// Primary text.
  final Color text;

  /// Secondary text — labels, units, specs.
  final Color dim;

  /// Tertiary text — disabled, watermarks.
  final Color faint;

  /// Fill of primary actions (a solid block of [text]).
  final Color inverse;

  /// Label colour on [inverse].
  final Color onInverse;

  final Color danger;
  final bool isDark;

  static const _darkTokens = CP._(
    bg: Color(0xFF0B0C0E),
    panel: Color(0xFF121417),
    hover: Color(0xFF1A1D21),
    line: Color(0xFF24272C),
    text: Color(0xFFE9EBED),
    dim: Color(0xFF8A9199),
    faint: Color(0xFF565C63),
    inverse: Color(0xFFE9EBED),
    onInverse: Color(0xFF0B0C0E),
    danger: Color(0xFFFF6B5E),
    isDark: true,
  );

  static const _lightTokens = CP._(
    bg: Color(0xFFF6F6F7),
    panel: Color(0xFFFFFFFF),
    hover: Color(0xFFF0F1F3),
    line: Color(0xFFE3E5E9),
    text: Color(0xFF111316),
    dim: Color(0xFF6B7178),
    faint: Color(0xFFA8AEB5),
    inverse: Color(0xFF111316),
    onInverse: Color(0xFFFFFFFF),
    danger: Color(0xFFC8322A),
    isDark: false,
  );

  static CP of(BuildContext context) =>
      Theme.of(context).brightness == Brightness.dark
          ? _darkTokens
          : _lightTokens;

  // ── Geometry ─────────────────────────────────────────────────────────────
  //
  // Sharp, not rounded: a hint of a radius on panels, less on controls, none
  // on bars and ticks.
  static const rPanel = 3.0;
  static const rControl = 2.0;

  // ── Typography ───────────────────────────────────────────────────────────
  //
  // No fonts are bundled; this walks the platform monospace faces in order and
  // Flutter skips the ones that aren't installed.
  static const monoStack = <String>[
    'SF Mono',
    'SFMono-Regular',
    'Menlo',
    'Monaco',
    'Consolas',
    'DejaVu Sans Mono',
    'Liberation Mono',
    'Roboto Mono',
    'Courier New',
    'monospace',
  ];

  static const _tabular = <FontFeature>[FontFeature.tabularFigures()];

  /// Uppercase micro-label — section headings, buttons, column heads.
  TextStyle get micro => TextStyle(
        fontFamilyFallback: monoStack,
        fontSize: 10.5,
        height: 1.2,
        fontWeight: FontWeight.w600,
        letterSpacing: 1.4,
        color: dim,
      );

  TextStyle get microStrong => micro.copyWith(color: text);

  /// Monospace running text — device names, test names, specs.
  TextStyle get mono => TextStyle(
        fontFamilyFallback: monoStack,
        fontSize: 12.5,
        height: 1.35,
        color: text,
        fontFeatures: _tabular,
      );

  TextStyle get monoDim => mono.copyWith(color: dim);

  TextStyle get monoSmall => mono.copyWith(fontSize: 11, height: 1.3);

  TextStyle get monoSmallDim => monoSmall.copyWith(color: dim);

  /// A measured value — the loudest thing in the table.
  TextStyle get value => TextStyle(
        fontFamilyFallback: monoStack,
        fontSize: 15,
        height: 1.1,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.3,
        color: text,
        fontFeatures: _tabular,
      );

  /// Panel and screen titles.
  TextStyle get title => TextStyle(
        fontFamilyFallback: monoStack,
        fontSize: 14,
        height: 1.25,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.1,
        color: text,
      );

  /// The wordmark.
  TextStyle get wordmark => TextStyle(
        fontFamilyFallback: monoStack,
        fontSize: 25,
        height: 1.05,
        fontWeight: FontWeight.w700,
        letterSpacing: -1.4,
        color: text,
      );

  /// Prose — the one place a proportional face is used.
  TextStyle get body => TextStyle(fontSize: 13.5, height: 1.5, color: dim);

  TextStyle get bodyStrong => body.copyWith(color: text);
}

/// Material glue.  The app's look lives in [CP] and `ui/common/kit.dart`; this
/// only neutralizes the framework bits that still show through — dialog
/// barriers, text selection, scrollbars, snack bars.
class ClpeakTheme {
  static ThemeData dark() => _base(CP._darkTokens, Brightness.dark);
  static ThemeData light() => _base(CP._lightTokens, Brightness.light);

  static ThemeData _base(CP t, Brightness brightness) {
    final scheme = ColorScheme(
      brightness: brightness,
      primary: t.text,
      onPrimary: t.onInverse,
      secondary: t.text,
      onSecondary: t.onInverse,
      error: t.danger,
      onError: t.onInverse,
      surface: t.panel,
      onSurface: t.text,
      outline: t.dim,
      outlineVariant: t.line,
    );

    final typography = Typography.material2021(colorScheme: scheme);

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      scaffoldBackgroundColor: t.bg,
      canvasColor: t.bg,
      dividerColor: t.line,
      // Ripples animate for ~300ms per tap — frames the benchmark would
      // otherwise be rendering.  The kit draws its own flat press states, so
      // nothing here should be splashing anyway; this is the backstop.
      splashFactory: NoSplash.splashFactory,
      highlightColor: Colors.transparent,
      splashColor: Colors.transparent,
      textTheme: (brightness == Brightness.dark
              ? typography.white
              : typography.black)
          .apply(bodyColor: t.text, displayColor: t.text),
      textSelectionTheme: TextSelectionThemeData(
        cursorColor: t.text,
        selectionColor: t.text.withValues(alpha: 0.22),
        selectionHandleColor: t.text,
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: t.panel,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        barrierColor: t.isDark
            ? const Color(0xCC000000)
            : const Color(0x66000000),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(CP.rPanel),
          side: BorderSide(color: t.line),
        ),
      ),
      popupMenuTheme: PopupMenuThemeData(
        color: t.panel,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(CP.rPanel),
          side: BorderSide(color: t.line),
        ),
        textStyle: t.mono,
      ),
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        backgroundColor: t.inverse,
        contentTextStyle: t.mono.copyWith(color: t.onInverse),
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(CP.rControl),
        ),
      ),
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: t.inverse,
          borderRadius: BorderRadius.circular(CP.rControl),
        ),
        textStyle: t.monoSmall.copyWith(color: t.onInverse),
        waitDuration: const Duration(milliseconds: 400),
      ),
      scrollbarTheme: ScrollbarThemeData(
        thickness: const WidgetStatePropertyAll(8),
        radius: const Radius.circular(0),
        thumbColor: WidgetStatePropertyAll(t.line),
      ),
      progressIndicatorTheme: ProgressIndicatorThemeData(
        color: t.text,
        linearTrackColor: t.hover,
      ),
    );
  }

  /// Stable per-category tints — the only colour in the interface.
  ///
  /// Two palettes: luminous on the near-black ground, inked down for white,
  /// where the dark-mode amber and teal fall below readable contrast.
  static Color categoryColor(BenchCategory c, {Brightness? brightness}) =>
      brightness == Brightness.light
          ? switch (c) {
              BenchCategory.compute => const Color(0xFF1F63C8), // blue
              BenchCategory.crypto => const Color(0xFF1F63C8),
              BenchCategory.string => const Color(0xFF1F63C8),
              BenchCategory.bandwidth => const Color(0xFF1F63C8),
              BenchCategory.latency => const Color(0xFF1F63C8),
              BenchCategory.ai => const Color(0xFF1F63C8),
              BenchCategory.unknown => const Color(0xFF1F63C8),
            }
          : switch (c) {
              BenchCategory.compute => const Color(0xFF4FA8FF), // blue
              BenchCategory.crypto => const Color(0xFF4FA8FF),
              BenchCategory.string => const Color(0xFF4FA8FF),
              BenchCategory.bandwidth => const Color(0xFF4FA8FF),
              BenchCategory.latency => const Color(0xFF4FA8FF),
              BenchCategory.ai => const Color(0xFF4FA8FF),
              BenchCategory.unknown => const Color(0xFF4FA8FF),
            };

  static IconData categoryIcon(BenchCategory c) => switch (c) {
        BenchCategory.compute => Icons.speed,
        BenchCategory.crypto => Icons.lock_outline,
        BenchCategory.string => Icons.text_fields,
        BenchCategory.bandwidth => Icons.swap_vert,
        BenchCategory.latency => Icons.timer_outlined,
        BenchCategory.ai => Icons.auto_awesome,
        BenchCategory.unknown => Icons.help_outline,
      };

  static IconData backendIcon(String backend) => switch (backend) {
        'CPU' => Icons.memory,
        'ONNX' => Icons.hub_outlined,
        'Metal' || 'Vulkan' || 'OpenCL' || 'CUDA' || 'ROCm' || 'oneAPI' =>
          Icons.developer_board,
        _ => Icons.device_unknown,
      };
}
