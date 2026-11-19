import 'package:flutter/material.dart';

/// The colours of one processor kind: a pale tinted card with dark text on
/// it in light mode, a deep tint with light text in dark mode.
@immutable
class KindHue {
  const KindHue({
    required this.tint,
    required this.tintInk,
    required this.tintSub,
    required this.accent,
    required this.rule,
    required this.onAccent,
  });

  /// The kind card's fill, and a KIND label's.
  final Color tint;

  /// Text on [tint]: the card's name and its numbers.
  final Color tintInk;

  /// Secondary text on [tint]: labels and notes.
  final Color tintSub;

  /// The kind's accent: a KIND label's text, units, links, a device chip's
  /// dot and outline.
  final Color accent;

  /// A hairline inside the card.
  final Color rule;

  /// Text on an [accent] fill (a "Best" badge).
  final Color onAccent;
}

/// One run of a compare, A or B.  Colour is never text: [fill] fills the
/// run's letter badge, [tint] its card and the soft highlight behind its
/// better value, so the numbers stay ink in either theme.
@immutable
class RunHue {
  const RunHue({required this.fill, required this.onFill, required this.tint});

  /// The letter badge.
  final Color fill;

  /// The letter on [fill].
  final Color onFill;

  /// The run's card, and the highlight behind its better value.
  final Color tint;
}

/// XpuPeak's design tokens: a white (near-black) ground with neutral UI, a
/// pale tinted card per processor kind (CPU blue, GPU green, NPU violet), and
/// two run colours for a compare (A blue, B orange), used only as fills.
/// One typeface, Sora, with tabular figures in numbers; rounded geometry
/// with 1 px hairlines and no elevation.
///
/// Nothing here uses `ColorScheme.fromSeed`: both palettes are fixed, so
/// the two brightnesses stay predictable instead of being derived.
class CP {
  const CP._({
    required this.ground,
    required this.side,
    required this.surface,
    required this.surface2,
    required this.well,
    required this.ink,
    required this.inkOn,
    required this.muted,
    required this.line,
    required this.countOn,
    required this.segOn,
    required this.segShadow,
    required this.barShadow,
    required this.danger,
    required this.warnTint,
    required this.warnInk,
    required this.tagOnTint,
    required this.isDark,
    required this.cpu,
    required this.gpu,
    required this.npu,
    required this.otherKind,
    required this.runA,
    required this.runB,
  });

  /// The page ground.
  final Color ground;

  /// The desktop sidebar's fill.
  final Color side;

  /// Cards, lists and panels.
  final Color surface;

  /// Tracks (segmented controls, search field), unselected chips, the
  /// selected nav item, callouts.
  final Color surface2;

  /// An inner panel inside a card: expanded readings, a value tile.
  final Color well;

  /// Text, and the fill of primary actions and selected tabs.
  final Color ink;

  /// Text on [ink].
  final Color inkOn;

  /// Secondary text: notes, counts, captions.
  final Color muted;

  /// Hairline borders and rules.
  final Color line;

  /// A count's pill inside a selected (ink) tab.
  final Color countOn;

  /// The selected option of a segmented control.
  final Color segOn;

  /// The selected segment's small shadow (none in dark mode).
  final List<BoxShadow> segShadow;

  /// The floating pick bar's shadow, the one elevated surface.
  final List<BoxShadow> barShadow;

  /// Destructive actions and errors.
  final Color danger;

  /// A warning tag ("Cancelled"): its fill and its text.
  final Color warnTint;
  final Color warnInk;

  /// A tag's fill on a run's tinted card.
  final Color tagOnTint;

  final bool isDark;

  /// The processor kinds' hues.
  final KindHue cpu;
  final KindHue gpu;
  final KindHue npu;

  /// Any other kind (`accelerator`, `unknown`, one this build does not know):
  /// neutral.
  final KindHue otherKind;

  /// The two runs of a compare.
  final RunHue runA;
  final RunHue runB;

  /// The hue of a processor kind or device type, as the file spells it
  /// (`cpu`, `GPU`, …); a kind without a colour of its own is neutral.
  KindHue kind(String kind) => switch (kind.toLowerCase()) {
        'cpu' => cpu,
        'gpu' => gpu,
        'npu' => npu,
        _ => otherKind,
      };

  /// A run's hue by its letter: `A` blue, `B` orange.
  RunHue run(String letter) => letter.toUpperCase() == 'B' ? runB : runA;

  /// The dot of something found and working (a runtime that loaded).
  Color get ok => gpu.accent;

  static const _light = CP._(
    ground: Color(0xFFFFFFFF),
    side: Color(0xFFF6F6F8),
    surface: Color(0xFFFFFFFF),
    surface2: Color(0xFFEDEDF1),
    well: Color(0xFFF4F4F6),
    ink: Color(0xFF111114),
    inkOn: Color(0xFFFFFFFF),
    muted: Color(0xFF5B5E66),
    line: Color(0xFFE3E3E8),
    countOn: Color(0x33FFFFFF),
    segOn: Color(0xFFFFFFFF),
    segShadow: [
      BoxShadow(color: Color(0x29111114), offset: Offset(0, 1), blurRadius: 3),
    ],
    barShadow: [
      BoxShadow(
          color: Color(0x38111114), offset: Offset(0, 12), blurRadius: 30),
    ],
    danger: Color(0xFFC8322A),
    warnTint: Color(0xFFFBEAD9),
    warnInk: Color(0xFF7A3E0A),
    tagOnTint: Color(0xC7FFFFFF),
    isDark: false,
    cpu: KindHue(
      tint: Color(0xFFE7EDF8),
      tintInk: Color(0xFF14213D),
      tintSub: Color(0xFF4A5773),
      accent: Color(0xFF34508C),
      rule: Color(0x1A111128),
      onAccent: Color(0xFFFFFFFF),
    ),
    gpu: KindHue(
      tint: Color(0xFFE3F0EA),
      tintInk: Color(0xFF0F2A20),
      tintSub: Color(0xFF475E55),
      accent: Color(0xFF2F6A56),
      rule: Color(0x1A111128),
      onAccent: Color(0xFFFFFFFF),
    ),
    npu: KindHue(
      tint: Color(0xFFECE8F6),
      tintInk: Color(0xFF1F1936),
      tintSub: Color(0xFF585170),
      accent: Color(0xFF5B4B91),
      rule: Color(0x1A111128),
      onAccent: Color(0xFFFFFFFF),
    ),
    otherKind: KindHue(
      tint: Color(0xFFEDEDF1),
      tintInk: Color(0xFF111114),
      tintSub: Color(0xFF5B5E66),
      accent: Color(0xFF5B5E66),
      rule: Color(0x1A111128),
      onAccent: Color(0xFFFFFFFF),
    ),
    runA: RunHue(
      fill: Color(0xFF2F6FEB),
      onFill: Color(0xFFFFFFFF),
      tint: Color(0xFFDCE8FF),
    ),
    runB: RunHue(
      fill: Color(0xFFF59331),
      onFill: Color(0xFF111114),
      tint: Color(0xFFFFE4C7),
    ),
  );

  static const _dark = CP._(
    ground: Color(0xFF0F1013),
    side: Color(0xFF141519),
    surface: Color(0xFF1A1B20),
    surface2: Color(0xFF25262C),
    well: Color(0xFF131418),
    ink: Color(0xFFECEDF1),
    inkOn: Color(0xFF111114),
    muted: Color(0xFFA2A5AE),
    line: Color(0xFF2A2B31),
    countOn: Color(0x24111114),
    segOn: Color(0xFF3A3B42),
    segShadow: [],
    barShadow: [
      BoxShadow(
          color: Color(0x80000000), offset: Offset(0, 12), blurRadius: 30),
    ],
    danger: Color(0xFFFF6B5E),
    warnTint: Color(0xFF3A2A1A),
    warnInk: Color(0xFFF2C79A),
    tagOnTint: Color(0x14FFFFFF),
    isDark: true,
    cpu: KindHue(
      tint: Color(0xFF1B2436),
      tintInk: Color(0xFFE6EBF5),
      tintSub: Color(0xFFAEB8CC),
      accent: Color(0xFF9FB4E3),
      rule: Color(0x1AFFFFFF),
      onAccent: Color(0xFF111114),
    ),
    gpu: KindHue(
      tint: Color(0xFF182821),
      tintInk: Color(0xFFE3EFEA),
      tintSub: Color(0xFFA9BDB4),
      accent: Color(0xFF8DC6AF),
      rule: Color(0x1AFFFFFF),
      onAccent: Color(0xFF111114),
    ),
    npu: KindHue(
      tint: Color(0xFF231F33),
      tintInk: Color(0xFFECE8F6),
      tintSub: Color(0xFFB7AFCB),
      accent: Color(0xFFB8A9E5),
      rule: Color(0x1AFFFFFF),
      onAccent: Color(0xFF111114),
    ),
    otherKind: KindHue(
      tint: Color(0xFF25262C),
      tintInk: Color(0xFFECEDF1),
      tintSub: Color(0xFFA2A5AE),
      accent: Color(0xFFA2A5AE),
      rule: Color(0x1AFFFFFF),
      onAccent: Color(0xFF111114),
    ),
    runA: RunHue(
      fill: Color(0xFF6FA0FF),
      onFill: Color(0xFF0F1013),
      tint: Color(0xFF1D2B47),
    ),
    runB: RunHue(
      fill: Color(0xFFF5A456),
      onFill: Color(0xFF0F1013),
      tint: Color(0xFF3D2B18),
    ),
  );

  static CP of(BuildContext context) =>
      forBrightness(Theme.of(context).brightness);

  /// The tokens of one brightness, where there is no context to read it from.
  static CP forBrightness(Brightness brightness) =>
      brightness == Brightness.dark ? _dark : _light;

  // ── Geometry ─────────────────────────────────────────────────────────────

  /// Cards, lists and settings panels.
  static const rCard = 24.0;

  /// The big cards: a processor's kind card, a compare's headline card.
  static const rCardLarge = 26.0;

  /// Inner panels: fold rows, callouts.
  static const rPanel = 20.0;

  /// A well inside a card (expanded readings).
  static const rWell = 18.0;

  /// Rows that select as a block (a history row, a recent run) and value
  /// tiles.
  static const rRow = 16.0;

  /// Text fields and nav items.
  static const rControl = 14.0;

  /// The highlight behind a value.
  static const rHighlight = 10.0;

  /// Pills: buttons, chips, tabs, tags.
  static const rPill = 999.0;

  // ── Typography ───────────────────────────────────────────────────────────
  //
  // Sora throughout, bundled (assets/fonts/sora); numbers in tabular
  // figures so readings never re-flow as digits change.  Letter spacing is
  // given in em, as the design states it.

  static const font = 'Sora';

  static const _tabular = <FontFeature>[FontFeature.tabularFigures()];

  TextStyle _style(double size, FontWeight weight,
          {double em = 0,
          double height = 1.4,
          Color? color,
          bool tabular = false}) =>
      TextStyle(
        fontFamily: font,
        fontSize: size,
        fontWeight: weight,
        letterSpacing: em * size,
        height: height,
        color: color ?? ink,
        fontFeatures: tabular ? _tabular : null,
      );

  /// A desktop page title: 38/800, -0.04em.
  TextStyle get h1 =>
      _style(38, FontWeight.w800, em: -0.04, height: 1.08);

  /// A phone page title: 26/800.
  TextStyle get h1Phone =>
      _style(26, FontWeight.w800, em: -0.04, height: 1.1);

  /// A section title: 20/800.
  TextStyle get h2 =>
      _style(20, FontWeight.w800, em: -0.03, height: 1.2);

  /// A group title inside a section: 16/800.
  TextStyle get h3 =>
      _style(16, FontWeight.w800, em: -0.02, height: 1.25);

  /// A row's or an item's title: 15/700.
  TextStyle get title => _style(15, FontWeight.w700, height: 1.3);

  /// Running text: 14.
  TextStyle get body => _style(14, FontWeight.w400);

  TextStyle get bodyMuted => body.copyWith(color: muted);

  /// What buttons, tabs, chips and nav items say: 14.5/700.
  TextStyle get label => _style(14.5, FontWeight.w700, height: 1.2);

  /// A paragraph of explanation: 13.5, muted.
  TextStyle get prose =>
      _style(13.5, FontWeight.w400, height: 1.55, color: muted);

  /// The line under a title: 13, muted.
  TextStyle get caption =>
      _style(13, FontWeight.w400, height: 1.35, color: muted);

  /// Small text: 12.
  TextStyle get small => _style(12, FontWeight.w400, height: 1.3);

  TextStyle get smallMuted => small.copyWith(color: muted);

  /// A KIND label and a column head: 11.5/800, 0.08em, muted.  Callers
  /// write the text in capitals.
  TextStyle get caps => _style(11.5, FontWeight.w800,
      em: 0.08, height: 1.2, color: muted);

  /// A settings section's head (APPEARANCE): 12.5/800, 0.12em, muted.
  TextStyle get capsHead => _style(12.5, FontWeight.w800,
      em: 0.12, height: 1.2, color: muted);

  /// The wordmark beside the app mark: 21/800.
  TextStyle get wordmark =>
      _style(21, FontWeight.w800, em: -0.03, height: 1.1);

  /// A reading in a row: 22/800, tabular.
  TextStyle get number => figure(22);

  /// A number at any size, tabular, tracked tighter the larger it is: the
  /// flagship numbers (44 desktop, 30 phone), a value tile's (26), a
  /// compare row's (17).
  TextStyle figure(double size,
          {FontWeight weight = FontWeight.w800, Color? color}) =>
      _style(size, weight,
          em: size >= 28
              ? -0.045
              : size >= 23
                  ? -0.035
                  : size >= 20
                      ? -0.03
                      : 0,
          height: 1.1,
          color: color,
          tabular: true);
}

/// Material glue.  The app's look lives in [CP] and `ui/common/kit.dart`;
/// this sets Sora as the font of everything and neutralizes the framework
/// bits that still show through -- dialog barriers, text selection,
/// scrollbars, snack bars, menus, tooltips.
class XpuPeakTheme {
  static ThemeData dark() => _base(CP._dark, Brightness.dark);
  static ThemeData light() => _base(CP._light, Brightness.light);

  static ThemeData _base(CP t, Brightness brightness) {
    final scheme = ColorScheme(
      brightness: brightness,
      primary: t.ink,
      onPrimary: t.inkOn,
      secondary: t.ink,
      onSecondary: t.inkOn,
      error: t.danger,
      onError: t.inkOn,
      surface: t.surface,
      onSurface: t.ink,
      outline: t.muted,
      outlineVariant: t.line,
    );

    final typography = Typography.material2021(colorScheme: scheme);

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      fontFamily: CP.font,
      scaffoldBackgroundColor: t.ground,
      canvasColor: t.ground,
      dividerColor: t.line,
      // Ripples animate for ~300ms per tap -- frames the benchmark would
      // otherwise be rendering.  The kit draws its own flat press states, so
      // nothing here should be splashing anyway; this is the backstop.
      splashFactory: NoSplash.splashFactory,
      highlightColor: Colors.transparent,
      splashColor: Colors.transparent,
      textTheme: (brightness == Brightness.dark
              ? typography.white
              : typography.black)
          .apply(
              fontFamily: CP.font, bodyColor: t.ink, displayColor: t.ink),
      textSelectionTheme: TextSelectionThemeData(
        cursorColor: t.ink,
        selectionColor: t.ink.withValues(alpha: 0.22),
        selectionHandleColor: t.ink,
      ),
      dialogTheme: DialogThemeData(
        backgroundColor: t.surface,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        barrierColor:
            t.isDark ? const Color(0xCC000000) : const Color(0x66000000),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(CP.rCardLarge),
          side: BorderSide(color: t.line),
        ),
      ),
      popupMenuTheme: PopupMenuThemeData(
        color: t.surface,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(CP.rRow),
          side: BorderSide(color: t.line),
        ),
        textStyle: t.body,
        labelTextStyle: WidgetStatePropertyAll(t.body),
      ),
      snackBarTheme: SnackBarThemeData(
        behavior: SnackBarBehavior.floating,
        backgroundColor: t.ink,
        contentTextStyle: t.body.copyWith(color: t.inkOn),
        actionTextColor: t.inkOn,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(CP.rRow),
        ),
      ),
      tooltipTheme: TooltipThemeData(
        decoration: BoxDecoration(
          color: t.ink,
          borderRadius: BorderRadius.circular(CP.rHighlight),
        ),
        textStyle: t.small.copyWith(color: t.inkOn),
        waitDuration: const Duration(milliseconds: 400),
      ),
      scrollbarTheme: ScrollbarThemeData(
        thickness: const WidgetStatePropertyAll(8),
        radius: const Radius.circular(CP.rPill),
        thumbColor: WidgetStatePropertyAll(t.line),
      ),
      progressIndicatorTheme: ProgressIndicatorThemeData(
        color: t.ink,
        linearTrackColor: t.surface2,
      ),
    );
  }

  static IconData backendIcon(String backend) => switch (backend) {
        'CPU' => Icons.memory,
        'ONNX' => Icons.hub_outlined,
        'CoreML' => Icons.psychology_outlined,
        'LiteRT' => Icons.bolt_outlined,
        'Metal' || 'Vulkan' || 'OpenCL' || 'CUDA' || 'ROCm' || 'oneAPI' =>
          Icons.developer_board,
        _ => Icons.device_unknown,
      };
}
