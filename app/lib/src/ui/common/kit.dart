/// The XpuPeak widget kit.
///
/// Built on raw Flutter primitives (`Container`, `GestureDetector`,
/// `MouseRegion`) rather than Material's `Card` / `Chip` / `Switch` /
/// `AppBar` / `NavigationRail` / `ListTile`.  The look is the tokens in
/// [CP]: neutral UI on a white (near-black) ground, a tinted card per
/// processor kind, ink-filled pill actions, Sora throughout.  Two rules hold
/// for everything here:
///
///   * **Nothing animates.**  Press and hover are flat colour swaps, not ink
///     splashes, and disclosure is instant rather than a cross-fade.  Every
///     animated frame is GPU work stolen from the benchmark running
///     underneath (see `app/AGENTS.md`), so the design and the measurement
///     agree here.
///   * **Every control is a real control.**  [CTap] gives each one a 44 px
///     target, keyboard focus (Enter / Space activate it, a ring shows where
///     focus is) and the semantics of what it is: a button, a selected tab,
///     a toggled switch, an expanded fold.
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../theme/xpupeak_theme.dart';

/// The window width from which the shell shows the desktop sidebar, and
/// screens their desktop layout.
const double cWideWidth = 900;

/// Whether the window is wide enough for the desktop layout.
bool cIsWide(BuildContext context) =>
    MediaQuery.sizeOf(context).width >= cWideWidth;

/// A hover wash over [over]: flat, one step towards `surface2`.
Color _wash(CP t, Color over) =>
    Color.alphaBlend(t.surface2.withValues(alpha: t.isDark ? 0.7 : 0.6), over);

// ── Pressable base ─────────────────────────────────────────────────────────

/// Hover, press and keyboard focus without a splash; a control's semantics.
class CTap extends StatefulWidget {
  const CTap({
    super.key,
    required this.builder,
    this.onTap,
    this.cursor = SystemMouseCursors.click,
    this.semanticLabel,
    this.button = true,
    this.selected,
    this.toggled,
    this.checked,
    this.expanded,
    this.focusRadius = const BorderRadius.all(Radius.circular(CP.rControl)),
    this.insetFocus = false,
    this.autofocus = false,
  });

  final Widget Function(BuildContext context, bool hovered, bool pressed)
      builder;
  final VoidCallback? onTap;
  final MouseCursor cursor;

  /// Read in place of the child's text; for a control that shows none.
  final String? semanticLabel;

  /// Announced as a button (the default); false for a row that only
  /// sometimes takes a tap, or a switch.
  final bool button;

  /// A tab's or a chip's state.
  final bool? selected;

  /// A switch's state.
  final bool? toggled;

  /// A checkbox's state.
  final bool? checked;

  /// A fold's state.
  final bool? expanded;

  /// The shape the focus ring follows.
  final BorderRadius focusRadius;

  /// Draw the focus ring inside the bounds, for a row inside a card that
  /// clips its children.
  final bool insetFocus;

  final bool autofocus;

  @override
  State<CTap> createState() => _CTapState();
}

class _CTapState extends State<CTap> {
  bool _hovered = false;
  bool _pressed = false;
  bool _focused = false;

  late final Map<Type, Action<Intent>> _actions = {
    ActivateIntent: CallbackAction<ActivateIntent>(onInvoke: (_) => _activate()),
    ButtonActivateIntent:
        CallbackAction<ButtonActivateIntent>(onInvoke: (_) => _activate()),
  };

  Object? _activate() {
    widget.onTap?.call();
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final enabled = widget.onTap != null;
    final control = widget.button ||
        widget.toggled != null ||
        widget.checked != null ||
        widget.selected != null;
    Widget child = widget.builder(
        context, enabled && _hovered, enabled && _pressed);
    if (_focused) {
      child = CustomPaint(
        foregroundPainter: _FocusRing(
          color: CP.of(context).ink,
          radius: widget.focusRadius,
          inset: widget.insetFocus,
        ),
        child: child,
      );
    }
    return Semantics(
      container: control || enabled,
      button: widget.button &&
          widget.toggled == null &&
          widget.checked == null,
      enabled: control ? enabled : null,
      selected: widget.selected,
      toggled: widget.toggled,
      checked: widget.checked,
      expanded: widget.expanded,
      label: widget.semanticLabel,
      onTap: widget.onTap,
      child: FocusableActionDetector(
        enabled: enabled,
        autofocus: widget.autofocus,
        actions: _actions,
        onShowFocusHighlight: (on) {
          if (on != _focused) setState(() => _focused = on);
        },
        child: MouseRegion(
          cursor: enabled ? widget.cursor : MouseCursor.defer,
          onEnter: enabled ? (_) => setState(() => _hovered = true) : null,
          onExit: enabled ? (_) => setState(() => _hovered = false) : null,
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            excludeFromSemantics: true,
            onTap: widget.onTap,
            onTapDown: enabled ? (_) => setState(() => _pressed = true) : null,
            onTapUp: enabled ? (_) => setState(() => _pressed = false) : null,
            onTapCancel:
                enabled ? () => setState(() => _pressed = false) : null,
            child: child,
          ),
        ),
      ),
    );
  }
}

/// The keyboard focus ring: 2 px of ink around (or just inside) the control.
class _FocusRing extends CustomPainter {
  const _FocusRing(
      {required this.color, required this.radius, required this.inset});

  final Color color;
  final BorderRadius radius;
  final bool inset;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    canvas.drawRRect(
      radius.toRRect(inset ? rect.deflate(1.5) : rect.inflate(3)),
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
  }

  @override
  bool shouldRepaint(_FocusRing old) =>
      old.color != color || old.radius != radius || old.inset != inset;
}

// ── Glyphs ─────────────────────────────────────────────────────────────────

/// The app's line icons, as SVG path data on a 24-unit grid (the design's
/// own icons; [CGlyph] draws them).
abstract final class CGlyphs {
  // Navigation.
  static const bench = 'M4 15a8 8 0 1 1 16 0M12 15l4-5M7 19h10';
  static const history =
      'M3.5 12a8.5 8.5 0 1 0 2.5-6M3.5 4.5v4h4M12 8v4.5l3 2';
  static const settings = 'M4 8h9M17 8h3M4 16h3M11 16h9'
      'M13 8a2 2 0 1 0 4 0a2 2 0 1 0-4 0M7 16a2 2 0 1 0 4 0a2 2 0 1 0-4 0';
  static const about = 'M4 12a8 8 0 1 0 16 0a8 8 0 1 0-16 0M12 11v5M12 8v0.5';

  /// The app mark's peak line.
  static const peak = 'M3 18l6-10 4 6 2-3 6 7';

  /// Run (drawn filled).
  static const play = 'M7 4.5v15l13-7.5z';

  static const back = 'M15 6l-6 6 6 6';
  static const forward = 'M9 6l6 6-6 6';
  static const chevronDown = 'M6 9l6 6 6-6';
  static const expandAll = 'M7 4.5l5 5 5-5M7 14.5l5 5 5-5';
  static const collapseAll = 'M7 9.5l5-5 5 5M7 19.5l5-5 5 5';
  static const search =
      'M10.5 17a6.5 6.5 0 1 0 0-13a6.5 6.5 0 0 0 0 13zM15.5 15.5L20 20';

  /// Share / export: an arrow up out of a tray.
  static const upload = 'M12 15V3.5M7.5 8L12 3.5 16.5 8M5 12v7.5h14V12';

  /// Import: an arrow down into a tray.
  static const download = 'M12 3.5V15M7.5 10.5L12 15l4.5-4.5M5 15v4.5h14V15';

  /// Swap A and B.
  static const swap = 'M4 8h15l-4-4M20 16H5l4 4';

  /// Two processors paired.
  static const pair = 'M4 12h16M8 8l-4 4 4 4M16 8l4 4-4 4';

  static const check = 'M5 12.5l4.5 4.5L19 7.5';

  /// Opens outside the app.
  static const external = 'M14 4h6v6M20 4l-9 9M18 14v5.5H4.5V6H10';

  /// A row's actions menu (drawn filled).
  static const more = 'M6 13.5a1.5 1.5 0 1 0 0-3a1.5 1.5 0 0 0 0 3z'
      'M12 13.5a1.5 1.5 0 1 0 0-3a1.5 1.5 0 0 0 0 3z'
      'M18 13.5a1.5 1.5 0 1 0 0-3a1.5 1.5 0 0 0 0 3z';
}

/// A line icon drawn from SVG path data ([CGlyphs]), in the text colour.
class CGlyph extends StatelessWidget {
  const CGlyph(
    this.path, {
    super.key,
    this.size = 20,
    this.color,
    this.stroke = 1.9,
    this.fill = false,
    this.semanticLabel,
  });

  final String path;
  final double size;
  final Color? color;

  /// Stroke width in grid units (of 24).
  final double stroke;
  final bool fill;
  final String? semanticLabel;

  static final _parsed = <String, Path>{};

  @override
  Widget build(BuildContext context) {
    final c = color ??
        DefaultTextStyle.of(context).style.color ??
        CP.of(context).ink;
    final glyph = SizedBox.square(
      dimension: size,
      child: CustomPaint(
        painter: _GlyphPainter(
          _parsed.putIfAbsent(path, () => _parseSvgPath(path)),
          c,
          stroke,
          fill,
        ),
      ),
    );
    return semanticLabel == null
        ? ExcludeSemantics(child: glyph)
        : Semantics(label: semanticLabel, image: true, child: glyph);
  }
}

class _GlyphPainter extends CustomPainter {
  const _GlyphPainter(this.path, this.color, this.stroke, this.fill);

  final Path path;
  final Color color;
  final double stroke;
  final bool fill;

  @override
  void paint(Canvas canvas, Size size) {
    canvas.save();
    canvas.scale(size.width / 24, size.height / 24);
    canvas.drawPath(
      path,
      Paint()
        ..color = color
        ..style = fill ? PaintingStyle.fill : PaintingStyle.stroke
        ..strokeWidth = stroke
        ..strokeCap = StrokeCap.round
        ..strokeJoin = StrokeJoin.round,
    );
    canvas.restore();
  }

  @override
  bool shouldRepaint(_GlyphPainter old) =>
      old.path != path ||
      old.color != color ||
      old.stroke != stroke ||
      old.fill != fill;
}

/// SVG path data -- the commands the design's icons use (M L H V A C Z,
/// absolute and relative, with implicit repeats) -- as a [Path].  Stops at
/// anything else, keeping what it drew.
Path _parseSvgPath(String d) {
  final tokens = <Object>[];
  final number = RegExp(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?');
  var at = 0;
  while (at < d.length) {
    final c = d[at];
    if ('MmLlHhVvAaCcZz'.contains(c)) {
      tokens.add(c);
      at++;
    } else if (' ,\t\r\n'.contains(c)) {
      at++;
    } else {
      final m = number.matchAsPrefix(d, at);
      if (m == null) break;
      tokens.add(double.parse(m[0]!));
      at = m.end;
    }
  }

  final path = Path();
  var i = 0;
  var cmd = '';
  double x = 0, y = 0, startX = 0, startY = 0;
  double n() => tokens[i++] as double;
  try {
    while (i < tokens.length) {
      final token = tokens[i];
      if (token is String) {
        cmd = token;
        i++;
        if (cmd == 'Z' || cmd == 'z') {
          path.close();
          x = startX;
          y = startY;
          continue;
        }
      } else if (cmd.isEmpty || cmd == 'Z' || cmd == 'z') {
        break;
      }
      final rel = cmd == cmd.toLowerCase();
      switch (cmd.toUpperCase()) {
        case 'M':
          final dx = n(), dy = n();
          x = rel ? x + dx : dx;
          y = rel ? y + dy : dy;
          path.moveTo(x, y);
          startX = x;
          startY = y;
          // Further pairs after a move are lines.
          cmd = rel ? 'l' : 'L';
        case 'L':
          final dx = n(), dy = n();
          x = rel ? x + dx : dx;
          y = rel ? y + dy : dy;
          path.lineTo(x, y);
        case 'H':
          final dx = n();
          x = rel ? x + dx : dx;
          path.lineTo(x, y);
        case 'V':
          final dy = n();
          y = rel ? y + dy : dy;
          path.lineTo(x, y);
        case 'A':
          final rx = n(), ry = n(), rotation = n();
          final large = n() != 0, sweep = n() != 0;
          final dx = n(), dy = n();
          x = rel ? x + dx : dx;
          y = rel ? y + dy : dy;
          path.arcToPoint(Offset(x, y),
              radius: Radius.elliptical(rx, ry),
              rotation: rotation,
              largeArc: large,
              clockwise: sweep);
        case 'C':
          final x1 = n(), y1 = n(), x2 = n(), y2 = n(), dx = n(), dy = n();
          final ox = rel ? x : 0.0, oy = rel ? y : 0.0;
          path.cubicTo(ox + x1, oy + y1, ox + x2, oy + y2, ox + dx, oy + dy);
          x = ox + dx;
          y = oy + dy;
        default:
          return path;
      }
    }
  } on Object {
    // Malformed data: keep what was drawn.
  }
  return path;
}

// ── Surfaces ───────────────────────────────────────────────────────────────

/// A card: surface fill, 1 px hairline, 24 px corners, no shadow.
class CPanel extends StatelessWidget {
  const CPanel({
    super.key,
    required this.child,
    this.padding,
    this.radius = CP.rCard,
    this.color,
    this.outlined = true,
  });

  final Widget child;
  final EdgeInsetsGeometry? padding;
  final double radius;

  /// The fill, `surface` unless given (`surface2` for a callout).
  final Color? color;

  /// Draw the hairline; a callout has none.
  final bool outlined;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final shape = BorderRadius.circular(radius);
    return Container(
      decoration: BoxDecoration(color: color ?? t.surface, borderRadius: shape),
      // The hairline over the children: a row's square corners would
      // otherwise cover the rounded ones.
      foregroundDecoration: outlined
          ? BoxDecoration(border: Border.all(color: t.line), borderRadius: shape)
          : null,
      clipBehavior: Clip.antiAlias,
      padding: padding,
      child: child,
    );
  }
}

/// A card's head: a marker, the title, an action, and a hairline under it.
class CPanelHead extends StatelessWidget {
  const CPanelHead({
    super.key,
    required this.title,
    this.tag,
    this.icon,
    this.trailing,
    this.color,
  });

  final String title;

  /// Marker on the left.  Use [tag] when the title doesn't already name the
  /// thing (a device titled with its backend), [icon] when it does.
  final String? tag;
  final IconData? icon;
  final Widget? trailing;

  /// The marker's colour.
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      constraints: const BoxConstraints(minHeight: 56),
      padding: const EdgeInsets.fromLTRB(20, 10, 14, 10),
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: t.line)),
      ),
      child: Row(
        children: [
          if (tag != null) ...[
            CTag(text: tag!, color: color, upper: false),
            const SizedBox(width: 10),
          ] else if (icon != null) ...[
            Icon(icon, size: 18, color: color ?? t.muted),
            const SizedBox(width: 10),
          ],
          Expanded(child: Text(title, style: t.title)),
          ?trailing,
        ],
      ),
    );
  }
}

/// A section's head in capitals (APPEARANCE), with a note on the right.
class CSection extends StatelessWidget {
  const CSection({
    super.key,
    required this.label,
    this.color,
    this.trailing,
  });

  final String label;
  final Color? color;
  final String? trailing;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Semantics(
      header: true,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.baseline,
          textBaseline: TextBaseline.alphabetic,
          children: [
            Expanded(
              child: Text(label.toUpperCase(),
                  style: t.capsHead.copyWith(color: color ?? t.muted)),
            ),
            if (trailing != null) ...[
              const SizedBox(width: 12),
              Text(trailing!, style: t.caption),
            ],
          ],
        ),
      ),
    );
  }
}

/// A section's h2 (or a group's h3) with a muted note and an optional control.
class CSectionHeader extends StatelessWidget {
  const CSectionHeader({
    super.key,
    required this.title,
    this.note,
    this.trailing,
    this.level = 2,
  }) : assert(level == 2 || level == 3);

  final String title;
  final String? note;
  final Widget? trailing;

  /// 2: a section (20/800); 3: a group inside one (16/800).
  final int level;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final head = Semantics(
      header: true,
      headingLevel: level,
      child: Text(title, style: level == 2 ? t.h2 : t.h3),
    );
    final noteStyle =
        level == 2 ? t.caption.copyWith(fontSize: 13.5) : t.caption;
    final block = LayoutBuilder(builder: (context, constraints) {
      if (note == null) return head;
      if (constraints.maxWidth < 520) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [head, const SizedBox(height: 2), Text(note!, style: noteStyle)],
        );
      }
      return Row(
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          head,
          SizedBox(width: level == 2 ? 14 : 10),
          Flexible(child: Text(note!, style: noteStyle)),
        ],
      );
    });
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 6),
      child: trailing == null
          ? block
          : Row(
              children: [
                Expanded(child: block),
                const SizedBox(width: 12),
                trailing!,
              ],
            ),
    );
  }
}

/// A processor kind's tinted card: a KIND line and a name, then [child].
class CKindCard extends StatelessWidget {
  const CKindCard({
    super.key,
    required this.kind,
    this.label,
    this.title,
    required this.child,
    this.compact = false,
  });

  /// `cpu`, `gpu`, `npu`, …: picks the hue ([CP.kind]).
  final String kind;

  /// The capitals line at the head (`CPU · 10 CORES`); the kind itself
  /// unless given.
  final String? label;

  /// The processor's name.
  final String? title;
  final Widget child;

  /// The phone's tighter padding and smaller name.
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = t.kind(kind);
    return Container(
      decoration: BoxDecoration(
        color: h.tint,
        borderRadius: BorderRadius.circular(CP.rCardLarge),
      ),
      clipBehavior: Clip.antiAlias,
      child: DefaultTextStyle.merge(
        style: TextStyle(color: h.tintInk),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisSize: MainAxisSize.min,
          children: [
            Padding(
              padding: compact
                  ? const EdgeInsets.fromLTRB(20, 17, 20, 4)
                  : const EdgeInsets.fromLTRB(24, 22, 24, 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    label ?? kind.toUpperCase(),
                    style: t.capsHead.copyWith(
                      fontSize: compact ? 12 : 12.5,
                      fontWeight: FontWeight.w700,
                      color: h.accent,
                    ),
                  ),
                  if (title != null) ...[
                    const SizedBox(height: 2),
                    Text(
                      title!,
                      style: t.title.copyWith(
                        fontSize: compact ? 17.5 : 21,
                        height: 1.2,
                        letterSpacing: (compact ? 17.5 : 21) * -0.02,
                        color: h.tintInk,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            Padding(
              padding: compact
                  ? const EdgeInsets.fromLTRB(20, 10, 20, 18)
                  : const EdgeInsets.fromLTRB(24, 6, 24, 22),
              child: child,
            ),
          ],
        ),
      ),
    );
  }
}

/// A KIND label: CPU, GPU or NPU in a small pill of its tint (or grey).
class CKindLabel extends StatelessWidget {
  const CKindLabel(
    this.kind, {
    super.key,
    this.text,
    this.neutral = false,
    this.small = false,
  });

  final String kind;

  /// What it says; the kind in capitals unless given.
  final String? text;
  final bool neutral;

  /// The phone's 22 px pill.
  final bool small;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = t.kind(kind);
    final size = small ? 10.5 : 11.5;
    // Centred across the pill's height only: as wide as its text, wherever
    // it is placed.
    return Container(
      height: small ? 22 : 24,
      padding: EdgeInsets.symmetric(horizontal: small ? 8 : 10),
      decoration: BoxDecoration(
        color: neutral ? t.surface2 : h.tint,
        borderRadius: BorderRadius.circular(CP.rPill),
      ),
      child: Center(
        widthFactor: 1,
        child: Text(
          text ?? kind.toUpperCase(),
          style: t.caps.copyWith(
            fontSize: size,
            letterSpacing: size * (small ? 0.06 : 0.08),
            height: 1,
            color: neutral ? t.ink : h.accent,
          ),
        ),
      ),
    );
  }
}

/// A small pill tag: a status (Imported, Cancelled) or a name (a backend).
class CTag extends StatelessWidget {
  const CTag({
    super.key,
    required this.text,
    this.color,
    this.filled = false,
    this.upper = true,
    this.fill,
  });

  final String text;

  /// The text's colour (and, softly, the fill's, unless [fill] is given).
  final Color? color;

  /// A solid pill of [color] (ink unless given) with inverse text.
  final bool filled;

  /// Capitals with tracking, as a KIND label.  Names with meaningful casing
  /// (`oneAPI`, `ROCm`) and sentence-case statuses pass false.
  final bool upper;

  /// The pill's fill (`warnTint` for a warning).
  final Color? fill;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final c = color ?? t.ink;
    final Color bg, fg;
    if (filled) {
      bg = c;
      fg = t.inkOn;
    } else {
      bg = fill ??
          (color == null
              ? t.surface2
              : c.withValues(alpha: t.isDark ? 0.2 : 0.12));
      fg = c;
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(CP.rPill),
      ),
      child: Text(
        upper ? text.toUpperCase() : text,
        style: upper
            ? t.caps.copyWith(color: fg, height: 1.2)
            : t.small.copyWith(
                fontSize: 11.5, fontWeight: FontWeight.w700, color: fg),
      ),
    );
  }
}

/// A count in a small pill (a nav item's, a tab's).
class CCount extends StatelessWidget {
  const CCount(this.text, {super.key, this.fill, this.color});

  final String text;

  /// `surface2` unless given.
  final Color? fill;

  /// `muted` unless given.
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      height: 22,
      constraints: const BoxConstraints(minWidth: 24),
      padding: const EdgeInsets.symmetric(horizontal: 7),
      decoration: BoxDecoration(
        color: fill ?? t.surface2,
        borderRadius: BorderRadius.circular(CP.rPill),
      ),
      child: Center(
        widthFactor: 1,
        child: Text(
          text,
          style: t.small.copyWith(
            fontWeight: FontWeight.w700,
            height: 1,
            color: color ?? t.muted,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
      ),
    );
  }
}

/// The floating ink bar of a pick in progress, with the app's one shadow.
class CFloatingBar extends StatelessWidget {
  const CFloatingBar({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.fromLTRB(22, 10, 10, 10),
  });

  final Widget child;
  final EdgeInsetsGeometry padding;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: t.ink,
        borderRadius: BorderRadius.circular(32),
        boxShadow: t.barShadow,
      ),
      child: DefaultTextStyle.merge(
        style: t.label.copyWith(fontWeight: FontWeight.w600, color: t.inkOn),
        child: child,
      ),
    );
  }
}

// ── Buttons ────────────────────────────────────────────────────────────────

enum CButtonKind {
  /// An ink-filled pill with inverse text: the screen's main action.
  primary,

  /// A pill outlined in 1.5 px of ink: a secondary action.
  ghost,

  /// Underlined text: a quiet action.
  quiet,
}

/// A pill button, or an underlined text button: 44 px tall, 56 when stretched.
class CButton extends StatelessWidget {
  const CButton({
    super.key,
    required this.label,
    this.icon,
    this.glyph,
    this.onPressed,
    this.kind = CButtonKind.ghost,
    this.danger = false,
    this.stretch = false,
  });

  final String label;

  /// A leading icon: a Material one, or a [CGlyphs] path.
  final IconData? icon;
  final String? glyph;
  final VoidCallback? onPressed;
  final CButtonKind kind;

  /// A destructive action, in the danger colour.
  final bool danger;

  /// Fill the width given by the parent and centre the label, for a button
  /// that stands on its own instead of sitting in a row of actions.
  final bool stretch;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final enabled = onPressed != null;

    return CTap(
      onTap: onPressed,
      focusRadius: BorderRadius.circular(CP.rPill),
      builder: (context, hovered, pressed) {
        Color fg;
        Color bg = Colors.transparent;
        Color? border;
        switch (kind) {
          case CButtonKind.primary:
            final fill = danger ? t.danger : t.ink;
            fg = enabled ? t.inkOn : t.muted;
            bg = !enabled
                ? t.surface2
                : pressed
                    ? fill.withValues(alpha: 0.78)
                    : hovered
                        ? fill.withValues(alpha: 0.88)
                        : fill;
          case CButtonKind.ghost:
            fg = !enabled ? t.muted : (danger ? t.danger : t.ink);
            border = !enabled ? t.line : fg;
            if (pressed) {
              bg = t.surface2;
            } else if (hovered) {
              bg = t.surface2.withValues(alpha: 0.6);
            }
          case CButtonKind.quiet:
            fg = !enabled ? t.muted : (danger ? t.danger : t.ink);
            if (pressed) {
              bg = t.surface2;
            } else if (hovered) {
              bg = t.surface2.withValues(alpha: 0.6);
            }
        }
        final style = (stretch ? t.label.copyWith(fontSize: 16) : t.label)
            .copyWith(color: fg);
        final text = kind == CButtonKind.quiet
            // An underline set off from the text, as the design draws it.
            ? Container(
                padding: const EdgeInsets.only(bottom: 1),
                decoration: BoxDecoration(
                  border: Border(bottom: BorderSide(color: fg, width: 1.5)),
                ),
                child: Text(label, style: style),
              )
            : Text(label, style: style);
        final hasIcon = icon != null || glyph != null;
        return Container(
          constraints: BoxConstraints(
              minHeight: stretch ? 56 : 44, minWidth: 44),
          padding: EdgeInsets.fromLTRB(
              kind == CButtonKind.quiet ? 14 : (hasIcon ? 16 : 20),
              0,
              kind == CButtonKind.quiet ? 14 : 20,
              0),
          decoration: BoxDecoration(
            color: bg,
            border: border == null
                ? null
                : Border.all(color: border, width: 1.5),
            borderRadius: BorderRadius.circular(CP.rPill),
          ),
          child: Row(
            mainAxisSize: stretch ? MainAxisSize.max : MainAxisSize.min,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (glyph != null) ...[
                CGlyph(glyph!,
                    size: 18,
                    color: fg,
                    stroke: 2.1,
                    fill: glyph == CGlyphs.play),
                const SizedBox(width: 8),
              ] else if (icon != null) ...[
                Icon(icon, size: 18, color: fg),
                const SizedBox(width: 8),
              ],
              Flexible(child: text),
            ],
          ),
        );
      },
    );
  }
}

/// A round 44 px icon button, bare or [filled] with `surface2`.
class CIconButton extends StatelessWidget {
  const CIconButton({
    super.key,
    this.icon,
    this.glyph,
    required this.tooltip,
    this.onPressed,
    this.filled = false,
    this.stroke = 2.1,
  }) : assert(icon != null || glyph != null);

  /// A Material icon, or a [CGlyphs] path.
  final IconData? icon;
  final String? glyph;

  /// Shown on hover and read as the button's name.
  final String tooltip;
  final VoidCallback? onPressed;
  final bool filled;
  final double stroke;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final enabled = onPressed != null;
    final fg = !enabled
        ? t.muted.withValues(alpha: 0.5)
        : (filled ? t.ink : t.muted);
    return Tooltip(
      message: tooltip,
      excludeFromSemantics: true,
      child: CTap(
        onTap: onPressed,
        semanticLabel: tooltip,
        focusRadius: BorderRadius.circular(CP.rPill),
        builder: (context, hovered, pressed) => Container(
          width: 44,
          height: 44,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: filled
                ? (pressed || hovered ? _wash(t, t.surface2) : t.surface2)
                : (pressed
                    ? t.surface2
                    : hovered
                        ? t.surface2.withValues(alpha: 0.6)
                        : Colors.transparent),
          ),
          child: glyph != null
              ? CGlyph(glyph!,
                  size: 20,
                  color: fg,
                  stroke: stroke,
                  fill: glyph == CGlyphs.more || glyph == CGlyphs.play)
              : Icon(icon, size: 20, color: fg),
        ),
      ),
    );
  }
}

/// The round back button of a phone header.
class CBackButton extends StatelessWidget {
  const CBackButton({super.key, required this.onPressed, this.tooltip = 'Back'});

  final VoidCallback? onPressed;
  final String tooltip;

  @override
  Widget build(BuildContext context) => CIconButton(
        glyph: CGlyphs.back,
        tooltip: tooltip,
        onPressed: onPressed,
        filled: true,
        stroke: 2.3,
      );
}

// ── Selection controls ─────────────────────────────────────────────────────

/// A pill switch: an ink track when on, a grey one when off.
class CSwitch extends StatelessWidget {
  const CSwitch({
    super.key,
    required this.value,
    required this.onChanged,
    this.semanticLabel,
  });

  final bool value;
  final ValueChanged<bool>? onChanged;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final enabled = onChanged != null;
    return CTap(
      onTap: enabled ? () => onChanged!(!value) : null,
      button: false,
      toggled: value,
      semanticLabel: semanticLabel,
      focusRadius: BorderRadius.circular(CP.rPill),
      builder: (context, hovered, pressed) => SizedBox(
        width: 60,
        height: 44,
        child: Center(
          child: Opacity(
            opacity: enabled ? 1 : 0.45,
            child: Container(
              width: 48,
              height: 28,
              padding: const EdgeInsets.all(3),
              decoration: BoxDecoration(
                color: value
                    ? t.ink
                    : (hovered ? _wash(t, t.surface2) : t.surface2),
                borderRadius: BorderRadius.circular(CP.rPill),
              ),
              alignment:
                  value ? Alignment.centerRight : Alignment.centerLeft,
              child: Container(
                width: 22,
                height: 22,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: value ? t.inkOn : t.ground,
                  // The knob's own edge, as the design draws it: a white
                  // knob on a pale track needs it to be seen.
                  boxShadow: const [
                    BoxShadow(
                        color: Color(0x40000000),
                        offset: Offset(0, 1),
                        blurRadius: 3),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// A rounded checkbox, ink with a check when on, on a 44 px target.
class CCheckbox extends StatelessWidget {
  const CCheckbox({
    super.key,
    required this.value,
    required this.onChanged,
    this.semanticLabel,
  });

  final bool value;
  final ValueChanged<bool>? onChanged;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final enabled = onChanged != null;
    return CTap(
      onTap: enabled ? () => onChanged!(!value) : null,
      button: false,
      checked: value,
      semanticLabel: semanticLabel,
      focusRadius: BorderRadius.circular(CP.rHighlight),
      builder: (context, hovered, pressed) => SizedBox.square(
        dimension: 44,
        child: Center(
          child: Opacity(
            opacity: enabled ? 1 : 0.45,
            child: Container(
              width: 22,
              height: 22,
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: value
                    ? t.ink
                    : (hovered ? t.surface2 : Colors.transparent),
                border: Border.all(
                    color: value ? t.ink : t.muted, width: 2),
                borderRadius: BorderRadius.circular(7),
              ),
              child: value
                  ? CGlyph(CGlyphs.check, size: 16, color: t.inkOn, stroke: 3)
                  : null,
            ),
          ),
        ),
      ),
    );
  }
}

/// A filter chip with an optional count: ink when selected, grey when not.
class CChip extends StatelessWidget {
  const CChip({
    super.key,
    required this.label,
    required this.selected,
    required this.onTap,
    this.color,
    this.icon,
    this.count,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;

  /// The selected fill, ink unless given.
  final Color? color;
  final IconData? icon;
  final String? count;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final on = color ?? t.ink;
    return CTap(
      onTap: onTap,
      selected: selected,
      focusRadius: BorderRadius.circular(CP.rPill),
      builder: (context, hovered, pressed) {
        final fg = selected ? t.inkOn : t.ink;
        return Container(
          constraints: const BoxConstraints(minHeight: 44),
          padding: EdgeInsets.fromLTRB(16, 8, count == null ? 16 : 14, 8),
          decoration: BoxDecoration(
            color: selected
                ? on
                : (hovered || pressed ? _wash(t, t.surface2) : t.surface2),
            borderRadius: BorderRadius.circular(CP.rPill),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (icon != null) ...[
                Icon(icon, size: 16, color: fg),
                const SizedBox(width: 7),
              ],
              // Flexible + wrapping, not a fixed-width ellipsis: a chip can
              // carry whatever name a driver reports for its device, and an
              // unflexed Text keeps its intrinsic width and overflows the
              // line the Wrap gave it.  Truncating instead would hide exactly
              // the tail that tells two otherwise identical devices apart, so
              // a long chip grows downwards.
              Flexible(
                child: Text(label,
                    style: t.label.copyWith(fontSize: 14, color: fg)),
              ),
              if (count != null) ...[
                const SizedBox(width: 7),
                Text(count!,
                    style: t.small.copyWith(
                        fontWeight: FontWeight.w600,
                        color: selected ? t.inkOn : t.muted)),
              ],
            ],
          ),
        );
      },
    );
  }
}

/// An outlined device chip, in its kind's tint when selected (or neutral).
class CDeviceChip extends StatelessWidget {
  const CDeviceChip({
    super.key,
    required this.label,
    required this.selected,
    required this.onTap,
    this.kind,
    this.count,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;
  final String? kind;
  final String? count;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = kind == null ? null : t.kind(kind!);
    return CTap(
      onTap: onTap,
      selected: selected,
      focusRadius: BorderRadius.circular(CP.rPill),
      builder: (context, hovered, pressed) {
        final Color bg, fg, edge;
        if (h != null) {
          bg = selected ? h.tint : (hovered ? _wash(t, t.surface) : t.surface);
          fg = selected ? h.tintInk : t.ink;
          edge = selected ? h.accent : t.line;
        } else {
          bg = selected
              ? t.surface2
              : (hovered ? _wash(t, t.surface) : t.surface);
          fg = t.ink;
          edge = selected ? t.ink : t.line;
        }
        return Container(
          constraints: const BoxConstraints(minHeight: 44),
          padding: EdgeInsets.fromLTRB(h != null ? 14 : 16, 8, 18, 8),
          decoration: BoxDecoration(
            color: bg,
            border: Border.all(color: edge, width: h != null ? 1 : 1.5),
            borderRadius: BorderRadius.circular(CP.rPill),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (h != null) ...[
                Container(
                  width: 8,
                  height: 8,
                  decoration:
                      BoxDecoration(color: h.accent, shape: BoxShape.circle),
                ),
                const SizedBox(width: 9),
              ],
              Flexible(
                child: Text(label,
                    style: t.label.copyWith(fontSize: 14, color: fg)),
              ),
              if (count != null) ...[
                const SizedBox(width: 8),
                Text(count!,
                    style: t.small
                        .copyWith(fontWeight: FontWeight.w600, color: t.muted)),
              ],
            ],
          ),
        );
      },
    );
  }
}

/// One option of a [CSegmented].
class CSegment<T> {
  const CSegment(this.value, this.label);

  final T value;
  final String label;
}

/// A segmented control: a grey track, the selected option raised on `segOn`.
class CSegmented<T> extends StatelessWidget {
  const CSegmented({
    super.key,
    required this.segments,
    required this.selected,
    required this.onChanged,
    this.expand = false,
    this.semanticLabel,
  });

  final List<CSegment<T>> segments;
  final T selected;
  final ValueChanged<T> onChanged;

  /// Equal widths across the width given (Theme: System / Light / Dark);
  /// otherwise each option is as wide as its label.
  final bool expand;
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    Widget option(CSegment<T> s) {
      final on = s.value == selected;
      return CTap(
        onTap: () => onChanged(s.value),
        selected: on,
        focusRadius: BorderRadius.circular(CP.rPill),
        builder: (context, hovered, pressed) => Container(
          height: 40,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          decoration: BoxDecoration(
            color: on
                ? t.segOn
                : (hovered
                    ? t.surface.withValues(alpha: t.isDark ? 0.12 : 0.45)
                    : Colors.transparent),
            borderRadius: BorderRadius.circular(CP.rPill),
            boxShadow: on ? t.segShadow : null,
          ),
          child: Center(
            widthFactor: 1,
            child: Text(
              s.label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: t.label
                  .copyWith(fontSize: 14, color: on ? t.ink : t.muted),
            ),
          ),
        ),
      );
    }

    return Semantics(
      container: true,
      label: semanticLabel,
      child: Container(
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          color: t.surface2,
          borderRadius: BorderRadius.circular(CP.rPill),
        ),
        child: Row(
          mainAxisSize: expand ? MainAxisSize.max : MainAxisSize.min,
          children: [
            for (var i = 0; i < segments.length; i++) ...[
              if (i > 0) const SizedBox(width: 4),
              expand
                  ? Expanded(child: option(segments[i]))
                  : option(segments[i]),
            ],
          ],
        ),
      ),
    );
  }
}

/// One tab of a [CTabStrip].
class CTab {
  const CTab(this.label, {this.count, this.key});

  final String label;

  /// Shown beside the label (a backend's test count).
  final String? count;

  /// Keys the tab's widget, for a test or a screenshot to find it by.
  final Key? key;
}

/// A tab strip, the selected tab an ink pill; framed, or [scroll] on phones.
class CTabStrip extends StatelessWidget {
  const CTabStrip({
    super.key,
    required this.tabs,
    required this.selected,
    required this.onSelected,
    this.scroll = false,
  });

  final List<CTab> tabs;
  final int selected;
  final ValueChanged<int> onSelected;
  final bool scroll;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    Widget tab(int i) {
      final x = tabs[i];
      final on = i == selected;
      return CTap(
        key: x.key,
        onTap: () => onSelected(i),
        selected: on,
        focusRadius: BorderRadius.circular(CP.rPill),
        builder: (context, hovered, pressed) => Container(
          height: 40,
          padding: x.count == null
              ? const EdgeInsets.symmetric(horizontal: 18)
              : EdgeInsets.fromLTRB(16, 0, scroll ? 14 : 10, 0),
          decoration: BoxDecoration(
            color: on
                ? t.ink
                : scroll
                    ? (hovered ? _wash(t, t.surface2) : t.surface2)
                    : (hovered
                        ? t.surface2.withValues(alpha: 0.6)
                        : Colors.transparent),
            borderRadius: BorderRadius.circular(CP.rPill),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(x.label,
                  style: t.label.copyWith(
                      fontSize: scroll ? 14 : 14.5,
                      color: on ? t.inkOn : t.ink)),
              if (x.count != null) ...[
                SizedBox(width: scroll ? 7 : 8),
                scroll
                    ? Text(x.count!,
                        style: t.small.copyWith(
                            fontWeight: FontWeight.w600,
                            color: on ? t.inkOn : t.muted))
                    : CCount(x.count!,
                        fill: on ? t.countOn : t.surface2,
                        color: on ? t.inkOn : t.muted),
              ],
            ],
          ),
        ),
      );
    }

    if (scroll) {
      return SingleChildScrollView(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.fromLTRB(16, 0, 16, 2),
        child: Row(
          children: [
            for (var i = 0; i < tabs.length; i++) ...[
              if (i > 0) const SizedBox(width: 6),
              tab(i),
            ],
          ],
        ),
      );
    }
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(5),
      decoration: BoxDecoration(
        color: t.surface,
        border: Border.all(color: t.line),
        borderRadius: BorderRadius.circular(CP.rCardLarge),
      ),
      child: Wrap(
        spacing: 4,
        runSpacing: 4,
        children: [for (var i = 0; i < tabs.length; i++) tab(i)],
      ),
    );
  }
}

/// A pill search field on the grey track, with its magnifier.
class CSearchField extends StatelessWidget {
  const CSearchField({
    super.key,
    this.controller,
    this.focusNode,
    this.hint = 'Search',
    this.onChanged,
    this.semanticLabel,
  });

  final TextEditingController? controller;
  final FocusNode? focusNode;
  final String hint;
  final ValueChanged<String>? onChanged;

  /// The field's name for a screen reader, where the hint does not say it.
  final String? semanticLabel;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final style = t.body.copyWith(fontSize: 14.5);
    return Container(
      height: 46,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: t.surface2,
        borderRadius: BorderRadius.circular(CP.rPill),
      ),
      child: Row(
        children: [
          CGlyph(CGlyphs.search, size: 18, color: t.muted, stroke: 2.1),
          const SizedBox(width: 10),
          Expanded(
            child: Semantics(
              label: semanticLabel,
              child: TextField(
                controller: controller,
                focusNode: focusNode,
                onChanged: onChanged,
                style: style,
                cursorColor: t.ink,
                cursorWidth: 1.5,
                textInputAction: TextInputAction.search,
                decoration: InputDecoration.collapsed(
                  hintText: hint,
                  hintStyle: style.copyWith(color: t.muted),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ── Data display ───────────────────────────────────────────────────────────

/// A flat fraction bar with round ends, never indeterminate.
class CMeter extends StatelessWidget {
  const CMeter({super.key, required this.fraction, required this.color});

  final double fraction;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      height: 6,
      decoration: BoxDecoration(
        color: t.surface2,
        borderRadius: BorderRadius.circular(3),
      ),
      alignment: Alignment.centerLeft,
      child: FractionallySizedBox(
        widthFactor: fraction.clamp(0.0, 1.0),
        child: Container(
          decoration: BoxDecoration(
            color: color,
            borderRadius: BorderRadius.circular(3),
          ),
        ),
      ),
    );
  }
}

/// A value and its unit in tabular figures, optionally on a tint [highlight].
class CValue extends StatelessWidget {
  const CValue({
    super.key,
    required this.value,
    required this.unit,
    this.color,
    this.small = false,
    this.size,
    this.weight,
    this.unitColor,
    this.highlight,
    this.padded = false,
  });

  final String value;
  final String unit;

  /// The number's colour, ink unless given (a kind card's ink).
  final Color? color;

  /// A table cell's size (14.5) rather than a row's (22).
  final bool small;

  /// The number's size, overriding [small]: 44 for a flagship number, 17
  /// for a compare row's.
  final double? size;
  final FontWeight? weight;

  /// The unit's colour, muted unless given (a kind's accent).
  final Color? unitColor;
  final Color? highlight;

  /// Keep the highlight's padding without one, so the values of a column
  /// line up whether or not they are highlighted.  The padding is
  /// outdented: the number itself lines up with the text above and below.
  final bool padded;

  static const _padX = 9.0;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final s = size ?? (small ? 14.5 : 22);
    final w = highlight != null
        ? FontWeight.w800
        : (weight ?? (s >= 20 ? FontWeight.w800 : FontWeight.w700));
    final unitSize = s >= 40 ? 15.0 : (s >= 20 ? 13.0 : 12.5);
    final text = Text.rich(
      TextSpan(children: [
        TextSpan(
          text: value,
          style: t.figure(s, weight: w, color: color ?? t.ink),
        ),
        if (unit.isNotEmpty)
          TextSpan(
            text: s >= 40 ? '  $unit' : ' $unit',
            style: t.small.copyWith(
              fontSize: unitSize,
              fontWeight: s >= 20 ? FontWeight.w700 : FontWeight.w600,
              color: unitColor ?? t.muted,
            ),
          ),
      ]),
      maxLines: 1,
    );
    if (highlight == null && !padded) return text;
    return Transform.translate(
      offset: const Offset(-_padX, 0),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: _padX, vertical: 3),
        decoration: BoxDecoration(
          color: highlight,
          borderRadius: BorderRadius.circular(CP.rHighlight),
        ),
        child: text,
      ),
    );
  }
}

/// A value tile with a caption: on the well, or on a run's tint when better.
class CValueTile extends StatelessWidget {
  const CValueTile({
    super.key,
    required this.value,
    required this.unit,
    this.caption,
    this.highlight,
    this.weight,
    this.size = 26,
  });

  final String value;
  final String unit;
  final String? caption;
  final Color? highlight;

  /// 600 unless given; 800 with a [highlight].
  final FontWeight? weight;
  final double size;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 9, 12, 10),
      decoration: BoxDecoration(
        color: highlight ?? t.well,
        borderRadius: BorderRadius.circular(CP.rRow),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          CValue(
            value: value,
            unit: unit,
            size: size,
            weight: weight ??
                (highlight != null ? FontWeight.w800 : FontWeight.w600),
            unitColor: t.ink,
          ),
          if (caption != null && caption!.isNotEmpty)
            Text(caption!, style: t.smallMuted),
        ],
      ),
    );
  }
}

/// A compare's run letter (A or B) on its run's colour, in five sizes.
enum CBadgeSize {
  xs(18, 10),
  sm(20, 10.5),
  md(22, 11.5),
  lg(30, 14),
  xl(36, 16);

  const CBadgeSize(this.diameter, this.fontSize);

  final double diameter;
  final double fontSize;
}

/// A round letter badge: A on run A's blue, B on run B's orange.
class CLetterBadge extends StatelessWidget {
  const CLetterBadge(
    this.letter, {
    super.key,
    this.size = CBadgeSize.sm,
    this.hue,
  });

  final String letter;
  final CBadgeSize size;
  final RunHue? hue;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = hue ?? t.run(letter);
    return Container(
      width: size.diameter,
      height: size.diameter,
      alignment: Alignment.center,
      decoration: BoxDecoration(color: h.fill, shape: BoxShape.circle),
      child: Text(
        letter,
        style: t.small.copyWith(
          fontSize: size.fontSize,
          fontWeight: FontWeight.w800,
          height: 1,
          color: h.onFill,
        ),
      ),
    );
  }
}

/// One row of a card's list, with its hairline under it.
class CRow extends StatelessWidget {
  const CRow({
    super.key,
    required this.child,
    this.onTap,
    this.padding = const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
    this.rule = true,
    this.accent,
  });

  final Widget child;
  final VoidCallback? onTap;
  final EdgeInsetsGeometry padding;
  final bool rule;

  /// An optional mark down the row's left edge.
  final Color? accent;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onTap,
      button: onTap != null,
      insetFocus: true,
      focusRadius: BorderRadius.zero,
      cursor: onTap == null ? MouseCursor.defer : SystemMouseCursors.click,
      builder: (context, hovered, pressed) => Container(
        decoration: BoxDecoration(
          color: hovered || pressed ? _wash(t, t.surface) : t.surface,
          border: Border(
            bottom: rule ? BorderSide(color: t.line) : BorderSide.none,
            left: accent == null
                ? BorderSide.none
                : BorderSide(color: accent!, width: 3),
          ),
        ),
        padding: padding,
        child: child,
      ),
    );
  }
}

// ── Screen chrome ──────────────────────────────────────────────────────────

/// A page's head: a big title and subtitle, a round back button, the actions.
class CHeader extends StatelessWidget {
  const CHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.onBack,
    this.actions = const [],
  });

  final String title;
  final String? subtitle;
  final VoidCallback? onBack;
  final List<Widget> actions;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final heading = Semantics(
      header: true,
      headingLevel: 1,
      child: Text(
        title,
        style: wide ? t.h1 : t.h1Phone,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
      ),
    );
    final titleBlock = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        heading,
        if (subtitle != null && subtitle!.isNotEmpty) ...[
          const SizedBox(height: 4),
          Text(subtitle!,
              style: t.caption.copyWith(fontSize: wide ? 15 : 14),
              maxLines: 2,
              overflow: TextOverflow.ellipsis),
        ],
      ],
    );
    final actionRow = Wrap(
      spacing: 8,
      runSpacing: 8,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: actions,
    );

    if (wide) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(34, 30, 34, 12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            if (onBack != null) ...[
              CBackButton(onPressed: onBack),
              const SizedBox(width: 16),
            ],
            Expanded(child: titleBlock),
            if (actions.isNotEmpty) ...[
              const SizedBox(width: 20),
              actionRow,
            ],
          ],
        ),
      );
    }
    if (onBack == null) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(24, 14, 16, 8),
        child: Row(
          children: [
            Expanded(child: titleBlock),
            if (actions.isNotEmpty) ...[
              const SizedBox(width: 8),
              actionRow,
            ],
          ],
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              CBackButton(onPressed: onBack),
              const Spacer(),
              actionRow,
            ],
          ),
          const SizedBox(height: 16),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: titleBlock,
          ),
        ],
      ),
    );
  }
}

/// Instant (unanimated) disclosure block.
class CExpander extends StatefulWidget {
  const CExpander({
    super.key,
    required this.header,
    required this.child,
    this.initiallyOpen = false,
  });

  /// Built with the current open state so the caller can flip its own marker.
  final Widget Function(BuildContext context, bool open) header;
  final Widget child;
  final bool initiallyOpen;

  @override
  State<CExpander> createState() => _CExpanderState();
}

class _CExpanderState extends State<CExpander> {
  late bool _open = widget.initiallyOpen;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CTap(
          onTap: () => setState(() => _open = !_open),
          expanded: _open,
          insetFocus: true,
          builder: (context, hovered, pressed) =>
              widget.header(context, _open),
        ),
        if (_open) widget.child,
      ],
    );
  }
}

/// A fold row (badge, label, count, chevron) that opens beneath it, instantly.
class CFold extends StatefulWidget {
  const CFold({
    super.key,
    required this.label,
    this.leading,
    this.count,
    required this.child,
    this.initiallyOpen = false,
  });

  final String label;

  /// Before the label: a [CLetterBadge] ("Only on A").
  final Widget? leading;
  final String? count;
  final Widget child;
  final bool initiallyOpen;

  @override
  State<CFold> createState() => _CFoldState();
}

class _CFoldState extends State<CFold> {
  late bool _open = widget.initiallyOpen;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CTap(
          onTap: () => setState(() => _open = !_open),
          expanded: _open,
          focusRadius: BorderRadius.circular(CP.rPanel),
          builder: (context, hovered, pressed) => Container(
            constraints: const BoxConstraints(minHeight: 54),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: hovered || pressed ? _wash(t, t.surface) : t.surface,
              border: Border.all(color: t.line),
              borderRadius: BorderRadius.circular(CP.rPanel),
            ),
            child: Row(
              children: [
                if (widget.leading != null) ...[
                  widget.leading!,
                  const SizedBox(width: 10),
                ],
                Expanded(
                  child: Text(widget.label,
                      style: t.label.copyWith(fontSize: 14)),
                ),
                if (widget.count != null) ...[
                  const SizedBox(width: 8),
                  Text(widget.count!,
                      style: t.caption.copyWith(fontSize: 12.5)),
                ],
                const SizedBox(width: 8),
                Transform.rotate(
                  angle: _open ? math.pi : 0,
                  child: CGlyph(CGlyphs.chevronDown,
                      size: 18, color: t.ink, stroke: 2.2),
                ),
              ],
            ),
          ),
        ),
        if (_open) ...[
          const SizedBox(height: 8),
          widget.child,
        ],
      ],
    );
  }
}

/// Empty / waiting state: a centred title and a muted line, no spinner.
class CEmpty extends StatelessWidget {
  const CEmpty({super.key, required this.title, this.detail, this.icon});

  final String title;
  final String? detail;
  final IconData? icon;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[
              Icon(icon, size: 28, color: t.muted),
              const SizedBox(height: 14),
            ],
            Text(title,
                style: t.title.copyWith(fontSize: 16),
                textAlign: TextAlign.center),
            if (detail != null) ...[
              const SizedBox(height: 6),
              Text(detail!,
                  style: t.prose, textAlign: TextAlign.center),
            ],
          ],
        ),
      ),
    );
  }
}

/// An empty state in a dashed box: "No saved run matches."
class CEmptyBox extends StatelessWidget {
  const CEmptyBox({
    super.key,
    this.text,
    this.child,
    this.padding = const EdgeInsets.symmetric(horizontal: 20, vertical: 28),
    this.radius = CP.rCard,
    this.center = true,
  }) : assert(text != null || child != null);

  final String? text;
  final Widget? child;
  final EdgeInsetsGeometry padding;
  final double radius;

  /// Centre [text]; a box listing things sets it left.
  final bool center;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CustomPaint(
      painter: _DashedBorder(color: t.line, radius: radius),
      child: Padding(
        padding: padding,
        child: child ??
            Text(
              text!,
              textAlign: center ? TextAlign.center : TextAlign.start,
              style: t.bodyMuted.copyWith(fontSize: 14.5),
            ),
      ),
    );
  }
}

class _DashedBorder extends CustomPainter {
  const _DashedBorder({required this.color, required this.radius});

  final Color color;
  final double radius;

  @override
  void paint(Canvas canvas, Size size) {
    final outline = Path()
      ..addRRect(RRect.fromRectAndRadius(
          (Offset.zero & size).deflate(0.5), Radius.circular(radius)));
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    const dash = 4.0, gap = 3.0;
    for (final metric in outline.computeMetrics()) {
      for (var d = 0.0; d < metric.length; d += dash + gap) {
        canvas.drawPath(
            metric.extractPath(d, math.min(d + dash, metric.length)), paint);
      }
    }
  }

  @override
  bool shouldRepaint(_DashedBorder old) =>
      old.color != color || old.radius != radius;
}

/// A dialog in the app's frame: a sentence-case title, actions at the foot.
class CDialog extends StatelessWidget {
  const CDialog({
    super.key,
    required this.title,
    required this.child,
    required this.actions,
  });

  final String title;
  final Widget child;
  final List<Widget> actions;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Dialog(
      backgroundColor: t.surface,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      insetPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(CP.rCardLarge),
        side: BorderSide(color: t.line),
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 440),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 22, 24, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Semantics(
                header: true,
                child: Text(title, style: t.h2),
              ),
              const SizedBox(height: 14),
              Flexible(child: child),
              const SizedBox(height: 20),
              Wrap(
                alignment: WrapAlignment.end,
                spacing: 8,
                runSpacing: 8,
                children: actions,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
