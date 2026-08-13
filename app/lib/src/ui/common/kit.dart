/// The clpeak widget kit.
///
/// Deliberately built on raw Flutter primitives (`Container`, `GestureDetector`,
/// `MouseRegion`) rather than Material's `Card` / `Chip` / `Switch` /
/// `NavigationRail`, so the app reads as an instrument console instead of a
/// stock Material app.  Two consequences worth knowing:
///
///   * **Nothing here animates.**  Press and hover are flat colour swaps, not
///     ink splashes, and disclosure is instant rather than a cross-fade.  Every
///     animated frame is GPU work stolen from the benchmark running underneath
///     (see `app/AGENTS.md`), so the design and the measurement agree here.
///   * **Hit targets are explicit.**  `GestureDetector` + `MouseRegion` give
///     the same pointer behaviour as `InkWell` minus the ripple.
library;

import 'package:flutter/material.dart';

import '../../theme/clpeak_theme.dart';

// ── Pressable base ─────────────────────────────────────────────────────────

/// Hover/press state without an ink splash.  Rebuilds only itself.
class CTap extends StatefulWidget {
  const CTap({
    super.key,
    required this.builder,
    this.onTap,
    this.cursor = SystemMouseCursors.click,
  });

  final Widget Function(BuildContext context, bool hovered, bool pressed)
      builder;
  final VoidCallback? onTap;
  final MouseCursor cursor;

  @override
  State<CTap> createState() => _CTapState();
}

class _CTapState extends State<CTap> {
  bool _hovered = false;
  bool _pressed = false;

  @override
  Widget build(BuildContext context) {
    final enabled = widget.onTap != null;
    return MouseRegion(
      cursor: enabled ? widget.cursor : MouseCursor.defer,
      onEnter: enabled ? (_) => setState(() => _hovered = true) : null,
      onExit: enabled ? (_) => setState(() => _hovered = false) : null,
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: widget.onTap,
        onTapDown: enabled ? (_) => setState(() => _pressed = true) : null,
        onTapUp: enabled ? (_) => setState(() => _pressed = false) : null,
        onTapCancel: enabled ? () => setState(() => _pressed = false) : null,
        child: widget.builder(
            context, enabled && _hovered, enabled && _pressed),
      ),
    );
  }
}

// ── Surfaces ───────────────────────────────────────────────────────────────

/// A bordered panel: flat fill, 1px rule, square-ish corners, no shadow.
class CPanel extends StatelessWidget {
  const CPanel({super.key, required this.child, this.padding});

  final Widget child;
  final EdgeInsetsGeometry? padding;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      decoration: BoxDecoration(
        color: t.panel,
        border: Border.all(color: t.line),
        borderRadius: BorderRadius.circular(CP.rPanel),
      ),
      clipBehavior: Clip.antiAlias,
      padding: padding,
      child: child,
    );
  }
}

/// The strip that titles a panel, separated from its body by a hairline.
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
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 9, 12, 9),
      decoration: BoxDecoration(
        color: t.isDark ? t.hover : t.hover.withValues(alpha: 0.7),
        border: Border(bottom: BorderSide(color: t.line)),
      ),
      child: Row(
        children: [
          if (tag != null) ...[
            CTag(text: tag!, color: color, upper: false),
            const SizedBox(width: 10),
          ] else if (icon != null) ...[
            Icon(icon, size: 15, color: color ?? t.dim),
            const SizedBox(width: 9),
          ],
          Expanded(child: Text(title, style: t.title)),
          ?trailing,
        ],
      ),
    );
  }
}

/// `▍ SECTION ─────────────────── trailing` — the app's signature heading.
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
    return Row(
      children: [
        Container(width: 3, height: 12, color: color ?? t.text),
        const SizedBox(width: 9),
        Text(label.toUpperCase(),
            style: t.micro.copyWith(color: color ?? t.text)),
        const SizedBox(width: 12),
        Expanded(child: Container(height: 1, color: t.line)),
        if (trailing != null) ...[
          const SizedBox(width: 12),
          Text(trailing!.toUpperCase(), style: t.micro),
        ],
      ],
    );
  }
}

/// Small uppercase marker — backend tags, status flags.
class CTag extends StatelessWidget {
  const CTag({
    super.key,
    required this.text,
    this.color,
    this.filled = false,
    this.upper = true,
  });

  final String text;
  final Color? color;
  final bool filled;

  /// Backend names (`oneAPI`, `ROCm`) carry meaningful casing — pass false to
  /// keep it.
  final bool upper;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final c = color ?? t.dim;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 2),
      decoration: BoxDecoration(
        color: filled ? c : Colors.transparent,
        border: Border.all(color: filled ? c : c.withValues(alpha: 0.55)),
        borderRadius: BorderRadius.circular(CP.rControl),
      ),
      child: Text(
        upper ? text.toUpperCase() : text,
        style: t.micro.copyWith(
          fontSize: 9.5,
          color: filled ? t.panel : c,
        ),
      ),
    );
  }
}

// ── Buttons ────────────────────────────────────────────────────────────────

enum CButtonKind {
  /// Solid block of the text colour — one per screen.
  primary,

  /// Hairline outline.
  ghost,

  /// No frame until hovered.
  quiet,
}

class CButton extends StatelessWidget {
  const CButton({
    super.key,
    required this.label,
    this.icon,
    this.onPressed,
    this.kind = CButtonKind.ghost,
    this.danger = false,
  });

  final String label;
  final IconData? icon;
  final VoidCallback? onPressed;
  final CButtonKind kind;
  final bool danger;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final enabled = onPressed != null;

    return CTap(
      onTap: onPressed,
      builder: (context, hovered, pressed) {
        final Color fg, bg;
        Color? border;

        switch (kind) {
          case CButtonKind.primary:
            fg = t.onInverse;
            bg = pressed
                ? t.inverse.withValues(alpha: 0.75)
                : hovered
                    ? t.inverse.withValues(alpha: 0.88)
                    : t.inverse;
          case CButtonKind.ghost:
            fg = danger ? t.danger : t.text;
            bg = pressed || hovered ? t.hover : Colors.transparent;
            border = danger ? t.danger.withValues(alpha: 0.5) : t.line;
          case CButtonKind.quiet:
            fg = danger ? t.danger : t.dim;
            bg = pressed || hovered ? t.hover : Colors.transparent;
        }

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
          decoration: BoxDecoration(
            color: enabled ? bg : t.hover.withValues(alpha: 0.5),
            border: border == null ? null : Border.all(color: border),
            borderRadius: BorderRadius.circular(CP.rControl),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (icon != null) ...[
                Icon(icon, size: 13, color: enabled ? fg : t.faint),
                const SizedBox(width: 7),
              ],
              Text(
                label.toUpperCase(),
                style: t.micro.copyWith(color: enabled ? fg : t.faint),
              ),
            ],
          ),
        );
      },
    );
  }
}

/// Square icon-only button (screen header actions).
class CIconButton extends StatelessWidget {
  const CIconButton({
    super.key,
    required this.icon,
    required this.tooltip,
    this.onPressed,
  });

  final IconData icon;
  final String tooltip;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Tooltip(
      message: tooltip,
      child: CTap(
        onTap: onPressed,
        builder: (context, hovered, pressed) => Container(
          width: 30,
          height: 30,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: hovered || pressed ? t.hover : Colors.transparent,
            borderRadius: BorderRadius.circular(CP.rControl),
          ),
          child: Icon(icon,
              size: 16, color: onPressed == null ? t.faint : t.dim),
        ),
      ),
    );
  }
}

// ── Selection controls ─────────────────────────────────────────────────────

/// Square two-state switch — a track with a block that sits left or right.
class CSwitch extends StatelessWidget {
  const CSwitch({super.key, required this.value, required this.onChanged});

  final bool value;
  final ValueChanged<bool>? onChanged;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onChanged == null ? null : () => onChanged!(!value),
      builder: (context, hovered, pressed) => Container(
        width: 34,
        height: 18,
        padding: const EdgeInsets.all(2),
        decoration: BoxDecoration(
          color: value ? t.inverse : (hovered ? t.hover : Colors.transparent),
          border: Border.all(color: value ? t.inverse : t.line),
          borderRadius: BorderRadius.circular(CP.rControl),
        ),
        child: Align(
          alignment: value ? Alignment.centerRight : Alignment.centerLeft,
          child: Container(
            width: 12,
            height: 12,
            color: value ? t.onInverse : t.dim,
          ),
        ),
      ),
    );
  }
}

/// Square checkbox with a hairline frame.
class CCheckbox extends StatelessWidget {
  const CCheckbox({super.key, required this.value, required this.onChanged});

  final bool value;
  final ValueChanged<bool>? onChanged;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onChanged == null ? null : () => onChanged!(!value),
      builder: (context, hovered, pressed) => Container(
        width: 15,
        height: 15,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: value ? t.inverse : (hovered ? t.hover : Colors.transparent),
          border: Border.all(color: value ? t.inverse : t.line),
          borderRadius: BorderRadius.circular(1),
        ),
        child: value
            ? Icon(Icons.check, size: 11, color: t.onInverse)
            : null,
      ),
    );
  }
}

/// Selectable chip — square, hairline, tinted by its category when on.
class CChip extends StatelessWidget {
  const CChip({
    super.key,
    required this.label,
    required this.selected,
    required this.onTap,
    this.color,
    this.icon,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;
  final Color? color;
  final IconData? icon;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final c = color ?? t.text;
    return CTap(
      onTap: onTap,
      builder: (context, hovered, pressed) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
        decoration: BoxDecoration(
          color: selected
              ? c.withValues(alpha: t.isDark ? 0.16 : 0.10)
              : (hovered ? t.hover : Colors.transparent),
          border: Border.all(
              color: selected ? c.withValues(alpha: 0.85) : t.line),
          borderRadius: BorderRadius.circular(CP.rControl),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            // A filled square, not a check mark: selection reads at a glance
            // across the whole row of chips.
            Container(
              width: 7,
              height: 7,
              color: selected ? c : t.faint,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: t.monoSmall.copyWith(
                color: selected ? t.text : t.dim,
                fontWeight: selected ? FontWeight.w600 : FontWeight.w400,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Data display ───────────────────────────────────────────────────────────

/// Flat fraction bar — square ends, static fill, no indeterminate mode.
class CMeter extends StatelessWidget {
  const CMeter({super.key, required this.fraction, required this.color});

  final double fraction;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      height: 5,
      color: t.isDark ? t.hover : t.line,
      child: Align(
        alignment: Alignment.centerLeft,
        child: FractionallySizedBox(
          widthFactor: fraction.clamp(0.0, 1.0),
          child: Container(color: color),
        ),
      ),
    );
  }
}

/// A value with its unit — value in the loud style, unit dimmed beside it.
class CValue extends StatelessWidget {
  const CValue({
    super.key,
    required this.value,
    required this.unit,
    this.color,
    this.small = false,
  });

  final String value;
  final String unit;
  final Color? color;
  final bool small;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Text.rich(
      TextSpan(children: [
        TextSpan(
          text: value,
          style: small
              ? t.mono.copyWith(
                  fontWeight: FontWeight.w600, color: color ?? t.text)
              : t.value.copyWith(color: color ?? t.text),
        ),
        TextSpan(text: ' $unit', style: t.monoSmallDim),
      ]),
      maxLines: 1,
    );
  }
}

/// One hairline-separated table row.  Rows draw their own bottom rule so a
/// lazily-built list still reads as a continuous table.
class CRow extends StatelessWidget {
  const CRow({
    super.key,
    required this.child,
    this.onTap,
    this.padding = const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
    this.rule = true,
    this.accent,
  });

  final Widget child;
  final VoidCallback? onTap;
  final EdgeInsetsGeometry padding;
  final bool rule;

  /// Optional 2px left edge marker (category tint).
  final Color? accent;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onTap,
      cursor: onTap == null ? MouseCursor.defer : SystemMouseCursors.click,
      builder: (context, hovered, pressed) => Container(
        decoration: BoxDecoration(
          color: hovered || pressed ? t.hover : t.panel,
          border: Border(
            bottom: rule
                ? BorderSide(color: t.line)
                : BorderSide.none,
            left: accent == null
                ? BorderSide.none
                : BorderSide(color: accent!, width: 2),
          ),
        ),
        padding: padding,
        child: child,
      ),
    );
  }
}

// ── Screen chrome ──────────────────────────────────────────────────────────

/// Flat screen header — no AppBar fill, no elevation, one hairline underneath.
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
    return Container(
      decoration: BoxDecoration(
        border: Border(bottom: BorderSide(color: t.line)),
      ),
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      child: Row(
        children: [
          if (onBack != null) ...[
            CIconButton(
                icon: Icons.arrow_back,
                tooltip: 'Back',
                onPressed: onBack),
            const SizedBox(width: 8),
          ],
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: t.title, maxLines: 1,
                    overflow: TextOverflow.ellipsis),
                if (subtitle != null) ...[
                  const SizedBox(height: 3),
                  Text(subtitle!.toUpperCase(),
                      style: t.micro, maxLines: 1,
                      overflow: TextOverflow.ellipsis),
                ],
              ],
            ),
          ),
          ...actions,
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
          builder: (context, hovered, pressed) =>
              widget.header(context, _open),
        ),
        if (_open) widget.child,
      ],
    );
  }
}

/// Empty / waiting state: a centred mono line, no spinner.
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
              Icon(icon, size: 28, color: t.faint),
              const SizedBox(height: 14),
            ],
            Text(title, style: t.mono, textAlign: TextAlign.center),
            if (detail != null) ...[
              const SizedBox(height: 6),
              Text(detail!,
                  style: t.monoSmallDim, textAlign: TextAlign.center),
            ],
          ],
        ),
      ),
    );
  }
}

/// Dialog body in the app's frame — square, hairline, mono title.
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
      backgroundColor: t.panel,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(CP.rPanel),
        side: BorderSide(color: t.line),
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
              decoration: BoxDecoration(
                color: t.hover,
                border: Border(bottom: BorderSide(color: t.line)),
              ),
              child: Text(title.toUpperCase(), style: t.microStrong),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
              child: child,
            ),
            Container(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  for (var i = 0; i < actions.length; i++) ...[
                    if (i > 0) const SizedBox(width: 8),
                    actions[i],
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
