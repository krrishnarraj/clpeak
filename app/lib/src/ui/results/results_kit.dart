/// Pieces the results screens share that the kit (`ui/common/kit.dart`)
/// does not have, built on its tokens.  Kept here, beside their only users;
/// one that another screen needs can move to the kit as it is.
library;

import 'package:flutter/material.dart';

import '../../model/run_document.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/kit.dart';

/// A device's kind as its hue knows it: its processor's where the file links
/// one, else the type its own API reported.  Empty where neither says.
String deviceKindOf(RunDocument doc, DeviceRun run) {
  final p = doc.processorById(run.processor);
  if (p != null && p.kind.isNotEmpty) return p.kind;
  return run.type;
}

/// The first sentence of a test's or a reading's description, for the line
/// under its name; the whole text is a tap on the info glyph away.  The
/// native text writes a dash as " -- ", shown here as one.
String descriptionPreview(String text) {
  final s = text.trim().replaceAll(' -- ', ' — ');
  if (s.isEmpty) return '';
  final end = RegExp(r'[.;](\s|$)').firstMatch(s);
  final first = end == null ? s : s.substring(0, end.start);
  return first.trim();
}

/// What hover and press do to a row or a pill: one flat step towards
/// `surface2`, as the kit's own controls wash.
Color hoverWash(CP t, Color over) =>
    Color.alphaBlend(t.surface2.withValues(alpha: t.isDark ? 0.7 : 0.6), over);

/// Facts as key over value -- a device's properties, the system's -- in a
/// wrapping flow on a wide window and two columns on a phone.  On a kind
/// card, [hue] gives the card's own ink.  Values wrap, never ellipsize: a
/// driver version's tail is what tells two installs apart.
class FactGrid extends StatelessWidget {
  const FactGrid({
    super.key,
    required this.entries,
    this.hue,
    this.compact = false,
  });

  final List<(String, String)> entries;
  final KindHue? hue;

  /// The phone's two columns and smaller type.
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final keyStyle = t.small.copyWith(
        fontSize: compact ? 12 : 12.5, color: hue?.tintSub ?? t.muted);
    final valueStyle = t.title.copyWith(
        fontSize: compact ? 14 : 15, color: hue?.tintInk ?? t.ink);
    Widget fact((String, String) e) => Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(e.$1, style: keyStyle),
            const SizedBox(height: 1),
            Text(e.$2, style: valueStyle),
          ],
        );
    if (!compact) {
      return Wrap(
        spacing: 36,
        runSpacing: 12,
        children: [for (final e in entries) fact(e)],
      );
    }
    return LayoutBuilder(builder: (context, c) {
      final column = ((c.maxWidth - 16) / 2).floorToDouble();
      return Wrap(
        spacing: 16,
        runSpacing: 10,
        children: [
          for (final e in entries) SizedBox(width: column, child: fact(e)),
        ],
      );
    });
  }
}

/// A quiet pill: a 1 px hairline on the surface, for an action beside a
/// heading that is neither the screen's main one nor a secondary one (Expand
/// all, Collapse all).
class QuietPill extends StatelessWidget {
  const QuietPill({
    super.key,
    required this.label,
    this.glyph,
    this.onPressed,
    this.stretch = false,
  });

  final String label;

  /// A [CGlyphs] path before the label.
  final String? glyph;
  final VoidCallback? onPressed;

  /// Fill the width given and centre the label.
  final bool stretch;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final enabled = onPressed != null;
    return CTap(
      onTap: onPressed,
      focusRadius: BorderRadius.circular(CP.rPill),
      builder: (context, hovered, pressed) {
        final fg = enabled ? t.ink : t.muted;
        return Container(
          height: 44,
          padding: EdgeInsets.fromLTRB(glyph == null ? 16 : 12, 0, 16, 0),
          decoration: BoxDecoration(
            color: pressed
                ? t.surface2
                : hovered
                    ? hoverWash(t, t.surface)
                    : t.surface,
            border: Border.all(color: t.line),
            borderRadius: BorderRadius.circular(CP.rPill),
          ),
          child: Row(
            mainAxisSize: stretch ? MainAxisSize.max : MainAxisSize.min,
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (glyph != null) ...[
                CGlyph(glyph!, size: 18, color: fg, stroke: 2.2),
                const SizedBox(width: 6),
              ],
              Flexible(
                child: Text(label,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: t.label.copyWith(fontSize: 13.5, color: fg)),
              ),
            ],
          ),
        );
      },
    );
  }
}

/// A heading over a group of rows: its name in the h3 weight and a muted
/// count on the same baseline ("Compute  11 tests").
class GroupHeading extends StatelessWidget {
  const GroupHeading({
    super.key,
    required this.label,
    this.count,
    this.wide = true,
    this.muted = false,
  });

  final String label;
  final String? count;
  final bool wide;

  /// For a group of what was not measured.
  final bool muted;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: wide ? 6 : 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.baseline,
        textBaseline: TextBaseline.alphabetic,
        children: [
          Flexible(
            child: Semantics(
              header: true,
              headingLevel: 3,
              child: Text(
                label,
                style: t.h3.copyWith(
                  fontSize: wide ? 17 : 16,
                  color: muted ? t.muted : t.ink,
                ),
              ),
            ),
          ),
          if (count != null && count!.isNotEmpty) ...[
            SizedBox(width: wide ? 10 : 8),
            Text(count!,
                style: t.caption.copyWith(fontSize: wide ? 13 : 12.5)),
          ],
        ],
      ),
    );
  }
}

/// "3 tests", "1 test".
String countOf(int n, String one, [String? many]) =>
    '$n ${n == 1 ? one : (many ?? '${one}s')}';
