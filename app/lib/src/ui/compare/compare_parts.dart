part of 'compare_screen.dart';

// The compare's own pieces.  Built here rather than in the kit because only
// the compare uses them (yet): a slice of a card for lazily built rows, the
// value cell and tile of a verdict, the run cards and their swap button.

/// One entry of the screen's lazily built list: [top] above it, the page's
/// side gutters unless [inset] is false.
class _Item {
  _Item(this.build, {this.top = 0, this.inset = true, this.key});

  final WidgetBuilder build;
  final double top;
  final bool inset;

  /// Keys a row, so a stateful one (a fold) never takes over another's
  /// state when a tab changes what sits at its index.
  final Key? key;
}

/// A hover wash over [over], as the kit's controls use.
Color _wash(CP t, Color over) =>
    Color.alphaBlend(t.surface2.withValues(alpha: t.isDark ? 0.7 : 0.6), over);

/// How the two values of a verdict are set: the better one bold on its
/// run's tint, the other light.  Where the two read the same, were measured
/// another way, or are not like for like ([comparable]), neither is marked.
class _Marks {
  const _Marks._(this.winner);

  /// One side's value alone: nothing to mark.
  const _Marks.none() : winner = null;

  factory _Marks.of(Verdict? v, {bool comparable = true}) => _Marks._(
      v == null || !comparable || v.methodChanged || v.readsSame
          ? null
          : v.winner);

  final CompareSide? winner;

  bool on(CompareSide side) => winner == side;

  FontWeight weight(CompareSide side) => winner == null
      ? FontWeight.w600
      : (winner == side ? FontWeight.w800 : FontWeight.w500);
}

/// A part of a card: one row of a list built lazily, which draws its own
/// share of the card's edge -- the rounded top on the [first], the rounded
/// bottom on the [last], the sides on all, and a hairline above every row
/// but the first where [rule] says.
class _Slice extends StatelessWidget {
  const _Slice({
    required this.child,
    this.first = false,
    this.last = false,
    this.rule = true,
  });

  final Widget child;
  final bool first;
  final bool last;
  final bool rule;

  static const radius = 22.0;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    const r = Radius.circular(radius);
    Widget body = ColoredBox(color: t.surface, child: child);
    if (first || last) {
      body = ClipRRect(
        borderRadius: BorderRadius.vertical(
          top: first ? r : Radius.zero,
          bottom: last ? r : Radius.zero,
        ),
        child: body,
      );
    }
    return CustomPaint(
      foregroundPainter: _SliceEdge(
        color: t.line,
        first: first,
        last: last,
        rule: rule && !first,
      ),
      child: body,
    );
  }
}

class _SliceEdge extends CustomPainter {
  const _SliceEdge({
    required this.color,
    required this.first,
    required this.last,
    required this.rule,
  });

  final Color color;
  final bool first;
  final bool last;
  final bool rule;

  @override
  void paint(Canvas canvas, Size size) {
    const r = _Slice.radius;
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;
    canvas.save();
    canvas.clipRect(Offset.zero & size);
    // The card's outline, its rounding pushed past the ends this slice does
    // not have, so only its sides show there.
    final outline = Rect.fromLTRB(0, first ? 0 : -r - 2, size.width,
        last ? size.height : size.height + r + 2);
    canvas.drawRRect(
        RRect.fromRectAndRadius(outline.deflate(0.5), const Radius.circular(r)),
        paint);
    if (rule) canvas.drawLine(const Offset(0, 0.5), Offset(size.width, 0.5), paint);
    canvas.restore();
  }

  @override
  bool shouldRepaint(_SliceEdge old) =>
      old.color != color ||
      old.first != first ||
      old.last != last ||
      old.rule != rule;
}

/// One side's value in a row: the number, on its run's tint where it is
/// the better one, and what it came from beneath.
class _ValueCell extends StatelessWidget {
  const _ValueCell({
    required this.value,
    required this.side,
    required this.marks,
    this.caption = '',
    this.size = 17,
  });

  final CompareValue? value;
  final CompareSide side;
  final _Marks marks;
  final String caption;
  final double size;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final v = value;
    if (v == null) {
      return Text('—',
          style: t.figure(size, weight: FontWeight.w500, color: t.muted));
    }
    final text = valueText(v);
    final on = marks.on(side);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        CValue(
          value: text.value,
          unit: text.unit,
          size: size,
          weight: marks.weight(side),
          unitColor: t.ink,
          highlight: on ? t.run(side == CompareSide.a ? 'A' : 'B').tint : null,
          padded: true,
        ),
        if (caption.isNotEmpty)
          Text(caption,
              style: t.smallMuted.copyWith(fontSize: size < 17 ? 11.5 : 12)),
      ],
    );
  }
}

/// One side's headline: a value tile, on its run's tint where it is the
/// better one.
class _Tile extends StatelessWidget {
  const _Tile({
    required this.value,
    required this.side,
    required this.marks,
    required this.caption,
    this.size = 26,
  });

  final CompareValue? value;
  final CompareSide side;
  final _Marks marks;
  final String caption;
  final double size;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final v = value;
    if (v == null) {
      return Container(
        padding: const EdgeInsets.fromLTRB(12, 9, 12, 10),
        decoration: BoxDecoration(
          color: t.well,
          borderRadius: BorderRadius.circular(CP.rRow),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('—',
                style:
                    t.figure(size, weight: FontWeight.w600, color: t.muted)),
            if (caption.isNotEmpty) Text(caption, style: t.smallMuted),
          ],
        ),
      );
    }
    final text = valueText(v);
    final on = marks.on(side);
    return CValueTile(
      value: text.value,
      unit: text.unit,
      caption: caption,
      size: size,
      weight: marks.weight(side),
      highlight: on ? t.run(side == CompareSide.a ? 'A' : 'B').tint : null,
    );
  }
}

/// What a headline tile, or a pair of them, shows: `Floating point`.
class _TileLabel extends StatelessWidget {
  const _TileLabel(this.text, {this.small = false});

  final String text;
  final bool small;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 2),
      child: Text(text,
          style: t.caption.copyWith(
              fontSize: small ? 12.5 : 13, fontWeight: FontWeight.w600)),
    );
  }
}

/// A card of headlines, A's value beside B's: a pair's, or one processor's
/// in two runs of one machine ([sameMode]), whose values are marked only
/// where they are readings of one measure ([likeForLike]).
class _HeadCard extends StatelessWidget {
  const _HeadCard({
    required this.headlines,
    required this.sameMode,
    required this.wide,
    this.kind,
    this.title,
    this.caps,
    this.a,
    this.b,
  });

  final List<HeadlineCompare> headlines;
  final bool sameMode;
  final bool wide;

  /// The kind chip at the head (`GPU`).
  final String? kind;

  /// The processor's name beside it.
  final String? title;

  /// A capitals head in place of the kind (the phone's HEADLINES).
  final String? caps;

  /// What each column is: the processor, or the run's day.
  final String? a;
  final String? b;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final tile = wide ? 26.0 : 22.0;
    final columnStyle = t.body
        .copyWith(fontSize: wide ? 13 : 12.5, fontWeight: FontWeight.w700);
    final head = <Widget>[
      if (caps != null)
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 2),
          child: Text(caps!, style: t.capsHead.copyWith(fontSize: 12)),
        ),
      if (kind != null || title != null)
        Wrap(
          spacing: wide ? 10 : 8,
          runSpacing: 6,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            if (kind != null)
              CKindLabel('', text: kind, neutral: true, small: !wide),
            if (title != null)
              Text(title!,
                  style: t.body.copyWith(
                      fontSize: wide ? 14 : 13.5,
                      fontWeight: FontWeight.w700)),
          ],
        ),
      if (a != null && b != null)
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
                child: _Badged('A', a!,
                    style: columnStyle,
                    size: wide ? CBadgeSize.sm : CBadgeSize.xs)),
            SizedBox(width: wide ? 10 : 8),
            Expanded(
                child: _Badged('B', b!,
                    style: columnStyle,
                    size: wide ? CBadgeSize.sm : CBadgeSize.xs)),
          ],
        ),
    ];
    Widget row(int i, HeadlineCompare h) {
      final both = h.a != null && h.b != null;
      final like = !sameMode || !both || likeForLike(h);
      final marks = _Marks.of(h.verdict, comparable: like);
      return Container(
        padding: const EdgeInsets.only(top: 10),
        decoration: BoxDecoration(
          border: Border(
              top: BorderSide(color: i == 0 ? Colors.transparent : t.line)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _TileLabel(headlineName(h.headline), small: !wide),
            const SizedBox(height: 6),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: _Tile(
                    value: h.a,
                    side: CompareSide.a,
                    marks: marks,
                    caption: h.a == null ? 'Not measured' : headCaption(h.a!),
                    size: tile,
                  ),
                ),
                SizedBox(width: wide ? 10 : 8),
                Expanded(
                  child: _Tile(
                    value: h.b,
                    side: CompareSide.b,
                    marks: marks,
                    caption: h.b == null ? 'Not measured' : headCaption(h.b!),
                    size: tile,
                  ),
                ),
              ],
            ),
            if (!like) ...[
              const SizedBox(height: 6),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 2),
                child: Text(
                  'Not like for like: different measures '
                  '(${unlikeWhy(h.a!, h.b!)})',
                  style: t.caption.copyWith(fontSize: wide ? 12.5 : 12),
                ),
              ),
            ],
          ],
        ),
      );
    }

    return CPanel(
      radius: wide ? CP.rCardLarge : CP.rCard,
      padding: wide
          ? const EdgeInsets.fromLTRB(18, 18, 18, 16)
          : const EdgeInsets.fromLTRB(16, 16, 16, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (final w in head) ...[w, const SizedBox(height: 10)],
          if (headlines.isEmpty)
            Text('No headline measured on either side.', style: t.caption)
          else
            for (final (i, h) in headlines.indexed) ...[
              if (i > 0) const SizedBox(height: 10),
              row(i, h),
            ],
        ],
      ),
    );
  }
}

/// A letter badge and a name beside it.
class _Badged extends StatelessWidget {
  const _Badged(
    this.letter,
    this.text, {
    this.style,
    this.muted = false,
    this.size = CBadgeSize.sm,
    this.gap = 7,
  });

  final String letter;
  final String text;
  final TextStyle? style;
  final bool muted;
  final CBadgeSize size;
  final double gap;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final s = (style ?? t.small.copyWith(fontSize: 13, fontWeight: FontWeight.w700))
        .copyWith(color: muted ? t.muted : t.ink);
    return Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        CLetterBadge(letter, size: size),
        SizedBox(width: gap),
        Flexible(child: Text(text, style: s)),
      ],
    );
  }
}

/// A line under a heading that starts with a run's badge: "Only on
/// MacBook Pro: Metal, Core ML".
class _Note extends StatelessWidget {
  const _Note(this.letter, this.text, {required this.wide});

  final String letter;
  final String text;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: wide ? 6 : 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 1),
            child: CLetterBadge(letter, size: CBadgeSize.xs),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(text,
                style: t.caption
                    .copyWith(fontSize: wide ? 13 : 12, height: 1.45)),
          ),
        ],
      ),
    );
  }
}

/// The two columns' heads above a list: what the rows are, then each run.
class _ColumnHead extends StatelessWidget {
  const _ColumnHead({
    required this.label,
    required this.a,
    required this.b,
    required this.wide,
    required this.chevron,
  });

  final String label;
  final String a;
  final String b;
  final bool wide;

  /// Leave the rows' chevron its column.
  final bool chevron;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final style = t.caps.copyWith(
      fontSize: wide ? 11.5 : 11,
      fontWeight: FontWeight.w700,
      letterSpacing: wide ? 11.5 * 0.08 : 11 * 0.06,
    );
    Widget side(String letter, String text) => Row(
          children: [
            CLetterBadge(letter, size: CBadgeSize.xs),
            const SizedBox(width: 7),
            Expanded(
              child: Text(text,
                  style: style, maxLines: 1, overflow: TextOverflow.ellipsis),
            ),
          ],
        );
    if (!wide) {
      return Padding(
        padding: EdgeInsets.fromLTRB(17, 0, chevron ? 39 : 17, 0),
        child: Row(
          children: [
            Expanded(child: side('A', a)),
            const SizedBox(width: 10),
            Expanded(child: side('B', b)),
          ],
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 19),
      child: Row(
        children: [
          Expanded(child: Text(label, style: style)),
          const SizedBox(width: 16),
          SizedBox(width: _valueColumn, child: side('A', a)),
          const SizedBox(width: 16),
          SizedBox(width: _valueColumn, child: side('B', b)),
          if (chevron) const SizedBox(width: 16 + 18),
        ],
      ),
    );
  }
}

/// A value column's width in a wide row.
const _valueColumn = 170.0;

/// Cards in columns at least [minWidth] wide, as many as fit, each row as
/// tall as its tallest.
class _Grid extends StatelessWidget {
  const _Grid({
    required this.children,
    required this.minWidth,
    this.gap = 16,
    this.runGap = 16,
  });

  final List<Widget> children;
  final double minWidth;
  final double gap;
  final double runGap;

  @override
  Widget build(BuildContext context) {
    if (children.isEmpty) return const SizedBox.shrink();
    return LayoutBuilder(builder: (context, box) {
      final fit = math.max(
          1,
          ((box.maxWidth + gap) / (math.min(minWidth, box.maxWidth) + gap))
              .floor());
      final cols = math.min(fit, children.length);
      final rows = <Widget>[];
      for (var i = 0; i < children.length; i += cols) {
        if (rows.isNotEmpty) rows.add(SizedBox(height: runGap));
        if (cols == 1) {
          rows.add(children[i]);
          continue;
        }
        rows.add(IntrinsicHeight(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              for (var j = 0; j < cols; j++) ...[
                if (j > 0) SizedBox(width: gap),
                Expanded(
                  child: i + j < children.length
                      ? children[i + j]
                      : const SizedBox.shrink(),
                ),
              ],
            ],
          ),
        ));
      }
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: rows,
      );
    });
  }
}

// ── The two runs ───────────────────────────────────────────────────────────

/// One run of the compare on its run's tint: its badge, what tells it apart,
/// and Change, which goes back to History to pick another.
class _RunCard extends StatelessWidget {
  const _RunCard({
    required this.letter,
    required this.heading,
    required this.tag,
    required this.line1,
    required this.line2,
    required this.onChange,
    required this.compact,
  });

  final String letter;
  final String heading;

  /// Where it came from: "This Mac", "Imported"; null where unknown.
  final String? tag;
  final String line1;
  final String line2;
  final VoidCallback onChange;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final hue = t.run(letter);
    final size = compact ? 16.0 : 18.0;
    final headStyle = t.title.copyWith(
      fontSize: size,
      fontWeight: FontWeight.w800,
      letterSpacing: size * -0.02,
      height: 1.3,
    );
    final lineStyle = t.caption.copyWith(fontSize: compact ? 12.5 : 13);
    final tagPill = tag == null
        ? null
        : CTag(text: tag!, upper: false, color: t.ink, fill: t.tagOnTint);
    Widget line(String s) => Text(
          s.isEmpty ? ' ' : s,
          style: lineStyle,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
        );
    final heading = Wrap(
      spacing: 8,
      runSpacing: 4,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [Text(this.heading, style: headStyle), ?tagPill],
    );
    return Semantics(
      container: true,
      label: 'Run $letter',
      child: Container(
        padding: compact
            ? const EdgeInsets.fromLTRB(16, 14, 2, 14)
            : const EdgeInsets.fromLTRB(20, 18, 10, 18),
        decoration: BoxDecoration(
          color: hue.tint,
          borderRadius: BorderRadius.circular(compact ? 22 : CP.rCard),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CLetterBadge(letter,
                size: compact ? CBadgeSize.lg : CBadgeSize.xl),
            SizedBox(width: compact ? 12 : 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  heading,
                  SizedBox(height: compact ? 1 : 2),
                  line(line1),
                  line(line2),
                ],
              ),
            ),
            CButton(
                label: 'Change',
                kind: CButtonKind.quiet,
                onPressed: onChange),
          ],
        ),
      ),
    );
  }
}

/// A phone's two run cards, one over the other, the swap button on the seam
/// between them however tall each card grows: it follows a mark at the
/// seam, and is painted over both.
class _StackedCards extends StatefulWidget {
  const _StackedCards({required this.a, required this.b, required this.swap});

  final Widget a;
  final Widget b;
  final Widget swap;

  @override
  State<_StackedCards> createState() => _StackedCardsState();
}

class _StackedCardsState extends State<_StackedCards> {
  final _seam = LayerLink();

  @override
  Widget build(BuildContext context) => Stack(
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              widget.a,
              CompositedTransformTarget(
                link: _seam,
                child: const SizedBox(height: 8),
              ),
              widget.b,
            ],
          ),
          CompositedTransformFollower(
            link: _seam,
            showWhenUnlinked: false,
            targetAnchor: Alignment.center,
            followerAnchor: Alignment.center,
            child: widget.swap,
          ),
        ],
      );
}

/// Arrows up and down: the swap between two stacked cards.
const _swapUpDown = 'M8 4v15l-4-4M16 20V5l4 4';

/// Swap A and B: a round button with a hairline, between the run cards.
class _SwapButton extends StatelessWidget {
  const _SwapButton({required this.onTap, this.across = false});

  final VoidCallback onTap;

  /// Between two cards side by side (arrows left and right), rather than
  /// stacked.
  final bool across;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final fill = across ? t.surface : t.ground;
    return Tooltip(
      message: 'Swap A and B',
      excludeFromSemantics: true,
      child: CTap(
        onTap: onTap,
        semanticLabel: 'Swap A and B',
        focusRadius: BorderRadius.circular(CP.rPill),
        builder: (context, hovered, pressed) => Container(
          width: across ? 46 : 44,
          height: across ? 46 : 44,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: hovered || pressed ? _wash(t, fill) : fill,
            border: Border.all(color: t.line),
          ),
          child: CGlyph(across ? CGlyphs.swap : _swapUpDown,
              size: across ? 20 : 18, color: t.ink, stroke: 2.1),
        ),
      ),
    );
  }
}

/// What kind of compare this is, and what the colours mean.
class _Banner extends StatelessWidget {
  const _Banner({required this.same, required this.wide});

  final bool same;
  final bool wide;

  static const _sameBody =
      'The same processors in both runs, so each reading is set against the '
      'same reading of the other run. What differs is run-to-run variation, '
      'or a driver or setting that changed.';
  static const _crossBody =
      'Processors pair by kind. By processor, each side shows its best from '
      'any backend. By backend, each side shows its best through that one '
      'backend.';
  static const _legend =
      'The better value is bold, on its run’s colour. Within 0.05% counts as '
      'the same, and so is neither.';

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final body = same ? _sameBody : _crossBody;
    if (!wide) {
      return Container(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
        decoration: BoxDecoration(
          color: t.surface2,
          borderRadius: BorderRadius.circular(CP.rPanel),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(body, style: t.prose.copyWith(fontSize: 12.5, height: 1.5)),
            const SizedBox(height: 8),
            const _Legend(small: true),
            const SizedBox(height: 8),
            Text(_legend, style: t.prose.copyWith(fontSize: 12.5, height: 1.45)),
          ],
        ),
      );
    }
    final text = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(same ? 'Same hardware' : 'Different hardware',
            style: t.title.copyWith(fontWeight: FontWeight.w800)),
        const SizedBox(height: 2),
        Text(body, style: t.prose.copyWith(height: 1.5)),
      ],
    );
    final key = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _Legend(small: false),
        const SizedBox(height: 6),
        Text(_legend, style: t.prose.copyWith(fontSize: 13, height: 1.45)),
      ],
    );
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 16),
      decoration: BoxDecoration(
        color: t.surface2,
        borderRadius: BorderRadius.circular(22),
      ),
      child: LayoutBuilder(builder: (context, box) {
        if (box.maxWidth < 720) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [text, const SizedBox(height: 12), key],
          );
        }
        return Row(
          children: [
            Expanded(flex: 42, child: text),
            const SizedBox(width: 28),
            Expanded(flex: 30, child: key),
          ],
        );
      }),
    );
  }
}

/// A and B, each by its badge and a swatch of its tint.
class _Legend extends StatelessWidget {
  const _Legend({required this.small});

  final bool small;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    Widget swatch(Color c) => Container(
          width: small ? 30 : 34,
          height: small ? 18 : 20,
          decoration: BoxDecoration(
            color: c,
            borderRadius: BorderRadius.circular(small ? 6 : 7),
          ),
        );
    final badge = small ? CBadgeSize.sm : CBadgeSize.md;
    return ExcludeSemantics(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          CLetterBadge('A', size: badge),
          SizedBox(width: small ? 7 : 8),
          swatch(t.runA.tint),
          SizedBox(width: small ? 15 : 18),
          CLetterBadge('B', size: badge),
          SizedBox(width: small ? 7 : 8),
          swatch(t.runB.tint),
        ],
      ),
    );
  }
}

// ── Rows ───────────────────────────────────────────────────────────────────

/// A measure both sides have: what it is, each side's value, and on a tap
/// where each came from.
class _MeasureRow extends StatelessWidget {
  const _MeasureRow({
    required this.label,
    required this.words,
    required this.a,
    required this.b,
    required this.verdict,
    required this.captionA,
    required this.captionB,
    required this.open,
    required this.onToggle,
    required this.wide,
  });

  final String label;
  final String words;
  final CompareValue a;
  final CompareValue b;
  final Verdict verdict;
  final String captionA;
  final String captionB;
  final bool open;
  final VoidCallback onToggle;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final marks = _Marks.of(verdict);
    final size = wide ? 17.0 : 16.0;
    final what = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(label,
            style: t.label.copyWith(fontSize: wide ? 14.5 : 14, height: 1.3)),
        if (words.isNotEmpty)
          Text(words, style: t.caption.copyWith(fontSize: wide ? 12.5 : 12)),
        if (verdict.methodChanged)
          Text(
            wide
                ? 'Method changed between the runs: not compared'
                : 'Method changed: not compared',
            style: t.caption.copyWith(
                fontSize: wide ? 12.5 : 12, fontStyle: FontStyle.italic),
          ),
      ],
    );
    final cellA = _ValueCell(
        value: a,
        side: CompareSide.a,
        marks: marks,
        caption: captionA,
        size: size);
    final cellB = _ValueCell(
        value: b,
        side: CompareSide.b,
        marks: marks,
        caption: captionB,
        size: size);
    final chevron = Transform.rotate(
      angle: open ? math.pi : 0,
      child: CGlyph(CGlyphs.chevronDown, size: 18, color: t.muted, stroke: 2.2),
    );
    final head = wide
        ? Row(
            children: [
              Expanded(child: what),
              const SizedBox(width: 16),
              SizedBox(width: _valueColumn, child: cellA),
              const SizedBox(width: 16),
              SizedBox(width: _valueColumn, child: cellB),
              const SizedBox(width: 16),
              chevron,
            ],
          )
        : Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    what,
                    const SizedBox(height: 6),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(child: cellA),
                        const SizedBox(width: 10),
                        Expanded(child: cellB),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              chevron,
            ],
          );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CTap(
          onTap: onToggle,
          expanded: open,
          insetFocus: true,
          focusRadius: BorderRadius.zero,
          builder: (context, hovered, pressed) => Container(
            constraints: BoxConstraints(minHeight: wide ? 60 : 0),
            color: hovered || pressed ? _wash(t, t.surface) : t.surface,
            padding: wide
                ? const EdgeInsets.symmetric(horizontal: 18, vertical: 10)
                : const EdgeInsets.fromLTRB(16, 11, 12, 12),
            child: head,
          ),
        ),
        if (open)
          Container(
            margin: wide
                ? const EdgeInsets.fromLTRB(14, 0, 14, 14)
                : const EdgeInsets.fromLTRB(10, 0, 10, 12),
            padding: wide
                ? const EdgeInsets.fromLTRB(16, 12, 16, 12)
                : const EdgeInsets.fromLTRB(12, 10, 12, 10),
            decoration: BoxDecoration(
              color: t.well,
              borderRadius: BorderRadius.circular(wide ? CP.rRow : 14),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _Badged('A', sourceOf(a),
                    size: CBadgeSize.xs,
                    gap: 10,
                    style: t.body.copyWith(fontSize: wide ? 13 : 12)),
                const SizedBox(height: 7),
                _Badged('B', sourceOf(b),
                    size: CBadgeSize.xs,
                    gap: 10,
                    style: t.body.copyWith(fontSize: wide ? 13 : 12)),
              ],
            ),
          ),
      ],
    );
  }
}

/// A row of a fold: a measure, or a test, only one side has.
class _OnlyRow extends StatelessWidget {
  const _OnlyRow({
    required this.label,
    required this.words,
    required this.value,
    required this.unit,
    required this.caption,
    required this.wide,
  });

  final String label;
  final String words;
  final String value;
  final String unit;
  final String caption;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final number = CValue(
      value: value,
      unit: unit,
      size: wide ? 15 : 14,
      weight: FontWeight.w600,
      unitColor: t.ink,
    );
    final what = [
      Text(label,
          style: t.body.copyWith(
              fontSize: wide ? 14 : 13, fontWeight: FontWeight.w600)),
      if (words.isNotEmpty)
        Text(words, style: t.caption.copyWith(fontSize: wide ? 12.5 : 11.5)),
    ];
    if (wide) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 10),
        child: Row(
          children: [
            Expanded(
              child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: what),
            ),
            const SizedBox(width: 16),
            number,
            const SizedBox(width: 16),
            SizedBox(
              width: 110,
              child: Text(caption,
                  style: t.caption.copyWith(fontSize: 12.5)),
            ),
          ],
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 9),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ...what,
          const SizedBox(height: 2),
          Row(
            children: [
              number,
              const SizedBox(width: 8),
              Flexible(
                child: Text(caption,
                    style: t.caption.copyWith(fontSize: 11.5)),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// What a fold opens on: its rows in a dashed card.
class _FoldList extends StatelessWidget {
  const _FoldList({required this.rows});

  final List<Widget> rows;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CEmptyBox(
      padding: EdgeInsets.zero,
      radius: _Slice.radius,
      center: false,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (final (i, r) in rows.indexed)
            Container(
              decoration: i == 0
                  ? null
                  : BoxDecoration(
                      border: Border(top: BorderSide(color: t.line))),
              child: r,
            ),
        ],
      ),
    );
  }
}

/// A fold for what only one side has: its badge, its label and how many.
Widget _fold({
  required Key key,
  required String letter,
  required String label,
  required String count,
  required List<Widget> rows,
}) =>
    CFold(
      key: key,
      label: label,
      count: count,
      leading: CLetterBadge(letter, size: CBadgeSize.sm),
      child: _FoldList(rows: rows),
    );

// ── Re-pairing ─────────────────────────────────────────────────────────────

/// The person's pick in [_PartnerDialog]: a processor of the other run, or
/// none.
class _Partner {
  const _Partner(this.group);

  final ProcessorMeasures? group;
}

/// Which of the other run's processors to pair one with.
class _PartnerDialog extends StatelessWidget {
  const _PartnerDialog({
    required this.name,
    required this.others,
    required this.current,
    required this.paired,
  });

  final String name;
  final List<ProcessorMeasures> others;
  final ProcessorMeasures? current;
  final bool paired;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final rows = <Widget>[
      for (final (i, g) in others.indexed)
        CRow(
          rule: paired || i < others.length - 1,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          onTap: () => Navigator.pop(context, _Partner(g)),
          child: Row(
            children: [
              CKindLabel(g.kind.isEmpty ? 'device' : g.kind,
                  neutral: true, small: true),
              const SizedBox(width: 10),
              Expanded(
                child: Text(g.name.isEmpty ? g.id : g.name,
                    style: t.body.copyWith(fontWeight: FontWeight.w600)),
              ),
              if (identical(g, current))
                CGlyph(CGlyphs.check, size: 18, color: t.ink, stroke: 2.4),
            ],
          ),
        ),
      if (paired)
        CRow(
          rule: false,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          onTap: () => Navigator.pop(context, const _Partner(null)),
          child: Text('Leave it unpaired',
              style: t.body
                  .copyWith(fontWeight: FontWeight.w600, color: t.muted)),
        ),
    ];
    return CDialog(
      title: 'Pair ${name.isEmpty ? 'this processor' : name} with',
      actions: [
        CButton(
          label: 'Cancel',
          kind: CButtonKind.quiet,
          onPressed: () => Navigator.pop(context),
        ),
      ],
      child: SingleChildScrollView(
        child: CPanel(
          radius: CP.rPanel,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: rows,
          ),
        ),
      ),
    );
  }
}
