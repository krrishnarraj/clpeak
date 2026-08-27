import 'package:flutter/material.dart';

import '../../model/result_entry.dart';
import '../../model/run_document.dart';
import '../../theme/clpeak_theme.dart';
import '../common/kit.dart';

/// Renders one RunDocument: run-selector chips (when a session covered
/// multiple backends/devices), a device header, then each category as a
/// hairline table of tests with expandable metric detail, and a de-emphasized
/// unsupported section.  Used by the live run view, the just-finished view,
/// and the history viewer.
class ResultsBody extends StatefulWidget {
  const ResultsBody({
    super.key,
    required this.document,
    this.header,
  });

  final RunDocument document;

  /// Optional slivers-above widget (e.g. the live-run banner).
  final Widget? header;

  @override
  State<ResultsBody> createState() => _ResultsBodyState();
}

class _ResultsBodyState extends State<ResultsBody> {
  String? _selectedRunKey;

  @override
  Widget build(BuildContext context) {
    final runs = widget.document.runs;
    if (runs.isEmpty) {
      return ListView(
        padding: const EdgeInsets.all(20),
        children: [
          if (widget.header != null) widget.header!,
          const SizedBox(height: 40),
          const CEmpty(title: 'Waiting for results…'),
        ],
      );
    }

    final selected = runs.firstWhere((r) => r.key == _selectedRunKey,
        orElse: () => runs.last);
    final brightness = Theme.of(context).brightness;

    // Flattened one level: category sections used to be non-lazy Columns
    // holding every test card, so all of them were built, laid out and
    // painted on each frame even when off-screen.  As sibling rows of the
    // sliver list only the visible ones cost anything — which matters during
    // a live run, where each frame is GPU work competing with the benchmark.
    //
    // The table look is drawn per row (each carries its own rules) rather than
    // by wrapping a category in one container, which would undo the laziness.
    final rows = <_Row>[
      if (widget.header != null) _WidgetRow(widget.header!, padBottom: 16),
      if (runs.length > 1)
        _WidgetRow(
          _RunSelector(
            runs: runs,
            selectedKey: selected.key,
            onSelected: (key) => setState(() => _selectedRunKey = key),
          ),
          padBottom: 16,
        ),
      _WidgetRow(_DeviceHeader(run: selected)),
      for (final group in selected.categories)
        if (group.supported.isNotEmpty) ...[
          _SectionRow(group, brightness),
          for (var i = 0; i < group.supported.length; i++)
            _TestRow(
              group,
              group.supported[i],
              brightness,
              last: i == group.supported.length - 1,
            ),
        ],
      if (selected.categories.any((g) => g.unsupported.isNotEmpty))
        _WidgetRow(_UnsupportedSection(run: selected), padTop: 22),
    ];

    return ListView.builder(
      // Bottom pad carries the old trailing SizedBox(32).
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 52),
      itemCount: rows.length,
      itemBuilder: (context, i) => RepaintBoundary(child: rows[i].build()),
    );
  }
}

/// One row of the lazily-built results list.  Descriptors, not widgets: the
/// subtree is only inflated for rows the viewport actually reaches.
sealed class _Row {
  const _Row();

  Widget build();
}

class _WidgetRow extends _Row {
  const _WidgetRow(this.child, {this.padTop = 0, this.padBottom = 0});

  final Widget child;
  final double padTop;
  final double padBottom;

  @override
  Widget build() => Padding(
        padding: EdgeInsets.only(top: padTop, bottom: padBottom),
        child: child,
      );
}

class _SectionRow extends _Row {
  const _SectionRow(this.group, this.brightness);

  final CategoryGroup group;
  final Brightness brightness;

  @override
  Widget build() => Padding(
        padding: const EdgeInsets.only(top: 22, bottom: 9),
        child: CSection(
          label: group.category.label,
          color: ClpeakTheme.categoryColor(group.category,
              brightness: brightness),
          trailing: '${group.supported.length} tests',
        ),
      );
}

class _TestRow extends _Row {
  const _TestRow(this.group, this.test, this.brightness, {required this.last});

  final CategoryGroup group;
  final TestResult test;
  final Brightness brightness;
  final bool last;

  @override
  Widget build() => _TestLine(
        // Keyed by test tag: rows are inserted as results stream in during a
        // live run, and without a key a row's expansion state would follow
        // the index and land on the wrong test.
        key: ValueKey(test.test),
        test: test,
        color: ClpeakTheme.categoryColor(group.category,
            brightness: brightness),
        first: identical(test, group.supported.first),
        last: last,
      );
}

/// Which backend/device the readings below belong to.
class _RunSelector extends StatelessWidget {
  const _RunSelector({
    required this.runs,
    required this.selectedKey,
    required this.onSelected,
  });

  final List<DeviceRun> runs;
  final String selectedKey;
  final ValueChanged<String> onSelected;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('DEVICE', style: t.micro),
        const SizedBox(height: 8),
        // Wrap, not a horizontal scroller: with many devices the trailing
        // chips used to sit off-screen with no way (mouse wheel included) to
        // reach them.
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final run in runs)
              CChip(
                label: '${run.backend} · ${run.device}',
                selected: run.key == selectedKey,
                onTap: () => onSelected(run.key),
              ),
          ],
        ),
      ],
    );
  }
}

/// Device identity and its properties.  Props are shown from the moment the
/// device opens — they arrive with the very first event of a device's run, so
/// there is nothing to wait for — and are static thereafter.
class _DeviceHeader extends StatelessWidget {
  const _DeviceHeader({required this.run});

  final DeviceRun run;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final showPlatform =
        run.platform.isNotEmpty && run.platform != run.backend;
    final trail = [
      if (showPlatform) run.platform,
      if (run.driver.isNotEmpty) run.driver,
    ];

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // The platform/driver line is sized off the panel, not off a fixed
          // column: a 260px column ellipsized a driver version away even on a
          // wide desktop window, where the panel had hundreds of pixels to
          // spare.  It wraps rather than ellipsizes for the same reason names
          // do — the version tail is what tells two installs of the same
          // driver apart — so `ellipsis` here only ever bites a single token
          // too long to break.
          LayoutBuilder(
            builder: (context, c) {
              // Head padding (12 + 12) off the panel, then half of what is
              // left, so the device name keeps a column of its own.  A loose
              // constraint: a shorter line still shrink-wraps, so the cap only
              // decides where a very long one wraps.
              final trailMax = ((c.maxWidth - 24) / 2).clamp(0.0, 640.0);
              return CPanelHead(
                title: run.device,
                tag: run.backend,
                trailing: trail.isEmpty
                    ? null
                    : ConstrainedBox(
                        constraints: BoxConstraints(maxWidth: trailMax),
                        child: Text(trail.join('  ·  '),
                            style: t.monoSmallDim,
                            textAlign: TextAlign.right,
                            overflow: TextOverflow.ellipsis),
                      ),
              );
            },
          ),
          if (run.props.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 11, 12, 11),
              child: Wrap(
                spacing: 22,
                runSpacing: 7,
                children: [
                  for (final prop in run.props)
                    Text.rich(TextSpan(children: [
                      TextSpan(
                          text: '${prop.key.toUpperCase()}  ', style: t.micro),
                      TextSpan(text: prop.value, style: t.monoSmall),
                    ])),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// One test: name, peak reading, and — when opened — a metric breakdown.
///
/// Draws its own top/bottom rules and side borders so a run of these reads as
/// a single bordered table even though each is an independent list row.
class _TestLine extends StatefulWidget {
  const _TestLine({
    super.key,
    required this.test,
    required this.color,
    required this.first,
    required this.last,
  });

  final TestResult test;
  final Color color;
  final bool first;
  final bool last;

  @override
  State<_TestLine> createState() => _TestLineState();
}

class _TestLineState extends State<_TestLine> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final test = widget.test;
    final peak = formatMetric(test.peakValue, test.unit);

    return Container(
      decoration: BoxDecoration(
        color: t.panel,
        border: Border(
          top: BorderSide(color: t.line),
          left: BorderSide(color: t.line),
          right: BorderSide(color: t.line),
          bottom: widget.last ? BorderSide(color: t.line) : BorderSide.none,
        ),
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(widget.first ? CP.rPanel : 0),
          bottom: Radius.circular(widget.last ? CP.rPanel : 0),
        ),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          CTap(
            onTap: () => setState(() => _expanded = !_expanded),
            builder: (context, hovered, pressed) => Container(
              color: hovered || pressed ? t.hover : Colors.transparent,
              padding: const EdgeInsets.fromLTRB(12, 10, 10, 10),
              child: Row(
                children: [
                  Container(width: 3, height: 15, color: widget.color),
                  const SizedBox(width: 10),
                  // The title wraps rather than ellipsizing: the row is as
                  // tall as its name needs, which on a phone is the
                  // difference between "Half-precision compute fp1…" and
                  // knowing which half-precision test this is.
                  Expanded(
                    child: _NameWithInfo(
                      name: test.display,
                      description: test.description,
                      style: t.mono,
                    ),
                  ),
                  const SizedBox(width: 12),
                  CValue(
                      value: peak.value,
                      unit: peak.unit,
                      color: widget.color),
                  const SizedBox(width: 6),
                  Icon(
                    _expanded ? Icons.remove : Icons.add,
                    size: 13,
                    color: t.faint,
                  ),
                ],
              ),
            ),
          ),
          // Instant, not a cross-fade: expansion happens mid-run, and an
          // animated one would cost the benchmark a burst of frames.
          if (_expanded)
            Container(
              decoration: BoxDecoration(
                color: t.isDark
                    ? t.bg.withValues(alpha: 0.45)
                    : t.hover.withValues(alpha: 0.5),
                border: Border(top: BorderSide(color: t.line)),
              ),
              padding: const EdgeInsets.fromLTRB(25, 9, 12, 10),
              child: Column(
                children: [
                  for (final metric in test.metrics)
                    _MetricLine(
                      entry: metric,
                      maxValue: test.maxValue,
                      color: widget.color,
                      glyphColumn: test.hasMetricNotes,
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

/// A name, plus its info glyph when it has something to explain.
///
/// The glyph rides in the text flow, right after the last word, rather than
/// in a slot reserved at the edge of the name's column: on a wide desktop
/// window the name column can stretch far past a short name, and a glyph
/// anchored to that far edge reads as unrelated to the name it explains. A
/// column-aligned glyph is nice when it happens for free, but not at the
/// cost of a visible gap on every short row — inline never has that gap.
class _NameWithInfo extends StatelessWidget {
  const _NameWithInfo({
    required this.name,
    required this.description,
    required this.style,
    this.small = false,
  });

  final String name;
  final String description;
  final TextStyle style;

  /// Sized for the breakdown's smaller type.
  final bool small;

  @override
  Widget build(BuildContext context) {
    return Text.rich(
      TextSpan(children: [
        TextSpan(text: name),
        if (description.isNotEmpty)
          WidgetSpan(
            alignment: PlaceholderAlignment.middle,
            child: _InfoGlyph(
              title: name,
              description: description,
              small: small,
            ),
          ),
      ]),
      style: style,
    );
  }
}

/// The info affordance: a faint glyph beside a name, present only when that
/// name has an explanation.  Its own tap target, so it opens the description
/// instead of expanding the row it sits in.
///
/// Used at both levels — beside a test's title for what the test measures, and
/// beside a reading's label for what that one reading means, each opening its
/// own text.
class _InfoGlyph extends StatelessWidget {
  const _InfoGlyph({
    required this.title,
    required this.description,
    this.small = false,
  });

  final String title;
  final String description;

  /// Sized for the breakdown's smaller type.
  final bool small;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: () => _showInfoDialog(context, title, description),
      builder: (context, hovered, pressed) => Padding(
        padding: EdgeInsets.only(left: small ? 5 : 7, right: 2),
        child: Icon(
          Icons.info_outline,
          size: small ? 12 : 13,
          color: hovered || pressed ? t.text : t.faint,
        ),
      ),
    );
  }
}

/// One explanation, in plain language.
///
/// Opened without a transition: a description can be asked for mid-run, and
/// the GUI holds a graphics context on the device being benchmarked, so an
/// animated route would cost the running kernel a burst of frames (see the
/// live-run note in app/AGENTS.md).
Future<void> _showInfoDialog(
    BuildContext context, String title, String description) {
  return showGeneralDialog<void>(
    context: context,
    barrierDismissible: true,
    barrierLabel: MaterialLocalizations.of(context).modalBarrierDismissLabel,
    barrierColor: Colors.black.withValues(alpha: 0.55),
    transitionDuration: Duration.zero,
    pageBuilder: (context, _, _) => _InfoDialog(
      title: title,
      description: description,
    ),
  );
}

class _InfoDialog extends StatelessWidget {
  const _InfoDialog({required this.title, required this.description});

  final String title;
  final String description;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CDialog(
      title: title,
      actions: [
        CButton(label: 'Close', onPressed: () => Navigator.pop(context)),
      ],
      // Long prose on a short phone screen.
      child: ConstrainedBox(
        constraints:
            BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.6),
        child: SingleChildScrollView(
          child: Text(description, style: t.body),
        ),
      ),
    );
  }
}

class _MetricLine extends StatelessWidget {
  const _MetricLine({
    required this.entry,
    required this.maxValue,
    required this.color,
    required this.glyphColumn,
  });

  final ResultEntry entry;
  final double maxValue;
  final Color color;

  /// Some reading of this test is documented, so every row of it leaves room
  /// for the glyph and the meters stay in one column.
  final bool glyphColumn;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final ok = entry.status == ResultStatus.ok;
    final fraction =
        ok && maxValue > 0 ? (entry.value / maxValue).clamp(0.0, 1.0) : 0.0;
    final f = ok ? formatMetric(entry.value, entry.unit) : null;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3.5),
      child: Row(
        children: [
          SizedBox(
            // Fixed width, so the meters stay in one column; a label longer
            // than it wraps inside that width rather than being cut.
            width: glyphColumn ? 121 : 104,
            child: _NameWithInfo(
              name: entry.metric,
              description: entry.metricDescription,
              style: t.monoSmallDim,
              small: true,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: ok
                ? CMeter(fraction: fraction, color: color)
                : Text(
                    '${entry.status.name} — ${entry.reason}',
                    style: t.monoSmall.copyWith(color: t.faint),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
          ),
          if (f != null) ...[
            const SizedBox(width: 12),
            SizedBox(
              width: 94,
              child: Align(
                alignment: Alignment.centerRight,
                child: CValue(value: f.value, unit: f.unit, small: true),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _UnsupportedSection extends StatelessWidget {
  const _UnsupportedSection({required this.run});

  final DeviceRun run;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final items = <TestResult>[
      for (final g in run.categories) ...g.unsupported,
    ];
    if (items.isEmpty) return const SizedBox.shrink();

    return CPanel(
      child: CExpander(
        header: (context, open) => Container(
          padding: const EdgeInsets.fromLTRB(12, 11, 12, 11),
          child: Row(
            children: [
              Icon(Icons.block, size: 13, color: t.faint),
              const SizedBox(width: 9),
              Expanded(
                child: Text(
                  'NOT SUPPORTED ON THIS DEVICE (${items.length})',
                  style: t.micro,
                ),
              ),
              Icon(open ? Icons.remove : Icons.add, size: 13, color: t.faint),
            ],
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(height: 1, color: t.line),
            for (final test in items)
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 7, 12, 7),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      flex: 4,
                      child: _NameWithInfo(
                        name: test.display,
                        description: test.description,
                        style: t.monoSmall.copyWith(color: t.dim),
                        small: true,
                      ),
                    ),
                    if (test.skipReason.isNotEmpty) ...[
                      const SizedBox(width: 12),
                      Expanded(
                        flex: 5,
                        child: Text(test.skipReason,
                            style: t.monoSmall.copyWith(color: t.faint),
                            textAlign: TextAlign.right),
                      ),
                    ],
                  ],
                ),
              ),
            const SizedBox(height: 4),
          ],
        ),
      ),
    );
  }
}
