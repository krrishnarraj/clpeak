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
          CPanelHead(
            title: run.device,
            tag: run.backend,
            trailing: trail.isEmpty
                ? null
                : ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 260),
                    child: Text(trail.join('  ·  '),
                        style: t.monoSmallDim,
                        maxLines: 1,
                        textAlign: TextAlign.right,
                        overflow: TextOverflow.ellipsis),
                  ),
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
                  Expanded(
                    child: Text(test.display,
                        style: t.mono,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis),
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
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}

class _MetricLine extends StatelessWidget {
  const _MetricLine({
    required this.entry,
    required this.maxValue,
    required this.color,
  });

  final ResultEntry entry;
  final double maxValue;
  final Color color;

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
            width: 104,
            child: Text(entry.metric,
                style: t.monoSmallDim,
                maxLines: 1,
                overflow: TextOverflow.ellipsis),
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
                      child: Text(test.display,
                          style: t.monoSmall.copyWith(color: t.dim)),
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
