import 'package:flutter/material.dart';

import '../../model/result_entry.dart';
import '../../model/run_document.dart';
import '../../theme/clpeak_theme.dart';

/// Renders one RunDocument: run-selector chips (when a session covered
/// multiple backends/devices), device header, hero score tiles, category
/// sections with expandable test cards, and a de-emphasized unsupported
/// section.  Used by the live run view, the just-finished view, and the
/// history viewer.
class ResultsBody extends StatefulWidget {
  const ResultsBody({
    super.key,
    required this.document,
    this.compact = false,
    this.header,
  });

  final RunDocument document;

  /// Compact mode (live run): no hero tiles, collapsed props.
  final bool compact;

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
          Center(
            child: Text('Waiting for results…',
                style: Theme.of(context)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(color: Theme.of(context).colorScheme.outline)),
          ),
        ],
      );
    }

    final selected = runs.firstWhere((r) => r.key == _selectedRunKey,
        orElse: () => runs.last);

    // Flattened one level: category sections used to be non-lazy Columns
    // holding every test card, so all of them were built, laid out and
    // painted on each frame even when off-screen.  As sibling rows of the
    // sliver list only the visible ones cost anything — which matters during
    // a live run, where each frame is GPU work competing with the benchmark.
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
      _WidgetRow(_DeviceHeaderCard(run: selected, compact: widget.compact)),
      for (final group in selected.categories)
        if (group.supported.isNotEmpty) ...[
          _SectionRow(group),
          for (final test in group.supported) _TestRow(group, test),
        ],
      if (selected.categories.any((g) => g.unsupported.isNotEmpty))
        _WidgetRow(_UnsupportedSection(run: selected), padTop: 20),
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
  const _SectionRow(this.group);

  final CategoryGroup group;

  @override
  Widget build() => Padding(
        padding: const EdgeInsets.only(top: 20, bottom: 10),
        child: _CategoryHeader(group: group),
      );
}

class _TestRow extends _Row {
  const _TestRow(this.group, this.test);

  final CategoryGroup group;
  final TestResult test;

  @override
  Widget build() => Padding(
        padding: const EdgeInsets.only(bottom: 8),
        // Keyed by test tag: rows are inserted as results stream in during a
        // live run, and without a key a card's expansion state would follow
        // the index and land on the wrong test.
        child: _TestCard(
          key: ValueKey(test.test),
          test: test,
          color: ClpeakTheme.categoryColor(group.category),
        ),
      );
}

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
    // Wrap, not a horizontal scroller: with many devices the trailing chips
    // used to sit off-screen with no way (mouse wheel included) to reach them.
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        for (final run in runs)
          ChoiceChip(
            avatar: Icon(ClpeakTheme.backendIcon(run.backend), size: 18),
            label: Text(
              '${run.backend} · ${run.device}',
              overflow: TextOverflow.ellipsis,
            ),
            selected: run.key == selectedKey,
            onSelected: (_) => onSelected(run.key),
          ),
      ],
    );
  }
}

class _DeviceHeaderCard extends StatelessWidget {
  const _DeviceHeaderCard({required this.run, required this.compact});

  final DeviceRun run;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final showPlatform =
        run.platform.isNotEmpty && run.platform != run.backend;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(ClpeakTheme.backendIcon(run.backend),
                    color: scheme.primary),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(run.device,
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(fontWeight: FontWeight.w600)),
                      Text(
                        [
                          run.backend,
                          if (showPlatform) run.platform,
                          if (run.driver.isNotEmpty) run.driver,
                        ].join(' · '),
                        style: Theme.of(context)
                            .textTheme
                            .bodySmall
                            ?.copyWith(color: scheme.outline),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            if (!compact && run.props.isNotEmpty) ...[
              const SizedBox(height: 12),
              Wrap(
                spacing: 16,
                runSpacing: 6,
                children: [
                  for (final prop in run.props)
                    Text.rich(
                      TextSpan(children: [
                        TextSpan(
                          text: '${prop.key}  ',
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: scheme.outline),
                        ),
                        TextSpan(
                          text: prop.value,
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                      ]),
                    ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _CategoryHeader extends StatelessWidget {
  const _CategoryHeader({required this.group});

  final CategoryGroup group;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(ClpeakTheme.categoryIcon(group.category),
            size: 18, color: ClpeakTheme.categoryColor(group.category)),
        const SizedBox(width: 8),
        Text(group.category.label,
            style: Theme.of(context)
                .textTheme
                .titleSmall
                ?.copyWith(fontWeight: FontWeight.w600)),
      ],
    );
  }
}

class _TestCard extends StatefulWidget {
  const _TestCard({super.key, required this.test, required this.color});

  final TestResult test;
  final Color color;

  @override
  State<_TestCard> createState() => _TestCardState();
}

class _TestCardState extends State<_TestCard> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final test = widget.test;
    final scheme = Theme.of(context).colorScheme;
    final peak = formatMetric(test.peakValue, test.unit);

    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          InkWell(
            onTap: () => setState(() => _expanded = !_expanded),
            child: Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  Expanded(
                    child: Text(test.display,
                        style: Theme.of(context).textTheme.bodyMedium),
                  ),
                  const SizedBox(width: 12),
                  Text.rich(
                    TextSpan(children: [
                      TextSpan(
                        text: peak.value,
                        style: Theme.of(context)
                            .textTheme
                            .titleMedium
                            ?.copyWith(
                              fontWeight: FontWeight.w700,
                              color: widget.color,
                              fontFeatures: const [
                                FontFeature.tabularFigures()
                              ],
                            ),
                      ),
                      TextSpan(
                        text: ' ${peak.unit}',
                        style: Theme.of(context)
                            .textTheme
                            .bodySmall
                            ?.copyWith(color: scheme.outline),
                      ),
                    ]),
                  ),
                  const SizedBox(width: 4),
                  Icon(
                    _expanded ? Icons.expand_less : Icons.expand_more,
                    size: 20,
                    color: scheme.outline,
                  ),
                ],
              ),
            ),
          ),
          AnimatedCrossFade(
            duration: const Duration(milliseconds: 180),
            crossFadeState: _expanded
                ? CrossFadeState.showSecond
                : CrossFadeState.showFirst,
            firstChild: const SizedBox(width: double.infinity),
            secondChild: Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Column(
                children: [
                  for (final metric in test.metrics)
                    _MetricRow(
                      entry: metric,
                      maxValue: test.maxValue,
                      color: widget.color,
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricRow extends StatelessWidget {
  const _MetricRow({
    required this.entry,
    required this.maxValue,
    required this.color,
  });

  final ResultEntry entry;
  final double maxValue;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final ok = entry.status == ResultStatus.ok;
    final fraction =
        ok && maxValue > 0 ? (entry.value / maxValue).clamp(0.0, 1.0) : 0.0;

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 110,
            child: Text(
              entry.metric,
              style: Theme.of(context).textTheme.bodySmall,
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: ok
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: LinearProgressIndicator(
                      value: fraction,
                      minHeight: 6,
                      backgroundColor: scheme.surfaceContainerHighest,
                      valueColor: AlwaysStoppedAnimation(
                          color.withValues(alpha: 0.85)),
                    ),
                  )
                : Text(
                    '${entry.status.name} — ${entry.reason}',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: scheme.outline),
                    overflow: TextOverflow.ellipsis,
                  ),
          ),
          if (ok) ...[
            const SizedBox(width: 10),
            SizedBox(
              width: 90,
              child: Builder(builder: (context) {
                final f = formatMetric(entry.value, entry.unit);
                return Text(
                  '${f.value} ${f.unit}',
                  textAlign: TextAlign.right,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    fontFeatures: const [FontFeature.tabularFigures()],
                  ),
                );
              }),
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
    final scheme = Theme.of(context).colorScheme;
    final items = <TestResult>[
      for (final g in run.categories) ...g.unsupported,
    ];
    if (items.isEmpty) return const SizedBox.shrink();

    return Card(
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          leading: Icon(Icons.block, size: 18, color: scheme.outline),
          title: Text(
            'Not supported on this device (${items.length})',
            style: Theme.of(context)
                .textTheme
                .bodyMedium
                ?.copyWith(color: scheme.outline),
          ),
          children: [
            for (final test in items)
              ListTile(
                dense: true,
                title: Text(test.display,
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: scheme.outline)),
                subtitle: test.skipReason.isEmpty
                    ? null
                    : Text(test.skipReason,
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: scheme.outline.withValues(alpha: 0.7))),
              ),
          ],
        ),
      ),
    );
  }
}
