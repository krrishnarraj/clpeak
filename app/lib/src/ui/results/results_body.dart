import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';

import '../../model/headlines.dart';
import '../../model/measures.dart';
import '../../model/result_model.dart';
import '../../model/run_document.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import 'results_kit.dart';
import 'summary_view.dart';

/// Renders one RunDocument as tabs: for a finished run a Summary first --
/// a kind card of flagship numbers per processor, a way into every backend,
/// and the system -- then a tab per backend: its devices (chips where it
/// reached several), the device's card, each category's tests as a table of
/// rows that open onto their readings, and what could not be measured.
/// Used by the live run view, the just-finished view, and the history
/// viewer.
class ResultsBody extends StatefulWidget {
  const ResultsBody({
    super.key,
    required this.document,
    this.header,
    this.showDiagnostics = true,
    this.catalog,
  });

  final RunDocument document;

  /// Optional first row above the results (e.g. the live-run status).
  final Widget? header;

  /// Whether to show the diagnostics section when the log is non-empty.
  /// The live run passes false unless the run was started with verbose
  /// diagnostics enabled; saved runs leave the default true.
  final bool showDiagnostics;

  /// The headlines the Summary shows; the app's bundled catalog
  /// ([HeadlineCatalog.bundled]) when null.
  final HeadlineCatalog? catalog;

  @override
  State<ResultsBody> createState() => _ResultsBodyState();
}

class _ResultsBodyState extends State<ResultsBody> {
  /// The tab on show: [_summary] or a backend's name; null until the user
  /// picks one, which shows the default.
  String? _tab;

  /// Per backend, the device on show ([DeviceRun.key]).
  final Map<String, String> _deviceOf = {};

  /// No backend is named this.
  static const _summary = '\u0000summary';

  HeadlineCatalog? _bundled;
  bool _loadingCatalog = false;

  HeadlineCatalog? get _catalog => widget.catalog ?? _bundled;

  void _loadCatalog() {
    if (_catalog != null || _loadingCatalog) return;
    _loadingCatalog = true;
    HeadlineCatalog.bundled().then((c) {
      if (mounted) setState(() => _bundled = c);
    });
  }

  /// A run whose document came from a file -- saved, from history, or what a
  /// run that stopped saved.  A live run's has no meta until it is over,
  /// and nothing to summarise until it is linked and saved: its tabs are
  /// its backends, following the device in flight.
  bool get _finished => widget.document.meta != null;

  void _openBackend(String backend) => setState(() => _tab = backend);

  void _openDevice(DeviceRun run) => setState(() {
        _tab = run.backend;
        _deviceOf[run.backend] = run.key;
      });

  // Expand all / Collapse all.  Null follows each test's default
  // (heterogeneous open, everything else closed); true or false holds every
  // test that way until the next one.  A tap on one test overrides it for
  // that test until the next Expand all or Collapse all, which bumps the
  // generation.
  bool? _forcedExpanded;
  int _forcedGen = 0;

  void _expandAll() => setState(() {
        _forcedGen++;
        _forcedExpanded = true;
      });

  void _collapseAll() => setState(() {
        _forcedGen++;
        _forcedExpanded = false;
      });

  // The phone's tab strip scrolls sideways, so the tab on show -- opened by
  // a "via" link or a tile, or the backend a live run moved on to -- can sit
  // past its edge.  Brought into view once per change, by a jump: nothing
  // here animates (app/AGENTS.md, the live-run trap).
  final _strip = GlobalKey();
  String? _revealed;

  void _reveal(String tab) {
    if (tab == _revealed) return;
    _revealed = tab;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      final root = _strip.currentContext;
      if (root == null) return;
      final key = ValueKey('results-tab:${tab == _summary ? 'Summary' : tab}');
      Element? found;
      void visit(Element e) {
        if (found != null) return;
        if (e.widget.key == key) {
          found = e;
        } else {
          e.visitChildren(visit);
        }
      }

      (root as Element).visitChildren(visit);
      final e = found;
      final box = e?.renderObject;
      final scrollable = e == null ? null : Scrollable.maybeOf(e);
      if (box == null || scrollable == null) return;
      final viewport = RenderAbstractViewport.maybeOf(box);
      if (viewport == null) return;
      final p = scrollable.position;
      // Whole already: leave the strip where the user put it.
      if (p.pixels >= viewport.getOffsetToReveal(box, 1).offset &&
          p.pixels <= viewport.getOffsetToReveal(box, 0).offset) {
        return;
      }
      p.jumpTo(viewport
          .getOffsetToReveal(box, 0.5)
          .offset
          .clamp(p.minScrollExtent, p.maxScrollExtent));
    });
  }

  @override
  Widget build(BuildContext context) {
    final wide = cIsWide(context);
    final padding = wide
        ? const EdgeInsets.fromLTRB(34, 22, 34, 44)
        : const EdgeInsets.fromLTRB(16, 16, 16, 28);
    final runs = widget.document.runs;
    if (runs.isEmpty) {
      return ListView(
        padding: padding,
        children: [
          if (widget.header != null) widget.header!,
          const SizedBox(height: 40),
          const CEmpty(title: 'Waiting for results…'),
        ],
      );
    }

    final backends = <String>[];
    final testCount = <String, int>{};
    for (final r in runs) {
      if (!backends.contains(r.backend)) backends.add(r.backend);
      testCount[r.backend] = (testCount[r.backend] ?? 0) + r.tests.length;
    }
    final tabs = [if (_finished) _summary, ...backends];
    final tab = tabs.contains(_tab)
        ? _tab!
        : (_finished ? _summary : runs.last.backend);

    final rows = tab == _summary ? _summaryRows(wide) : _backendRows(tab, wide);
    final list = ListView.builder(
      // A list per tab, each keeping its own place: a backend opened from
      // the Summary's foot starts at its top, not at the Summary's offset.
      key: PageStorageKey<String>('results:$tab'),
      padding: padding,
      itemCount: rows.length,
      itemBuilder: (context, i) => RepaintBoundary(child: rows[i].build()),
    );
    if (tabs.length < 2) return list;
    if (!wide) _reveal(tab);
    // Pinned over the list, so another backend is a tap away however far
    // down this one has been read.  Framed and wrapping on a wide window;
    // a scrolling row of pills on a phone, where a swipe reaches the rest.
    return Column(
      children: [
        Padding(
          key: _strip,
          padding: wide
              ? const EdgeInsets.fromLTRB(34, 10, 34, 0)
              : const EdgeInsets.only(top: 8),
          child: CTabStrip(
            tabs: [
              for (final name in tabs)
                name == _summary
                    ? const CTab('Summary',
                        key: ValueKey('results-tab:Summary'))
                    : CTab(backendName(name),
                        count: '${testCount[name]}',
                        key: ValueKey('results-tab:$name')),
            ],
            selected: tabs.indexOf(tab),
            onSelected: (i) => setState(() => _tab = tabs[i]),
            scroll: !wide,
          ),
        ),
        Expanded(child: list),
      ],
    );
  }

  /// The Summary: each processor's flagship numbers, the way into every
  /// backend, then the system.
  List<_Row> _summaryRows(bool wide) {
    final doc = widget.document;
    final catalog = _catalog;
    if (catalog == null) _loadCatalog();
    final measures = RunMeasures.of(doc);
    final unlinked = doc.processors.isEmpty;
    final gap = wide ? 30.0 : 26.0;
    return [
      if (widget.header != null) _WidgetRow(widget.header!, padBottom: 16),
      if (unlinked)
        const _WidgetRow(
          _Note('This file does not link its devices to processors, so each '
              "backend's view of one is listed on its own."),
          padBottom: 14,
        ),
      if (catalog != null && measures.groups.isNotEmpty)
        _WidgetRow(FlagshipGrid(
          cards: [
            for (final group in measures.groups)
              (
                group: group,
                values: [
                  for (final h in catalog.headlines)
                    if (h.appliesTo(group.kind)) (h, group.headline(h)),
                ],
              ),
          ],
          onDevice: _openDevice,
        )),
      _WidgetRow(
        const CSectionHeader(
          title: 'Explore each backend',
          note: 'Every test, on every device, as each backend measured it',
        ),
        padTop: gap,
        padBottom: wide ? 14 : 12,
      ),
      _WidgetRow(BackendTiles(document: doc, onOpen: _openBackend)),
      if (doc.meta != null)
        _WidgetRow(SystemPanel(meta: doc.meta!), padTop: gap),
      if (widget.showDiagnostics && doc.log.isNotEmpty)
        _WidgetRow(_DiagnosticsSection(log: doc.log), padTop: gap),
    ];
  }

  /// One backend's tab, for the device picked among those it reached.
  List<_Row> _backendRows(String backend, bool wide) {
    final doc = widget.document;
    final mine = [for (final r in doc.runs) if (r.backend == backend) r];
    final selected = mine.firstWhere((r) => r.key == _deviceOf[backend],
        // A live run follows the device in flight; a finished one opens on
        // the backend's first.
        orElse: () => _finished ? mine.first : mine.last);
    final kind = deviceKindOf(doc, selected);
    final groups = [
      for (final g in selected.categories)
        if (g.supported case final tests when tests.isNotEmpty) (g, tests),
    ];
    final unavailable = selected.unavailable;
    final total = mine.fold(0, (n, r) => n + r.tests.length);
    final gap = wide ? 20.0 : 14.0;

    // Flat: every test is a row of the lazily built list, so only the ones
    // on screen are built, laid out and painted -- which matters during a
    // live run, where each frame is GPU work competing with the benchmark.
    // A category's table is drawn by its rows, each with its own edges,
    // since wrapping the category in one card would undo that.
    return <_Row>[
      if (widget.header != null) _WidgetRow(widget.header!, padBottom: 16),
      _WidgetRow(
        _BackendHeading(
          backend: backend,
          tests: total,
          devices: mine.length,
          wide: wide,
          onExpandAll: groups.isEmpty ? null : _expandAll,
          onCollapseAll: groups.isEmpty ? null : _collapseAll,
        ),
        padBottom: gap,
      ),
      if (mine.length > 1)
        _WidgetRow(
          _DeviceChips(
            runs: mine,
            kinds: [for (final r in mine) deviceKindOf(doc, r)],
            selectedKey: selected.key,
            onSelected: (key) => setState(() => _deviceOf[backend] = key),
          ),
          padBottom: gap,
        ),
      _WidgetRow(_DeviceCard(
        run: selected,
        processor: doc.processorById(selected.processor),
        kind: kind,
        wide: wide,
      )),
      for (final (group, tests) in groups) ...[
        _WidgetRow(
          GroupHeading(
            label: group.category.label,
            count: countOf(tests.length, 'test'),
            wide: wide,
          ),
          padTop: gap,
          padBottom: wide ? 10 : 8,
        ),
        for (var i = 0; i < tests.length; i++)
          _TestRow(
            selected,
            tests[i],
            kind: kind,
            wide: wide,
            first: i == 0,
            last: i == tests.length - 1,
            forcedExpanded: _forcedExpanded,
            forcedGen: _forcedGen,
          ),
      ],
      if (unavailable.isNotEmpty)
        _WidgetRow(_UnavailableSection(items: unavailable, wide: wide),
            padTop: gap + 6),
      if (widget.showDiagnostics && doc.log.isNotEmpty)
        _WidgetRow(_DiagnosticsSection(log: doc.log), padTop: gap + 6),
    ];
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

class _TestRow extends _Row {
  const _TestRow(
    this.run,
    this.test, {
    required this.kind,
    required this.wide,
    required this.first,
    required this.last,
    this.forcedExpanded,
    this.forcedGen = 0,
  });

  final DeviceRun run;
  final TestResult test;
  final String kind;
  final bool wide;
  final bool first;
  final bool last;
  final bool? forcedExpanded;
  final int forcedGen;

  @override
  Widget build() => _TestLine(
        // Keyed by device and test tag: rows are inserted as results stream
        // in during a live run, and without a key a row's expansion state
        // would follow the index and land on the wrong test -- or, with the
        // tag alone, on the same test of the next device shown.
        key: ValueKey('${run.key}/${test.key}'),
        test: test,
        kind: kind,
        wide: wide,
        first: first,
        last: last,
        forcedExpanded: forcedExpanded,
        forcedGen: forcedGen,
      );
}

/// A muted line of explanation.
class _Note extends StatelessWidget {
  const _Note(this.text);

  final String text;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 6),
        child: Text(text, style: CP.of(context).caption),
      );
}

/// The backend's name and reach, with Expand all and Collapse all.
class _BackendHeading extends StatelessWidget {
  const _BackendHeading({
    required this.backend,
    required this.tests,
    required this.devices,
    required this.wide,
    this.onExpandAll,
    this.onCollapseAll,
  });

  final String backend;
  final int tests;
  final int devices;
  final bool wide;

  /// Null where the device has no test to open.
  final VoidCallback? onExpandAll;
  final VoidCallback? onCollapseAll;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final size = wide ? 28.0 : 24.0;
    final block = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Semantics(
          header: true,
          headingLevel: 2,
          child: Text(
            backendName(backend),
            style: t.h2.copyWith(
                fontSize: size, height: 1.15, letterSpacing: size * -0.035),
          ),
        ),
        Text(
          '${countOf(tests, 'test')} on ${countOf(devices, 'device')}',
          style: t.bodyMuted.copyWith(fontSize: wide ? 14 : 13.5),
        ),
      ],
    );
    final controls = onExpandAll != null;
    final expand = QuietPill(
      label: 'Expand all',
      glyph: CGlyphs.expandAll,
      onPressed: onExpandAll,
      stretch: !wide,
    );
    final collapse = QuietPill(
      label: 'Collapse all',
      glyph: CGlyphs.collapseAll,
      onPressed: onCollapseAll,
      stretch: !wide,
    );
    if (wide) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4),
        child: Row(
          children: [
            Expanded(child: block),
            if (controls) ...[
              const SizedBox(width: 16),
              expand,
              const SizedBox(width: 8),
              collapse,
            ],
          ],
        ),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(padding: const EdgeInsets.symmetric(horizontal: 8), child: block),
        if (controls) ...[
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(child: expand),
              const SizedBox(width: 8),
              Expanded(child: collapse),
            ],
          ),
        ],
      ],
    );
  }
}

/// Which of one backend's devices the readings below belong to.  A Wrap,
/// not a horizontal scroller: with many devices the trailing chips sat
/// off-screen with no way (mouse wheel included) to reach them, and a long
/// driver-given name wraps inside its chip rather than being cut.
class _DeviceChips extends StatelessWidget {
  const _DeviceChips({
    required this.runs,
    required this.kinds,
    required this.selectedKey,
    required this.onSelected,
  });

  /// One backend's devices, and each one's kind.
  final List<DeviceRun> runs;
  final List<String> kinds;
  final String selectedKey;
  final ValueChanged<String> onSelected;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      label: 'Device',
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          for (var i = 0; i < runs.length; i++)
            CDeviceChip(
              // A name is not unique -- MoltenVK shows one GPU twice -- so
              // the backend's index tells such twins apart.
              label: runs.where((r) => r.device == runs[i].device).length > 1
                  ? '${runs[i].device} #${runs[i].index}'
                  : runs[i].device,
              kind: kinds[i].isEmpty ? null : kinds[i],
              selected: runs[i].key == selectedKey,
              onTap: () => onSelected(runs[i].key),
            ),
        ],
      ),
    );
  }
}

/// The device: a card in its kind's tint, headed by the kind and backend,
/// with everything its API reported.  The properties arrive with the very
/// first event of a device's run, so a live run shows them from the start;
/// a saved file adds the ML runtime's provider or accelerator, and which
/// processor the device is a view of and what linked it there.
class _DeviceCard extends StatelessWidget {
  const _DeviceCard({
    required this.run,
    required this.processor,
    required this.kind,
    required this.wide,
  });

  final DeviceRun run;
  final ProcessorInfo? processor;
  final String kind;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final p = processor;
    final rt = run.runtime;
    final entries = <(String, String)>[
      if (run.platform.isNotEmpty && run.platform != run.backend)
        ('Platform', run.platform),
      if (run.driver.isNotEmpty) ('Driver', run.driver),
      if (rt.provider.isNotEmpty) ('Provider', rt.provider),
      if (rt.accelerator.isNotEmpty) ('Accelerator', rt.accelerator),
      if (rt.version.isNotEmpty) ('Runtime', rt.version),
      for (final prop in run.props) (prop.key, prop.value),
      if (run.processor.isNotEmpty)
        (
          'Processor',
          p == null || p.name.isEmpty || p.name == run.device
              ? run.processor
              : '${run.processor} · ${p.name}',
        ),
      if (run.link.isNotEmpty) ('Linked by', run.link),
    ];
    final word = kind.isEmpty ? 'DEVICE' : kind.toUpperCase();
    return CKindCard(
      kind: kind,
      label: '$word · ${backendName(run.backend).toUpperCase()}',
      title: run.device,
      compact: !wide,
      child: entries.isEmpty
          ? const SizedBox.shrink()
          : FactGrid(entries: entries, hue: t.kind(kind), compact: !wide),
    );
  }
}

/// One test: a row of a category's table that opens onto its readings.
///
/// A homogeneous test collapses to its best reading, with the readings a
/// tap away: they are interchangeable variants of one measurement, so one
/// number really is the answer, and the readings mark which one it is.
///
/// A heterogeneous test starts open -- its readings measure different
/// things, and the largest of them is not the test's result but merely its
/// largest number.  Its row names what varies across the readings ("DATA
/// TYPE", "CACHE LEVEL") instead of a value, and it can be closed to that.
///
/// Draws its own edges -- the table's top and rounded corners on the first
/// row, its foot on the last, a hairline between -- so a run of these reads
/// as one card even though each is an independent list row.
class _TestLine extends StatefulWidget {
  const _TestLine({
    super.key,
    required this.test,
    required this.kind,
    required this.wide,
    required this.first,
    required this.last,
    this.forcedExpanded,
    this.forcedGen = 0,
  });

  final TestResult test;

  /// The device's kind, whose accent marks units and the best reading.
  final String kind;
  final bool wide;
  final bool first;
  final bool last;
  final bool? forcedExpanded;
  final int forcedGen;

  @override
  State<_TestLine> createState() => _TestLineState();
}

class _TestLineState extends State<_TestLine> {
  // Null follows the default: heterogeneous with several readings open,
  // everything else closed.  Once the user taps, their choice sticks even
  // as readings stream in during a live run.  Expand all / Collapse all
  // bump [widget.forcedGen] and override it until the next tap.
  bool? _userExpanded;

  @override
  void initState() {
    super.initState();
    _userExpanded = widget.forcedExpanded;
  }

  @override
  void didUpdateWidget(covariant _TestLine oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.forcedGen != oldWidget.forcedGen &&
        widget.forcedExpanded != null) {
      _userExpanded = widget.forcedExpanded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = t.kind(widget.kind);
    final test = widget.test;
    final ok = test.okMetrics;
    final collapsible = test.collapsible;
    final expanded = _userExpanded ?? !collapsible;
    final peak = collapsible ? test.peakMetric : null;
    // Only a homogeneous test has a best reading to mark, and only among
    // several: on a one-row table the mark says nothing.
    final best = peak != null && ok.length > 1 ? peak : null;
    final results = countOf(ok.length, 'result');

    final radius = Radius.circular(widget.wide ? CP.rCard : 22);
    final shape = BorderRadius.vertical(
      top: widget.first ? radius : Radius.zero,
      bottom: widget.last ? radius : Radius.zero,
    );
    final edge = BorderSide(color: t.line);

    final head = widget.wide
        ? _wideHead(t, h, test, peak, results, expanded)
        : _phoneHead(t, h, test, peak, results, expanded);

    return Container(
      decoration: BoxDecoration(color: t.surface, borderRadius: shape),
      // The edges over the children, so a hovered row's wash does not
      // cover them.
      foregroundDecoration: BoxDecoration(
        border: Border(
          top: edge,
          left: edge,
          right: edge,
          bottom: widget.last ? edge : BorderSide.none,
        ),
        borderRadius: shape,
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          CTap(
            onTap: () => setState(() => _userExpanded = !expanded),
            expanded: expanded,
            insetFocus: true,
            focusRadius: shape,
            builder: (context, hovered, pressed) => Container(
              color: hovered || pressed ? hoverWash(t, t.surface) : null,
              child: head,
            ),
          ),
          // Instant, not a cross-fade: expansion happens mid-run, and an
          // animated one would cost the benchmark a burst of frames.
          if (expanded && ok.isNotEmpty)
            Container(
              margin: widget.wide
                  ? const EdgeInsets.fromLTRB(14, 0, 14, 14)
                  : const EdgeInsets.fromLTRB(10, 0, 10, 12),
              padding: EdgeInsets.symmetric(vertical: widget.wide ? 6 : 4),
              decoration: BoxDecoration(
                color: t.well,
                borderRadius:
                    BorderRadius.circular(widget.wide ? CP.rWell : 16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Only measurements: a reading that could not be taken is
                  // listed once, at the foot of the page, rather than
                  // sitting in the middle of a table of numbers.
                  for (final metric in ok)
                    _MetricLine(
                      test: test,
                      metric: metric,
                      best: identical(metric, best),
                      hue: h,
                      wide: widget.wide,
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  /// What stands where a value would: the best reading, or for a
  /// heterogeneous test what its readings vary by.
  Widget? _answer(CP t, KindHue h, TestResult test, MetricResult? peak,
      {required double size, double axisWidth = 170}) {
    if (peak != null) {
      final f = formatValue(peak.value, test.unitsOf(peak));
      return CValue(value: f.value, unit: f.unit, size: size, unitColor: h.accent);
    }
    if (test.axis.isEmpty) return null;
    // "DATA TYPE, PHASE AND CONTEXT LENGTH" explains, it is not the answer:
    // capped, so the title keeps the width.
    return ConstrainedBox(
      constraints: BoxConstraints(maxWidth: axisWidth),
      child: Text(
        test.axis.toUpperCase(),
        style: t.caps,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
      ),
    );
  }

  Widget _wideHead(CP t, KindHue h, TestResult test, MetricResult? peak,
      String results, bool expanded) {
    final preview = descriptionPreview(test.description);
    final answer = _answer(t, h, test, peak, size: 22);
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 13, 16, 13),
      child: ConstrainedBox(
        constraints: const BoxConstraints(minHeight: 42),
        child: Row(
          children: [
            // The title wraps rather than ellipsizing: its tail is what
            // tells two variants of one test apart.
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  _NameWithInfo(
                    name: test.displayTitle,
                    description: test.description,
                    style: t.title,
                  ),
                  if (preview.isNotEmpty) ...[
                    const SizedBox(height: 2),
                    Text(preview,
                        style: t.caption,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis),
                  ],
                ],
              ),
            ),
            if (answer != null) ...[const SizedBox(width: 18), answer],
            const SizedBox(width: 18),
            ConstrainedBox(
              constraints: const BoxConstraints(minWidth: 100),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Text(results,
                      style: t.small.copyWith(fontSize: 12.5, color: t.muted)),
                  const SizedBox(width: 6),
                  _Chevron(open: expanded, size: 18),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _phoneHead(CP t, KindHue h, TestResult test, MetricResult? peak,
      String results, bool expanded) {
    final answer = _answer(t, h, test, peak, size: 20, axisWidth: 190);
    return Padding(
      padding: const EdgeInsets.fromLTRB(18, 12, 14, 12),
      child: ConstrainedBox(
        constraints: const BoxConstraints(minHeight: 48),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  _NameWithInfo(
                    name: test.displayTitle,
                    description: test.description,
                    style: t.title.copyWith(fontSize: 14.5),
                  ),
                  const SizedBox(height: 3),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.baseline,
                    textBaseline: TextBaseline.alphabetic,
                    children: [
                      if (answer != null) ...[
                        Flexible(child: answer),
                        const SizedBox(width: 8),
                      ],
                      Text(results,
                          style:
                              t.small.copyWith(fontSize: 12.5, color: t.muted)),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(width: 10),
            _Chevron(open: expanded, size: 20),
          ],
        ),
      ),
    );
  }
}

/// A row's disclosure mark, turned (not animated) when open.
class _Chevron extends StatelessWidget {
  const _Chevron({required this.open, required this.size});

  final bool open;
  final double size;

  @override
  Widget build(BuildContext context) => Transform.rotate(
        angle: open ? math.pi : 0,
        child: CGlyph(CGlyphs.chevronDown,
            size: size, color: CP.of(context).muted, stroke: 2.2),
      );
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

  /// Sized for a reading's smaller type.
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

/// The info affordance: a glyph beside a name, present only when that name
/// has an explanation.  Its own tap target, so it opens the description
/// instead of opening the row it sits in.
///
/// Used at both levels — beside a test's title for what the test measures,
/// and beside a reading's label for what that one reading means, each
/// opening its own text.  The line under a name previews that text's first
/// sentence; the glyph has all of it.
class _InfoGlyph extends StatelessWidget {
  const _InfoGlyph({
    required this.title,
    required this.description,
    this.small = false,
  });

  final String title;
  final String description;

  /// Sized for a reading's smaller type.
  final bool small;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: () => _showInfoDialog(context, title, description),
      semanticLabel: 'About $title',
      focusRadius: BorderRadius.circular(CP.rPill),
      builder: (context, hovered, pressed) => Padding(
        padding: EdgeInsets.fromLTRB(small ? 5 : 6, 2, 2, 2),
        child: CGlyph(
          CGlyphs.about,
          size: small ? 14 : 16,
          stroke: 2.1,
          color: hovered || pressed ? t.ink : t.muted,
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
    barrierColor: Theme.of(context).dialogTheme.barrierColor ??
        Colors.black.withValues(alpha: 0.55),
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
          child: Text(description,
              style: t.prose.copyWith(fontSize: 14, color: t.ink)),
        ),
      ),
    );
  }
}

/// One reading in an open test: its label and what it means, the value,
/// and on a homogeneous test the mark of the one the row collapses to.
class _MetricLine extends StatelessWidget {
  const _MetricLine({
    required this.test,
    required this.metric,
    required this.best,
    required this.hue,
    required this.wide,
  });

  final TestResult test;
  final MetricResult metric;
  final bool best;
  final KindHue hue;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final f = formatValue(metric.value, test.unitsOf(metric));
    final preview = descriptionPreview(metric.description);
    final label = _NameWithInfo(
      name: metric.displayLabel,
      description: metric.description,
      style: t.label.copyWith(fontSize: wide ? 13.5 : 13, height: 1.3),
      small: true,
    );
    final note = preview.isEmpty
        ? null
        : Text(preview,
            style: t.small.copyWith(fontSize: wide ? 12.5 : 12, color: t.muted),
            maxLines: 2,
            overflow: TextOverflow.ellipsis);
    final value =
        CValue(value: f.value, unit: f.unit, small: true, size: wide ? 14.5 : 13.5);
    final badge = best ? _BestBadge(hue: hue, small: !wide) : null;

    if (wide) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 8),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [label, ?note],
              ),
            ),
            if (badge != null) ...[const SizedBox(width: 12), badge],
            const SizedBox(width: 12),
            ConstrainedBox(
              constraints: const BoxConstraints(minWidth: 112),
              child: Align(
                alignment: Alignment.centerRight,
                widthFactor: 1,
                child: value,
              ),
            ),
          ],
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(child: label),
              if (badge != null) ...[const SizedBox(width: 8), badge],
              const SizedBox(width: 8),
              value,
            ],
          ),
          if (note != null) ...[const SizedBox(height: 1), note],
        ],
      ),
    );
  }
}

/// "Best": the reading a homogeneous test collapses to, on its kind's accent.
class _BestBadge extends StatelessWidget {
  const _BestBadge({required this.hue, this.small = false});

  final KindHue hue;
  final bool small;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      height: small ? 20 : 22,
      padding: EdgeInsets.symmetric(horizontal: small ? 8 : 9),
      decoration: BoxDecoration(
        color: hue.accent,
        borderRadius: BorderRadius.circular(CP.rPill),
      ),
      child: Center(
        widthFactor: 1,
        child: Text(
          'Best',
          style: t.small.copyWith(
            fontSize: small ? 11 : 11.5,
            fontWeight: FontWeight.w700,
            height: 1,
            color: hue.onAccent,
          ),
        ),
      ),
    );
  }
}

/// Everything the device could not measure, in one place at the foot of the
/// page: whole tests it does not support, and the individual readings missing
/// from tests that otherwise ran.  Both answer the same question, and keeping
/// a dead row out of the tables above is what lets them be nothing but
/// measurements.
class _UnavailableSection extends StatelessWidget {
  const _UnavailableSection({required this.items, required this.wide});

  final List<UnavailableItem> items;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final whole = items.where((i) => i.label.isEmpty).length;
    final readings = items.length - whole;
    final count = [
      if (whole > 0) countOf(whole, 'test'),
      if (readings > 0) countOf(readings, 'reading'),
    ].join(' · ');
    final titleStyle = t.body.copyWith(
        fontSize: wide ? 14 : 13.5, fontWeight: FontWeight.w600, color: t.muted);
    final reasonStyle =
        t.small.copyWith(fontSize: wide ? 13 : 12, color: t.muted);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        GroupHeading(
          label: 'Not available on this device',
          count: count,
          wide: wide,
          muted: true,
        ),
        SizedBox(height: wide ? 10 : 8),
        CEmptyBox(
          padding: EdgeInsets.zero,
          radius: wide ? CP.rCard : 22,
          center: false,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              for (var i = 0; i < items.length; i++)
                Container(
                  decoration: i == 0
                      ? null
                      : BoxDecoration(
                          border: Border(top: BorderSide(color: t.line))),
                  padding: wide
                      ? const EdgeInsets.fromLTRB(20, 12, 20, 12)
                      : const EdgeInsets.fromLTRB(18, 11, 18, 11),
                  child: Builder(builder: (context) {
                    final item = items[i];
                    final title = _NameWithInfo(
                      name: item.title,
                      description: item.description,
                      style: titleStyle,
                      small: true,
                    );
                    final why = item.reason.isNotEmpty
                        ? item.reason
                        : switch (item.status) {
                            ResultStatus.unsupported => 'Not supported',
                            ResultStatus.skipped => 'Skipped',
                            ResultStatus.error => 'Failed',
                            ResultStatus.ok => '',
                          };
                    if (wide) {
                      return Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(flex: 5, child: title),
                          const SizedBox(width: 16),
                          Expanded(
                              flex: 4, child: Text(why, style: reasonStyle)),
                        ],
                      );
                    }
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        title,
                        if (why.isNotEmpty) Text(why, style: reasonStyle),
                      ],
                    );
                  }),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

/// The run's diagnostic stream, at the foot of the page.  Counts-only by
/// design: rendering per-line log rows (up to hundreds of rich-text rows,
/// rebuilt on every live tick) made the GUI sluggish on phones during heavy
/// runs.  The full log — problems, debug lines, runtimes' own messages —
/// stays in the document and in the file Share sends; the section just says
/// how much of it there is.
///
/// Run-level, not per device: the stream is one run's, in order, and a line
/// between two devices belongs to neither.
class _DiagnosticsSection extends StatelessWidget {
  const _DiagnosticsSection({required this.log});

  final List<LogEntry> log;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CSection(label: 'Diagnostics', trailing: countOf(log.length, 'line')),
        const SizedBox(height: 10),
        CPanel(
          color: t.surface2,
          outlined: false,
          radius: CP.rPanel,
          padding: const EdgeInsets.fromLTRB(18, 13, 18, 14),
          child: Text("The run's file holds the full log.", style: t.prose),
        ),
      ],
    );
  }
}
