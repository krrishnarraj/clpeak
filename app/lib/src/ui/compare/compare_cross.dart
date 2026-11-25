part of 'compare_screen.dart';

// Two machines (or one with a processor the other run did not see): the
// processors paired, each pair's headlines, then every measure both runs
// have, by processor or backend by backend.

const _crossHeadNote = "Each processor's peaks, which can be at different "
    'data types. The measures below compare like with like.';

/// A pair's kind, as its tab and its chip say it: `GPU`, or for two kinds a
/// person paired, `CPU · GPU`.
String _pairKind(PairCompare p) => p.a.kind == p.b.kind
    ? kindLabel(p.a.kind)
    : '${kindLabel(p.a.kind)} · ${kindLabel(p.b.kind)}';

/// Its tab's label: the kind, and where two pairs share one, the A side's
/// processor too.
String _pairTab(RunCompare c, PairCompare p) {
  final kind = _pairKind(p);
  final twice = c.pairs.where((x) => _pairKind(x) == kind).length > 1;
  return twice ? '$kind · ${p.a.name}' : kind;
}

/// A backend's devices on one side, as a line names them.
String _devicesText(List<DeviceRun> devices) =>
    devices.map((d) => d.device).join(' + ');

extension _Cross on _CompareScreenState {
  List<_Item> _crossItems({required bool wide, required bool rowsWide}) {
    final c = _c;
    final pair = _currentPair;
    final items = <_Item>[];

    if (wide) {
      items
        ..add(_Item((ctx) => const CSectionHeader(title: 'Processors'),
            top: 34))
        ..add(_Item((ctx) => _strip(), top: 10))
        ..add(_Item(
            (ctx) => const CSectionHeader(
                title: 'Headlines', note: _crossHeadNote),
            top: 34))
        ..add(_Item(
            (ctx) => _Grid(minWidth: 320, children: [
                  for (final p in c.pairs)
                    _HeadCard(
                      kind: _pairKind(p),
                      a: p.a.name,
                      b: p.b.name,
                      headlines: p.headlines,
                      sameMode: false,
                      wide: true,
                    ),
                  for (final g in c.unpairedA)
                    _UnpairedCard(
                        run: c.a,
                        group: g,
                        letter: 'A',
                        catalog: c.catalog,
                        wide: true),
                  for (final g in c.unpairedB)
                    _UnpairedCard(
                        run: c.b,
                        group: g,
                        letter: 'B',
                        catalog: c.catalog,
                        wide: true),
                ]),
            top: 12));
    } else {
      if (c.pairs.length > 1) {
        items.add(_Item((ctx) => _pairTabs(wide: false), top: 22, inset: false));
      }
      if (pair != null) {
        items
          ..add(_Item(
              (ctx) => _PairLine(
                    pair: pair,
                    onChange: () => _repair(pair.a, CompareSide.a),
                  ),
              top: c.pairs.length > 1 ? 8 : 22))
          ..add(_Item(
              (ctx) => _HeadCard(
                    caps: 'HEADLINES',
                    headlines: pair.headlines,
                    sameMode: false,
                    wide: false,
                  ),
              top: 10));
      }
      for (final (letter, run, groups) in [
        ('A', c.a, c.unpairedA),
        ('B', c.b, c.unpairedB),
      ]) {
        for (final g in groups) {
          items.add(_Item(
              (ctx) => _UnpairedCard(
                    run: run,
                    group: g,
                    letter: letter,
                    catalog: c.catalog,
                    wide: false,
                    onChange: () => _repair(
                        g, letter == 'A' ? CompareSide.a : CompareSide.b),
                  ),
              top: 10));
        }
      }
    }

    final measures = pair == null ? 0 : pair.both.length;
    items.addAll(_detailHead(
      title: _byBackend ? 'Backend by backend' : 'Measured on both',
      note: _byBackend
          ? "Each side's best through one backend, for the backends both "
              'runs have'
          : '${plural(measures, 'measure')} in both runs · each side\'s '
              'best, from whichever backend reached it',
      phoneNote: _byBackend
          ? 'The backends both runs have'
          : '${plural(measures, 'measure')} in both runs',
      wide: wide,
      cross: true,
    ));
    items.addAll(_byBackend
        ? _crossByBackend(wide: wide, rowsWide: rowsWide)
        : _crossByProcessor(wide: wide, rowsWide: rowsWide));
    return items;
  }

  /// Which pair the measures below are of.
  Widget _pairTabs({required bool wide}) => CTabStrip(
        scroll: !wide,
        selected: _c.pairs.indexOf(_currentPair!),
        onSelected: (i) => _set(() {
          _pair = i;
          _open.clear();
        }),
        tabs: [
          for (final (i, p) in _c.pairs.indexed)
            CTab(_pairTab(_c, p),
                count: '${p.both.length}', key: ValueKey('compare-pair:$i')),
        ],
      );

  /// Every pair, then every processor no pair holds, each with Change.
  Widget _strip() {
    final c = _c;
    final rows = <Widget>[
      for (final p in c.pairs)
        _StripRow(
          kind: _pairKind(p),
          a: p.a.name,
          b: p.b.name,
          how: pairedByText(p.pairedBy),
          onChange: () => _repair(p.a, CompareSide.a),
        ),
      for (final g in c.unpairedA)
        _StripRow(
          kind: kindLabel(g.kind),
          a: g.name,
          b: 'Nothing to pair with',
          bMuted: true,
          how: 'unpaired',
          onChange: () => _repair(g, CompareSide.a),
        ),
      for (final g in c.unpairedB)
        _StripRow(
          kind: kindLabel(g.kind),
          a: 'Nothing to pair with',
          aMuted: true,
          b: g.name,
          how: 'unpaired',
          onChange: () => _repair(g, CompareSide.b),
        ),
    ];
    return CPanel(
      radius: CP.rCard,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (final (i, r) in rows.indexed)
            i == 0 ? r : _Ruled(child: r),
        ],
      ),
    );
  }

  /// A value's row of a measure both sides have, opened by [key].
  Widget _keyRow(KeyCompare r, String key, bool rowsWide,
      String Function(CompareValue v) caption) {
    return _MeasureRow(
      label: familyTitle(r.key.family),
      words: r.label.join(' · '),
      a: r.a.best,
      b: r.b.best,
      verdict: r.verdict,
      captionA: caption(r.a.best),
      captionB: caption(r.b.best),
      open: _open.contains(key),
      onToggle: () => _toggle(key),
      wide: rowsWide,
    );
  }

  Widget _onlyRow(MeasureRow r, bool rowsWide, String caption) {
    final v = valueText(r.best);
    return _OnlyRow(
      label: familyTitle(r.key.family),
      words: r.label.join(' · '),
      value: v.value,
      unit: v.unit,
      caption: caption,
      wide: rowsWide,
    );
  }

  List<_Item> _crossByProcessor({required bool wide, required bool rowsWide}) {
    final c = _c;
    final pair = _currentPair;
    final items = <_Item>[];
    if (wide && c.pairs.length > 1) {
      items.add(_Item((ctx) => _pairTabs(wide: true), top: 16));
    }
    if (pair == null) {
      items.add(_Item(
          (ctx) => const CEmptyBox(
              text: 'No processor of one run is paired with one of the '
                  'other. Pair two with Change.'),
          top: 16));
      return items;
    }
    final nameA = sideName(c.a), nameB = sideName(c.b);
    final scope = 'P:${pair.a.id}:${pair.b.id}';
    if (pair.both.isEmpty) {
      items.add(_Item(
          (ctx) => const CEmptyBox(text: 'Nothing both processors measured.'),
          top: 16));
    } else {
      items.add(_columns('MEASURE', nameA, nameB,
          rowsWide: rowsWide, chevron: true));
      for (final g in pair.bothByCategory) {
        items.addAll(_category<KeyCompare>(
          g,
          'measure',
          (r) => _keyRow(r, '$scope:${r.key.id}', rowsWide,
              (v) => backendName(v.backend)),
          (r) => '$scope:${r.key.id}',
        ));
      }
    }
    items.addAll(_folds(
      scope: scope,
      labelA: 'Only on $nameA',
      labelB: 'Only on $nameB',
      countA: '${pair.onlyA.length}',
      countB: '${pair.onlyB.length}',
      hasA: pair.onlyA.isNotEmpty,
      hasB: pair.onlyB.isNotEmpty,
      rowsA: () => [
        for (final r in pair.onlyA)
          _onlyRow(r, rowsWide, backendName(r.best.backend)),
      ],
      rowsB: () => [
        for (final r in pair.onlyB)
          _onlyRow(r, rowsWide, backendName(r.best.backend)),
      ],
    ));
    return items;
  }

  List<_Item> _crossByBackend({required bool wide, required bool rowsWide}) {
    final c = _c;
    final nameA = sideName(c.a), nameB = sideName(c.b);
    final items = <_Item>[];
    final names = c.backends;

    Widget onlyNotes() => _Notes(wide: wide, notes: [
          if (c.backendsOnlyA.isNotEmpty)
            ('A', 'Only on $nameA: ${c.backendsOnlyA.map(backendName).join(', ')}'),
          if (c.backendsOnlyB.isNotEmpty)
            ('B', 'Only on $nameB: ${c.backendsOnlyB.map(backendName).join(', ')}'),
        ]);
    final hasOnly = c.backendsOnlyA.isNotEmpty || c.backendsOnlyB.isNotEmpty;

    if (names.isEmpty) {
      items.add(_Item(
          (ctx) => const CEmptyBox(
              text: 'The two runs have no backend in common.'),
          top: 16));
      if (hasOnly) items.add(_Item((ctx) => onlyNotes(), top: 12));
      return items;
    }
    final be = names.contains(_backend) ? _backend! : names.first;
    final shared = [
      for (final p in c.pairs)
        for (final x in p.backends)
          if (x.backend == be) (p, x),
    ];
    int countOf(String name) => [
          for (final p in c.pairs)
            for (final x in p.backends)
              if (x.backend == name) x.both.length,
        ].fold(0, (a, b) => a + b);

    items.add(_Item(
        (ctx) => CTabStrip(
              scroll: !wide,
              selected: names.indexOf(be),
              onSelected: (i) => _set(() {
                _backend = names[i];
                _open.clear();
              }),
              tabs: [
                for (final n in names)
                  CTab(backendName(n),
                      count: '${countOf(n)}',
                      key: ValueKey('compare-backend:$n')),
              ],
            ),
        top: 16,
        inset: wide));
    if (hasOnly) items.add(_Item((ctx) => onlyNotes(), top: 10));

    final si = shared.isEmpty ? 0 : (_sub[be] ?? 0).clamp(0, shared.length - 1);
    if (shared.length > 1) {
      items.add(_Item(
          (ctx) => _ChipRow(
                wide: wide,
                chips: [
                  for (final (i, (p, x)) in shared.indexed)
                    CDeviceChip(
                      key: ValueKey('compare-backend-pair:$be:$i'),
                      label: '${_pairKind(p)} · ${_devicesText(x.devicesA)} '
                          'and ${_devicesText(x.devicesB)}',
                      kind: p.a.kind == p.b.kind ? p.a.kind : null,
                      count: '${x.both.length}',
                      selected: i == si,
                      onTap: () => _set(() {
                        _sub[be] = i;
                        _open.clear();
                      }),
                    ),
                ],
              ),
          top: 10,
          inset: wide));
    }

    if (shared.isEmpty) {
      items.add(_Item(
          (ctx) => CEmptyBox(
              text: 'Nothing measured through ${backendName(be)} on a '
                  'processor of both runs.'),
          top: 14));
    } else {
      final (pair, x) = shared[si];
      final scope = 'B:$be:${pair.a.id}:${pair.b.id}';
      items.add(_Item(
          (ctx) => _BackendPeaks(
                backend: be,
                kind: _pairKind(pair),
                a: _devicesText(x.devicesA),
                b: _devicesText(x.devicesB),
                headlines: x.headlines,
                wide: wide,
              ),
          top: 14));
      if (x.both.isEmpty) {
        items.add(_Item(
            (ctx) => CEmptyBox(
                text: 'Nothing both measured through ${backendName(be)}.'),
            top: 16));
      } else {
        items.add(_columns('MEASURE', nameA, nameB,
            rowsWide: rowsWide, chevron: true));
        for (final g in x.bothByCategory) {
          items.addAll(_category<KeyCompare>(
            g,
            'measure',
            (r) => _keyRow(
                r, '$scope:${r.key.id}', rowsWide, (v) => v.metric.id),
            (r) => '$scope:${r.key.id}',
          ));
        }
      }
      // A shared backend's devices no pair sets side by side through it.
      String alone(String name, CompareRun run, List<DeviceRun> devices) =>
          '$name alone, through ${backendName(be)}: '
          '${devices.map((d) => '${d.device} (${kindLabel(deviceKind(run, d))})').join(', ')}';
      final da = [for (final d in c.unpairedDevicesA) if (d.backend == be) d];
      final db = [for (final d in c.unpairedDevicesB) if (d.backend == be) d];
      if (da.isNotEmpty || db.isNotEmpty) {
        items.add(_Item(
            (ctx) => _Notes(wide: wide, column: true, notes: [
                  if (da.isNotEmpty) ('A', alone(nameA, c.a, da)),
                  if (db.isNotEmpty) ('B', alone(nameB, c.b, db)),
                ]),
            top: 14));
      }
      final through = ', through ${backendName(be)}';
      items.addAll(_folds(
        scope: scope,
        labelA: 'Only on $nameA$through',
        labelB: 'Only on $nameB$through',
        countA: '${x.onlyA.length}',
        countB: '${x.onlyB.length}',
        hasA: x.onlyA.isNotEmpty,
        hasB: x.onlyB.isNotEmpty,
        rowsA: () => [
          for (final r in x.onlyA) _onlyRow(r, rowsWide, r.best.metric.id),
        ],
        rowsB: () => [
          for (final r in x.onlyB) _onlyRow(r, rowsWide, r.best.metric.id),
        ],
      ));
    }
    return items;
  }
}

// ── Pieces ─────────────────────────────────────────────────────────────────

/// [child] under a hairline: a row of a card after its first.
class _Ruled extends StatelessWidget {
  const _Ruled({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) => DecoratedBox(
        decoration: BoxDecoration(
          border: Border(top: BorderSide(color: CP.of(context).line)),
        ),
        child: child,
      );
}

/// One row of the processors strip: the kind, A's processor and B's, why
/// they are side by side, and Change.
class _StripRow extends StatelessWidget {
  const _StripRow({
    required this.kind,
    required this.a,
    required this.b,
    required this.how,
    required this.onChange,
    this.aMuted = false,
    this.bMuted = false,
  });

  final String kind;
  final String a;
  final String b;
  final String how;
  final VoidCallback onChange;
  final bool aMuted;
  final bool bMuted;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final name = t.body.copyWith(fontSize: 14.5, fontWeight: FontWeight.w700);
    return Padding(
      padding: const EdgeInsets.fromLTRB(18, 8, 10, 8),
      child: Row(
        children: [
          CKindLabel('', text: kind, neutral: true),
          const SizedBox(width: 14),
          Expanded(
            child: Wrap(
              spacing: 10,
              runSpacing: 4,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                _Badged('A', a, style: name, muted: aMuted),
                CGlyph(CGlyphs.pair, size: 18, color: t.muted, stroke: 2),
                _Badged('B', b, style: name, muted: bMuted),
              ],
            ),
          ),
          const SizedBox(width: 14),
          Text(how, style: t.caption.copyWith(fontSize: 12.5)),
          const SizedBox(width: 4),
          CButton(
              label: 'Change', kind: CButtonKind.quiet, onPressed: onChange),
        ],
      ),
    );
  }
}

/// A phone's pair under the pair tabs: A's processor, B's, why, and Change.
class _PairLine extends StatelessWidget {
  const _PairLine({required this.pair, required this.onChange});

  final PairCompare pair;
  final VoidCallback onChange;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final name = t.body.copyWith(fontSize: 13, fontWeight: FontWeight.w700);
    return Padding(
      padding: const EdgeInsets.only(left: 8),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _Badged('A', pair.a.name, style: name, size: CBadgeSize.xs),
                const SizedBox(height: 3),
                _Badged('B', pair.b.name, style: name, size: CBadgeSize.xs),
                const SizedBox(height: 3),
                Text(pairedByText(pair.pairedBy),
                    style: t.smallMuted.copyWith(fontWeight: FontWeight.w600)),
              ],
            ),
          ),
          CButton(
              label: 'Change', kind: CButtonKind.quiet, onPressed: onChange),
        ],
      ),
    );
  }
}

/// Lines that start with a run's badge, in a row on a wide window (or
/// [column]), one under the other on a phone.
class _Notes extends StatelessWidget {
  const _Notes({required this.notes, required this.wide, this.column = false});

  final List<(String, String)> notes;
  final bool wide;
  final bool column;

  @override
  Widget build(BuildContext context) {
    final lines = [for (final (l, s) in notes) _Note(l, s, wide: wide)];
    if (wide && !column) {
      return Wrap(spacing: 20, runSpacing: 4, children: lines);
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (final (i, l) in lines.indexed) ...[
          if (i > 0) const SizedBox(height: 4),
          l,
        ],
      ],
    );
  }
}

/// Chips that wrap on a wide window and scroll sideways on a phone.
class _ChipRow extends StatelessWidget {
  const _ChipRow({required this.chips, required this.wide});

  final List<Widget> chips;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    if (wide) return Wrap(spacing: 8, runSpacing: 8, children: chips);
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Row(
        children: [
          for (final (i, c) in chips.indexed) ...[
            if (i > 0) const SizedBox(width: 6),
            ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 300), child: c),
          ],
        ],
      ),
    );
  }
}

/// A processor of one run no pair holds: its headlines alone, in a dashed
/// card.
class _UnpairedCard extends StatelessWidget {
  const _UnpairedCard({
    required this.run,
    required this.group,
    required this.letter,
    required this.catalog,
    required this.wide,
    this.onChange,
  });

  final CompareRun run;

  /// The headlines to show it by: the compare's.
  final HeadlineCatalog catalog;
  final ProcessorMeasures group;
  final String letter;
  final bool wide;

  /// A phone's Change; the desktop's is in the processors strip.
  final VoidCallback? onChange;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final other = letter == 'A' ? 'B' : 'A';
    final heads = [
      for (final h in catalog.headlines)
        if (h.appliesTo(group.kind))
          (h, group.headline(h)),
    ];
    final name = t.body.copyWith(
        fontSize: wide ? 13 : 13.5, fontWeight: FontWeight.w700);
    final head = wide
        ? Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CKindLabel('', text: kindLabel(group.kind), neutral: true),
              const SizedBox(height: 6),
              _Badged(letter, group.name, style: name),
              const SizedBox(height: 4),
              Text('Nothing in $other to pair with', style: t.caption),
            ],
          )
        : Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      spacing: 8,
                      runSpacing: 4,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        CKindLabel('',
                            text: kindLabel(group.kind),
                            neutral: true,
                            small: true),
                        _Badged(letter, group.name,
                            style: name, size: CBadgeSize.xs),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text('Nothing in $other to pair with',
                        style: t.caption.copyWith(fontSize: 12.5)),
                  ],
                ),
              ),
              if (onChange != null)
                CButton(
                    label: 'Change',
                    kind: CButtonKind.quiet,
                    onPressed: onChange),
            ],
          );
    final rows = <Widget>[
      for (final (i, (h, v)) in heads.indexed)
        wide
            ? Container(
                margin: const EdgeInsets.only(top: 10),
                padding: const EdgeInsets.only(top: 10),
                decoration: BoxDecoration(
                  border: Border(
                      top: BorderSide(
                          color: i == 0 ? Colors.transparent : t.line)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _TileLabel(headlineLabel(h)),
                    const SizedBox(height: 6),
                    _Tile(
                      value: v == null
                          ? null
                          : CompareValue.of(run, v.best),
                      side: letter == 'A' ? CompareSide.a : CompareSide.b,
                      marks: const _Marks.none(),
                      caption: v == null
                          ? 'Not measured'
                          : headCaption(CompareValue.of(run, v.best)),
                    ),
                  ],
                ),
              )
            : Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.baseline,
                  textBaseline: TextBaseline.alphabetic,
                  children: [
                    Expanded(
                      child: Text(headlineLabel(h),
                          style: t.caption.copyWith(fontSize: 12.5)),
                    ),
                    if (v == null)
                      Text('—',
                          style: t.figure(15,
                              weight: FontWeight.w600, color: t.muted))
                    else
                      CValue(
                        value: valueText(CompareValue.of(run, v.best)).value,
                        unit: valueText(CompareValue.of(run, v.best)).unit,
                        size: 15,
                        weight: FontWeight.w600,
                        unitColor: t.ink,
                      ),
                  ],
                ),
              ),
    ];
    return CEmptyBox(
      padding: wide
          ? const EdgeInsets.fromLTRB(20, 18, 20, 16)
          : const EdgeInsets.fromLTRB(16, 14, 8, 14),
      radius: wide ? CP.rCardLarge : 22,
      center: false,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          head,
          if (wide) ...rows else
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: rows,
              ),
            ),
        ],
      ),
    );
  }
}

/// Each side's peaks through one backend: three headline columns on a wide
/// window, one under the other on a phone.
class _BackendPeaks extends StatelessWidget {
  const _BackendPeaks({
    required this.backend,
    required this.kind,
    required this.a,
    required this.b,
    required this.headlines,
    required this.wide,
  });

  final String backend;
  final String kind;
  final String a;
  final String b;
  final List<HeadlineCompare> headlines;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final name = t.body.copyWith(
        fontSize: wide ? 13.5 : 12.5, fontWeight: FontWeight.w700);
    final title = Text('Peaks through ${backendName(backend)}',
        style: t.h3.copyWith(fontSize: wide ? 16 : 15));
    final head = wide
        ? Wrap(
            spacing: 12,
            runSpacing: 6,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              title,
              CKindLabel('', text: kind, neutral: true),
              _Badged('A', a, style: name),
              _Badged('B', b, style: name),
            ],
          )
        : Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Wrap(
                spacing: 8,
                runSpacing: 4,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  title,
                  CKindLabel('', text: kind, neutral: true, small: true),
                ],
              ),
              const SizedBox(height: 4),
              _Badged('A', a, style: name, size: CBadgeSize.xs, gap: 6),
              const SizedBox(height: 4),
              _Badged('B', b, style: name, size: CBadgeSize.xs, gap: 6),
            ],
          );
    Widget column(HeadlineCompare h) {
      final marks = _Marks.of(h.verdict);
      String cap(CompareValue? v) => v == null ? 'Not measured' : wordsOf(v);
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _TileLabel(headlineLabel(h.headline), small: !wide),
          const SizedBox(height: 6),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _Tile(
                  value: h.a,
                  side: CompareSide.a,
                  marks: marks,
                  caption: cap(h.a),
                  size: wide ? 23 : 20,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _Tile(
                  value: h.b,
                  side: CompareSide.b,
                  marks: marks,
                  caption: cap(h.b),
                  size: wide ? 23 : 20,
                ),
              ),
            ],
          ),
        ],
      );
    }

    return CPanel(
      radius: wide ? CP.rCardLarge : 22,
      padding: wide
          ? const EdgeInsets.fromLTRB(20, 18, 20, 16)
          : const EdgeInsets.fromLTRB(16, 14, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          head,
          const SizedBox(height: 12),
          if (headlines.isEmpty)
            Text('No headline measured through ${backendName(backend)}.',
                style: t.caption)
          else if (wide)
            _Grid(
              minWidth: 280,
              gap: 16,
              runGap: 12,
              children: [for (final h in headlines) column(h)],
            )
          else
            for (final (i, h) in headlines.indexed)
              Container(
                padding: EdgeInsets.only(top: i == 0 ? 0 : 8),
                margin: EdgeInsets.only(top: i == 0 ? 0 : 8),
                decoration: BoxDecoration(
                  border: Border(
                      top: BorderSide(
                          color: i == 0 ? Colors.transparent : t.line)),
                ),
                child: column(h),
              ),
        ],
      ),
    );
  }
}
