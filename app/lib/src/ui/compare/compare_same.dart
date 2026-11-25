part of 'compare_screen.dart';

// One machine twice: each processor's headlines in each run, then every
// reading both runs took, backend by backend and device by device.  No
// processors strip -- every processor is its own partner -- and no derived
// numbers: the two readings, the better one marked.

const _sameHeadNote = "Each processor's peaks in each run. Where the two "
    'came from different tests, they are not like for like.';

/// A device both runs saw, and the pair of processors it is a view of.
typedef _SameDevice = ({PairCompare pair, DeviceCompare device});

extension _Same on _CompareScreenState {
  /// Every device both runs saw, by backend in A's run order, each
  /// backend's in A's order.
  Map<String, List<_SameDevice>> _sameDevices() {
    final runs = _c.a.document.runs;
    final out = <String, List<_SameDevice>>{
      for (final r in runs) r.backend: [],
    };
    for (final p in _c.pairs) {
      for (final d in p.devices) {
        out.putIfAbsent(d.backend, () => []).add((pair: p, device: d));
      }
    }
    for (final list in out.values) {
      list.sort(
          (x, y) => runs.indexOf(x.device.a).compareTo(runs.indexOf(y.device.a)));
    }
    out.removeWhere((_, list) => list.isEmpty);
    return out;
  }

  List<_Item> _sameItems({required bool wide, required bool rowsWide}) {
    final c = _c;
    final (whenA, whenB) = sideWhen(c.a, c.b);
    final items = <_Item>[];

    if (wide) {
      items
        ..add(_Item(
            (ctx) =>
                const CSectionHeader(title: 'Headlines', note: _sameHeadNote),
            top: 34))
        ..add(_Item(
            (ctx) => _Grid(minWidth: 320, children: [
                  for (final p in c.pairs)
                    _HeadCard(
                      kind: kindLabel(p.a.kind),
                      title: p.a.name,
                      a: whenA,
                      b: whenB,
                      headlines: p.headlines,
                      sameMode: true,
                      wide: true,
                    ),
                ]),
            top: 12));
    } else {
      for (final (i, p) in c.pairs.indexed) {
        items.add(_Item(
            (ctx) => _HeadCard(
                  kind: kindLabel(p.a.kind),
                  title: p.a.name,
                  a: whenA,
                  b: whenB,
                  headlines: p.headlines,
                  sameMode: true,
                  wide: false,
                ),
            top: i == 0 ? 16 : 12));
      }
    }

    final byBackend = _sameDevices();
    final total = [
      for (final list in byBackend.values)
        for (final d in list) d.device.readings.length,
    ].fold(0, (a, b) => a + b);
    items.addAll(_detailHead(
      title: 'Reading by reading',
      note: '${plural(total, 'reading')} in both runs, by backend and device',
      phoneNote: '${plural(total, 'reading')} in both runs',
      wide: wide,
      cross: false,
    ));
    if (byBackend.isEmpty) {
      items.add(_Item(
          (ctx) =>
              const CEmptyBox(text: 'No reading was taken in both runs.'),
          top: 16));
      return items;
    }

    final names = byBackend.keys.toList();
    final be = names.contains(_backend) ? _backend! : names.first;
    final devices = byBackend[be]!;
    int countOf(String name) => byBackend[name]!
        .fold(0, (n, d) => n + d.device.readings.length);
    items.add(_Item(
        (ctx) => CTabStrip(
              scroll: !wide,
              selected: names.indexOf(be),
              onSelected: (i) => _set(() => _backend = names[i]),
              tabs: [
                for (final n in names)
                  CTab(backendName(n),
                      count: '${countOf(n)}',
                      key: ValueKey('compare-backend:$n')),
              ],
            ),
        top: 16,
        inset: wide));

    // What one run alone has: a backend, or a device of one.
    final onlyA = [
      ...c.backendsOnlyA.map(backendName),
      for (final p in c.pairs)
        for (final d in p.devicesOnlyA)
          if (d.backend == be) '${d.device} (${backendName(be)})',
    ];
    final onlyB = [
      ...c.backendsOnlyB.map(backendName),
      for (final p in c.pairs)
        for (final d in p.devicesOnlyB)
          if (d.backend == be) '${d.device} (${backendName(be)})',
    ];
    if (onlyA.isNotEmpty || onlyB.isNotEmpty) {
      items.add(_Item(
          (ctx) => _Notes(wide: wide, notes: [
                if (onlyA.isNotEmpty)
                  ('A', 'Only in the $whenA run: ${onlyA.join(', ')}'),
                if (onlyB.isNotEmpty)
                  ('B', 'Only in the $whenB run: ${onlyB.join(', ')}'),
              ]),
          top: 10));
    }

    final si = (_sub[be] ?? 0).clamp(0, devices.length - 1);
    String chipName(DeviceCompare d) =>
        d.occurrence == 0 ? d.name : '${d.name} (${d.occurrence + 1})';
    if (devices.length > 1) {
      items.add(_Item(
          (ctx) => _ChipRow(
                wide: wide,
                chips: [
                  for (final (i, d) in devices.indexed)
                    CDeviceChip(
                      key: ValueKey('compare-device:$be:$i'),
                      label: chipName(d.device),
                      kind: d.pair.a.kind,
                      count: '${d.device.readings.length}',
                      selected: i == si,
                      onTap: () => _set(() => _sub[be] = i),
                    ),
                ],
              ),
          top: 10,
          inset: wide));
    }

    final (:pair, :device) = devices[si];
    final platform = device.a.platform;
    items.add(_Item(
        (ctx) => CSectionHeader(
              title: chipName(device),
              note: [
                kindLabel(pair.a.kind),
                if (platform.isNotEmpty && platform != be) platform,
                '${plural(device.readings.length, 'reading')} in both runs',
              ].join(' · '),
              level: 3,
            ),
        top: 18));

    final scope = 'S:$be:${device.name}:${device.occurrence}';
    if (device.readings.isNotEmpty) {
      items.add(_columns('READING', whenA, whenB,
          rowsWide: rowsWide, chevron: false));
      // The readings, test by test, in one card: a head per test, then its
      // readings, each a slice of the card so the list stays lazy.
      final tests = <List<ReadingCompare>>[];
      for (final r in device.readings) {
        if (tests.isEmpty || tests.last.first.testKey != r.testKey) {
          tests.add([]);
        }
        tests.last.add(r);
      }
      for (final (gi, rows) in tests.indexed) {
        final test = rows.first.a.test;
        items.add(_Item(
            (ctx) => _Slice(
                  first: gi == 0,
                  child: _TestHead(
                    title: test.title.isNotEmpty ? test.title : test.key,
                    count: plural(rows.length, 'reading'),
                    wide: rowsWide,
                  ),
                ),
            top: gi == 0 ? 8 : 0,
            key: ValueKey('$scope:${test.key}')));
        for (final (ri, r) in rows.indexed) {
          final end = ri == rows.length - 1;
          items.add(_Item(
              (ctx) => _Slice(
                    rule: false,
                    last: end && gi == tests.length - 1,
                    child: _ReadingRow(
                        reading: r, wide: rowsWide, end: end),
                  ),
              key: ValueKey('$scope:${r.testKey}:${r.readingId}')));
        }
      }
    }

    // Per test, how many readings only one run took.
    List<(String, int)> byTest(List<CompareValue> values) {
      final out = <(String, int)>[];
      String? last;
      for (final v in values) {
        if (v.test.key != last) {
          last = v.test.key;
          out.add((v.test.title.isNotEmpty ? v.test.title : v.test.key, 0));
        }
        out.last = (out.last.$1, out.last.$2 + 1);
      }
      return out;
    }

    List<Widget> rowsOf(List<CompareValue> values) => [
          for (final (title, n) in byTest(values))
            _OnlyRow(
              label: title,
              words: '',
              value: '$n',
              unit: n == 1 ? 'reading' : 'readings',
              caption: '',
              wide: rowsWide,
            ),
        ];
    items.addAll(_folds(
      scope: scope,
      labelA: 'Only in the $whenA run',
      labelB: 'Only in the $whenB run',
      countA: plural(device.onlyA.length, 'reading'),
      countB: plural(device.onlyB.length, 'reading'),
      hasA: device.onlyA.isNotEmpty,
      hasB: device.onlyB.isNotEmpty,
      rowsA: () => rowsOf(device.onlyA),
      rowsB: () => rowsOf(device.onlyB),
    ));
    return items;
  }
}

/// A test's head in the reading-by-reading card: its title and how many
/// readings both runs took of it.
class _TestHead extends StatelessWidget {
  const _TestHead(
      {required this.title, required this.count, required this.wide});

  final String title;
  final String count;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Padding(
      padding: wide
          ? const EdgeInsets.fromLTRB(18, 12, 18, 4)
          : const EdgeInsets.fromLTRB(16, 11, 16, 2),
      child: Wrap(
        spacing: wide ? 10 : 8,
        runSpacing: 2,
        crossAxisAlignment: WrapCrossAlignment.end,
        children: [
          Text(title,
              style: t.body.copyWith(
                  fontSize: wide ? 14 : 13, fontWeight: FontWeight.w800)),
          Text(count,
              style: t.caption.copyWith(fontSize: wide ? 12.5 : 11.5)),
        ],
      ),
    );
  }
}

/// One reading both runs took: its id, A's value and B's, the better one
/// marked -- or, measured two ways, neither.
class _ReadingRow extends StatelessWidget {
  const _ReadingRow(
      {required this.reading, required this.wide, required this.end});

  final ReadingCompare reading;
  final bool wide;

  /// The last of its test's, with the test's bottom margin under it.
  final bool end;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final r = reading;
    final marks = _Marks.of(r.verdict);
    Widget cell(CompareValue v, CompareSide side) {
      final text = valueText(v);
      // Loose, so the tint hugs the number rather than filling the column.
      return Align(
        alignment: Alignment.centerLeft,
        child: CValue(
          value: text.value,
          unit: text.unit,
          size: wide ? 16 : 15,
          weight: marks.weight(side),
          unitColor: t.ink,
          highlight: marks.on(side)
              ? t.run(side == CompareSide.a ? 'A' : 'B').tint
              : null,
          padded: true,
        ),
      );
    }

    final label = Wrap(
      spacing: wide ? 10 : 8,
      runSpacing: 2,
      crossAxisAlignment: WrapCrossAlignment.end,
      children: [
        Text(r.readingId,
            style: t.caption.copyWith(
                fontSize: wide ? 13.5 : 12.5, fontWeight: FontWeight.w600)),
        if (r.verdict.methodChanged)
          Text('method changed: not compared',
              style: t.caption.copyWith(
                  fontSize: wide ? 12.5 : 12, fontStyle: FontStyle.italic)),
      ],
    );
    if (wide) {
      return Padding(
        padding: EdgeInsets.fromLTRB(18, 4, 18, end ? 12 : 4),
        child: Row(
          children: [
            Expanded(child: label),
            const SizedBox(width: 16),
            SizedBox(width: _valueColumn, child: cell(r.a, CompareSide.a)),
            const SizedBox(width: 16),
            SizedBox(width: _valueColumn, child: cell(r.b, CompareSide.b)),
          ],
        ),
      );
    }
    return Padding(
      padding: EdgeInsets.fromLTRB(16, 5, 16, end ? 13 : 5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          label,
          const SizedBox(height: 3),
          Row(
            children: [
              Expanded(child: cell(r.a, CompareSide.a)),
              const SizedBox(width: 10),
              Expanded(child: cell(r.b, CompareSide.b)),
            ],
          ),
        ],
      ),
    );
  }
}
