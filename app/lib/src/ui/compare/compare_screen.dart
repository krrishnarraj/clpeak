/// Two runs side by side (lib/src/model/compare.dart), as History opens
/// them: the two runs, what kind of compare it is, then
///
/// - for two machines: the processors paired, each pair's headlines, and
///   every measure both runs have -- by processor (each side's best from any
///   backend) or backend by backend (each side's best through one backend);
/// - for one machine twice: each processor's headlines, and every reading
///   both runs took, backend by backend and device by device.
///
/// Colour says which run a thing is, and only ever as a fill: the letter
/// badges, the run cards, and a soft tint behind the better value, which is
/// also bold; the numbers stay ink.  Which value is better, which two read
/// the same and which were measured another way is the model's (`Verdict`);
/// this screen lays it out and adds only its own words (compare_words.dart).
/// Its rows build lazily: one device of one run can hold hundreds of
/// readings.
library;

import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../model/compare.dart';
import '../../model/headlines.dart';
import '../../model/measures.dart';
import '../../model/run_document.dart';
import '../../model/run_summary.dart';
import '../../services/run_history_store.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import 'compare_words.dart';

part 'compare_cross.dart';
part 'compare_parts.dart';
part 'compare_same.dart';

/// What a run card's Change asks History for: another run for [side],
/// with [keep] staying on the other side.
class CompareChange {
  const CompareChange({required this.side, this.keep});

  final CompareSide side;

  /// The other side's run; null for a run History does not hold.
  final RunSummary? keep;
}

/// A compare, pushed by History over its list.  Pops with a [CompareChange]
/// when a run card's Change is pressed, and with nothing for Back.
class CompareScreen extends StatefulWidget {
  const CompareScreen({
    super.key,
    required this.compare,
    this.origins = const {},
  });

  final RunCompare compare;

  /// Where each saved run came from, by id (`RunHistoryStore.origins`).
  final Map<String, RunOrigin> origins;

  @override
  State<CompareScreen> createState() => _CompareScreenState();
}

class _CompareScreenState extends State<CompareScreen> {
  late RunCompare _c = widget.compare;

  /// Two machines: backend by backend rather than by processor.
  bool _byBackend = false;

  /// The pair on show, by processor (and the phone's headlines).
  int _pair = 0;

  /// The backend tab on show; the first unless picked.
  String? _backend;

  /// Per backend, the pair or device chip on show.
  final _sub = <String, int>{};

  /// The rows tapped open, by key.
  final _open = <String>{};

  /// For the parts of this library, which build the screen's sections.
  void _set(VoidCallback change) => setState(change);

  PairCompare? get _currentPair =>
      _c.pairs.isEmpty ? null : _c.pairs[_pair.clamp(0, _c.pairs.length - 1)];

  void _swap() {
    final old = _currentPair;
    setState(() {
      _c = _c.swapped();
      final i = old == null
          ? -1
          : _c.pairs.indexWhere(
              (p) => identical(p.a, old.b) && identical(p.b, old.a));
      _pair = math.max(0, i);
      _sub.clear();
      _open.clear();
    });
  }

  void _change(CompareSide side) => Navigator.of(context).pop(CompareChange(
      side: side, keep: side == CompareSide.a ? _c.b.summary : _c.a.summary));

  void _toggle(String key) => setState(() {
        if (!_open.remove(key)) _open.add(key);
      });

  /// Re-pair [g], one of [side]'s processors, with one the person picks of
  /// the other run's -- or with none.
  Future<void> _repair(ProcessorMeasures g, CompareSide side) async {
    final mine = side == CompareSide.a;
    final others = mine ? _c.b.measures.groups : _c.a.measures.groups;
    ProcessorMeasures? partner;
    PairCompare? pair;
    for (final p in _c.pairs) {
      if (identical(mine ? p.a : p.b, g)) {
        pair = p;
        partner = mine ? p.b : p.a;
      }
    }
    final picked = await showDialog<_Partner>(
      context: context,
      builder: (context) => _PartnerDialog(
        name: g.name,
        others: others,
        current: partner,
        paired: pair != null,
      ),
    );
    if (picked == null || !mounted) return;
    final choice = picked.group;
    if (identical(choice, partner)) return;
    RunCompare next;
    ProcessorMeasures? focus;
    if (choice == null) {
      if (pair == null) return;
      next = _c.withoutPair(pair.a);
    } else if (mine) {
      next = _c.withPair(g, choice);
      focus = g;
    } else {
      next = _c.withPair(choice, g);
      focus = choice;
    }
    setState(() {
      _c = next;
      final i = focus == null
          ? -1
          : next.pairs.indexWhere((p) => identical(p.a, focus));
      _pair = math.max(0, i);
      _sub.clear();
      _open.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final same = isSameHardware(_c);
    final pad = wide ? 34.0 : 16.0;
    return Scaffold(
      backgroundColor: t.ground,
      body: SafeArea(
        child: LayoutBuilder(builder: (context, box) {
          // Measure rows set their two values in columns of their own where
          // the content has room for them.
          final rowsWide = box.maxWidth - 2 * pad >= 620;
          final items = <_Item>[
            _Item((c) => _head(c, same, wide), inset: false),
            _Item((c) => _cards(c, same, wide), top: wide ? 14 : 4),
            _Item((c) => _Banner(same: same, wide: wide), top: wide ? 22 : 14),
            if (same)
              ..._sameItems(wide: wide, rowsWide: rowsWide)
            else
              ..._crossItems(wide: wide, rowsWide: rowsWide),
          ];
          return ListView.builder(
            key: const ValueKey('compare-list'),
            padding: const EdgeInsets.only(bottom: 48),
            itemCount: items.length,
            itemBuilder: (context, i) {
              final it = items[i];
              return KeyedSubtree(
                key: it.key,
                child: Padding(
                  padding: EdgeInsets.fromLTRB(
                      it.inset ? pad : 0, it.top, it.inset ? pad : 0, 0),
                  child: it.build(context),
                ),
              );
            },
          );
        }),
      ),
    );
  }

  Widget _head(BuildContext context, bool same, bool wide) {
    void back() => Navigator.of(context).maybePop();
    if (wide) {
      return CHeader(
        title: 'Compare',
        subtitle: same
            ? 'Two runs of the same hardware, reading by reading'
            : 'Two runs on different hardware, by processor and by backend',
        actions: [CButton(label: 'Back to history', onPressed: back)],
      );
    }
    return CHeader(
      title: 'Compare',
      subtitle: same ? 'Same hardware' : 'Different hardware',
      onBack: back,
    );
  }

  Widget _cards(BuildContext context, bool same, bool wide) {
    Widget card(CompareRun side, String letter, CompareSide which) {
      final started = sideStarted(side);
      final name = sideName(side);
      final s = side.summary;
      final tag = switch (s == null ? null : widget.origins[s.id]) {
        RunOrigin.measured => thisMachine(capital: true),
        RunOrigin.imported => 'Imported',
        null => null,
      };
      // One machine twice: the runs are told apart by when they ran.
      return _RunCard(
        letter: letter,
        heading: same && started != null ? dayAndTime(started) : name,
        tag: tag,
        line1: same && started != null ? name : systemLine(side.document),
        line2: same
            ? runNote(side.document)
            : (started == null ? '' : formatDay(started)),
        onChange: () => _change(which),
        compact: !wide,
      );
    }

    final a = card(_c.a, 'A', CompareSide.a);
    final b = card(_c.b, 'B', CompareSide.b);
    if (!wide) {
      return _StackedCards(a: a, b: b, swap: _SwapButton(onTap: _swap));
    }
    return LayoutBuilder(builder: (context, box) {
      if (box.maxWidth < 660) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            a,
            const SizedBox(height: 8),
            Center(child: _SwapButton(onTap: _swap, across: false)),
            const SizedBox(height: 8),
            b,
          ],
        );
      }
      return IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(child: a),
            const SizedBox(width: 12),
            Center(child: _SwapButton(onTap: _swap, across: true)),
            const SizedBox(width: 12),
            Expanded(child: b),
          ],
        ),
      );
    });
  }

  /// The detail section's title, its note and, for two machines, the
  /// By processor | By backend switch.
  List<_Item> _detailHead(
      {required String title,
      required String note,
      required String phoneNote,
      required bool wide,
      required bool cross}) {
    Widget views() => CSegmented<bool>(
          semanticLabel: 'Compare by',
          segments: const [
            CSegment(false, 'By processor'),
            CSegment(true, 'By backend'),
          ],
          selected: _byBackend,
          onChanged: (v) => setState(() {
            _byBackend = v;
            _open.clear();
          }),
          expand: !wide,
        );
    if (wide) {
      return [
        _Item(
            (c) => CSectionHeader(
                title: title, note: note, trailing: cross ? views() : null),
            top: 34),
      ];
    }
    return [
      _Item((c) => CSectionHeader(title: title, note: phoneNote), top: 26),
      if (cross) _Item((c) => views(), top: 12),
    ];
  }

  /// The two columns' heads above a list of rows.
  _Item _columns(String label, String a, String b,
          {required bool rowsWide, required bool chevron}) =>
      _Item(
          (c) => _ColumnHead(
                label: label,
                a: a,
                b: b,
                wide: rowsWide,
                chevron: chevron,
              ),
          top: 16);

  /// A category's head, then its rows, one card.
  List<_Item> _category<T>(CategoryRows<T> g, String noun,
      Widget Function(T row) row, String Function(T row) key) {
    return [
      _Item(
          (c) => CSectionHeader(
                title: g.category.label,
                note: plural(g.rows.length, noun),
                level: 3,
              ),
          top: 18),
      for (final (i, r) in g.rows.indexed)
        _Item(
          (c) => _Slice(
            first: i == 0,
            last: i == g.rows.length - 1,
            child: row(r),
          ),
          top: i == 0 ? 8 : 0,
          key: ValueKey(key(r)),
        ),
    ];
  }

  /// Two folds of what one side alone has, under a list: [a]'s rows, then
  /// [b]'s, each only where it has any.
  List<_Item> _folds({
    required String scope,
    required String labelA,
    required String labelB,
    required String countA,
    required String countB,
    required List<Widget> Function() rowsA,
    required List<Widget> Function() rowsB,
    required bool hasA,
    required bool hasB,
  }) =>
      [
        if (hasA)
          _Item(
              (c) => _fold(
                    key: ValueKey('fold-a:$scope'),
                    letter: 'A',
                    label: labelA,
                    count: countA,
                    rows: rowsA(),
                  ),
              top: 22),
        if (hasB)
          _Item(
              (c) => _fold(
                    key: ValueKey('fold-b:$scope'),
                    letter: 'B',
                    label: labelB,
                    count: countB,
                    rows: rowsB(),
                  ),
              top: hasA ? 10 : 22),
      ];
}
