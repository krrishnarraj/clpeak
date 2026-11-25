/// History: every saved run, newest first by month, the runs the app died
/// in among them -- and the one way into a compare: Compare runs, which
/// turns the list into a pick of two (the first A, the second B) under a
/// floating bar that says what kind of compare the two make.
library;

import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/compare.dart';
import '../../model/headlines.dart';
import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/compare_loading.dart';
import '../../services/export_service.dart';
import '../../services/resume_offer.dart';
import '../../services/run_history_store.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import '../common/run_route.dart';
import '../compare/compare_screen.dart';
import '../compare/compare_words.dart';
import '../results/results_screen.dart';
import 'details_editor.dart';

/// Which runs the list shows.
enum _Filter { all, mine, imported }

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  late Future<_HistoryContents> _contents;

  /// [BenchmarkService.runsEnded] when the current list was built; the tab
  /// lives in an IndexedStack, so initState alone would leave it stale
  /// forever.
  int _seenRunsEnded = 0;

  _Filter _filter = _Filter.all;
  final _search = TextEditingController();
  String _query = '';

  /// Picking two runs to compare, and the two picked: A first, then B.
  bool _picking = false;
  RunSummary? _pickA;
  RunSummary? _pickB;

  /// The compare of the two picks, read as soon as both are picked: it is
  /// what says whether they are one machine's.
  String? _preparedFor;
  Future<RunCompare?>? _prepared;
  RunCompare? _ready;

  @override
  void initState() {
    super.initState();
    _contents = _load();
  }

  @override
  void dispose() {
    _search.dispose();
    super.dispose();
  }

  Future<_HistoryContents> _load() async {
    final history = context.read<RunHistoryStore>();
    final service = context.read<BenchmarkService>();
    final runs = await history.list();
    // A cancelled run keeps its checkpoint, to be continued.
    final kept = <String, (RunCheckpointInfo, RunResumeState?)>{};
    for (final run in runs) {
      if (!run.cancelled || run.id == service.inFlightRunId) continue;
      final checkpoint = await history.checkpointInfo(run.id);
      if (checkpoint != null) {
        kept[run.id] = (checkpoint, await history.resumeState(run.id));
      }
    }
    return _HistoryContents(
      runs: runs,
      crashLogs:
          await history.listCrashLogs(inFlightId: service.inFlightRunId),
      kept: kept,
      origins: await history.origins(),
    );
  }

  void _refresh() {
    setState(() {
      _contents = _load();
    });
  }

  // ── Picking two runs ─────────────────────────────────────────────────────

  String? get _pickKey {
    final a = _pickA, b = _pickB;
    return a == null || b == null ? null : '${a.id}\u0000${b.id}';
  }

  void _startPicking() => setState(() => _picking = true);

  void _stopPicking() => setState(() {
        _picking = false;
        _pickA = _pickB = null;
      });

  /// A tap on a row while picking: unpicks a picked one, fills the empty
  /// side, A first; a third pick drops the oldest, so B becomes A.
  void _togglePick(RunSummary run) {
    setState(() {
      if (_pickA?.id == run.id) {
        _pickA = null;
      } else if (_pickB?.id == run.id) {
        _pickB = null;
      } else if (_pickA == null) {
        _pickA = run;
      } else if (_pickB == null) {
        _pickB = run;
      } else {
        _pickA = _pickB;
        _pickB = run;
      }
    });
    _prepare();
  }

  /// Read the two picks into a compare, once per pair of picks.
  Future<RunCompare?>? _prepare() {
    final key = _pickKey;
    if (key == null) return null;
    if (key == _preparedFor && _prepared != null) return _prepared;
    final a = _pickA!, b = _pickB!;
    final history = context.read<RunHistoryStore>();
    _preparedFor = key;
    _ready = null;
    final future = () async {
      try {
        final catalog = await HeadlineCatalog.bundled();
        return await history.compareRuns(a, b, catalog: catalog);
      } catch (_) {
        return null;
      }
    }();
    _prepared = future;
    future.then((c) {
      if (mounted && _preparedFor == key) setState(() => _ready = c);
    });
    return future;
  }

  Future<void> _compare() async {
    final prepared = _prepare();
    if (prepared == null) return;
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);
    final compare = await prepared;
    if (!mounted) return;
    if (compare == null) {
      // Reading it failed: let the next press try again.
      _preparedFor = null;
      messenger.showSnackBar(const SnackBar(
          content: Text('One of these runs can no longer be read.')));
      return;
    }
    final origins = (await _contents).origins;
    if (!mounted) return;
    final change = await navigator.push<CompareChange>(MaterialPageRoute(
        builder: (_) => CompareScreen(compare: compare, origins: origins)));
    if (!mounted) return;
    if (change == null) {
      _stopPicking();
      return;
    }
    // Change on a run card: back to the pick, the other run kept on its
    // side, this side's to choose again.
    setState(() {
      _picking = true;
      if (change.side == CompareSide.a) {
        _pickA = null;
        _pickB = change.keep;
      } else {
        _pickB = null;
        _pickA = change.keep;
      }
    });
  }

  // ── Import ───────────────────────────────────────────────────────────────

  Future<void> _import() async {
    try {
      const typeGroup = XTypeGroup(
        label: 'XpuPeak results',
        extensions: ['xpupeak.json', 'json'],
      );
      final file = await openFile(acceptedTypeGroups: [typeGroup]);
      if (file == null) return;
      if (!mounted) return;
      final raw = await file.readAsString();
      if (!mounted) return;
      final history = context.read<RunHistoryStore>();
      final messenger = ScaffoldMessenger.of(context);
      final name = file.name;

      Future<void> doImport(String rawContent, String fileName,
          {bool overwrite = false}) async {
        try {
          final summary = await history.importContent(rawContent,
              fileName: fileName, overwrite: overwrite);
          _refresh();
          messenger.showSnackBar(
            SnackBar(content: Text('Imported ${summary.fileName}')),
          );
        } on FileSystemException {
          final choice = await _showImportConflictDialog(fileName);
          if (!mounted) return;
          if (choice == _ImportConflictChoice.cancel) return;
          if (choice == _ImportConflictChoice.overwrite) {
            await doImport(rawContent, fileName, overwrite: true);
          } else {
            final unique = await history.nextAvailableFileName(fileName);
            await doImport(rawContent, unique, overwrite: false);
          }
        }
      }

      await doImport(raw, name);
    } on FormatException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Import failed: ${e.message}')),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Import failed: $e')),
      );
    }
  }

  Future<_ImportConflictChoice> _showImportConflictDialog(
      String fileName) async {
    final result = await showDialog<_ImportConflictChoice>(
      context: context,
      builder: (context) => CDialog(
        title: 'File already exists',
        actions: [
          CButton(
            label: 'Cancel',
            kind: CButtonKind.quiet,
            onPressed: () =>
                Navigator.pop(context, _ImportConflictChoice.cancel),
          ),
          CButton(
            label: 'Rename',
            onPressed: () =>
                Navigator.pop(context, _ImportConflictChoice.rename),
          ),
          CButton(
            label: 'Overwrite',
            danger: true,
            onPressed: () =>
                Navigator.pop(context, _ImportConflictChoice.overwrite),
          ),
        ],
        child: Text(
          'A run named "$fileName" already exists in history. '
          'Overwrite it or import as a new file?',
          style: CP.of(context).prose,
        ),
      ),
    );
    return result ?? _ImportConflictChoice.cancel;
  }

  // ── The list ─────────────────────────────────────────────────────────────

  bool _shows(_Entry e, _HistoryContents contents) {
    final origin = e.run == null ? RunOrigin.measured : contents.origins[e.run!.id];
    final keep = switch (_filter) {
      _Filter.all => true,
      _Filter.mine => origin == RunOrigin.measured,
      _Filter.imported => origin == RunOrigin.imported,
    };
    if (!keep) return false;
    final q = _query.trim().toLowerCase();
    if (q.isEmpty) return true;
    return e.searchText.toLowerCase().contains(q);
  }

  @override
  Widget build(BuildContext context) {
    // Re-read the files whenever a run ends (its summary is written to disk
    // before the count moves, so a new row is always there).
    final runsEnded =
        context.select<BenchmarkService, int>((s) => s.runsEnded);
    if (runsEnded != _seenRunsEnded) {
      _seenRunsEnded = runsEnded;
      _contents = _load();
    }
    // Whether a run that stopped part-way can be continued turns on the
    // device catalog, and never while another run is in flight; nothing
    // else of the service's state reaches this screen, which a run's
    // events must not rebuild.
    context.select<BenchmarkService, bool>(
        (s) => s.catalogReady && !s.isRunning);
    final service = context.read<BenchmarkService>();
    final t = CP.of(context);
    final wide = cIsWide(context);
    final pad = wide ? 34.0 : 16.0;

    return Scaffold(
      backgroundColor: t.ground,
      body: SafeArea(
        child: FutureBuilder<_HistoryContents>(
          // Recreate the future on every finished run / manual refresh.
          future: _contents,
          builder: (context, snapshot) {
            final contents = snapshot.data;
            if (contents == null) {
              return Column(
                children: [
                  _header(context, null, wide),
                  // Static text, not a spinner — nothing in this app spins.
                  const Expanded(child: CEmpty(title: 'Reading index…')),
                ],
              );
            }
            // One list, newest first: a run that did not finish sits where
            // it started, among the runs that did.  A cancelled run that was
            // continued and died in turn is the one run that did not finish,
            // its first part's result included.
            final unfinished = {for (final log in contents.crashLogs) log.id};
            final entries = <_Entry>[
              for (final log in contents.crashLogs) _Entry.crash(log),
              for (final run in contents.runs)
                if (!unfinished.contains(run.id)) _Entry.run(run),
            ]..sort((a, b) => b.startedAt.compareTo(a.startedAt));
            if (entries.isEmpty) {
              return Column(
                children: [
                  _header(context, contents, wide),
                  Expanded(
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const CEmpty(
                            title: 'No saved runs yet',
                            detail: 'Every benchmark run is saved here '
                                'automatically.',
                          ),
                          CButton(
                            label: 'Import run…',
                            glyph: CGlyphs.download,
                            onPressed: _import,
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              );
            }

            final shown = [for (final e in entries) if (_shows(e, contents)) e];
            final months = <_Month>[];
            for (final e in shown) {
              final m = DateTime(e.startedAt.year, e.startedAt.month);
              if (months.isEmpty || months.last.month != m) {
                months.add(_Month(m, []));
              }
              months.last.entries.add(e);
            }

            Widget entryRow(_Entry e) {
              final log = e.log;
              if (log != null) {
                return _CrashRow(
                  log: log,
                  offer: service.isRunning ? null : service.decideResume(log),
                  onChanged: _refresh,
                  picking: _picking,
                  wide: wide,
                );
              }
              final run = e.run!;
              final letter = _pickA?.id == run.id
                  ? 'A'
                  : (_pickB?.id == run.id ? 'B' : null);
              return _RunRow(
                summary: run,
                origin: contents.origins[run.id],
                offer: _offerFor(service, contents, run),
                picking: _picking,
                letter: letter,
                onPick: () => _togglePick(run),
                onChanged: _refresh,
                wide: wide,
              );
            }

            final items = <Widget Function(BuildContext)>[
              (c) => _header(c, contents, wide),
              (c) => Padding(
                    padding: EdgeInsets.fromLTRB(pad, wide ? 8 : 2, pad, 0),
                    child: _Filters(
                      search: _search,
                      filter: _filter,
                      counts: _counts(entries, contents),
                      wide: wide,
                      onQuery: (q) => setState(() => _query = q),
                      onFilter: (f) => setState(() => _filter = f),
                    ),
                  ),
              if (_picking)
                (c) => Padding(
                      padding:
                          EdgeInsets.fromLTRB(pad, wide ? 20 : 14, pad, 0),
                      child: _PickHint(wide: wide),
                    ),
              for (final m in months)
                (c) => Padding(
                      padding:
                          EdgeInsets.fromLTRB(pad, wide ? 20 : 14, pad, 0),
                      child: _MonthGroup(
                        key: ValueKey('month:${m.month}'),
                        title: monthTitle(m.month),
                        count: plural(m.entries.length, 'run'),
                        wide: wide,
                        rows: [for (final e in m.entries) entryRow(e)],
                      ),
                    ),
              if (months.isEmpty)
                (c) => Padding(
                      padding:
                          EdgeInsets.fromLTRB(pad, wide ? 20 : 14, pad, 0),
                      child: CEmptyBox(
                        text: 'No saved run matches.',
                        radius: wide ? CP.rCard : 22,
                      ),
                    ),
            ];

            final list = RefreshIndicator(
              onRefresh: () async => _refresh(),
              child: ListView.builder(
                key: const ValueKey('history-list'),
                padding: EdgeInsets.only(bottom: _picking ? 150 : 40),
                itemCount: items.length,
                itemBuilder: (context, i) => items[i](context),
              ),
            );
            if (!_picking) return list;
            final key = _pickKey;
            final ready = key != null && key == _preparedFor ? _ready : null;
            return Stack(
              children: [
                Positioned.fill(child: list),
                Positioned(
                  left: pad,
                  right: pad,
                  bottom: wide ? 20 : 16,
                  child: _PickBar(
                    a: _pickA,
                    b: _pickB,
                    hint: ready == null
                        ? 'The first run you pick is A, the second B'
                        : isSameHardware(ready)
                            ? 'Same hardware: compared reading by reading, '
                                'backend by backend'
                            : 'Different hardware: compared by processor, '
                                'and by backend',
                    onCancel: wide ? _stopPicking : null,
                    onCompare: key == null ? null : _compare,
                    wide: wide,
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  Widget _header(BuildContext context, _HistoryContents? contents, bool wide) {
    final saved = contents == null
        ? 0
        : contents.runs.length -
            contents.runs
                .where((r) => contents.crashLogs.any((l) => l.id == r.id))
                .length;
    final imported = contents == null
        ? 0
        : contents.runs
            .where((r) => contents.origins[r.id] == RunOrigin.imported)
            .length;
    final dead = contents?.crashLogs.length ?? 0;
    final count = contents == null
        ? ''
        : [
            plural(saved, 'saved run'),
            '$imported imported',
            if (dead > 0) '$dead did not finish',
          ].join(' · ');
    final canPick = (contents?.runs.length ?? 0) >= 2;
    final pick = CButton(
      key: const ValueKey('history-compare'),
      label: _picking ? 'Cancel' : 'Compare runs',
      kind: CButtonKind.primary,
      onPressed: _picking ? _stopPicking : (canPick ? _startPicking : null),
    );
    return CHeader(
      title: 'History',
      subtitle: count,
      actions: [
        if (wide)
          CButton(
            label: 'Import run…',
            glyph: CGlyphs.download,
            onPressed: _import,
          )
        else
          CIconButton(
            glyph: CGlyphs.download,
            tooltip: 'Import run',
            filled: true,
            onPressed: _import,
          ),
        pick,
      ],
    );
  }

  Map<_Filter, int> _counts(List<_Entry> entries, _HistoryContents contents) {
    var mine = 0, imported = 0;
    for (final e in entries) {
      final origin =
          e.run == null ? RunOrigin.measured : contents.origins[e.run!.id];
      if (origin == RunOrigin.measured) mine++;
      if (origin == RunOrigin.imported) imported++;
    }
    return {
      _Filter.all: entries.length,
      _Filter.mine: mine,
      _Filter.imported: imported,
    };
  }

  /// Whether a cancelled run can be continued; null for any other run, and
  /// while another is in flight.
  static ResumeOffer? _offerFor(
      BenchmarkService service, _HistoryContents contents, RunSummary run) {
    final kept = contents.kept[run.id];
    if (kept == null || service.isRunning) return null;
    return service.decideCancelled(run, checkpoint: kept.$1, state: kept.$2);
  }
}

enum _ImportConflictChoice { overwrite, rename, cancel }

class _HistoryContents {
  const _HistoryContents({
    required this.runs,
    required this.crashLogs,
    required this.kept,
    required this.origins,
  });
  final List<RunSummary> runs;
  final List<CrashLog> crashLogs;

  /// By id, what each cancelled run kept to be continued.
  final Map<String, (RunCheckpointInfo, RunResumeState?)> kept;

  /// By id, where each saved run came from.
  final Map<String, RunOrigin> origins;
}

/// One row of the list: a saved run, or a run the app died in.
class _Entry {
  _Entry.run(RunSummary this.run)
      : log = null,
        startedAt = run.startedAt;
  _Entry.crash(CrashLog this.log)
      : run = null,
        startedAt = log.startedAt;

  final RunSummary? run;
  final CrashLog? log;
  final DateTime startedAt;

  /// What the search reads: the run's name and its processors (or
  /// devices); a dead run's id and backends.
  String get searchText {
    final r = run;
    if (r != null) {
      return [
        runName(r),
        r.id,
        for (final p in r.processors) p.name,
        ...r.devices,
      ].join('\n');
    }
    return [log!.id, ...log!.backends.map(backendName), log!.lastTestTitle]
        .join('\n');
  }
}

class _Month {
  _Month(this.month, this.entries);

  final DateTime month;
  final List<_Entry> entries;
}

/// A run's processors, as its row's second line names them; its devices
/// where the file links none.
String _hardwareOf(RunSummary run) {
  final names = <String>[];
  for (final p in run.processors) {
    if (p.name.isNotEmpty && !names.contains(p.name)) names.add(p.name);
  }
  return names.isNotEmpty ? names.join(' · ') : run.devices.join(', ');
}

/// The kinds of a run's processors, in its order.
List<String> _kindsOf(RunSummary run) => [
      ...{
        for (final p in run.processors)
          if (p.kind.isNotEmpty) p.kind,
      },
    ];

// ── Pieces ─────────────────────────────────────────────────────────────────

/// The search and the filter chips: All, this machine's, Imported.
class _Filters extends StatelessWidget {
  const _Filters({
    required this.search,
    required this.filter,
    required this.counts,
    required this.wide,
    required this.onQuery,
    required this.onFilter,
  });

  final TextEditingController search;
  final _Filter filter;
  final Map<_Filter, int> counts;
  final bool wide;
  final ValueChanged<String> onQuery;
  final ValueChanged<_Filter> onFilter;

  @override
  Widget build(BuildContext context) {
    final field = CSearchField(
      controller: search,
      hint: 'Search by device or processor',
      semanticLabel: 'Search history',
      onChanged: onQuery,
    );
    final chips = [
      for (final (f, label) in [
        (_Filter.all, 'All'),
        (_Filter.mine, thisMachine(capital: true)),
        (_Filter.imported, 'Imported'),
      ])
        CChip(
          key: ValueKey('history-filter:${f.name}'),
          label: label,
          count: '${counts[f] ?? 0}',
          selected: filter == f,
          onTap: () => onFilter(f),
        ),
    ];
    if (!wide) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          field,
          const SizedBox(height: 10),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                for (final (i, c) in chips.indexed) ...[
                  if (i > 0) const SizedBox(width: 6),
                  c,
                ],
              ],
            ),
          ),
        ],
      );
    }
    return LayoutBuilder(builder: (context, box) {
      final row = Wrap(spacing: 6, runSpacing: 6, children: chips);
      if (box.maxWidth < 640) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [field, const SizedBox(height: 10), row],
        );
      }
      return Row(
        children: [
          Expanded(child: field),
          const SizedBox(width: 14),
          row,
        ],
      );
    });
  }
}

/// What picking two runs does, over the list while picking.
class _PickHint extends StatelessWidget {
  const _PickHint({required this.wide});

  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CPanel(
      color: t.surface2,
      outlined: false,
      radius: wide ? CP.rPanel : 18,
      padding: wide
          ? const EdgeInsets.symmetric(horizontal: 18, vertical: 14)
          : const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Text(
        wide
            ? 'Pick two runs: the first is A, the second B. Two runs of the '
                'same hardware compare reading by reading, backend by '
                'backend, to show how much each result moves from run to '
                'run. Runs on different hardware compare by processor, and '
                'by backend where both ran one. Imported runs work like '
                'your own.'
            : 'Pick two runs: the first is A, the second B. Same hardware '
                'compares reading by reading; different hardware, by '
                'processor and by backend.',
        style: t.body.copyWith(fontSize: wide ? 14 : 13.5, height: 1.45),
      ),
    );
  }
}

/// A month's runs: its heading, then one card of rows.
class _MonthGroup extends StatelessWidget {
  const _MonthGroup({
    super.key,
    required this.title,
    required this.count,
    required this.rows,
    required this.wide,
  });

  final String title;
  final String count;
  final List<Widget> rows;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: EdgeInsets.symmetric(horizontal: wide ? 6 : 8),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Semantics(
                header: true,
                child: Text(title,
                    style: t.h3.copyWith(fontSize: wide ? 17 : 16)),
              ),
              SizedBox(width: wide ? 10 : 8),
              Text(count,
                  style: t.caption.copyWith(fontSize: wide ? 13 : 12.5)),
            ],
          ),
        ),
        SizedBox(height: wide ? 10 : 8),
        Container(
          padding: EdgeInsets.all(wide ? 6 : 4),
          decoration: BoxDecoration(
            color: t.surface,
            border: Border.all(color: t.line),
            borderRadius: BorderRadius.circular(wide ? CP.rCard : 22),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              for (final (i, r) in rows.indexed)
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  decoration: BoxDecoration(
                    border: Border(
                        top: BorderSide(
                            color: i == 0 ? Colors.transparent : t.line)),
                  ),
                  child: r,
                ),
            ],
          ),
        ),
      ],
    );
  }
}

/// While picking, a row's A or B: an empty ring until picked, then the
/// letter on its run's colour.
class _PickRing extends StatelessWidget {
  const _PickRing({required this.letter, required this.wide});

  final String? letter;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final size = wide ? 26.0 : 24.0;
    final l = letter;
    final hue = l == null ? null : t.run(l);
    return Container(
      width: size,
      height: size,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: hue?.fill,
        border: Border.all(color: hue?.fill ?? t.muted, width: 2),
      ),
      child: l == null
          ? null
          : Text(l,
              style: t.small.copyWith(
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                  height: 1,
                  color: hue!.onFill)),
    );
  }
}

/// A saved run: its name and tags, its day and processors, its kinds; then
/// Continue where it can be, and its actions.  While picking, a tap picks
/// it, and a picked row takes its run's tint.
class _RunRow extends StatelessWidget {
  const _RunRow({
    required this.summary,
    required this.origin,
    required this.offer,
    required this.picking,
    required this.letter,
    required this.onPick,
    required this.onChanged,
    required this.wide,
  });

  final RunSummary summary;
  final RunOrigin? origin;

  /// For a cancelled run, whether it can be continued; null otherwise.
  final ResumeOffer? offer;
  final bool picking;

  /// A or B where picked.
  final String? letter;
  final VoidCallback onPick;
  final VoidCallback onChanged;
  final bool wide;

  Future<void> _open(BuildContext context) async {
    final history = context.read<RunHistoryStore>();
    final navigator = Navigator.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final doc = await history.load(summary);
    if (doc == null) {
      messenger.showSnackBar(
          const SnackBar(content: Text('Could not read this result file.')));
      return;
    }
    await navigator.push(MaterialPageRoute(
        settings: savedRunRoute(summary.id),
        builder: (_) => SavedResultsScreen(document: doc, summary: summary)));
  }

  Future<void> _rename(BuildContext context) async {
    final t = CP.of(context);
    final history = context.read<RunHistoryStore>();
    final controller = TextEditingController(text: summary.name);
    final name = await showDialog<String>(
      context: context,
      builder: (context) => CDialog(
        title: 'Rename run',
        actions: [
          CButton(
              label: 'Cancel',
              kind: CButtonKind.quiet,
              onPressed: () => Navigator.pop(context)),
          CButton(
              label: 'Save',
              kind: CButtonKind.primary,
              onPressed: () => Navigator.pop(context, controller.text)),
        ],
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: controller,
              autofocus: true,
              style: t.body,
              cursorWidth: 1.5,
              cursorRadius: Radius.zero,
              decoration: InputDecoration(
                isDense: true,
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 11),
                hintText: summary.id,
                hintStyle: t.body.copyWith(color: t.muted),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide(color: t.line),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide(color: t.line),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide(color: t.ink, width: 1.5),
                ),
              ),
              onSubmitted: (v) => Navigator.pop(context, v),
            ),
            const SizedBox(height: 10),
            Text('Leave empty to use the run timestamp', style: t.caps),
          ],
        ),
      ),
    );
    if (name != null) {
      await history.rename(summary, name);
      onChanged();
    }
  }

  Future<void> _delete(BuildContext context) async {
    final t = CP.of(context);
    final history = context.read<RunHistoryStore>();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => CDialog(
        title: 'Delete run?',
        actions: [
          CButton(
              label: 'Cancel',
              kind: CButtonKind.quiet,
              onPressed: () => Navigator.pop(context, false)),
          CButton(
              label: 'Delete',
              danger: true,
              onPressed: () => Navigator.pop(context, true)),
        ],
        child: Text(
          'This permanently removes the saved results from '
          '${formatDate(summary.startedAt)}.',
          style: t.prose,
        ),
      ),
    );
    if (confirmed == true) {
      await history.delete(summary);
      onChanged();
    }
  }

  Future<void> _share(BuildContext context) async {
    final history = context.read<RunHistoryStore>();
    final export = context.read<ExportService>();
    final messenger = ScaffoldMessenger.of(context);
    try {
      final file = await history.documentFile(summary);
      await export.exportRun(file, suggestedName: summary.fileName);
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Share failed: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final offer = this.offer;
    // Not while the run is being continued: the continuation rewrites its
    // file when it ends.
    final inFlight =
        context.read<BenchmarkService>().inFlightRunId == summary.id;
    final name = runName(summary);
    final day = formatDay(summary.startedAt);
    final hardware = _hardwareOf(summary);
    final picked = letter != null;
    final tagFill = picked ? t.tagOnTint : null;
    final tags = [
      if (origin == RunOrigin.imported)
        CTag(text: 'Imported', upper: false, fill: tagFill),
      // Measured in parts: continued after it stopped part-way.
      if (summary.resumed) CTag(text: 'Resumed', upper: false, fill: tagFill),
      if (summary.cancelled)
        CTag(
            text: 'Cancelled',
            upper: false,
            color: t.warnInk,
            fill: t.warnTint),
    ];
    final kinds = wide ? _kindsOf(summary) : const <String>[];
    final body = Row(
      children: [
        if (picking) ...[
          _PickRing(letter: letter, wide: wide),
          SizedBox(width: wide ? 14 : 12),
        ],
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Wrap(
                spacing: wide ? 8 : 6,
                runSpacing: 4,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Text(name,
                      style: t.title.copyWith(
                          fontSize: wide ? 16 : 15,
                          letterSpacing: (wide ? 16 : 15) * -0.01)),
                  ...tags,
                ],
              ),
              SizedBox(height: wide ? 3 : 2),
              Text(
                [day, if (hardware.isNotEmpty) hardware].join(' · '),
                style: t.caption.copyWith(fontSize: wide ? 13 : 12.5),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
        if (kinds.isNotEmpty) ...[
          const SizedBox(width: 14),
          for (final (i, k) in kinds.indexed) ...[
            if (i > 0) const SizedBox(width: 4),
            CKindLabel(k),
          ],
        ],
      ],
    );
    return Row(
      children: [
        Expanded(
          child: CTap(
            onTap: picking ? onPick : () => _open(context),
            selected: picking ? picked : null,
            semanticLabel: picking && picked ? 'Run $letter: $name' : null,
            focusRadius: BorderRadius.circular(wide ? CP.rWell : CP.rRow),
            builder: (context, hovered, pressed) => Container(
              constraints: BoxConstraints(minHeight: wide ? 68 : 64),
              padding: wide
                  ? const EdgeInsets.symmetric(horizontal: 14, vertical: 10)
                  : const EdgeInsets.fromLTRB(12, 9, 10, 9),
              decoration: BoxDecoration(
                color: picked
                    ? t.run(letter!).tint
                    : (hovered || pressed
                        ? t.surface2.withValues(alpha: t.isDark ? 0.7 : 0.6)
                        : Colors.transparent),
                borderRadius:
                    BorderRadius.circular(wide ? CP.rWell : CP.rRow),
              ),
              child: body,
            ),
          ),
        ),
        if (!picking && offer != null && offer.canContinue) ...[
          SizedBox(width: wide ? 6 : 2),
          // Into this result: only what the run had not finished.
          CButton(
            label: 'Continue',
            kind: CButtonKind.primary,
            onPressed: () => context.read<BenchmarkService>().resume(offer),
          ),
        ],
        SizedBox(width: wide ? 6 : 2),
        _RowMenu(
          label: 'Actions for $name, $day',
          items: [
            _MenuItem('Rename', () => _rename(context)),
            _MenuItem(
                'Details',
                inFlight
                    ? null
                    : () => editRunDetails(context, summary,
                        onSaved: onChanged)),
            _MenuItem('Share', () => _share(context)),
            _MenuItem('Delete', () => _delete(context), danger: true),
          ],
        ),
      ],
    );
  }
}

/// A run the process died in: the run log the native side streamed while it
/// ran, and what it finished when it left a checkpoint.  The log shares, so
/// it can go on a bug report; what was saved opens as results; and while
/// the run can be continued, Continue runs the rest into the same result.
class _CrashRow extends StatelessWidget {
  const _CrashRow({
    required this.log,
    required this.offer,
    required this.onChanged,
    required this.picking,
    required this.wide,
  });

  final CrashLog log;

  /// Whether it can be continued; null while another run is in flight.
  final ResumeOffer? offer;
  final VoidCallback onChanged;

  /// While picking: it has no result to compare, so it offers nothing to
  /// run.
  final bool picking;
  final bool wide;

  Future<void> _share(BuildContext context) async {
    final history = context.read<RunHistoryStore>();
    final export = context.read<ExportService>();
    final messenger = ScaffoldMessenger.of(context);
    try {
      final file = await history.crashLogFile(log);
      await export.exportRunLog(file, suggestedName: log.fileName);
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Share failed: $e')));
    }
  }

  Future<void> _delete(BuildContext context) async {
    final t = CP.of(context);
    final history = context.read<RunHistoryStore>();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => CDialog(
        title: 'Delete run log?',
        actions: [
          CButton(
              label: 'Cancel',
              kind: CButtonKind.quiet,
              onPressed: () => Navigator.pop(context, false)),
          CButton(
              label: 'Delete',
              danger: true,
              onPressed: () => Navigator.pop(context, true)),
        ],
        child: Text(
          log.checkpoint == null
              ? 'This permanently removes the only record of the run from '
                  '${formatDate(log.startedAt)}.'
              : 'This permanently removes the run from '
                  '${formatDate(log.startedAt)}: its log and what it saved.',
          style: t.prose,
        ),
      ),
    );
    if (confirmed == true) {
      await history.deleteCrashLog(log);
      onChanged();
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final last = log.lastEntry;
    final exit = log.processExit;
    final checkpoint = log.checkpoint;
    final saved = checkpoint?.tests ?? 0;
    final offer = this.offer;
    final canContinue = !picking && offer != null && offer.canContinue;
    final note = t.caption.copyWith(fontSize: wide ? 13 : 12.5);
    return Padding(
      padding: wide
          ? const EdgeInsets.fromLTRB(14, 8, 0, 10)
          : const EdgeInsets.fromLTRB(12, 6, 0, 9),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Wrap(
                  spacing: wide ? 8 : 6,
                  runSpacing: 4,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    Text(log.id,
                        style: t.title.copyWith(fontSize: wide ? 16 : 15)),
                    CTag(text: 'did not finish', color: t.danger),
                  ],
                ),
              ),
              _RowMenu(
                label: 'Actions for the run of ${formatDay(log.startedAt)} '
                    'that did not finish',
                items: [
                  _MenuItem('Share log', () => _share(context)),
                  _MenuItem('Delete', () => _delete(context), danger: true),
                ],
              ),
            ],
          ),
          Padding(
            padding: const EdgeInsets.only(right: 14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  checkpoint == null
                      ? 'The app stopped before this run finished. Its '
                          'results were never written; the run log recorded '
                          'up to that point is what to attach to a bug '
                          'report.'
                      : 'The app stopped before this run finished. '
                          '$saved ${saved == 1 ? 'test was' : 'tests were'} '
                          'saved; the run log is what to attach to a bug '
                          'report.',
                  style: note,
                ),
                if (offer != null &&
                    checkpoint != null &&
                    !offer.canContinue &&
                    offer.whyNot.isNotEmpty) ...[
                  const SizedBox(height: 6),
                  Text("Can't be continued: ${offer.whyNot}.", style: note),
                ],
                if (exit != null) ...[
                  const SizedBox(height: 6),
                  Text(
                    exit.message.split('\n').first,
                    style: t.small.copyWith(color: t.danger),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
                if (last != null) ...[
                  const SizedBox(height: 6),
                  Text(
                    'last line: ${last.message}',
                    style: t.small,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
                if (canContinue || saved > 0) ...[
                  const SizedBox(height: 10),
                  Wrap(
                    spacing: 10,
                    runSpacing: 8,
                    children: [
                      if (canContinue)
                        CButton(
                          label: 'Continue',
                          kind: CButtonKind.primary,
                          onPressed: () =>
                              context.read<BenchmarkService>().resume(offer),
                        ),
                      if (saved > 0)
                        CButton(
                          label: "View what's saved",
                          onPressed: () => openCheckpoint(context,
                              id: log.id, startedAt: log.startedAt),
                        ),
                    ],
                  ),
                ],
                const SizedBox(height: 8),
                // A Wrap, not a Row with an inflexible metadata Text: date
                // + line count + version alone can outgrow a phone, and
                // then the metadata sits on its own line instead of
                // overflowing.
                Wrap(
                  spacing: 10,
                  runSpacing: 6,
                  crossAxisAlignment: WrapCrossAlignment.center,
                  children: [
                    if (log.backends.isNotEmpty || log.verbose)
                      Wrap(
                        spacing: 5,
                        runSpacing: 5,
                        children: [
                          for (final backend in log.backends)
                            CTag(text: backendName(backend), upper: false),
                          if (log.verbose) const CTag(text: 'verbose'),
                        ],
                      ),
                    Text(
                      [
                        formatDay(log.startedAt),
                        '${log.entries} log lines',
                        if (log.generator.version.isNotEmpty)
                          'v${log.generator.version}',
                      ].join(' · '),
                      style: t.smallMuted,
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _MenuItem {
  const _MenuItem(this.label, this.onTap, {this.danger = false});

  final String label;

  /// Null while the action is not on offer.
  final VoidCallback? onTap;
  final bool danger;
}

/// A row's actions behind its ⋯.
class _RowMenu extends StatelessWidget {
  const _RowMenu({required this.label, required this.items});

  final String label;
  final List<_MenuItem> items;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return PopupMenuButton<int>(
      tooltip: label,
      icon: CGlyph(CGlyphs.more, size: 20, color: t.muted, fill: true),
      splashRadius: 1,
      position: PopupMenuPosition.under,
      color: t.surface,
      surfaceTintColor: Colors.transparent,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(CP.rRow),
        side: BorderSide(color: t.line),
      ),
      elevation: 2,
      onSelected: (i) => items[i].onTap?.call(),
      itemBuilder: (_) => [
        for (final (i, item) in items.indexed)
          PopupMenuItem(
            value: i,
            height: 40,
            enabled: item.onTap != null,
            child: Text(item.label,
                style: t.body.copyWith(
                    fontWeight: FontWeight.w600,
                    color: item.danger
                        ? t.danger
                        : (item.onTap == null ? t.muted : t.ink))),
          ),
      ],
    );
  }
}

// ── The pick bar ───────────────────────────────────────────────────────────

/// The floating bar of a pick in progress: each pick's badge and "name,
/// day", or what to choose next; a line on what the two make; Cancel (the
/// phone's is in the header) and Compare, which waits for two.
class _PickBar extends StatelessWidget {
  const _PickBar({
    required this.a,
    required this.b,
    required this.hint,
    required this.onCancel,
    required this.onCompare,
    required this.wide,
  });

  final RunSummary? a;
  final RunSummary? b;
  final String hint;
  final VoidCallback? onCancel;
  final VoidCallback? onCompare;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    _BarPick pick(String letter, RunSummary? run, bool other) => _BarPick(
          letter: letter,
          text: run == null
              ? (other ? 'then run $letter' : 'Choose run $letter')
              : runName(run),
          day: run == null ? '' : ', ${dayMonth(run.startedAt)}',
          picked: run != null,
          wide: wide,
        );
    final picks = [pick('A', a, false), pick('B', b, a == null)];
    final hintText = Text(hint,
        style: t.small.copyWith(
            fontSize: wide ? 12.5 : 11.5,
            color: t.inkOn.withValues(alpha: 0.8)));
    final compare = _BarButton(
      key: const ValueKey('pick-bar-compare'),
      label: 'Compare',
      filled: true,
      onTap: onCompare,
      wide: wide,
    );
    return Semantics(
      container: true,
      label: 'Runs to compare',
      child: CFloatingBar(
        padding: wide
            ? const EdgeInsets.fromLTRB(22, 10, 10, 10)
            : const EdgeInsets.fromLTRB(18, 8, 8, 8),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (wide)
                    Wrap(spacing: 18, runSpacing: 4, children: picks)
                  else ...[
                    picks[0],
                    const SizedBox(height: 3),
                    picks[1],
                  ],
                  SizedBox(height: wide ? 4 : 3),
                  hintText,
                ],
              ),
            ),
            const SizedBox(width: 8),
            if (onCancel != null) ...[
              _BarButton(
                  label: 'Cancel', filled: false, onTap: onCancel, wide: wide),
              const SizedBox(width: 4),
            ],
            compare,
          ],
        ),
      ),
    );
  }
}

/// One pick in the bar: its badge and its run, or an outlined badge and
/// what to choose.
class _BarPick extends StatelessWidget {
  const _BarPick({
    required this.letter,
    required this.text,
    required this.day,
    required this.picked,
    required this.wide,
  });

  final String letter;
  final String text;

  /// `, 19 Nov` after a picked run's name, kept whole when the name is cut
  /// short: two runs of one machine differ by it.
  final String day;
  final bool picked;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final size = wide ? CBadgeSize.md : CBadgeSize.xs;
    final badge = picked
        ? CLetterBadge(letter, size: size)
        : Container(
            width: size.diameter,
            height: size.diameter,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: t.inkOn, width: 1.5),
            ),
            child: Text(letter,
                style: t.small.copyWith(
                    fontSize: size.fontSize,
                    fontWeight: FontWeight.w800,
                    height: 1,
                    color: t.inkOn)),
          );
    final style = t.body.copyWith(
        fontSize: wide ? 14.5 : 13,
        fontWeight: picked ? FontWeight.w700 : FontWeight.w600,
        color: t.inkOn);
    return Semantics(
      container: true,
      label: '$text$day',
      excludeSemantics: true,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          badge,
          SizedBox(width: wide ? 8 : 7),
          Flexible(
            child: Text(text,
                maxLines: 1, overflow: TextOverflow.ellipsis, style: style),
          ),
          if (day.isNotEmpty) Text(day, maxLines: 1, style: style),
        ],
      ),
    );
  }
}

/// A button on the ink bar: [filled] in the inverse ink (Compare), or
/// quiet text (Cancel).
class _BarButton extends StatelessWidget {
  const _BarButton({
    super.key,
    required this.label,
    required this.filled,
    required this.onTap,
    required this.wide,
  });

  final String label;
  final bool filled;
  final VoidCallback? onTap;
  final bool wide;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final enabled = onTap != null;
    return CTap(
      onTap: onTap,
      focusRadius: BorderRadius.circular(CP.rPill),
      builder: (context, hovered, pressed) {
        final Color bg, fg;
        if (filled) {
          bg = !enabled
              ? t.inkOn.withValues(alpha: 0.18)
              : (pressed || hovered
                  ? t.inkOn.withValues(alpha: pressed ? 0.78 : 0.88)
                  : t.inkOn);
          fg = enabled ? t.ink : t.inkOn.withValues(alpha: 0.6);
        } else {
          bg = pressed || hovered
              ? t.inkOn.withValues(alpha: 0.12)
              : Colors.transparent;
          fg = t.inkOn;
        }
        return Container(
          height: 44,
          padding: EdgeInsets.symmetric(
              horizontal: filled ? (wide ? 24 : 20) : 16),
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: bg,
            borderRadius: BorderRadius.circular(CP.rPill),
          ),
          child: Text(label,
              style: t.label.copyWith(
                  fontSize: wide ? 14.5 : 14,
                  fontWeight: filled ? FontWeight.w800 : FontWeight.w700,
                  color: fg)),
        );
      },
    );
  }
}
