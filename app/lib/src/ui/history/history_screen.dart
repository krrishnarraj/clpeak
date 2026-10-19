import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/export_service.dart';
import '../../services/resume_offer.dart';
import '../../services/run_history_store.dart';
import '../../theme/clpeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import '../results/results_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  late Future<_HistoryContents> _contents;

  /// Id of the finished run the current list was built for; the tab lives in
  /// an IndexedStack, so initState alone would leave it stale forever.
  String? _seenRunId;

  @override
  void initState() {
    super.initState();
    _contents = _load();
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
    );
  }

  void _refresh() {
    setState(() {
      _contents = _load();
    });
  }

  Future<void> _import() async {
    try {
      const typeGroup = XTypeGroup(
        label: 'clpeak results',
        extensions: ['clpeak.json', 'json'],
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
          style: CP.of(context).body,
        ),
      ),
    );
    return result ?? _ImportConflictChoice.cancel;
  }

  @override
  Widget build(BuildContext context) {
    // Re-read the index whenever a run finishes (the summary is written to
    // disk before lastSummary is published, so the new row is always there).
    final lastRunId =
        context.select<BenchmarkService, String?>((s) => s.lastSummary?.id);
    if (lastRunId != _seenRunId) {
      _seenRunId = lastRunId;
      _contents = _load();
    }
    // Whether a run that stopped part-way can be continued turns on the
    // device catalog, and never while another run is in flight; nothing
    // else of the service's state reaches this screen, which a run's
    // events must not rebuild.
    context.select<BenchmarkService, bool>(
        (s) => s.catalogReady && !s.isRunning);
    final service = context.read<BenchmarkService>();

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: 'History',
              subtitle: 'saved runs',
              actions: [
                CButton(
                  label: 'Import',
                  icon: Icons.file_upload,
                  onPressed: _import,
                ),
              ],
            ),
            Expanded(
              child: FutureBuilder<_HistoryContents>(
                // Recreate the future on every finished run / manual refresh.
                future: _contents,
                builder: (context, snapshot) {
                  final contents = snapshot.data;
                  if (contents == null) {
                    // Static text, not a spinner — nothing in this app spins.
                    return const CEmpty(title: 'Reading index…');
                  }
                  final runs = contents.runs;
                  final crashLogs = contents.crashLogs;
                  if (runs.isEmpty && crashLogs.isEmpty) {
                    return Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const CEmpty(
                            icon: Icons.history,
                            title: 'No saved runs yet',
                            detail:
                                'Every benchmark run is saved here automatically.',
                          ),
                          const SizedBox(height: 16),
                          CButton(
                            label: 'Import',
                            icon: Icons.file_upload,
                            onPressed: _import,
                          ),
                        ],
                      ),
                    );
                  }
                  // One list, newest first: a run that did not finish sits
                  // where it started, among the runs that did.  A cancelled
                  // run that was continued and died in turn is the one run
                  // that did not finish, its first part's result included.
                  final unfinished = {for (final log in crashLogs) log.id};
                  final items = <(DateTime, Widget)>[
                    for (final log in crashLogs)
                      (
                        log.startedAt,
                        _CrashLogTile(
                          log: log,
                          offer: service.isRunning
                              ? null
                              : service.decideResume(log),
                          onChanged: _refresh,
                        ),
                      ),
                    for (final run in runs)
                      if (!unfinished.contains(run.id))
                        (
                          run.startedAt,
                          _RunTile(
                            summary: run,
                            offer: _offerFor(service, contents, run),
                            onChanged: _refresh,
                          ),
                        ),
                  ]..sort((a, b) => b.$1.compareTo(a.$1));
                  final tiles = [
                    for (final (_, tile) in items)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: tile,
                      ),
                  ];
                  return RefreshIndicator(
                    onRefresh: () async => _refresh(),
                    child: ListView.builder(
                      padding: const EdgeInsets.fromLTRB(20, 20, 20, 40),
                      itemCount: tiles.length,
                      itemBuilder: (context, i) => tiles[i],
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
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
  const _HistoryContents(
      {required this.runs, required this.crashLogs, required this.kept});
  final List<RunSummary> runs;
  final List<CrashLog> crashLogs;

  /// By id, what each cancelled run kept to be continued.
  final Map<String, (RunCheckpointInfo, RunResumeState?)> kept;
}

/// A run the process died in: the run log the native side streamed while it
/// ran, and what it finished when it left a checkpoint.  The log exports, so
/// it can go on a bug report; what was saved opens as results; and while
/// the run can be continued, Continue runs the rest into the same result.
class _CrashLogTile extends StatelessWidget {
  const _CrashLogTile(
      {required this.log, required this.offer, required this.onChanged});

  final CrashLog log;

  /// Whether it can be continued; null while another run is in flight.
  final ResumeOffer? offer;
  final VoidCallback onChanged;

  Future<void> _export(BuildContext context) async {
    final history = context.read<RunHistoryStore>();
    final export = context.read<ExportService>();
    final messenger = ScaffoldMessenger.of(context);
    try {
      final file = await history.crashLogFile(log);
      await export.exportRunLog(file, suggestedName: log.fileName);
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Export failed: $e')));
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
          style: t.body,
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
    return CPanel(
      child: Container(
        padding: const EdgeInsets.fromLTRB(12, 11, 8, 12),
        decoration: BoxDecoration(
          border: Border(left: BorderSide(color: t.danger, width: 2)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    log.id,
                    style: t.title,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                CTag(text: 'did not finish', color: t.danger),
                const SizedBox(width: 6),
                PopupMenuButton<String>(
                  tooltip: 'Run log actions',
                  icon: Icon(Icons.more_horiz, size: 17, color: t.dim),
                  iconSize: 17,
                  splashRadius: 1,
                  position: PopupMenuPosition.under,
                  onSelected: (action) => switch (action) {
                    'export' => _export(context),
                    'delete' => _delete(context),
                    _ => null,
                  },
                  itemBuilder: (_) => [
                    PopupMenuItem(
                        value: 'export',
                        height: 38,
                        child: Text('Export log', style: t.monoSmall)),
                    PopupMenuItem(
                        value: 'delete',
                        height: 38,
                        child: Text('Delete',
                            style: t.monoSmall.copyWith(color: t.danger))),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 7),
            Text(
              checkpoint == null
                  ? 'The app stopped before this run finished. Its results '
                      'were never written; the run log recorded up to that '
                      'point is what to attach to a bug report.'
                  : 'The app stopped before this run finished. '
                      '$saved ${saved == 1 ? 'test was' : 'tests were'} saved; '
                      'the run log is what to attach to a bug report.',
              style: t.monoSmallDim,
            ),
            if (offer != null &&
                checkpoint != null &&
                !offer.canContinue &&
                offer.whyNot.isNotEmpty) ...[
              const SizedBox(height: 7),
              Text("Can't be continued: ${offer.whyNot}.",
                  style: t.monoSmallDim),
            ],
            if (exit != null) ...[
              const SizedBox(height: 7),
              Text(
                exit.message.split('\n').first,
                style: t.monoSmall.copyWith(color: t.danger),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
            if (last != null) ...[
              const SizedBox(height: 7),
              Text(
                'last line: ${last.message}',
                style: t.monoSmall,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
            if ((offer?.canContinue ?? false) || saved > 0) ...[
              const SizedBox(height: 11),
              Wrap(
                spacing: 10,
                runSpacing: 8,
                children: [
                  if (offer != null && offer.canContinue)
                    CButton(
                      label: 'Continue',
                      icon: Icons.play_arrow,
                      kind: CButtonKind.primary,
                      onPressed: () =>
                          context.read<BenchmarkService>().resume(offer),
                    ),
                  if (saved > 0)
                    CButton(
                      label: "View what's saved",
                      icon: Icons.visibility_outlined,
                      onPressed: () => openCheckpoint(context,
                          id: log.id, startedAt: log.startedAt),
                    ),
                ],
              ),
            ],
            const SizedBox(height: 9),
            // A Wrap, not a Row with an inflexible metadata Text: date +
            // line count + version alone can outgrow a phone, and then the
            // metadata sits on its own line instead of overflowing.
            Wrap(
              spacing: 10,
              runSpacing: 6,
              alignment: WrapAlignment.spaceBetween,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                if (log.backends.isNotEmpty || log.verbose)
                  Wrap(
                    spacing: 5,
                    runSpacing: 5,
                    children: [
                      for (final backend in log.backends)
                        CTag(text: backend, upper: false),
                      if (log.verbose) const CTag(text: 'verbose'),
                    ],
                  ),
                Text(
                  [
                    formatDate(log.startedAt),
                    '${log.entries} log lines',
                    if (log.clpeakVersion.isNotEmpty) 'v${log.clpeakVersion}',
                  ].join('  ·  '),
                  style: t.micro,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _RunTile extends StatelessWidget {
  const _RunTile(
      {required this.summary, required this.offer, required this.onChanged});

  final RunSummary summary;

  /// For a cancelled run, whether it can be continued; null otherwise.
  final ResumeOffer? offer;
  final VoidCallback onChanged;

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
              style: t.mono,
              cursorWidth: 1.5,
              cursorRadius: Radius.zero,
              decoration: InputDecoration(
                isDense: true,
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 11),
                hintText: summary.id,
                hintStyle: t.mono.copyWith(color: t.faint),
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
                  borderSide: BorderSide(color: t.text, width: 1.5),
                ),
              ),
              onSubmitted: (v) => Navigator.pop(context, v),
            ),
            const SizedBox(height: 10),
            Text('Leave empty to use the run timestamp', style: t.micro),
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
          style: t.body,
        ),
      ),
    );
    if (confirmed == true) {
      await history.delete(summary);
      onChanged();
    }
  }

  Future<void> _export(BuildContext context) async {
    final history = context.read<RunHistoryStore>();
    final export = context.read<ExportService>();
    final messenger = ScaffoldMessenger.of(context);
    try {
      final file = await history.documentFile(summary);
      await export.exportRun(file, suggestedName: summary.fileName);
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Export failed: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final offer = this.offer;
    return CPanel(
      child: CTap(
        onTap: () => _open(context),
        builder: (context, hovered, pressed) => Container(
          color: hovered || pressed ? t.hover : Colors.transparent,
          padding: const EdgeInsets.fromLTRB(12, 11, 8, 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      summary.displayTitle,
                      style: t.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (summary.cancelled) ...[
                    CTag(text: 'cancelled', color: t.danger),
                    const SizedBox(width: 6),
                  ],
                  _Menu(
                    onRename: () => _rename(context),
                    onExport: () => _export(context),
                    onDelete: () => _delete(context),
                  ),
                ],
              ),
              const SizedBox(height: 7),
              Padding(
                padding: const EdgeInsets.only(right: 4),
                child: Text(
                  [
                    // The title is a name or a timestamp id, so the devices
                    // always belong here.
                    if (summary.devices.isNotEmpty) summary.devices.join(', '),
                  ].join(),
                  style: t.monoSmall,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (offer != null && offer.canContinue) ...[
                const SizedBox(height: 11),
                // Into this result: only what the run had not finished.
                CButton(
                  label: 'Continue',
                  icon: Icons.play_arrow,
                  kind: CButtonKind.primary,
                  onPressed: () =>
                      context.read<BenchmarkService>().resume(offer),
                ),
              ],
              if (summary.clpeakVersion.isNotEmpty ||
                  summary.hostOs.isNotEmpty ||
                  summary.hostArch.isNotEmpty) ...[
                const SizedBox(height: 5),
                Padding(
                  padding: const EdgeInsets.only(right: 4),
                  child: Text(
                    [
                      if (summary.hostOs.isNotEmpty)
                        [
                          summary.hostOs,
                          summary.hostOsVersion,
                        ].where((s) => s.isNotEmpty).join(' '),
                      if (summary.hostArch.isNotEmpty) summary.hostArch,
                      if (summary.clpeakVersion.isNotEmpty)
                        'v${summary.clpeakVersion}',
                    ].join('  ·  '),
                    style: t.monoSmallDim,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
              const SizedBox(height: 9),
              // Same Wrap as the unfinished-run tile above: the metadata
              // Text must be allowed its own line on a narrow screen.
              Wrap(
                spacing: 10,
                runSpacing: 6,
                alignment: WrapAlignment.spaceBetween,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  if (summary.backends.isNotEmpty)
                    Wrap(
                      spacing: 5,
                      runSpacing: 5,
                      children: [
                        for (final backend in summary.backends)
                          CTag(text: backend, upper: false),
                      ],
                    ),
                  Text(
                    [
                      formatDate(summary.startedAt),
                      if (summary.durationMs > 0)
                        formatDuration(
                            Duration(milliseconds: summary.durationMs)),
                    ].join('  ·  '),
                    style: t.micro,
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Menu extends StatelessWidget {
  const _Menu({
    required this.onRename,
    required this.onExport,
    required this.onDelete,
  });

  final VoidCallback onRename;
  final VoidCallback onExport;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return PopupMenuButton<String>(
      tooltip: 'Run actions',
      icon: Icon(Icons.more_horiz, size: 17, color: t.dim),
      iconSize: 17,
      splashRadius: 1,
      position: PopupMenuPosition.under,
      onSelected: (action) => switch (action) {
        'rename' => onRename(),
        'export' => onExport(),
        'delete' => onDelete(),
        _ => null,
      },
      itemBuilder: (_) => [
        PopupMenuItem(
            value: 'rename',
            height: 38,
            child: Text('Rename', style: t.monoSmall)),
        PopupMenuItem(
            value: 'export',
            height: 38,
            child: Text('Export JSON', style: t.monoSmall)),
        PopupMenuItem(
            value: 'delete',
            height: 38,
            child: Text('Delete',
                style: t.monoSmall.copyWith(color: t.danger))),
      ],
    );
  }
}
