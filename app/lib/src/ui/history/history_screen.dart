import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/export_service.dart';
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
  late Future<List<RunSummary>> _runs;

  /// Id of the finished run the current list was built for; the tab lives in
  /// an IndexedStack, so initState alone would leave it stale forever.
  String? _seenRunId;

  @override
  void initState() {
    super.initState();
    _runs = context.read<RunHistoryStore>().list();
  }

  void _refresh() {
    setState(() {
      _runs = context.read<RunHistoryStore>().list();
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
      _runs = context.read<RunHistoryStore>().list();
    }

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: 'History',
              subtitle: 'saved runs',
              actions: [
                CIconButton(
                  icon: Icons.file_upload,
                  tooltip: 'Import',
                  onPressed: _import,
                ),
              ],
            ),
            Expanded(
              child: FutureBuilder<List<RunSummary>>(
                // Recreate the future on every finished run / manual refresh.
                future: _runs,
                builder: (context, snapshot) {
                  final runs = snapshot.data;
                  if (runs == null) {
                    // Static text, not a spinner — nothing in this app spins.
                    return const CEmpty(title: 'Reading index…');
                  }
                  if (runs.isEmpty) {
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
                  return RefreshIndicator(
                    onRefresh: () async => _refresh(),
                    child: ListView.builder(
                      padding: const EdgeInsets.fromLTRB(20, 20, 20, 40),
                      itemCount: runs.length,
                      itemBuilder: (context, i) => Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: _RunTile(
                          summary: runs[i],
                          onChanged: _refresh,
                        ),
                      ),
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
}

enum _ImportConflictChoice { overwrite, rename, cancel }

class _RunTile extends StatelessWidget {
  const _RunTile({required this.summary, required this.onChanged});

  final RunSummary summary;
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
              Row(
                children: [
                  Expanded(
                    child: Wrap(
                      spacing: 5,
                      runSpacing: 5,
                      children: [
                        for (final backend in summary.backends)
                          CTag(text: backend, upper: false),
                      ],
                    ),
                  ),
                  const SizedBox(width: 10),
                  Text(
                    [
                      formatDate(summary.startedAt),
                      if (summary.durationMs > 0)
                        formatDuration(
                            Duration(milliseconds: summary.durationMs)),
                    ].join('  ·  '),
                    style: t.micro,
                  ),
                  const SizedBox(width: 4),
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
            child: Text('Export', style: t.monoSmall)),
        PopupMenuItem(
            value: 'delete',
            height: 38,
            child: Text('Delete',
                style: t.monoSmall.copyWith(color: t.danger))),
      ],
    );
  }
}
