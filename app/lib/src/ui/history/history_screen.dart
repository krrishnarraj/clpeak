import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/export_service.dart';
import '../../services/run_history_store.dart';
import '../../theme/clpeak_theme.dart';
import '../common/format.dart';
import '../results/results_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  late Future<List<RunSummary>> _runs;

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

  @override
  Widget build(BuildContext context) {
    // Refresh the list whenever a run finishes.
    context.select<BenchmarkService, RunSummary?>((s) => s.lastSummary);

    return Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: FutureBuilder<List<RunSummary>>(
        // Recreate the future on every finished run / manual refresh.
        future: _runs,
        builder: (context, snapshot) {
          final runs = snapshot.data;
          if (runs == null) {
            return const Center(child: CircularProgressIndicator());
          }
          if (runs.isEmpty) {
            return Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.history,
                      size: 48,
                      color: Theme.of(context).colorScheme.outline),
                  const SizedBox(height: 12),
                  const Text('No saved runs yet'),
                  const SizedBox(height: 4),
                  Text('Every benchmark run is saved here automatically.',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Theme.of(context).colorScheme.outline)),
                ],
              ),
            );
          }
          return RefreshIndicator(
            onRefresh: () async => _refresh(),
            child: ListView.separated(
              padding: const EdgeInsets.all(20),
              itemCount: runs.length,
              separatorBuilder: (_, _) => const SizedBox(height: 10),
              itemBuilder: (context, i) => _RunTile(
                summary: runs[i],
                onChanged: _refresh,
              ),
            ),
          );
        },
      ),
    );
  }
}

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
    final history = context.read<RunHistoryStore>();
    final controller = TextEditingController(text: summary.name);
    final name = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Rename run'),
        content: TextField(
          controller: controller,
          autofocus: true,
          decoration: InputDecoration(
            hintText: summary.devices.join(', '),
            helperText: 'Leave empty to use the device name',
          ),
          onSubmitted: (v) => Navigator.pop(context, v),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, controller.text),
              child: const Text('Save')),
        ],
      ),
    );
    if (name != null) {
      await history.rename(summary, name);
      onChanged();
    }
  }

  Future<void> _delete(BuildContext context) async {
    final history = context.read<RunHistoryStore>();
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete run?'),
        content: Text(
            'This permanently removes the saved results from '
            '${formatDate(summary.startedAt)}.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context, false),
              child: const Text('Cancel')),
          FilledButton(
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Delete')),
        ],
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
      final xml = await history.xmlFile(summary);
      await export.exportXml(xml, suggestedName: summary.fileName);
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Export failed: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: () => _open(context),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: Text(
                      summary.displayTitle,
                      style: Theme.of(context)
                          .textTheme
                          .titleSmall
                          ?.copyWith(fontWeight: FontWeight.w600),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  PopupMenuButton<String>(
                    icon: Icon(Icons.more_vert,
                        size: 20, color: scheme.outline),
                    onSelected: (action) => switch (action) {
                      'rename' => _rename(context),
                      'export' => _export(context),
                      'delete' => _delete(context),
                      _ => null,
                    },
                    itemBuilder: (_) => const [
                      PopupMenuItem(value: 'rename', child: Text('Rename')),
                      PopupMenuItem(
                          value: 'export', child: Text('Export XML')),
                      PopupMenuItem(value: 'delete', child: Text('Delete')),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 2),
              Text(
                [
                  // Keep the device list visible when a custom name hides it.
                  if (summary.name.isNotEmpty) summary.devices.join(', '),
                  formatDate(summary.startedAt),
                  if (summary.durationMs > 0)
                    formatDuration(Duration(milliseconds: summary.durationMs)),
                  if (summary.cancelled) 'cancelled',
                ].join(' · '),
                style: Theme.of(context)
                    .textTheme
                    .bodySmall
                    ?.copyWith(color: scheme.outline),
              ),
              const SizedBox(height: 10),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  for (final backend in summary.backends)
                    _MiniChip(
                      icon: ClpeakTheme.backendIcon(backend),
                      text: backend,
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

class _MiniChip extends StatelessWidget {
  const _MiniChip({this.icon, required this.text});

  final IconData? icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        color: scheme.surfaceContainerHighest.withValues(alpha: 0.6),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 14, color: scheme.outline),
            const SizedBox(width: 4),
          ],
          Text(text, style: Theme.of(context).textTheme.labelSmall),
        ],
      ),
    );
  }
}
