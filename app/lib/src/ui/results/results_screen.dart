import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_document.dart';
import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/export_service.dart';
import '../../services/run_history_store.dart';
import '../common/format.dart';
import 'results_body.dart';

/// Results of the run that just finished (hosted inside the Benchmark tab).
class LiveResultsScreen extends StatelessWidget {
  const LiveResultsScreen({super.key, required this.service});

  final BenchmarkService service;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final summary = service.lastSummary;

    return Scaffold(
      appBar: AppBar(
        title: Text(service.cancelled ? 'Run cancelled' : 'Results'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          tooltip: 'Back to dashboard',
          onPressed: service.reset,
        ),
        actions: [
          if (summary != null) _ExportButton(summary: summary),
        ],
      ),
      body: Column(
        children: [
          if (service.cancelled)
            MaterialBanner(
              leading: Icon(Icons.info_outline, color: scheme.primary),
              content: const Text(
                  'The run was cancelled — partial results were saved.'),
              actions: const [SizedBox.shrink()],
            )
          else if (service.exitCode > 0)
            MaterialBanner(
              leading: Icon(Icons.warning_amber, color: scheme.error),
              content: Text(
                  'Some backends reported errors (status ${service.exitCode}). '
                  'Results below may be incomplete.'),
              actions: const [SizedBox.shrink()],
            ),
          Expanded(child: ResultsBody(document: service.document)),
        ],
      ),
    );
  }
}

/// A saved run opened from History.
class SavedResultsScreen extends StatelessWidget {
  const SavedResultsScreen({
    super.key,
    required this.document,
    required this.summary,
  });

  final RunDocument document;
  final RunSummary summary;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(summary.name.isNotEmpty ? summary.name : 'Results',
                overflow: TextOverflow.ellipsis),
            Text(
              formatDate(summary.startedAt),
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Theme.of(context).colorScheme.outline),
            ),
          ],
        ),
        actions: [_ExportButton(summary: summary)],
      ),
      body: ResultsBody(document: document),
    );
  }
}

class _ExportButton extends StatelessWidget {
  const _ExportButton({required this.summary});

  final RunSummary summary;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      icon: const Icon(Icons.ios_share),
      tooltip: 'Export XML',
      onPressed: () async {
        final history = context.read<RunHistoryStore>();
        final export = context.read<ExportService>();
        final messenger = ScaffoldMessenger.of(context);
        try {
          final xml = await history.xmlFile(summary);
          await export.exportXml(xml, suggestedName: summary.fileName);
        } catch (e) {
          messenger.showSnackBar(
              SnackBar(content: Text('Export failed: $e')));
        }
      },
    );
  }
}
