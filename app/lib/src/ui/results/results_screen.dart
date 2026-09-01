import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_document.dart';
import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/export_service.dart';
import '../../services/run_history_store.dart';
import '../../theme/clpeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import 'results_body.dart';

/// Results of the run that just finished (hosted inside the Benchmark tab).
class LiveResultsScreen extends StatelessWidget {
  const LiveResultsScreen({super.key, required this.service});

  final BenchmarkService service;

  @override
  Widget build(BuildContext context) {
    final summary = service.lastSummary;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: service.cancelled ? 'Run cancelled' : 'Results',
              subtitle: summary?.id,
              onBack: service.reset,
              actions: [
                if (summary != null) _ExportButton(summary: summary),
              ],
            ),
            if (service.cancelled)
              const _Notice(
                icon: Icons.info_outline,
                text: 'The run was cancelled — partial results were saved.',
              )
            else if (service.exitCode > 0)
              _Notice(
                icon: Icons.warning_amber,
                danger: true,
                text:
                    'Some backends reported errors (status ${service.exitCode}). '
                    'Results below may be incomplete.',
              ),
            Expanded(child: ResultsBody(document: service.document)),
          ],
        ),
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
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: summary.displayTitle.isNotEmpty
                  ? summary.displayTitle
                  : 'Results',
              subtitle: formatDate(summary.startedAt),
              onBack: () => Navigator.of(context).pop(),
              actions: [_ExportButton(summary: summary)],
            ),
            Expanded(child: ResultsBody(document: document)),
          ],
        ),
      ),
    );
  }
}

/// Flat inline banner — a tinted left edge, no Material surface.
class _Notice extends StatelessWidget {
  const _Notice({required this.icon, required this.text, this.danger = false});

  final IconData icon;
  final String text;
  final bool danger;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final accent = danger ? t.danger : t.dim;
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        color: t.panel,
        border: Border(
          left: BorderSide(color: accent, width: 2),
          bottom: BorderSide(color: t.line),
        ),
      ),
      padding: const EdgeInsets.fromLTRB(12, 10, 14, 10),
      child: Row(
        children: [
          Icon(icon, size: 14, color: accent),
          const SizedBox(width: 10),
          Expanded(child: Text(text, style: t.monoSmall)),
        ],
      ),
    );
  }
}

class _ExportButton extends StatelessWidget {
  const _ExportButton({required this.summary});

  final RunSummary summary;

  @override
  Widget build(BuildContext context) {
    return CIconButton(
      icon: Icons.ios_share,
      tooltip: 'Export',
      onPressed: () async {
        final history = context.read<RunHistoryStore>();
        final export = context.read<ExportService>();
        final messenger = ScaffoldMessenger.of(context);
        try {
          final file = await history.documentFile(summary);
          await export.exportRun(file, suggestedName: summary.fileName);
        } catch (e) {
          messenger.showSnackBar(
              SnackBar(content: Text('Export failed: $e')));
        }
      },
    );
  }
}
