import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_document.dart';
import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/export_service.dart';
import '../../services/resume_offer.dart';
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

    // The finished run lives inside the Benchmark tab, not on a pushed
    // route, so a system back button or swipe gesture would pop the app
    // itself.  Intercept it and do what the header's back arrow does.
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) service.reset();
      },
      child: Scaffold(
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
              if (service.runFailure != null)
                _Notice(
                  icon: Icons.error_outline,
                  danger: true,
                  text: '${service.runFailure}.'
                      '${service.document.isEmpty ? '' : ' The results below '
                          'are what it measured until then.'}'
                      '${service.runFailureLogKept ? ' Its log is in History, '
                          'marked "did not finish".' : ''}',
                )
              else if (service.cancelled)
                const _Notice(
                  icon: Icons.info_outline,
                  text: 'The run was cancelled — partial results were saved. '
                      'Continue it from History to run the rest.',
                )
              else if (service.wasResumed)
                const _Notice(
                  icon: Icons.restart_alt,
                  text: 'Continued from where the run stopped: the results '
                      'hold every part of it.',
                )
              else if (service.exitCode > 0)
                _Notice(
                  icon: Icons.warning_amber,
                  danger: true,
                  text:
                      'Some backends reported errors (status ${service.exitCode}). '
                      'Results below may be incomplete.',
                ),
              Expanded(
                  child: ResultsBody(
                      document: service.document,
                      showDiagnostics: service.runVerbose)),
            ],
          ),
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
    this.notice,
  });

  final RunDocument document;
  final RunSummary summary;

  /// A line over the results: what a run that did not finish saved.
  final String? notice;

  @override
  Widget build(BuildContext context) {
    final part = ResumeOffer.latestPart(document);
    final at = part == null ? null : DateTime.tryParse(part.at)?.toLocal();
    final parts = document.meta?.resumed.length ?? 0;
    // What ended the part before the latest: the app, or a cancel.
    final after = part?.after ?? '';
    final stopped = after == 'cancelled'
        ? 'it was cancelled part-way and continued'
        : 'the app stopped part-way and the run was continued';
    final why = after.isEmpty || after == 'cancelled' ? '' : ' ($after)';
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
            if (notice != null)
              _Notice(icon: Icons.info_outline, text: notice!)
            else if (part != null)
              _Notice(
                icon: Icons.restart_alt,
                text: 'Measured in ${parts + 1} parts: $stopped'
                    '${at == null ? '' : ' on ${formatDate(at)}'}$why.',
              ),
            Expanded(child: ResultsBody(document: document)),
          ],
        ),
      ),
    );
  }
}

/// Show what a run that did not finish saved: its checkpoint, as results.
Future<void> openCheckpoint(BuildContext context,
    {required String id, required DateTime startedAt}) async {
  final history = context.read<RunHistoryStore>();
  final navigator = Navigator.of(context);
  final messenger = ScaffoldMessenger.of(context);
  final doc = await history.loadCheckpoint(id);
  if (doc == null) {
    messenger.showSnackBar(
        const SnackBar(content: Text('Could not read what this run saved.')));
    return;
  }
  final summary = RunSummary.fromDocument(
    id: id,
    // Export hands over the checkpoint itself.
    fileName: RunHistoryStore.partialFileNameFor(id),
    doc: doc,
    startedAt: startedAt,
    durationMs: ((doc.meta?.durationSeconds ?? 0) * 1000).round(),
    cancelled: false,
  );
  final tests = doc.testCount;
  await navigator.push(MaterialPageRoute(
      builder: (_) => SavedResultsScreen(
            document: doc,
            summary: summary,
            notice: 'Not finished: the app stopped part-way, and these '
                '$tests ${tests == 1 ? 'test is' : 'tests are'} what it saved.',
          )));
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
      tooltip: 'Export JSON',
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
