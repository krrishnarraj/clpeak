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
    final meta = document.meta;
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
            if (meta != null &&
                (meta.clpeakVersion.isNotEmpty ||
                    meta.generatedAt.isNotEmpty ||
                    meta.host.isNotEmpty))
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 12, 20, 0),
                child: _RunMetaPanel(meta: meta),
              ),
            Expanded(child: ResultsBody(document: document)),
          ],
        ),
      ),
    );
  }
}

class _RunMetaPanel extends StatelessWidget {
  const _RunMetaPanel({required this.meta});

  final RunMeta meta;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final entries = <({String key, String value})>[];

    if (meta.clpeakVersion.isNotEmpty) {
      entries.add((key: 'CLPEAK VERSION', value: meta.clpeakVersion));
    }
    if (meta.generatedAt.isNotEmpty) {
      final dt = DateTime.tryParse(meta.generatedAt);
      final display =
          dt != null ? formatDate(dt) : meta.generatedAt;
      entries.add((key: 'GENERATED', value: display));
    }
    if (meta.durationSeconds > 0) {
      entries.add((
        key: 'DURATION',
        value: formatDuration(
            Duration(milliseconds: (meta.durationSeconds * 1000).round()))
      ));
    }
    for (final e in meta.host.entries) {
      final k = e.key.toString().trim();
      final v = e.value?.toString().trim() ?? '';
      if (k.isEmpty || v.isEmpty) continue;
      // Pretty-print memory_bytes as GB/MB
      if (k == 'memory_bytes') {
        final bytes = int.tryParse(v);
        if (bytes != null) {
          entries.add((key: k.toUpperCase(), value: formatBytes(bytes)));
          continue;
        }
      }
      entries.add((key: k.toUpperCase(), value: v));
    }

    if (entries.isEmpty) return const SizedBox.shrink();

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(12, 9, 12, 9),
            decoration: BoxDecoration(
              color: t.isDark ? t.hover : t.hover.withValues(alpha: 0.7),
              border: Border(bottom: BorderSide(color: t.line)),
            ),
            child: Row(
              children: [
                Icon(Icons.info_outline, size: 13, color: t.faint),
                const SizedBox(width: 9),
                Expanded(
                  child: Text('RUN INFO', style: t.micro),
                ),
                if (meta.cancelled)
                  CTag(text: 'cancelled', color: t.danger),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 11, 12, 11),
            child: Wrap(
              spacing: 22,
              runSpacing: 7,
              children: [
                for (final e in entries)
                  Text.rich(
                    TextSpan(children: [
                      TextSpan(text: '${e.key}  ', style: t.micro),
                      TextSpan(text: e.value, style: t.monoSmall),
                    ]),
                  ),
              ],
            ),
          ),
        ],
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
