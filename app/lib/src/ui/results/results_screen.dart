import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_document.dart';
import '../../model/run_summary.dart';
import '../../services/benchmark_service.dart';
import '../../services/export_service.dart';
import '../../services/resume_offer.dart';
import '../../services/run_history_store.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import 'results_body.dart';

/// A run's name and the line under it: the name the user gave the run, or
/// else the machine's -- "MacBook Pro", its parenthesised model year moved
/// to the line below -- then the OS and when the run was made.
({String title, String subtitle}) _heading({
  String name = '',
  RunMeta? meta,
  DateTime? at,
  String fallback = 'Results',
}) {
  final system = meta?.systemName ?? '';
  final line = <String>[];
  String title;
  if (name.isNotEmpty) {
    title = name;
    if (system.isNotEmpty) line.add(system);
  } else if (system.isNotEmpty) {
    // "MacBook Pro (14-inch, 2021)": the name, and what tells it apart.
    final m = RegExp(r'^(.*\S)\s*\(([^()]+)\)$').firstMatch(system);
    title = m?.group(1) ?? system;
    if (m != null) line.add(m.group(2)!);
  } else {
    title = fallback;
  }
  final s = meta?.systemInfo;
  if (s != null) {
    final os = s.osDistribution.isNotEmpty
        ? s.osDistribution
        : [s.osName, s.osVersion].where((v) => v.isNotEmpty).join(' ');
    if (os.isNotEmpty) line.add(os);
  }
  if (at != null) line.add(_when(at));
  return (title: title, subtitle: line.join(' · '));
}

/// "16 Nov 2026, 11:12".
String _when(DateTime t) {
  String two(int v) => v.toString().padLeft(2, '0');
  return '${formatDay(t)}, ${two(t.hour)}:${two(t.minute)}';
}

/// Results of the run that just finished (hosted inside the Benchmark tab).
class LiveResultsScreen extends StatelessWidget {
  const LiveResultsScreen({super.key, required this.service});

  final BenchmarkService service;

  @override
  Widget build(BuildContext context) {
    final summary = service.lastSummary;
    // The saved file once there is one: its Summary, the system and every
    // processor's peaks, is only in the file.  Until then the live
    // document, whose meta a continued run already has from its checkpoint.
    final shown = service.savedDocument ?? service.document;
    final heading = _heading(
      name: summary?.name ?? '',
      meta: shown.meta,
      at: summary?.startedAt ?? service.startedAt,
      fallback: service.cancelled ? 'Run cancelled' : 'Results',
    );

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
                title: heading.title,
                subtitle: heading.subtitle,
                onBack: service.reset,
                actions: [
                  if (summary != null) _ShareButton(summary: summary),
                ],
              ),
              if (service.runFailure != null)
                _Notice(
                  danger: true,
                  text: '${service.runFailure}.'
                      '${service.document.isEmpty ? '' : ' The results below '
                          'are what it measured until then.'}'
                      '${service.runFailureLogKept ? ' Its log is in History, '
                          'marked "did not finish".' : ''}',
                )
              else if (service.cancelled)
                const _Notice(
                  text: 'The run was cancelled — partial results were saved. '
                      'Continue it from History to run the rest.',
                )
              else if (service.wasResumed)
                const _Notice(
                  text: 'Continued from where the run stopped: the results '
                      'hold every part of it.',
                )
              else if (service.exitCode > 0)
                _Notice(
                  danger: true,
                  text:
                      'Some backends reported errors (status ${service.exitCode}). '
                      'Results below may be incomplete.',
                ),
              Expanded(
                  child: ResultsBody(
                      document: shown, showDiagnostics: service.runVerbose)),
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
    final heading = _heading(
        name: summary.name, meta: document.meta, at: summary.startedAt);
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: heading.title,
              subtitle: heading.subtitle,
              onBack: () => Navigator.of(context).pop(),
              actions: [_ShareButton(summary: summary)],
            ),
            if (notice != null)
              _Notice(text: notice!)
            else if (part != null)
              _Notice(
                text: 'Measured in ${parts + 1} parts: $stopped'
                    '${at == null ? '' : ' on ${_when(at)}'}$why.',
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
    // Share hands over the checkpoint itself.
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

/// A line over the results, in a soft callout: grey for what to know, the
/// warning tint for what went wrong.
class _Notice extends StatelessWidget {
  const _Notice({required this.text, this.danger = false});

  final String text;
  final bool danger;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Padding(
      padding: cIsWide(context)
          ? const EdgeInsets.fromLTRB(34, 4, 34, 6)
          : const EdgeInsets.fromLTRB(16, 6, 16, 4),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(18, 12, 18, 13),
        decoration: BoxDecoration(
          color: danger ? t.warnTint : t.surface2,
          borderRadius: BorderRadius.circular(CP.rPanel),
        ),
        child: Text(
          text,
          style: t.body
              .copyWith(fontSize: 13.5, color: danger ? t.warnInk : t.ink),
        ),
      ),
    );
  }
}

/// The header's one action: the run's file, by the share sheet on a phone
/// and a save dialog on desktop ([ExportService]).
class _ShareButton extends StatelessWidget {
  const _ShareButton({required this.summary});

  final RunSummary summary;

  @override
  Widget build(BuildContext context) {
    return CButton(
      label: 'Share',
      glyph: cIsWide(context) ? CGlyphs.upload : null,
      kind: CButtonKind.primary,
      onPressed: () async {
        final history = context.read<RunHistoryStore>();
        final export = context.read<ExportService>();
        final messenger = ScaffoldMessenger.of(context);
        try {
          final file = await history.documentFile(summary);
          await export.exportRun(file, suggestedName: summary.fileName);
        } catch (e) {
          messenger.showSnackBar(
              SnackBar(content: Text('Could not share the run: $e')));
        }
      },
    );
  }
}
