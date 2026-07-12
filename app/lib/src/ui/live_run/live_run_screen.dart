import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../services/benchmark_service.dart';
import '../common/format.dart';
import '../results/results_body.dart';

/// The in-flight run: current-test banner, elapsed / completed counters,
/// live results ticking in below, and Cancel.
class LiveRunScreen extends StatefulWidget {
  const LiveRunScreen({super.key});

  @override
  State<LiveRunScreen> createState() => _LiveRunScreenState();
}

class _LiveRunScreenState extends State<LiveRunScreen> {
  Timer? _ticker;

  @override
  void initState() {
    super.initState();
    // Redraw the elapsed clock once a second.
    _ticker = Timer.periodic(
        const Duration(seconds: 1), (_) => setState(() {}));
  }

  @override
  void dispose() {
    _ticker?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final cancelling = service.state == BenchmarkState.cancelling;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Benchmarking…'),
        automaticallyImplyLeading: false,
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: Center(
              child: OutlinedButton.icon(
                onPressed: cancelling ? null : service.cancel,
                icon: const Icon(Icons.stop, size: 18),
                label: Text(cancelling ? 'Cancelling…' : 'Cancel'),
              ),
            ),
          ),
        ],
        bottom: const PreferredSize(
          preferredSize: Size.fromHeight(3),
          child: LinearProgressIndicator(minHeight: 3),
        ),
      ),
      body: ResultsBody(
        document: service.document,
        compact: true,
        header: _StatusBanner(service: service, cancelling: cancelling),
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({required this.service, required this.cancelling});

  final BenchmarkService service;
  final bool cancelling;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final statusText = cancelling
        ? 'Cancelling — finishing the current test…'
        : service.currentTest.isNotEmpty
            ? service.currentTest
            : service.currentBackend.isNotEmpty
                ? 'Preparing ${service.currentBackend}…'
                : 'Starting…';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          children: [
            SizedBox(
              width: 22,
              height: 22,
              child: CircularProgressIndicator(
                strokeWidth: 2.5,
                color: cancelling ? scheme.outline : scheme.primary,
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(statusText,
                      style: Theme.of(context).textTheme.bodyMedium,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis),
                  const SizedBox(height: 2),
                  Text(
                    [
                      if (service.currentBackend.isNotEmpty)
                        service.currentBackend,
                      '${service.completedTests} tests done',
                      formatDuration(service.elapsed),
                    ].join(' · '),
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: scheme.outline),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
