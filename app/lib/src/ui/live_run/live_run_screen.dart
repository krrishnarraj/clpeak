import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../services/benchmark_service.dart';
import '../common/format.dart';
import '../results/results_body.dart';

/// The in-flight run: current-test banner, elapsed / completed counters,
/// live results ticking in below, and Cancel.
///
/// This screen is deliberately animation-free.  Every frame it presents is
/// GPU work on the very device being benchmarked (the process shows up as
/// `C+G` in nvidia-smi), and a continuously-animating indicator costs the
/// running benchmark 10-15% of its score.  Liveness is carried by the
/// once-a-second elapsed clock and the changing test name instead — see
/// `_ElapsedClock`, which repaints a single Text rather than the screen.
class LiveRunScreen extends StatelessWidget {
  const LiveRunScreen({super.key});

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
        // A static rule, not an indeterminate LinearProgressIndicator: the
        // latter animates for the whole run and steals GPU time from it.
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(3),
          child: Container(
            height: 3,
            color: Theme.of(context).colorScheme.primary,
          ),
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
    final subtle = Theme.of(context)
        .textTheme
        .bodySmall
        ?.copyWith(color: scheme.outline);
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
            // Static icon rather than a spinner, for the same reason the
            // app-bar rule is static.
            SizedBox(
              width: 22,
              height: 22,
              child: Icon(
                cancelling ? Icons.stop_circle_outlined : Icons.speed,
                size: 22,
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
                  // The clock is its own widget so its once-a-second tick
                  // repaints a single Text instead of the whole screen
                  // (which would drag the live results list along with it).
                  Row(
                    children: [
                      Text(
                        [
                          if (service.currentBackend.isNotEmpty)
                            service.currentBackend,
                          '${service.completedTests} tests done',
                          '',
                        ].join(' · '),
                        style: subtle,
                      ),
                      _ElapsedClock(style: subtle),
                    ],
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

/// The elapsed-time readout, ticking once a second on its own.
///
/// Isolated from the rest of the live screen so the tick costs one Text
/// repaint per second rather than a full rebuild of the results list — and
/// so the run never pays for a continuously-animating widget.
class _ElapsedClock extends StatefulWidget {
  const _ElapsedClock({this.style});

  final TextStyle? style;

  @override
  State<_ElapsedClock> createState() => _ElapsedClockState();
}

class _ElapsedClockState extends State<_ElapsedClock> {
  Timer? _ticker;

  @override
  void initState() {
    super.initState();
    _ticker = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _ticker?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Text(
        formatDuration(context.read<BenchmarkService>().elapsed),
        style: widget.style,
      );
}
