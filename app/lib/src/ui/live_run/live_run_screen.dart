import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../services/benchmark_service.dart';
import '../../theme/clpeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import '../results/results_body.dart';

/// The in-flight run: current-test readout, elapsed / completed counters,
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
    final t = CP.of(context);
    final service = context.watch<BenchmarkService>();
    final cancelling = service.state == BenchmarkState.cancelling;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: cancelling ? 'Cancelling' : 'Benchmarking',
              subtitle: 'run in progress',
              actions: [
                CButton(
                  label: cancelling ? 'Cancelling…' : 'Cancel',
                  icon: Icons.stop,
                  danger: !cancelling,
                  onPressed: cancelling ? null : service.cancel,
                ),
              ],
            ),
            // A static rule, not an indeterminate LinearProgressIndicator: the
            // latter animates for the whole run and steals GPU time from it.
            Container(height: 2, color: cancelling ? t.dim : t.text),
            Expanded(
              child: ResultsBody(
                document: service.document,
                compact: true,
                header: _StatusPanel(
                    service: service, cancelling: cancelling),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StatusPanel extends StatelessWidget {
  const _StatusPanel({required this.service, required this.cancelling});

  final BenchmarkService service;
  final bool cancelling;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final statusText = cancelling
        ? 'Cancelling — finishing the current test…'
        : service.currentTest.isNotEmpty
            ? service.currentTest
            : service.currentBackend.isNotEmpty
                ? 'Preparing ${service.currentBackend}…'
                : 'Starting…';

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(12, 11, 12, 11),
            child: Row(
              children: [
                // Static block rather than a spinner, for the same reason the
                // rule under the header is static.
                Container(
                  width: 3,
                  height: 15,
                  color: cancelling ? t.dim : t.text,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(statusText,
                      style: t.mono,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis),
                ),
              ],
            ),
          ),
          Container(height: 1, color: t.line),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 9, 12, 10),
            child: Row(
              children: [
                _Stat(
                  label: 'Backend',
                  value: service.currentBackend.isEmpty
                      ? '—'
                      : service.currentBackend,
                ),
                _Stat(label: 'Tests done', value: '${service.completedTests}'),
                // The clock is its own widget so its once-a-second tick
                // repaints a single Text instead of the whole screen (which
                // would drag the live results list along with it).
                const _Stat(label: 'Elapsed', value: null),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value});

  final String label;

  /// `null` renders the self-ticking elapsed clock instead of a static value.
  final String? value;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label.toUpperCase(), style: t.micro),
          const SizedBox(height: 4),
          value == null
              ? _ElapsedClock(style: t.mono)
              : Text(value!,
                  style: t.mono,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis),
        ],
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
