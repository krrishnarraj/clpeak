import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../services/benchmark_service.dart';
import '../../theme/clpeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import '../results/results_body.dart';

/// The in-flight run: current-test readout, elapsed / completed counters,
/// live results ticking in below, Pause and Cancel.
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
    final paused = service.paused;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: cancelling
                  ? 'Cancelling'
                  : paused
                      ? 'Paused'
                      : 'Benchmarking',
              subtitle: 'run in progress',
              actions: [
                if (!cancelling) ...[
                  // Held at the next boundary between readings, so no
                  // reading measures the pause (include/common/
                  // screen_hold.h).
                  CButton(
                    label: paused ? 'Resume' : 'Pause',
                    icon: paused ? Icons.play_arrow : Icons.pause,
                    onPressed: () => service.setPaused(!paused),
                  ),
                  const SizedBox(width: 8),
                ],
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
            Container(
                height: 2, color: cancelling || paused ? t.dim : t.text),
            Expanded(
              child: ResultsBody(
                document: service.document,
                header: _StatusPanel(
                    service: service, cancelling: cancelling),
                showDiagnostics: service.runVerbose,
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

  String _statusText() {
    if (cancelling) return 'Cancelling — finishing the current test…';
    if (service.paused) {
      return service.held
          ? 'Paused — the run waits before its next reading until you '
              'resume it.'
          : 'Pausing — the reading in progress finishes first…';
    }
    if (service.coolingDown) {
      return 'Paused — the device is hot; resuming once it cools…';
    }
    if (service.currentTest.isNotEmpty) return service.currentTest;
    if (service.currentBackend.isNotEmpty) {
      return 'Preparing ${service.currentBackend}…';
    }
    return 'Starting…';
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final statusText = _statusText();

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(12, 11, 12, 11),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Static block rather than a spinner, for the same reason the
                // rule under the header is static.
                Container(
                  width: 3,
                  height: 15,
                  margin: const EdgeInsets.only(top: 2),
                  color: cancelling || service.paused ? t.dim : t.text,
                ),
                const SizedBox(width: 10),
                Expanded(
                  // Wraps rather than ellipsizes: the tail ("once it cools…",
                  // a test name) is what tells this status apart, so cutting
                  // it hides exactly the meaningful part.
                  child: Text(statusText, style: t.mono),
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
          // A phone holds the run while the app is off screen
          // (include/common/screen_hold.h), and may end the app meanwhile.
          if (Platform.isAndroid || Platform.isIOS) ...[
            Container(height: 1, color: t.line),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 8, 12, 9),
              child: Text(
                  'Leaving clpeak pauses the run until you come back.',
                  style: t.monoSmallDim),
            ),
          ],
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
