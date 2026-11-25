import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_document.dart';
import '../../services/benchmark_service.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import '../dashboard/processor_groups.dart';
import '../results/results_body.dart';

/// The in-flight run: what it is doing now, the device and the counters,
/// live results ticking in below, Pause and Cancel.
///
/// This screen is deliberately animation-free.  Every frame it presents is
/// GPU work on the very device being benchmarked (the process shows up as
/// `C+G` in nvidia-smi), and a continuously-animating indicator costs the
/// running benchmark 10-15% of its score.  So: no spinner, no progress bar,
/// no implicit animation, no tooltip -- the state is a static dot and a
/// card's colour, and liveness is carried by the once-a-second elapsed
/// clock and the changing test name.  The screen rebuilds only on
/// [BenchmarkService]'s coalesced tick, and `_ElapsedClock` repaints a
/// single Text rather than the screen.
class LiveRunScreen extends StatelessWidget {
  const LiveRunScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final cancelling = service.state == BenchmarkState.cancelling;
    final wide = cIsWide(context);
    final actions = _actions(service, cancelling);

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: cancelling
                  ? 'Cancelling'
                  : service.paused
                      ? (service.held ? 'Paused' : 'Pausing')
                      : service.coolingDown
                          ? 'Cooling down'
                          : 'Benchmarking',
              subtitle: service.wasResumed
                  ? 'Continuing an earlier run'
                  : 'Run in progress',
              // A phone's header has no room beside its title: there the
              // actions sit in the status card.
              actions: wide ? actions : const [],
            ),
            Expanded(
              child: ResultsBody(
                document: service.document,
                header: _StatusCard(
                  service: service,
                  cancelling: cancelling,
                  actions: wide ? null : actions,
                ),
                showDiagnostics: service.runVerbose,
              ),
            ),
          ],
        ),
      ),
    );
  }

  static List<Widget> _actions(BenchmarkService service, bool cancelling) => [
        if (!cancelling)
          // Held at the next boundary between readings, so no reading
          // measures the pause (include/common/screen_hold.h).
          CButton(
            label: service.paused ? 'Resume' : 'Pause',
            glyph: service.paused ? CGlyphs.play : null,
            kind: service.paused ? CButtonKind.primary : CButtonKind.ghost,
            onPressed: () => service.setPaused(!service.paused),
          ),
        CButton(
          label: cancelling ? 'Cancelling…' : 'Cancel',
          danger: !cancelling,
          onPressed: cancelling ? null : service.cancel,
        ),
      ];
}

/// The run's state in a card of one colour: the kind of the device being
/// measured (CPU blue, GPU green, NPU violet), neutral while nothing is,
/// warm while the device cools.  Colour, a static dot and words -- nothing
/// that moves.
class _StatusCard extends StatelessWidget {
  const _StatusCard({
    required this.service,
    required this.cancelling,
    this.actions,
  });

  final BenchmarkService service;
  final bool cancelling;

  /// Pause and Cancel, on a phone.
  final List<Widget>? actions;

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

  /// The device the run is on: the newest the run has reached, once its
  /// backend is the one in flight -- a resumed run's checkpoint ends with
  /// the device it stopped in.
  DeviceRun? _inFlight() {
    final runs = service.document.runs;
    if (runs.isEmpty || service.currentBackend.isEmpty) return null;
    return runs.last.backend == service.currentBackend ? runs.last : null;
  }

  /// The device's kind, from the catalog it was listed in (a live run's
  /// devices carry none until the file is linked and saved).
  String _kindOf(DeviceRun run) {
    final catalog = service.catalog;
    for (final b in catalog.backends) {
      if (b.name != run.backend) continue;
      for (final p in b.platforms) {
        if (b.platforms.length > 1 && p.name != run.platform) continue;
        for (final d in p.devices) {
          if (run.index >= 0 ? d.index == run.index : d.name == run.device) {
            return kindOfDevice(catalog, d);
          }
        }
      }
    }
    return run.type.isEmpty ? 'other' : kindOfDeviceType(run.type);
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final s = service;
    final device = _inFlight();
    final measuring = !cancelling && !s.paused && !s.coolingDown;
    final kind = measuring && device != null ? _kindOf(device) : 'other';
    final cooling = !cancelling && !s.paused && s.coolingDown;

    final Color fill, ink, sub, accent;
    if (cooling) {
      fill = t.warnTint;
      ink = t.warnInk;
      sub = t.warnInk.withValues(alpha: 0.78);
      accent = t.warnInk;
    } else {
      final h = t.kind(kind);
      fill = h.tint;
      ink = h.tintInk;
      sub = h.tintSub;
      accent = h.accent;
    }

    final state = cancelling
        ? 'CANCELLING'
        : s.paused
            ? (s.held ? 'PAUSED' : 'PAUSING')
            : cooling
                ? 'COOLING DOWN'
                : kind == 'other'
                    ? 'RUNNING'
                    : 'RUNNING · ${kind.toUpperCase()}';

    final stats = [
      _Stat(
        label: 'Backend',
        value: s.currentBackend.isEmpty ? '—' : s.currentBackend,
        ink: ink,
        sub: sub,
      ),
      _Stat(
        label: 'Device',
        value: device == null || device.device.isEmpty ? '—' : device.device,
        ink: ink,
        sub: sub,
      ),
      _Stat(
        label: 'Tests done',
        value: '${s.completedTests}',
        number: true,
        ink: ink,
        sub: sub,
      ),
      // The clock is its own widget so its once-a-second tick repaints a
      // single Text instead of the whole screen (which would drag the
      // live results list along with it).
      _Stat(label: 'Elapsed', value: null, number: true, ink: ink, sub: sub),
    ];

    return Container(
      padding: wide
          ? const EdgeInsets.fromLTRB(24, 20, 24, 20)
          : const EdgeInsets.fromLTRB(20, 17, 20, 18),
      decoration: BoxDecoration(
        color: fill,
        borderRadius: BorderRadius.circular(CP.rCardLarge),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              // A static dot rather than a spinner, for the reason above:
              // filled while the run measures, a ring while it does not.
              Container(
                width: 9,
                height: 9,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: measuring ? accent : null,
                  border: measuring ? null : Border.all(color: accent, width: 1.5),
                ),
              ),
              const SizedBox(width: 9),
              Expanded(
                child: Text(state,
                    style: t.capsHead.copyWith(
                        fontSize: wide ? 12.5 : 12,
                        fontWeight: FontWeight.w700,
                        color: accent)),
              ),
            ],
          ),
          const SizedBox(height: 6),
          // Wraps rather than ellipsizes: the tail ("once it cools…", a
          // test name) is what tells this status apart, so cutting it hides
          // exactly the meaningful part.
          Text(
            _statusText(),
            style: t.title.copyWith(
              fontSize: wide ? 20 : 17.5,
              height: 1.25,
              letterSpacing: (wide ? 20 : 17.5) * -0.02,
              color: ink,
            ),
          ),
          SizedBox(height: wide ? 18 : 16),
          // Four across where they fit, two by two on a phone.
          if (wide)
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                for (var i = 0; i < stats.length; i++) ...[
                  if (i > 0) const SizedBox(width: 16),
                  Expanded(flex: i == 1 ? 2 : 1, child: stats[i]),
                ],
              ],
            )
          else ...[
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: stats[0]),
                const SizedBox(width: 12),
                Expanded(child: stats[2]),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: stats[1]),
                const SizedBox(width: 12),
                Expanded(child: stats[3]),
              ],
            ),
          ],
          // A phone holds the run while the app is off screen
          // (include/common/screen_hold.h), and may end the app meanwhile.
          if (Platform.isAndroid || Platform.isIOS) ...[
            const SizedBox(height: 14),
            Text('Leaving XpuPeak pauses the run until you come back.',
                style: t.small.copyWith(color: sub)),
          ],
          if (actions != null) ...[
            const SizedBox(height: 16),
            Wrap(spacing: 8, runSpacing: 8, children: actions!),
          ],
        ],
      ),
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({
    required this.label,
    required this.value,
    required this.ink,
    required this.sub,
    this.number = false,
  });

  final String label;

  /// `null` renders the self-ticking elapsed clock instead of a static value.
  final String? value;

  /// A count or a time, in tabular figures.
  final bool number;
  final Color ink;
  final Color sub;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final style = number
        ? t.figure(20, weight: FontWeight.w700, color: ink)
        : t.title.copyWith(fontSize: 15, height: 1.3, color: ink);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label.toUpperCase(),
            style: t.caps.copyWith(fontSize: 11, color: sub)),
        const SizedBox(height: 4),
        value == null
            // Its own layer too: the tick repaints the clock alone.
            ? RepaintBoundary(child: _ElapsedClock(style: style))
            // Wraps: a device's name is identity.
            : Text(value!, style: style),
      ],
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
