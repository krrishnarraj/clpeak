import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/run_config.dart';
import '../../services/benchmark_service.dart';
import '../../services/resume_offer.dart';
import '../../services/settings_service.dart';
import '../../theme/xpupeak_theme.dart';
import '../app.dart';
import '../common/kit.dart';
import '../results/results_screen.dart';
import '../run_config/run_config_screen.dart';
import 'processor_groups.dart';

/// The Benchmark tab while no run is in flight: a run the app died in, to
/// be continued; the full run and the way to a custom one; and this
/// system's processors, each with the devices that reach it.
class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final catalog = service.catalog;
    final usable = catalog.usable;
    final devices = usable.fold<int>(0, (n, b) => n + b.deviceCount);
    final wide = cIsWide(context);
    final side = wide ? 34.0 : 16.0;

    final sections = <Widget>[
      if (service.resumeOffer case final offer?)
        _ResumePanel(service: service, offer: offer),
      if (!service.catalogReady)
        // Static text, no animation: the frame budget belongs to the
        // benchmarks, and enumeration finishes in seconds anyway.
        const _Waiting(
          title: 'Detecting devices…',
          detail: 'Probing backends. The app is usable meanwhile.',
        )
      else if (usable.isEmpty)
        const _Waiting(
          title: 'No compute devices found',
          detail: 'No backend could enumerate a device on this system.',
        )
      else ...[
        _Labelled(
          label: 'Run benchmark',
          child: _RunLauncher(service: service, deviceCount: devices),
        ),
        _Labelled(
          label: 'This system',
          trailing: '${usable.length} '
              '${usable.length == 1 ? 'backend' : 'backends'} · '
              '$devices ${devices == 1 ? 'device' : 'devices'}',
          child: ProcessorGrid(children: [
            for (final group in groupDevices(catalog, empty: true))
              _ProcessorCard(group: group, compact: !wide),
          ]),
        ),
      ],
    ];

    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: EdgeInsets.only(bottom: wide ? 44 : 28),
          children: [
            // The sidebar carries the mark on desktop; a phone has only
            // this screen to.
            wide
                ? const CHeader(title: 'Benchmark')
                : _Masthead(version: service.version),
            for (var i = 0; i < sections.length; i++)
              Padding(
                padding: EdgeInsets.fromLTRB(side,
                    i == 0 ? (wide ? 14 : 12) : (wide ? 30 : 24), side, 0),
                child: sections[i],
              ),
          ],
        ),
      ),
    );
  }
}

/// The phone's head: the app mark, the name and the version.
class _Masthead extends StatelessWidget {
  const _Masthead({required this.version});

  final String version;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(22, 16, 22, 6),
      child: Row(
        children: [
          const CAppMark(size: 36),
          const SizedBox(width: 12),
          Expanded(
            child: Text('XpuPeak',
                style: t.wordmark.copyWith(fontSize: 24),
                maxLines: 1,
                overflow: TextOverflow.ellipsis),
          ),
          const SizedBox(width: 12),
          Text('v$version',
              style: t.caption.copyWith(
                  fontFeatures: const [FontFeature.tabularFigures()])),
        ],
      ),
    );
  }
}

/// A section's head in capitals over what it holds.
class _Labelled extends StatelessWidget {
  const _Labelled({required this.label, required this.child, this.trailing});

  final String label;
  final String? trailing;
  final Widget child;

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          CSection(label: label, trailing: trailing),
          SizedBox(height: cIsWide(context) ? 12 : 10),
          child,
        ],
      );
}

/// Detecting, or nothing found: a title and a line in a dashed box.
class _Waiting extends StatelessWidget {
  const _Waiting({required this.title, required this.detail});

  final String title;
  final String detail;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CEmptyBox(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 36),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(title,
              style: t.title.copyWith(fontSize: 16),
              textAlign: TextAlign.center),
          const SizedBox(height: 4),
          Text(detail, style: t.prose, textAlign: TextAlign.center),
        ],
      ),
    );
  }
}

/// The full run -- every detected device, every category, the default
/// budgets -- and the way to choose instead.
class _RunLauncher extends StatelessWidget {
  const _RunLauncher({required this.service, required this.deviceCount});

  final BenchmarkService service;
  final int deviceCount;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final deviceLabel = deviceCount == 1
        ? 'the detected device'
        : 'all $deviceCount detected devices';

    // Runs need a complete catalog: device indices are positions in the
    // enumerated list.  The button enables itself when probing lands.
    final ready = service.catalogReady;

    final run = CButton(
      label: ready ? 'Run benchmark' : 'Detecting…',
      glyph: CGlyphs.play,
      kind: CButtonKind.primary,
      stretch: !wide,
      onPressed: ready
          ? () => service.start(
              preset: RunPreset.full,
              verbose: context.read<SettingsService>().verbose)
          : null,
    );
    final customize = CButton(
      label: 'Customize',
      onPressed: () => Navigator.of(context).push(RunConfigScreen.route()),
    );

    return CPanel(
      padding: wide
          ? const EdgeInsets.fromLTRB(24, 20, 24, 22)
          : const EdgeInsets.fromLTRB(18, 16, 18, 18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Measures compute, bandwidth and latency on $deviceLabel.',
            style: t.title.copyWith(fontSize: wide ? 17 : 16, height: 1.35),
          ),
          const SizedBox(height: 4),
          Text('Customize picks the devices, the test categories and the '
              'time budgets.',
              style: t.caption.copyWith(fontSize: wide ? 13.5 : 13)),
          SizedBox(height: wide ? 18 : 16),
          if (wide)
            Wrap(
              spacing: 10,
              runSpacing: 10,
              children: [run, customize],
            )
          else ...[
            run,
            const SizedBox(height: 10),
            SizedBox(height: 46, child: customize),
          ],
        ],
      ),
    );
  }
}

/// The newest run the app died in, while it can be continued: what
/// happened, what it saved, and Continue -- which runs only what is left,
/// into the same result (`--resume`).
class _ResumePanel extends StatelessWidget {
  const _ResumePanel({required this.service, required this.offer});

  final BenchmarkService service;
  final ResumeOffer offer;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final saved = offer.testsSaved;
    final detail = [
      saved == 0
          ? 'Nothing had finished yet; Continue starts it again under the same name.'
          : '$saved ${saved == 1 ? 'test is' : 'tests are'} saved. Continue runs '
              'only what is left and keeps both parts as one result.',
      if (offer.skipCrashed)
        'The test it stopped in is not run again, since it would most likely '
            'stop it again.',
    ].join(' ');

    return CPanel(
      color: t.surface2,
      outlined: false,
      padding: wide
          ? const EdgeInsets.fromLTRB(24, 20, 24, 20)
          : const EdgeInsets.fromLTRB(18, 16, 18, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CTag(
            text: 'Did not finish',
            upper: false,
            color: t.warnInk,
            fill: t.warnTint,
          ),
          const SizedBox(height: 10),
          Text(offer.headline,
              style: t.title.copyWith(fontSize: wide ? 17 : 16, height: 1.35)),
          const SizedBox(height: 4),
          Text(detail, style: t.prose),
          const SizedBox(height: 16),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              CButton(
                label: 'Continue',
                glyph: CGlyphs.play,
                kind: CButtonKind.primary,
                onPressed: () => service.resume(offer),
              ),
              if (saved > 0)
                CButton(
                  label: "View what's saved",
                  onPressed: () => openCheckpoint(context,
                      id: offer.id, startedAt: offer.startedAt),
                ),
              CButton(
                label: 'Dismiss',
                kind: CButtonKind.quiet,
                onPressed: service.dismissResumeOffer,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// A processor in its kind's tint, and each backend's view of it: what the
/// backend calls it where that differs, and what its API reports.
class _ProcessorCard extends StatelessWidget {
  const _ProcessorCard({required this.group, required this.compact});

  final ProcessorGroup group;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = t.kind(group.kind);
    final sub = t.caption.copyWith(fontSize: compact ? 12.5 : 13, color: h.tintSub);
    return CKindCard(
      kind: group.kind,
      label: group.label,
      title: group.name,
      compact: compact,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (group.facts.isNotEmpty) Text(group.facts, style: sub),
          if (group.devices.isEmpty) ...[
            if (group.facts.isNotEmpty) const SizedBox(height: 10),
            Text('No backend reaches it.', style: sub),
          ] else ...[
            SizedBox(height: group.facts.isEmpty ? 2 : 12),
            for (final g in group.devices) _DeviceLine(device: g, group: group),
          ],
        ],
      ),
    );
  }
}

class _DeviceLine extends StatelessWidget {
  const _DeviceLine({required this.device, required this.group});

  final GroupDevice device;
  final ProcessorGroup group;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = t.kind(group.kind);
    final detail = deviceDetail(device, group);
    final caps = <String>[
      if (device.device.fp16) 'fp16',
      if (device.device.fp64) 'fp64',
    ];
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: h.rule)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(device.backend.name,
                    style: t.title.copyWith(fontSize: 14.5, color: h.tintInk)),
                // Wraps rather than ellipsizes — driver-reported device
                // names run long, and the tail is what distinguishes two
                // of them.
                if (detail.isNotEmpty) ...[
                  const SizedBox(height: 1),
                  Text(detail,
                      style: t.small.copyWith(fontSize: 12.5, color: h.tintSub)),
                ],
              ],
            ),
          ),
          for (final cap in caps) ...[
            const SizedBox(width: 6),
            CTag(text: cap, color: h.tintInk, fill: t.tagOnTint),
          ],
        ],
      ),
    );
  }
}
