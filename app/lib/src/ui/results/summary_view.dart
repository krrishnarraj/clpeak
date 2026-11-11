import 'package:flutter/material.dart';

import '../../model/headlines.dart';
import '../../model/measures.dart';
import '../../model/result_model.dart';
import '../../model/run_document.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';

// The Summary tab: which machine the run was on, then for every processor in
// it -- every piece of hardware, however many backends reached it -- its
// flagship numbers: the headline catalog's measures (assets/measures.json),
// each the best raw reading over every backend, with the data type it was
// reached at and the backend that reached it.  Nothing is scored or derived
// beyond picking the best reading; the readings are in the backends' tabs.

/// The machine: its name, what kind of machine it is, the OS, and how the
/// run went.  Above the processors, since every number below is this
/// machine's.
class SystemPanel extends StatelessWidget {
  const SystemPanel({super.key, required this.meta});

  final RunMeta meta;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final s = meta.systemInfo;
    final c = meta.conditions;
    final title = meta.systemName;
    String other(String v) => v.isNotEmpty && v != title ? v : '';
    final os = s.osDistribution.isNotEmpty
        ? s.osDistribution
        : [s.osName, s.osVersion].where((v) => v.isNotEmpty).join(' ');
    final thermal = c.thermalStart.isEmpty || c.thermalStart == c.thermalEnd
        ? c.thermalStart
        : [c.thermalStart, c.thermalEnd].where((v) => v.isNotEmpty).join(' → ');
    final generator = meta.generator;
    final entries = <(String, String)>[
      ('MODEL NAME', other(s.name)),
      ('MODEL', other(s.model)),
      ('BOARD', other(s.board)),
      ('MAKER', other(s.manufacturer)),
      ('OS', os),
      ('ARCH', s.arch),
      ('MEMORY', s.memoryBytes > 0 ? formatBytes(s.memoryBytes) : ''),
      (
        'POWER',
        [
          switch (c.powerSource) { 'ac' => 'AC', final p => p },
          if (c.lowPowerMode == true) 'low-power mode',
        ].where((v) => v.isNotEmpty).join(', ')
      ),
      ('PROFILE', c.powerProfile),
      ('GOVERNOR', c.cpuGovernor),
      ('THERMAL', thermal),
      // Named by the file's own generator: "XPUPEAK VERSION".
      (
        generator.name.isEmpty
            ? 'VERSION'
            : '${generator.name.toUpperCase()} VERSION',
        generator.version
      ),
      (
        'DURATION',
        meta.durationSeconds > 0
            ? formatDuration(Duration(
                milliseconds: (meta.durationSeconds * 1000).round()))
            : ''
      ),
    ].where((e) => e.$2.isNotEmpty).toList();
    // What only a person knows, under the keys they were saved with.
    final notes = [
      for (final e in meta.annotations.entries)
        if (e.key != 'system.name' && e.value.trim().isNotEmpty)
          (e.key.toUpperCase().replaceAll(RegExp(r'[._]'), ' '), e.value),
    ];

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
            decoration: BoxDecoration(
              color: t.isDark ? t.hover : t.hover.withValues(alpha: 0.7),
              border: Border(bottom: BorderSide(color: t.line)),
            ),
            child: Row(
              children: [
                Icon(_formIcon(s.formFactor, s.formFactorDetail),
                    size: 15, color: t.dim),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(title.isEmpty ? 'System' : title,
                      style: t.title),
                ),
                Wrap(
                  spacing: 5,
                  runSpacing: 5,
                  children: [
                    // Only what the file says: a file without a form factor
                    // is read as a desktop, but showing one would claim it.
                    if (s.formFactor.isNotEmpty) CTag(text: s.formFactor),
                    if (s.formFactorDetail.isNotEmpty)
                      CTag(text: s.formFactorDetail),
                    if (s.isVirtual) const CTag(text: 'virtual'),
                    if (s.translated) const CTag(text: 'translated'),
                    if (meta.cancelled) CTag(text: 'cancelled', color: t.danger),
                  ],
                ),
              ],
            ),
          ),
          if (entries.isNotEmpty || notes.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 11, 12, 11),
              child: Wrap(
                spacing: 22,
                runSpacing: 7,
                children: [
                  for (final (key, value) in [...entries, ...notes])
                    Text.rich(TextSpan(children: [
                      TextSpan(text: '$key  ', style: t.micro),
                      TextSpan(text: value, style: t.monoSmall),
                    ])),
                ],
              ),
            ),
        ],
      ),
    );
  }

  static IconData _formIcon(String formFactor, String detail) =>
      switch ((formFactor, detail)) {
        (_, 'phone') => Icons.smartphone,
        (_, 'tablet') => Icons.tablet_mac,
        ('mobile', _) => Icons.smartphone,
        ('laptop', _) => Icons.laptop_mac,
        ('server', _) => Icons.dns_outlined,
        (_, 'embedded') => Icons.developer_board,
        _ => Icons.desktop_windows_outlined,
      };
}

/// One processor's flagship numbers.
class FlagshipCard extends StatelessWidget {
  const FlagshipCard({
    super.key,
    required this.group,
    required this.values,
    this.package,
    this.onDevice,
  });

  final ProcessorMeasures group;

  /// Each headline that applies to this kind of processor, with its value
  /// here -- null where nothing here measured it.
  final List<(Headline, HeadlineValue?)> values;

  /// The package the processor is in, when the file says.
  final PackageInfo? package;

  /// Opens one of its devices in its backend's tab.
  final ValueChanged<DeviceRun>? onDevice;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final facts = processorFacts(group, package);
    final kind = group.kind.isEmpty ? 'device' : group.kind;

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.fromLTRB(12, 11, 12, 11),
            decoration: BoxDecoration(
              border: Border(bottom: BorderSide(color: t.line)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.only(top: 1),
                  child: CTag(text: kind, filled: true),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Wraps, never ellipsizes: a name is identity.
                      Text(group.name.isEmpty ? 'Unnamed device' : group.name,
                          style: t.title),
                      if (facts.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Text(facts, style: t.monoSmallDim),
                      ],
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                Text(group.linked ? group.id : group.devices.first.backend,
                    style: t.micro),
              ],
            ),
          ),
          if (values.isEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
              child: Text('No headline applies to this kind of device.',
                  style: t.monoSmallDim),
            )
          else
            LayoutBuilder(builder: (context, c) {
              final stats = [
                for (final (h, v) in values)
                  _Stat(headline: h, value: v, device: group.name),
              ];
              // Side by side where there is room for three big numbers;
              // stacked on a phone.
              if (c.maxWidth >= 600) {
                return IntrinsicHeight(
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      for (var i = 0; i < stats.length; i++) ...[
                        if (i > 0) Container(width: 1, color: t.line),
                        Expanded(child: stats[i]),
                      ],
                    ],
                  ),
                );
              }
              return Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (var i = 0; i < stats.length; i++) ...[
                    if (i > 0) Container(height: 1, color: t.line),
                    stats[i],
                  ],
                ],
              );
            }),
          if (group.devices.isNotEmpty)
            Container(
              padding: const EdgeInsets.fromLTRB(12, 9, 12, 10),
              decoration: BoxDecoration(
                border: Border(top: BorderSide(color: t.line)),
              ),
              child: Wrap(
                spacing: 6,
                runSpacing: 6,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: Text('MEASURED THROUGH', style: t.micro),
                  ),
                  for (final run in group.devices)
                    _DeviceLink(
                      label: _deviceLabel(run, group.devices),
                      onTap: onDevice == null ? null : () => onDevice!(run),
                    ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  /// A device by its backend, and by its backend's name for it where that
  /// backend reached the processor more than once (MoltenVK shows one GPU
  /// twice; Core ML reaches a CPU and a GPU of one SoC).
  static String _deviceLabel(DeviceRun run, List<DeviceRun> all) {
    final same = all.where((r) => r.backend == run.backend).length;
    return same > 1 ? '${run.backend} · ${run.device} #${run.index}' : run.backend;
  }
}

/// A processor's facts in one line: cores and clusters for a CPU, compute
/// units and memory for a GPU or an NPU, and the package it sits in when
/// that names something more.
String processorFacts(ProcessorMeasures group, PackageInfo? package) {
  final p = group.processor;
  final run = group.devices.isEmpty ? null : group.devices.first;
  // An unlinked device has only its own API's facts.
  final f = p?.facts ?? run?.facts ?? const HardwareFacts();
  String clock(int mhz) => '${(mhz / 1000).toStringAsFixed(1)} GHz';
  final parts = <String>[];
  if (group.kind == 'cpu') {
    final cpu = f.cpu;
    if (cpu.cores > 0) parts.add('${cpu.cores} cores');
    if (cpu.threads > 0 && cpu.threads != cpu.cores) {
      parts.add('${cpu.threads} threads');
    }
    final kinds = [
      for (final c in cpu.clusters)
        if (c.cores > 0 && c.kind.isNotEmpty)
          '${c.cores}${switch (c.kind) {
            'prime' => 'X',
            'performance' => 'P',
            'efficiency' => 'E',
            'low_power' => 'LP',
            final k => ' $k',
          }}',
    ];
    if (kinds.length > 1) parts.add(kinds.join(' + '));
    final maxMhz = cpu.clusters.fold(0, (m, c) => c.maxMhz > m ? c.maxMhz : m);
    if (maxMhz > 0) parts.add(clock(maxMhz));
    if (cpu.sockets > 1) parts.add('${cpu.sockets} sockets');
    if (cpu.isa.isNotEmpty) parts.add(cpu.isa);
    for (final e in cpu.engines) {
      if (e.name.isNotEmpty) parts.add(e.name);
    }
  } else {
    if (f.computeUnits > 0) {
      final unit = f.computeUnit.isEmpty ? 'unit' : f.computeUnit;
      parts.add('${f.computeUnits} $unit${f.computeUnits == 1 ? '' : 's'}');
    }
    if (f.arch.isNotEmpty) parts.add(f.arch);
    if (f.clockMhz > 0) parts.add(clock(f.clockMhz));
    if (f.integrated == true) {
      parts.add('integrated');
    } else if (f.memoryBytes > 0) {
      parts.add(formatBytes(f.memoryBytes));
    }
  }
  if (p == null && parts.isEmpty && run != null) {
    if (run.platform.isNotEmpty && run.platform != run.backend) {
      parts.add(run.platform);
    }
    if (run.driver.isNotEmpty) parts.add(run.driver);
  }
  if (p != null &&
      package != null &&
      package.name.isNotEmpty &&
      package.name != p.name) {
    parts.add('in ${package.name}');
  }
  // Each fact whole ("4.3 GHz", never "4.3" over "GHz"): a narrow screen
  // wraps between them.
  return [for (final s in parts) s.replaceAll(' ', ' ')].join('  ·  ');
}

/// One flagship number: what it is, the number, what it was reached at, and
/// which backend reached it.  Tapping it shows every backend's best.
class _Stat extends StatelessWidget {
  const _Stat({required this.headline, required this.value, required this.device});

  final Headline headline;
  final HeadlineValue? value;
  final String device;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final v = value;
    final tint = XpuPeakTheme.categoryColor(
        headline.quantity == 'bytes_per_second'
            ? BenchCategory.bandwidth
            : BenchCategory.compute,
        brightness: Theme.of(context).brightness);
    final big = t.value.copyWith(fontSize: 30, letterSpacing: -0.9, height: 1.05);

    final body = Padding(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 13),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(headline.title.toUpperCase(), style: t.micro),
          const SizedBox(height: 9),
          if (v == null) ...[
            Text('—', style: big.copyWith(color: t.faint)),
            const SizedBox(height: 6),
            Text('not measured on this device', style: t.monoSmallDim),
          ] else ...[
            Builder(builder: (context) {
              final f = formatValue(v.best.value, v.best.units);
              return Text.rich(
                TextSpan(children: [
                  TextSpan(text: f.value, style: big.copyWith(color: tint)),
                  TextSpan(
                      text: ' ${f.unit}',
                      style: t.mono.copyWith(color: t.dim, fontSize: 14)),
                ]),
              );
            }),
            const SizedBox(height: 6),
            Text(_words(v.best), style: t.monoSmall),
            const SizedBox(height: 7),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                CTag(text: v.best.backend, upper: false, color: tint),
                if (v.byBackend.length > 1)
                  Text('best of ${v.byBackend.length} backends',
                      style: t.micro.copyWith(fontSize: 9.5)),
              ],
            ),
          ],
        ],
      ),
    );
    if (v == null) return body;
    return CTap(
      onTap: () => showHeadlineDialog(context, device: device, value: v),
      builder: (context, hovered, pressed) => Container(
        color: hovered || pressed ? t.hover : Colors.transparent,
        child: body,
      ),
    );
  }
}

/// A reading's data type and qualifiers, or what its family measures where
/// it names neither.
String _words(MeasuredReading r) {
  final words = describeReading(r);
  return words.isEmpty ? r.test.title : words.join(' · ');
}

class _DeviceLink extends StatelessWidget {
  const _DeviceLink({required this.label, this.onTap});

  final String label;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onTap,
      builder: (context, hovered, pressed) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 4),
        decoration: BoxDecoration(
          color: hovered || pressed ? t.hover : Colors.transparent,
          border: Border.all(color: t.line),
          borderRadius: BorderRadius.circular(CP.rControl),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Flexible(
              child: Text(label,
                  style: t.monoSmall.copyWith(color: hovered ? t.text : t.dim)),
            ),
            if (onTap != null) ...[
              const SizedBox(width: 4),
              Icon(Icons.chevron_right, size: 12, color: t.faint),
            ],
          ],
        ),
      ),
    );
  }
}

/// Where a flagship number came from, and what every other backend reached
/// on the same device.  Opened without a transition, like every dialog on
/// the results screen (app/AGENTS.md).
Future<void> showHeadlineDialog(BuildContext context,
    {required String device, required HeadlineValue value}) {
  return showGeneralDialog<void>(
    context: context,
    barrierDismissible: true,
    barrierLabel: MaterialLocalizations.of(context).modalBarrierDismissLabel,
    barrierColor: Colors.black.withValues(alpha: 0.55),
    transitionDuration: Duration.zero,
    pageBuilder: (context, _, _) => _HeadlineDialog(device: device, value: value),
  );
}

class _HeadlineDialog extends StatelessWidget {
  const _HeadlineDialog({required this.device, required this.value});

  final String device;
  final HeadlineValue value;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final best = value.best;
    final f = formatValue(best.value, best.units);
    final ranked = value.ranked;
    return CDialog(
      title: '${value.headline.title} · $device',
      actions: [
        CButton(label: 'Close', onPressed: () => Navigator.pop(context)),
      ],
      child: ConstrainedBox(
        constraints:
            BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.6),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CValue(value: f.value, unit: f.unit),
              const SizedBox(height: 6),
              Text(_words(best), style: t.mono),
              const SizedBox(height: 10),
              Text(
                'Measured through ${best.backend} on ${best.device.device}: '
                '${best.test.displayTitle}, reading '
                '"${best.metric.displayLabel}"'
                '${best.revision > 1 ? ' (method revision ${best.revision})' : ''}.',
                style: t.body,
              ),
              if (ranked.length > 1) ...[
                const SizedBox(height: 14),
                Text("EACH BACKEND'S BEST", style: t.micro),
                const SizedBox(height: 6),
                for (final r in ranked)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 3),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        SizedBox(
                          width: 64,
                          child: Text(r.backend,
                              style: t.monoSmall.copyWith(
                                  fontWeight: identical(r, best)
                                      ? FontWeight.w700
                                      : FontWeight.w400)),
                        ),
                        Expanded(
                            child: Text(_words(r), style: t.monoSmallDim)),
                        const SizedBox(width: 10),
                        Builder(builder: (context) {
                          final g = formatValue(r.value, r.units);
                          return CValue(value: g.value, unit: g.unit, small: true);
                        }),
                      ],
                    ),
                  ),
              ],
              const SizedBox(height: 12),
              Text(
                'The best reading on this device over every backend that '
                'reached it, as measured: nothing is scored or derived.',
                style: t.monoSmallDim,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
