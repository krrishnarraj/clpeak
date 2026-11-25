import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../../model/headlines.dart';
import '../../model/measures.dart';
import '../../model/run_document.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import 'results_kit.dart';

// The Summary tab: for every processor in the run -- every piece of
// hardware, however many backends reached it -- a card in its kind's tint
// with its flagship numbers: the headline catalog's measures
// (assets/measures.json), each the best raw reading over every backend, with
// the data type it was reached at and a link to the backend that reached it.
// Nothing is scored or derived beyond picking the best reading.  Below the
// cards, a way into every backend's tab, where the readings are; then the
// machine the run was on.

/// One processor's card, as [FlagshipGrid] lays it out.
typedef FlagshipData = ({
  ProcessorMeasures group,
  List<(Headline, HeadlineValue?)> values,
});

/// The processors' cards: side by side in columns at least 250 px wide on a
/// wide window, each row's cards as tall as its tallest; stacked on a phone.
class FlagshipGrid extends StatelessWidget {
  const FlagshipGrid({super.key, required this.cards, this.onDevice});

  final List<FlagshipData> cards;

  /// Opens one of a processor's devices in its backend's tab.
  final ValueChanged<DeviceRun>? onDevice;

  @override
  Widget build(BuildContext context) {
    FlagshipCard card(FlagshipData d, {bool spread = false}) => FlagshipCard(
          group: d.group,
          values: d.values,
          onDevice: onDevice,
          spread: spread,
        );
    if (!cIsWide(context)) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (var i = 0; i < cards.length; i++) ...[
            if (i > 0) const SizedBox(height: 12),
            card(cards[i]),
          ],
        ],
      );
    }
    return LayoutBuilder(builder: (context, c) {
      const minWidth = 250.0, gap = 18.0;
      final fit = ((c.maxWidth + gap) / (minWidth + gap)).floor();
      final columns = math.max(1, math.min(cards.length, fit));
      final width = (c.maxWidth - gap * (columns - 1)) / columns;
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (var start = 0; start < cards.length; start += columns) ...[
            if (start > 0) const SizedBox(height: gap),
            IntrinsicHeight(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (var i = start;
                      i < math.min(cards.length, start + columns);
                      i++) ...[
                    if (i > start) const SizedBox(width: gap),
                    SizedBox(
                      width: width,
                      // A card as wide as two sets its numbers side by side.
                      child: card(cards[i], spread: width >= 600),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ],
      );
    });
  }
}

/// One processor's flagship numbers, on a card in its kind's tint.
class FlagshipCard extends StatelessWidget {
  const FlagshipCard({
    super.key,
    required this.group,
    required this.values,
    this.onDevice,
    this.spread = false,
  });

  final ProcessorMeasures group;

  /// Each headline that applies to this kind of processor, with its value
  /// here -- null where nothing here measured it.
  final List<(Headline, HeadlineValue?)> values;

  /// Opens one of its devices in its backend's tab.
  final ValueChanged<DeviceRun>? onDevice;

  /// The numbers side by side rather than stacked, on a wide card.
  final bool spread;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final kind = group.kind;
    final h = t.kind(kind);
    final name = group.name.isEmpty ? 'Unnamed device' : group.name;

    Widget body;
    if (values.isEmpty) {
      body = Text('No headline applies to this kind of device.',
          style: t.small.copyWith(fontSize: 13, color: h.tintSub));
    } else if (!wide) {
      // Three to a row, as many rows as the catalog has headlines.
      const perRow = 3;
      body = Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (var start = 0; start < values.length; start += perRow) ...[
            if (start > 0) const SizedBox(height: 14),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                for (var i = start; i < start + perRow; i++) ...[
                  if (i > start) const SizedBox(width: 10),
                  Expanded(
                    child: i < values.length
                        ? _Stat(
                            headline: values[i].$1,
                            value: values[i].$2,
                            device: name,
                            hue: h,
                            kind: kind,
                            compact: true,
                            onDevice: onDevice,
                          )
                        : const SizedBox.shrink(),
                  ),
                ],
              ],
            ),
          ],
        ],
      );
    } else {
      final stats = [
        for (final (headline, value) in values)
          _Stat(
            headline: headline,
            value: value,
            device: name,
            hue: h,
            kind: kind,
            onDevice: onDevice,
          ),
      ];
      body = spread
          ? Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                for (var i = 0; i < stats.length; i++) ...[
                  if (i > 0) const SizedBox(width: 24),
                  Expanded(child: stats[i]),
                ],
              ],
            )
          : Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                for (var i = 0; i < stats.length; i++) ...[
                  if (i > 0) const SizedBox(height: 14),
                  stats[i],
                ],
              ],
            );
    }

    return CKindCard(
      kind: kind,
      label: processorLabel(group),
      title: name,
      compact: !wide,
      child: body,
    );
  }
}

/// A card's head line, in capitals: the kind and how many cores or compute
/// units it has (`GPU · 16 CORES`, `GPU · 30 SM`); for a device nothing
/// linked to a processor, the kind and the backend whose view it is.
String processorLabel(ProcessorMeasures group) {
  final kind = group.kind.isEmpty ? 'DEVICE' : group.kind.toUpperCase();
  if (!group.linked) {
    final backend = group.devices.isEmpty ? '' : group.devices.first.backend;
    return backend.isEmpty ? kind : '$kind · ${backend.toUpperCase()}';
  }
  final f = group.processor!.facts;
  if (group.kind == 'cpu') {
    final n = f.cpu.cores;
    return n > 0 ? '$kind · ${countOf(n, 'core').toUpperCase()}' : kind;
  }
  final n = f.computeUnits;
  if (n <= 0) return kind;
  final unit = f.computeUnit.isEmpty ? 'unit' : f.computeUnit;
  // "16 cores", but "30 SM": an acronym takes no plural in capitals.
  final words =
      unit == unit.toUpperCase() || n == 1 ? '$n $unit' : '$n ${unit}s';
  return '$kind · ${words.toUpperCase()}';
}

/// What a card calls a headline: the catalog's own measures by their short
/// names, any other by its title.
String headlineLabel(Headline h, {bool short = false}) => switch (h.id) {
      'peak-float' => short ? 'Float' : 'Floating point',
      'peak-integer' => 'Integer',
      'peak-memory-bandwidth' => short ? 'Memory' : 'Memory bandwidth',
      _ => h.title,
    };

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

/// One flagship number: what it is, the number in the card's ink with its
/// unit in the kind's accent, what it was reached at, and a link to the
/// backend that reached it.  Tapping the number shows every backend's best.
class _Stat extends StatelessWidget {
  const _Stat({
    required this.headline,
    required this.value,
    required this.device,
    required this.hue,
    required this.kind,
    this.compact = false,
    this.onDevice,
  });

  final Headline headline;
  final HeadlineValue? value;
  final String device;
  final KindHue hue;
  final String kind;

  /// The phone's column: label, number, unit and note each on a line.
  final bool compact;
  final ValueChanged<DeviceRun>? onDevice;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = hue;
    final v = value;
    final f = v == null ? null : formatValue(v.best.value, v.best.units);
    // Words that only restate the headline -- "Global memory bandwidth"
    // under "Memory bandwidth" -- say nothing the label does not.
    final full = v == null ? '' : _words(v.best);
    final words = full.toLowerCase().contains(
            headlineLabel(headline).toLowerCase())
        ? ''
        : full;
    final label = headlineLabel(headline, short: compact);
    final open = v == null
        ? null
        : () => showHeadlineDialog(context,
            device: device,
            value: v,
            label: headlineLabel(headline),
            kind: kind);
    final via = v == null
        ? null
        : _ViaLink(
            label: 'via ${v.best.backend}',
            hue: h,
            size: compact ? 12 : 13.5,
            onTap: onDevice == null ? null : () => onDevice!(v.best.device),
          );
    final noteStyle =
        t.small.copyWith(fontSize: compact ? 12 : 13.5, color: h.tintSub);

    if (compact) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          _StatTap(
            hue: h,
            onTap: open,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(label,
                    style: t.small.copyWith(
                        fontWeight: FontWeight.w600, color: h.tintSub)),
                FittedBox(
                  fit: BoxFit.scaleDown,
                  alignment: Alignment.centerLeft,
                  child: Text(f?.value ?? '—',
                      maxLines: 1,
                      style: t.figure(30,
                          color: v == null ? h.tintSub : h.tintInk)),
                ),
                if (f != null && f.unit.isNotEmpty)
                  Text(f.unit,
                      style: t.small.copyWith(
                          fontWeight: FontWeight.w700, color: h.accent)),
              ],
            ),
          ),
          if (words.isNotEmpty) Text(words, style: noteStyle),
          via ?? Text('Not measured', style: noteStyle),
        ],
      );
    }

    return Container(
      padding: const EdgeInsets.only(top: 13),
      decoration: BoxDecoration(
        border: Border(top: BorderSide(color: h.rule)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          _StatTap(
            hue: h,
            onTap: open,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(label,
                    style: t.small.copyWith(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: h.tintSub)),
                const SizedBox(height: 1),
                CValue(
                  value: f?.value ?? '—',
                  unit: f?.unit ?? '',
                  size: 44,
                  color: v == null ? h.tintSub : h.tintInk,
                  unitColor: h.accent,
                ),
              ],
            ),
          ),
          const SizedBox(height: 1),
          ConstrainedBox(
            constraints: const BoxConstraints(minHeight: 22),
            child: via == null
                ? Text('Not measured', style: noteStyle)
                : Wrap(
                    spacing: 6,
                    runSpacing: 2,
                    crossAxisAlignment: WrapCrossAlignment.center,
                    children: [
                      if (words.isNotEmpty) Text('$words ·', style: noteStyle),
                      via,
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}

/// A flagship number as a control: tapped, it shows every backend's best.
/// Hover lays a soft wash of the card's rule behind it, outdented so the
/// number stays in line with the label above.
class _StatTap extends StatelessWidget {
  const _StatTap({required this.hue, required this.onTap, required this.child});

  final KindHue hue;
  final VoidCallback? onTap;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    if (onTap == null) return child;
    return Tooltip(
      message: "Each backend's best",
      excludeFromSemantics: true,
      child: CTap(
        onTap: onTap,
        focusRadius: BorderRadius.circular(CP.rHighlight),
        builder: (context, hovered, pressed) => Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned(
              left: -8,
              right: -8,
              top: -3,
              bottom: -3,
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: hovered || pressed ? hue.rule : Colors.transparent,
                  borderRadius: BorderRadius.circular(CP.rHighlight),
                ),
              ),
            ),
            child,
          ],
        ),
      ),
    );
  }
}

/// "via Metal": opens the reading's device in its backend's tab.
class _ViaLink extends StatelessWidget {
  const _ViaLink({
    required this.label,
    required this.hue,
    required this.size,
    this.onTap,
  });

  final String label;
  final KindHue hue;
  final double size;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onTap,
      focusRadius: BorderRadius.circular(6),
      builder: (context, hovered, pressed) {
        final c = hovered || pressed ? hue.tintInk : hue.accent;
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 2),
          child: Container(
            // An underline set off from the text, as the design draws it.
            padding: const EdgeInsets.only(bottom: 1),
            decoration: BoxDecoration(
              border: Border(bottom: BorderSide(color: c, width: 1.2)),
            ),
            child: Text(label,
                style: t.small.copyWith(
                    fontSize: size,
                    fontWeight: FontWeight.w700,
                    height: 1.25,
                    color: c)),
          ),
        );
      },
    );
  }
}

/// A reading's data type and qualifiers, or what its family measures where
/// it names neither.
String _words(MeasuredReading r) {
  final words = describeReading(r);
  return words.isEmpty ? r.test.title : words.join(' · ');
}

/// Every backend that ran tests, as a way into its tab: its name, how many
/// tests on how many devices, and the kinds of those devices.  Tiles in a
/// grid on a wide window, a list on a phone.
class BackendTiles extends StatelessWidget {
  const BackendTiles({super.key, required this.document, required this.onOpen});

  final RunDocument document;

  /// Opens a backend's tab, by its name.
  final ValueChanged<String> onOpen;

  @override
  Widget build(BuildContext context) {
    final order = <String>[];
    final tests = <String, int>{};
    final devices = <String, int>{};
    final kinds = <String, List<String>>{};
    for (final r in document.runs) {
      if (!order.contains(r.backend)) order.add(r.backend);
      tests[r.backend] = (tests[r.backend] ?? 0) + r.tests.length;
      devices[r.backend] = (devices[r.backend] ?? 0) + 1;
      final k = deviceKindOf(document, r);
      final list = kinds.putIfAbsent(r.backend, () => []);
      if (k.isNotEmpty && !list.contains(k)) list.add(k);
    }
    final tiles = [
      for (final b in order)
        (
          name: b,
          line:
              '${countOf(tests[b]!, 'test')} on ${countOf(devices[b]!, 'device')}',
          kinds: kinds[b]!,
        ),
    ];

    if (!cIsWide(context)) {
      final t = CP.of(context);
      return CPanel(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            for (var i = 0; i < tiles.length; i++)
              CRow(
                onTap: () => onOpen(tiles[i].name),
                rule: i < tiles.length - 1,
                padding: const EdgeInsets.fromLTRB(18, 11, 16, 11),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(minHeight: 42),
                  child: Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(tiles[i].name,
                                style: t.title.copyWith(
                                    fontSize: 16, letterSpacing: -0.16)),
                            Text(tiles[i].line,
                                style: t.small
                                    .copyWith(fontSize: 12.5, color: t.muted)),
                          ],
                        ),
                      ),
                      const SizedBox(width: 12),
                      Wrap(
                        spacing: 4,
                        runSpacing: 4,
                        children: [
                          for (final k in tiles[i].kinds)
                            CKindLabel(k, small: true),
                        ],
                      ),
                      const SizedBox(width: 12),
                      CGlyph(CGlyphs.forward,
                          size: 18, color: t.muted, stroke: 2.2),
                    ],
                  ),
                ),
              ),
          ],
        ),
      );
    }

    return LayoutBuilder(builder: (context, c) {
      const minWidth = 200.0, gap = 12.0;
      final columns =
          math.max(1, ((c.maxWidth + gap) / (minWidth + gap)).floor());
      final width = (c.maxWidth - gap * (columns - 1)) / columns;
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (var start = 0; start < tiles.length; start += columns) ...[
            if (start > 0) const SizedBox(height: gap),
            IntrinsicHeight(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (var i = start;
                      i < math.min(tiles.length, start + columns);
                      i++) ...[
                    if (i > start) const SizedBox(width: gap),
                    SizedBox(
                      width: width,
                      child: _BackendTile(
                        name: tiles[i].name,
                        line: tiles[i].line,
                        kinds: tiles[i].kinds,
                        onTap: () => onOpen(tiles[i].name),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ],
      );
    });
  }
}

class _BackendTile extends StatelessWidget {
  const _BackendTile({
    required this.name,
    required this.line,
    required this.kinds,
    required this.onTap,
  });

  final String name;
  final String line;
  final List<String> kinds;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onTap,
      focusRadius: BorderRadius.circular(22),
      builder: (context, hovered, pressed) => Container(
        constraints: const BoxConstraints(minHeight: 118),
        padding: const EdgeInsets.fromLTRB(18, 16, 16, 16),
        decoration: BoxDecoration(
          color: pressed
              ? t.surface2
              : hovered
                  ? hoverWash(t, t.surface)
                  : t.surface,
          border: Border.all(color: t.line),
          borderRadius: BorderRadius.circular(22),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(name,
                          style: t.title
                              .copyWith(fontSize: 17, letterSpacing: -0.34)),
                    ),
                    const SizedBox(width: 8),
                    CGlyph(CGlyphs.forward,
                        size: 18, color: t.muted, stroke: 2.2),
                  ],
                ),
                const SizedBox(height: 4),
                Text(line, style: t.caption),
              ],
            ),
            if (kinds.isNotEmpty) ...[
              const SizedBox(height: 12),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [for (final k in kinds) CKindLabel(k)],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// The machine the run was on, and what it was doing: its model and OS,
/// the power and thermal state, which XpuPeak measured it and how long it
/// took, and the details only a person knows (`annotations`).
class SystemPanel extends StatelessWidget {
  const SystemPanel({super.key, required this.meta});

  final RunMeta meta;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
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
      ('Model name', other(s.name)),
      ('Model', other(s.model)),
      ('Board', other(s.board)),
      ('Maker', other(s.manufacturer)),
      ('OS', os),
      ('Architecture', s.arch),
      ('Memory', s.memoryBytes > 0 ? formatBytes(s.memoryBytes) : ''),
      (
        'Power',
        [
          switch (c.powerSource) {
            'ac' => 'AC',
            'battery' => 'Battery',
            final p => p,
          },
          if (c.lowPowerMode == true) 'low-power mode',
        ].where((v) => v.isNotEmpty).join(', ')
      ),
      ('Power profile', c.powerProfile),
      ('CPU governor', c.cpuGovernor),
      ('Thermal state', thermal),
      // Named by the file's own generator: "XpuPeak version".
      (
        generator.name.isEmpty ? 'Version' : '${generator.name} version',
        generator.version
      ),
      (
        'Duration',
        meta.durationSeconds > 0
            ? formatDuration(Duration(
                milliseconds: (meta.durationSeconds * 1000).round()))
            : ''
      ),
      // What only a person knows, under the keys they were saved with.
      for (final e in meta.annotations.entries)
        if (e.key != 'system.name' && e.value.trim().isNotEmpty)
          (_keyWords(e.key), e.value),
    ].where((e) => e.$2.isNotEmpty).toList();
    final tags = [
      // Only what the file says: a file without a form factor is read as a
      // desktop, but showing one would claim it.
      if (s.formFactor.isNotEmpty) CTag(text: s.formFactor, upper: false),
      if (s.formFactorDetail.isNotEmpty)
        CTag(text: s.formFactorDetail, upper: false),
      if (s.isVirtual) const CTag(text: 'virtual', upper: false),
      if (s.translated) const CTag(text: 'translated', upper: false),
      if (meta.cancelled)
        CTag(text: 'Cancelled', upper: false, color: t.warnInk, fill: t.warnTint),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const CSectionHeader(title: 'System'),
        SizedBox(height: wide ? 14 : 12),
        CPanel(
          padding: wide
              ? const EdgeInsets.fromLTRB(24, 20, 24, 22)
              : const EdgeInsets.fromLTRB(20, 16, 20, 18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Wrap(
                spacing: 10,
                runSpacing: 8,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  Text(title.isEmpty ? 'Unnamed system' : title,
                      style: t.title.copyWith(fontSize: wide ? 17 : 16)),
                  ...tags,
                ],
              ),
              if (entries.isNotEmpty) ...[
                SizedBox(height: wide ? 16 : 14),
                FactGrid(entries: entries, compact: !wide),
              ],
            ],
          ),
        ),
      ],
    );
  }

  /// An annotation's key as a label: `cooling.fan` → "Cooling fan",
  /// `gpu.board` → "GPU board".
  static String _keyWords(String key) {
    const acronyms = {'cpu', 'gpu', 'npu', 'os', 'ram', 'bios', 'ssd'};
    final words = [
      for (final w in key.split(RegExp(r'[._\s]+')))
        if (w.isNotEmpty)
          acronyms.contains(w.toLowerCase()) ? w.toUpperCase() : w,
    ];
    if (words.isEmpty) return key;
    final s = words.join(' ');
    return s[0].toUpperCase() + s.substring(1);
  }
}

/// Where a flagship number came from, and what every other backend reached
/// on the same device.  Opened without a transition, like every dialog on
/// the results screen (app/AGENTS.md).
Future<void> showHeadlineDialog(
  BuildContext context, {
  required String device,
  required HeadlineValue value,
  String? label,
  String kind = '',
}) {
  return showGeneralDialog<void>(
    context: context,
    barrierDismissible: true,
    barrierLabel: MaterialLocalizations.of(context).modalBarrierDismissLabel,
    barrierColor: Theme.of(context).dialogTheme.barrierColor ??
        Colors.black.withValues(alpha: 0.55),
    transitionDuration: Duration.zero,
    pageBuilder: (context, _, _) => _HeadlineDialog(
      device: device,
      value: value,
      label: label ?? headlineLabel(value.headline),
      kind: kind,
    ),
  );
}

class _HeadlineDialog extends StatelessWidget {
  const _HeadlineDialog({
    required this.device,
    required this.value,
    required this.label,
    required this.kind,
  });

  final String device;
  final HeadlineValue value;
  final String label;
  final String kind;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = t.kind(kind);
    final best = value.best;
    final f = formatValue(best.value, best.units);
    final ranked = value.ranked;
    return CDialog(
      title: '$label · $device',
      actions: [
        CButton(label: 'Close', onPressed: () => Navigator.pop(context)),
      ],
      child: ConstrainedBox(
        constraints:
            BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.6),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              CValue(value: f.value, unit: f.unit, size: 34, unitColor: h.accent),
              const SizedBox(height: 4),
              Text(_words(best), style: t.body.copyWith(fontWeight: FontWeight.w600)),
              const SizedBox(height: 10),
              Text(
                'Measured through ${best.backend} on ${best.device.device}: '
                '${best.test.displayTitle}, reading '
                '"${best.metric.displayLabel}"'
                '${best.revision > 1 ? ' (method revision ${best.revision})' : ''}.',
                style: t.prose,
              ),
              if (ranked.length > 1) ...[
                const SizedBox(height: 18),
                Text("EACH BACKEND'S BEST", style: t.caps),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  decoration: BoxDecoration(
                    color: t.well,
                    borderRadius: BorderRadius.circular(CP.rRow),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      for (final r in ranked)
                        Padding(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 8),
                          child: Row(
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(r.backend,
                                        style: t.label.copyWith(fontSize: 13.5)),
                                    Text(_words(r),
                                        style: t.small.copyWith(color: t.muted)),
                                  ],
                                ),
                              ),
                              if (identical(r, best)) ...[
                                const SizedBox(width: 8),
                                _Badge(hue: h),
                              ],
                              const SizedBox(width: 10),
                              Builder(builder: (context) {
                                final g = formatValue(r.value, r.units);
                                return CValue(
                                    value: g.value, unit: g.unit, small: true);
                              }),
                            ],
                          ),
                        ),
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 14),
              Text(
                'The best reading on this device over every backend that '
                'reached it, as measured: nothing is scored or derived.',
                style: t.smallMuted,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// "Best", on the kind's accent.
class _Badge extends StatelessWidget {
  const _Badge({required this.hue});

  final KindHue hue;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      height: 22,
      padding: const EdgeInsets.symmetric(horizontal: 9),
      decoration: BoxDecoration(
        color: hue.accent,
        borderRadius: BorderRadius.circular(CP.rPill),
      ),
      child: Center(
        widthFactor: 1,
        child: Text('Best',
            style: t.small.copyWith(
                fontSize: 11.5,
                fontWeight: FontWeight.w700,
                height: 1,
                color: hue.onAccent)),
      ),
    );
  }
}
