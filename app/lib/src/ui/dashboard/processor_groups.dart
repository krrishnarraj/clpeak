/// The device catalog as the dashboard and the custom run show it: a card
/// per processor, holding every listed device that is a view of it.
library;

import 'dart:math' as math;

import 'package:flutter/widgets.dart';

import '../../model/catalog.dart';
import '../../model/hardware.dart';
import '../../model/run_config.dart';
import '../common/format.dart';

/// Cards side by side, as many to a row as fit at [minWidth] and the cards
/// of a row as tall as its tallest; one above the other below [twoUp].
class ProcessorGrid extends StatelessWidget {
  const ProcessorGrid({
    super.key,
    required this.children,
    this.minWidth = 280,
    this.gap = 16,
    this.twoUp = 600,
  });

  final List<Widget> children;
  final double minWidth;
  final double gap;

  /// The narrowest width that sets cards side by side.
  final double twoUp;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      if (c.maxWidth < twoUp || children.length < 2) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            for (var i = 0; i < children.length; i++) ...[
              if (i > 0) SizedBox(height: gap * 0.75),
              children[i],
            ],
          ],
        );
      }
      final columns = math.max(
          1,
          math.min(children.length,
              ((c.maxWidth + gap) / (minWidth + gap)).floor()));
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (var row = 0; row * columns < children.length; row++) ...[
            if (row > 0) SizedBox(height: gap),
            IntrinsicHeight(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (var col = 0; col < columns; col++) ...[
                    if (col > 0) SizedBox(width: gap),
                    Expanded(
                      child: row * columns + col < children.length
                          ? children[row * columns + col]
                          : const SizedBox.shrink(),
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

/// One listed device with the backend and platform that list it.
class GroupDevice {
  const GroupDevice(this.backend, this.platform, this.device);

  final CatalogBackend backend;
  final CatalogPlatform platform;
  final CatalogDevice device;

  /// How [RunConfig] names it within its backend.
  DeviceRef get ref =>
      (platformIndex: platform.index, deviceIndex: device.index);
}

/// One card: a processor and its devices in run order (by backend, then
/// index).  From a catalog without the hardware (an older native library),
/// or for devices nothing linked, the devices of one kind instead, with no
/// processor name: which of them share a processor is not known then.
class ProcessorGroup {
  const ProcessorGroup({
    required this.kind,
    required this.label,
    this.name,
    this.facts = '',
    this.devices = const [],
  });

  /// `cpu`, `gpu`, `npu`, …: the card's hue.
  final String kind;

  /// The card's capitals line: `GPU · 16 CORES`, or `GPU · 3 DEVICES` for
  /// the devices of a kind.
  final String label;

  /// The processor's name; null for the devices of a kind.
  final String? name;

  /// The rest of what the catalog knows of the processor, in one line.
  final String facts;

  final List<GroupDevice> devices;
}

/// A device's kind from the type its backend reports (`GPU`, `Integrated
/// GPU`, `NPU`, …): `cpu`, `gpu`, `npu`, `accelerator`, or `other`.
String kindOfDeviceType(String type) {
  final t = type.toLowerCase();
  if (t.contains('npu')) return 'npu';
  if (t.contains('gpu')) return 'gpu';
  if (t.contains('cpu')) return 'cpu';
  if (t.contains('accelerator')) return 'accelerator';
  return 'other';
}

/// The kind of a listed device: its processor's, else its type's.
String kindOfDevice(BackendCatalog catalog, CatalogDevice device) {
  final kind = catalog.processorById(device.processor)?.kind ?? '';
  return kind.isNotEmpty ? kind : kindOfDeviceType(device.type);
}

/// The catalog's usable devices, a card per processor in the order a run
/// numbers them, then the devices nothing linked, by kind.  With [empty],
/// a processor no listed device reaches gets a card too (the dashboard says
/// so; the custom run has nothing to offer on it).
List<ProcessorGroup> groupDevices(BackendCatalog catalog,
    {bool empty = false}) {
  final all = [
    for (final b in catalog.usable)
      for (final p in b.platforms)
        for (final d in p.devices) GroupDevice(b, p, d),
  ];
  final groups = <ProcessorGroup>[];
  final claimed = <GroupDevice>{};
  for (final processor in catalog.processors) {
    final mine = [
      for (final g in all)
        if (g.device.processor == processor.id) g,
    ];
    if (mine.isEmpty && !empty) continue;
    claimed.addAll(mine);
    final kind = processor.kind.isNotEmpty
        ? processor.kind
        : (mine.isEmpty ? 'other' : kindOfDeviceType(mine.first.device.type));
    groups.add(ProcessorGroup(
      kind: kind,
      label: processorLabel(processor, kind),
      name: processor.name.isEmpty ? null : processor.name,
      facts: processorFacts(processor, catalog.packageById(processor.package)),
      devices: mine,
    ));
  }
  final byKind = <String, List<GroupDevice>>{};
  for (final g in all) {
    if (claimed.contains(g)) continue;
    byKind.putIfAbsent(kindOfDeviceType(g.device.type), () => []).add(g);
  }
  for (final MapEntry(key: kind, value: devices) in byKind.entries) {
    groups.add(ProcessorGroup(
      kind: kind,
      label: '${_kindWord(kind)} · ${devices.length} '
          '${devices.length == 1 ? 'DEVICE' : 'DEVICES'}',
      devices: devices,
    ));
  }
  return groups;
}

String _kindWord(String kind) =>
    kind == 'other' ? 'OTHER' : kind.toUpperCase();

/// `GPU · 16 CORES`: the kind and how many cores (or compute units) the
/// processor has, where the catalog says.
String processorLabel(ProcessorInfo p, String kind) {
  final f = p.facts;
  String? count;
  if (kind == 'cpu') {
    final n = f.cpu.cores;
    if (n > 0) count = '$n ${n == 1 ? 'CORE' : 'CORES'}';
  } else if (f.computeUnits > 0) {
    final n = f.computeUnits;
    final unit = f.computeUnit.isEmpty ? 'unit' : f.computeUnit;
    // An abbreviation (SM, CU) takes no plural.
    final word = n == 1 || unit.toUpperCase() == unit ? unit : '${unit}s';
    count = '$n ${word.toUpperCase()}';
  }
  return count == null ? _kindWord(kind) : '${_kindWord(kind)} · $count';
}

/// The rest of a processor's facts in one line: its clusters, clock, ISA
/// and matrix engines for a CPU; architecture, clock and memory for the
/// others; and the package it sits in where that names something more.
String processorFacts(ProcessorInfo p, PackageInfo? package) {
  final f = p.facts;
  String clock(int mhz) => '${(mhz / 1000).toStringAsFixed(1)} GHz';
  final parts = <String>[];
  if (p.kind == 'cpu') {
    final cpu = f.cpu;
    if (cpu.threads > 0 && cpu.threads != cpu.cores) {
      parts.add('${cpu.threads} threads');
    }
    final clusters = [
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
    if (clusters.length > 1) parts.add(clusters.join(' + '));
    final maxMhz =
        cpu.clusters.fold(0, (m, c) => c.maxMhz > m ? c.maxMhz : m);
    if (maxMhz > 0) parts.add(clock(maxMhz));
    if (cpu.sockets > 1) parts.add('${cpu.sockets} sockets');
    if (cpu.isa.isNotEmpty) parts.add(cpu.isa);
    for (final e in cpu.engines) {
      if (e.name.isNotEmpty) parts.add(e.name);
    }
  } else {
    if (f.arch.isNotEmpty) parts.add(f.arch);
    if (f.clockMhz > 0) parts.add(clock(f.clockMhz));
    if (f.integrated == true) {
      parts.add('integrated');
    } else if (f.memoryBytes > 0) {
      parts.add(formatBytes(f.memoryBytes));
    }
  }
  if (package != null &&
      package.name.isNotEmpty &&
      package.name != p.name) {
    parts.add('in ${package.name}');
  }
  return parts.join(' · ');
}

/// What tells one of a card's devices apart, after its backend: its own
/// name where that is not the card's, its type where that says more than
/// the kind, its OpenCL platform where there are several, where an ONNX
/// plugin came from, and what its API reports of it.
String deviceDetail(GroupDevice g, ProcessorGroup group,
    {bool specs = true}) {
  final d = g.device;
  final type = d.type.trim();
  return [
    if (d.name.isNotEmpty && d.name != group.name) d.name,
    if (type.isNotEmpty && type.toLowerCase() != group.kind) type,
    if (g.backend.platforms.length > 1) 'platform ${g.platform.name}',
    if (d.origin.isNotEmpty) d.origin,
    if (specs && d.computeUnits > 0) '${d.computeUnits} CU',
    if (specs && d.clockMHz > 0) '${d.clockMHz} MHz',
    if (specs && d.globalMemBytes > 0) formatBytes(d.globalMemBytes),
  ].join(' · ');
}
