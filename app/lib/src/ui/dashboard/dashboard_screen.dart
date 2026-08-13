import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/catalog.dart';
import '../../model/run_config.dart';
import '../../services/benchmark_service.dart';
import '../../theme/clpeak_theme.dart';
import '../app.dart';
import '../common/format.dart';
import '../common/kit.dart';
import '../run_config/run_config_screen.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final catalog = service.catalog;
    final usable = catalog.usable;
    final devices = usable.fold<int>(0, (n, b) => n + b.deviceCount);

    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(20, 22, 20, 40),
          children: [
            _Masthead(version: service.version),
            const SizedBox(height: 26),
            if (usable.isEmpty)
              CPanel(
                child: const CEmpty(
                  icon: Icons.search_off,
                  title: 'No compute devices found',
                  detail:
                      'No backend could enumerate a device on this system.',
                ),
              )
            else ...[
              const CSection(label: 'RUN BENCHMARK'),
              const SizedBox(height: 10),
              _RunLauncher(service: service, deviceCount: devices),
              const SizedBox(height: 26),
              CSection(
                label: 'THIS SYSTEM',
                trailing: '${usable.length} backends · $devices devices',
              ),
              const SizedBox(height: 10),
              for (final backend in usable) ...[
                _BackendPanel(backend: backend),
                const SizedBox(height: 10),
              ],
            ],
          ],
        ),
      ),
    );
  }
}

/// Wordmark block — only shown on phones, where there's no sidebar to carry it.
class _Masthead extends StatelessWidget {
  const _Masthead({required this.version});

  final String version;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = MediaQuery.sizeOf(context).width >= 900;

    if (wide) {
      // The sidebar already carries the mark; here just name the view.
      return Text('Benchmark', style: t.wordmark.copyWith(fontSize: 22));
    }

    return Row(
      children: [
        const CAppMark(size: 30),
        const SizedBox(width: 12),
        Expanded(child: Text('clpeak', style: t.wordmark)),
        Text('v$version', style: t.micro.copyWith(letterSpacing: 0.6)),
      ],
    );
  }
}

class _RunLauncher extends StatelessWidget {
  const _RunLauncher({required this.service, required this.deviceCount});

  final BenchmarkService service;
  final int deviceCount;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final deviceLabel = deviceCount == 1
        ? 'the detected device'
        : 'all $deviceCount detected devices';

    return CPanel(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Measures peak compute, bandwidth and latency on $deviceLabel. '
            'Results stream in live and are saved to History. '
            'Custom… narrows the devices, test categories and time budgets.',
            style: t.body,
          ),
          const SizedBox(height: 16),
          Wrap(
            spacing: 10,
            runSpacing: 8,
            children: [
              CButton(
                label: 'Run',
                icon: Icons.play_arrow,
                kind: CButtonKind.primary,
                onPressed: () => service.start(preset: RunPreset.full),
              ),
              CButton(
                label: 'Custom…',
                icon: Icons.tune,
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const RunConfigScreen()),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// One backend, presented as a titled table of its devices.
class _BackendPanel extends StatelessWidget {
  const _BackendPanel({required this.backend});

  final CatalogBackend backend;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final devices = <CatalogDevice>[
      for (final platform in backend.platforms) ...platform.devices,
    ];

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          CPanelHead(
            title: backend.name,
            icon: ClpeakTheme.backendIcon(backend.name),
            trailing: Text(
              backend.deviceCount == 1
                  ? '1 DEVICE'
                  : '${backend.deviceCount} DEVICES',
              style: t.micro,
            ),
          ),
          for (var i = 0; i < devices.length; i++)
            CRow(
              rule: i != devices.length - 1,
              child: _DeviceLine(device: devices[i]),
            ),
        ],
      ),
    );
  }
}

class _DeviceLine extends StatelessWidget {
  const _DeviceLine({required this.device});

  final CatalogDevice device;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final specs = <String>[
      if (device.type.isNotEmpty) device.type,
      if (device.computeUnits > 0) '${device.computeUnits} CU',
      if (device.clockMHz > 0) '${device.clockMHz} MHz',
      if (device.globalMemBytes > 0) formatBytes(device.globalMemBytes),
    ];
    final caps = <String>[
      if (device.fp16) 'fp16',
      if (device.fp64) 'fp64',
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Text(device.name,
                  style: t.mono, maxLines: 1,
                  overflow: TextOverflow.ellipsis),
            ),
            for (final cap in caps) ...[
              const SizedBox(width: 6),
              CTag(text: cap),
            ],
          ],
        ),
        if (specs.isNotEmpty) ...[
          const SizedBox(height: 4),
          Text(specs.join('  ·  '), style: t.monoSmallDim),
        ],
      ],
    );
  }
}
