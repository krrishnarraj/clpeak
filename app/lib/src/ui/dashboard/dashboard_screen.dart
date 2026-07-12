import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/catalog.dart';
import '../../model/run_config.dart';
import '../../services/benchmark_service.dart';
import '../../theme/clpeak_theme.dart';
import '../common/format.dart';
import '../run_config/run_config_screen.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final catalog = service.catalog;
    final usable = catalog.usable;

    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text('clpeak',
                    style: Theme.of(context)
                        .textTheme
                        .headlineMedium
                        ?.copyWith(fontWeight: FontWeight.w700)),
                const SizedBox(width: 10),
                Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text('v${service.version}',
                      style: Theme.of(context)
                          .textTheme
                          .bodySmall
                          ?.copyWith(
                              color:
                                  Theme.of(context).colorScheme.outline)),
                ),
              ],
            ),
            Text('Cross-API compute benchmark',
                style: Theme.of(context)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(color: Theme.of(context).colorScheme.outline)),
            const SizedBox(height: 20),
            if (usable.isEmpty)
              const _EmptyState()
            else ...[
              _RunLauncher(service: service),
              const SizedBox(height: 24),
              Text('This system',
                  style: Theme.of(context).textTheme.titleMedium),
              const SizedBox(height: 12),
              for (final backend in usable) ...[
                _BackendCard(backend: backend),
                const SizedBox(height: 12),
              ],
            ],
          ],
        ),
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          children: [
            Icon(Icons.search_off,
                size: 48, color: Theme.of(context).colorScheme.outline),
            const SizedBox(height: 12),
            const Text('No compute devices found'),
            const SizedBox(height: 4),
            Text(
              'No backend could enumerate a device on this system.',
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Theme.of(context).colorScheme.outline),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}

class _RunLauncher extends StatelessWidget {
  const _RunLauncher({required this.service});

  final BenchmarkService service;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.bolt, color: scheme.primary),
                const SizedBox(width: 8),
                Text('Run benchmark',
                    style: Theme.of(context).textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              'Full uses the standard time budgets; Quick trims them for a '
              'fast overview. Custom picks devices, categories and budgets.',
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: scheme.outline),
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 12,
              runSpacing: 8,
              children: [
                FilledButton.icon(
                  onPressed: () => service.start(preset: RunPreset.full),
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Full run'),
                ),
                FilledButton.tonalIcon(
                  onPressed: () => service.start(preset: RunPreset.quick),
                  icon: const Icon(Icons.fast_forward),
                  label: const Text('Quick run'),
                ),
                OutlinedButton.icon(
                  onPressed: () => Navigator.of(context).push(
                    MaterialPageRoute(
                        builder: (_) => const RunConfigScreen()),
                  ),
                  icon: const Icon(Icons.tune),
                  label: const Text('Custom…'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _BackendCard extends StatelessWidget {
  const _BackendCard({required this.backend});

  final CatalogBackend backend;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(ClpeakTheme.backendIcon(backend.name),
                    size: 20, color: scheme.primary),
                const SizedBox(width: 8),
                Text(backend.name,
                    style: Theme.of(context)
                        .textTheme
                        .titleSmall
                        ?.copyWith(fontWeight: FontWeight.w600)),
                const Spacer(),
                Text(
                  backend.deviceCount == 1
                      ? '1 device'
                      : '${backend.deviceCount} devices',
                  style: Theme.of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(color: scheme.outline),
                ),
              ],
            ),
            for (final platform in backend.platforms)
              for (final device in platform.devices)
                Padding(
                  padding: const EdgeInsets.only(top: 10),
                  child: _DeviceRow(device: device),
                ),
          ],
        ),
      ),
    );
  }
}

class _DeviceRow extends StatelessWidget {
  const _DeviceRow({required this.device});

  final CatalogDevice device;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final details = <String>[
      if (device.type.isNotEmpty) device.type,
      if (device.computeUnits > 0) '${device.computeUnits} CUs',
      if (device.clockMHz > 0) '${device.clockMHz} MHz',
      if (device.globalMemBytes > 0) formatBytes(device.globalMemBytes),
      if (device.fp16) 'fp16',
      if (device.fp64) 'fp64',
    ];
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          margin: const EdgeInsets.only(top: 6),
          width: 6,
          height: 6,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: scheme.primary,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(device.name,
                  style: Theme.of(context).textTheme.bodyMedium),
              if (details.isNotEmpty)
                Text(details.join(' · '),
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: scheme.outline)),
            ],
          ),
        ),
      ],
    );
  }
}
