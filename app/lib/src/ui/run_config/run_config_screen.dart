import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/catalog.dart';
import '../../model/result_entry.dart';
import '../../model/run_config.dart';
import '../../services/benchmark_service.dart';
import '../../theme/clpeak_theme.dart';

/// Custom run configuration: devices, categories (chips only — individual
/// tests are intentionally not exposed), and the two time budgets.
class RunConfigScreen extends StatelessWidget {
  const RunConfigScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final config = service.config;
    final catalog = service.catalog;
    final canRun = config.hasSelection && config.categories.isNotEmpty;

    return Scaffold(
      appBar: AppBar(title: const Text('Custom run')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text('Devices', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            for (final backend in catalog.usable) ...[
              _BackendSelector(backend: backend),
              const SizedBox(height: 12),
            ],
            const SizedBox(height: 8),
            Text('Test categories',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            const _CategoryChips(),
            const SizedBox(height: 20),
            Text('Time budgets',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 4),
            Text(
              'Per-test measurement window. Longer budgets steady the '
              'numbers; shorter budgets finish faster.',
              style: Theme.of(context)
                  .textTheme
                  .bodySmall
                  ?.copyWith(color: Theme.of(context).colorScheme.outline),
            ),
            const SizedBox(height: 8),
            _BudgetSlider(
              label: 'GPU backends',
              value: config.maxTimeMs,
              min: 100,
              max: 2000,
              defaultValue: kDefaultMaxTimeMs,
              onChanged: (v) =>
                  service.updateConfig((c) => c.maxTimeMs = v),
            ),
            _BudgetSlider(
              label: 'CPU backend',
              value: config.maxTimeCpuMs,
              min: 250,
              max: 5000,
              defaultValue: kDefaultMaxTimeCpuMs,
              onChanged: (v) =>
                  service.updateConfig((c) => c.maxTimeCpuMs = v),
            ),
            const SizedBox(height: 80),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: canRun
            ? () {
                Navigator.of(context).pop();
                service.start();
              }
            : null,
        backgroundColor: canRun ? null : Theme.of(context).disabledColor,
        icon: const Icon(Icons.play_arrow),
        label: const Text('Run'),
      ),
    );
  }
}

class _BackendSelector extends StatelessWidget {
  const _BackendSelector({required this.backend});

  final CatalogBackend backend;

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final config = service.config;
    final scheme = Theme.of(context).colorScheme;

    final refs = <(DeviceRef, CatalogDevice)>[
      for (final p in backend.platforms)
        for (final d in p.devices)
          ((platformIndex: p.index, deviceIndex: d.index), d)
    ];
    final selectedCount =
        refs.where((r) => config.isDeviceSelected(backend.name, r.$1)).length;
    final allSelected = selectedCount == refs.length && refs.isNotEmpty;

    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Column(
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
                Switch(
                  value: selectedCount > 0,
                  onChanged: (on) => service.updateConfig((c) {
                    for (final (ref, _) in refs) {
                      c.toggleDevice(backend.name, ref, on);
                    }
                  }),
                ),
              ],
            ),
            if (refs.length > 1 || !allSelected)
              for (final (ref, device) in refs)
                CheckboxListTile(
                  dense: true,
                  contentPadding: const EdgeInsets.only(left: 28),
                  controlAffinity: ListTileControlAffinity.leading,
                  title: Text(device.name,
                      style: Theme.of(context).textTheme.bodyMedium),
                  subtitle: device.type.isEmpty
                      ? null
                      : Text(device.type,
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: scheme.outline)),
                  value: config.isDeviceSelected(backend.name, ref),
                  onChanged: (on) => service.updateConfig(
                      (c) => c.toggleDevice(backend.name, ref, on ?? false)),
                ),
          ],
        ),
      ),
    );
  }
}

class _CategoryChips extends StatelessWidget {
  const _CategoryChips();

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final config = service.config;
    final scheme = Theme.of(context).colorScheme;
    final selectedCount = config.categories.length;
    final total = BenchCategory.selectable.length;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final category in BenchCategory.selectable)
              Builder(builder: (context) {
                final selected = config.categories.contains(category);
                final color = ClpeakTheme.categoryColor(category);
                // Selection is made unmistakable: selected chips are filled
                // with the category tint and lead with a check mark;
                // unselected chips are hollow and muted.
                return FilterChip(
                  avatar: selected
                      ? null // FilterChip swaps in the check mark
                      : Icon(ClpeakTheme.categoryIcon(category),
                          size: 18, color: scheme.outline),
                  showCheckmark: true,
                  checkmarkColor: color,
                  label: Text(category.label),
                  labelStyle: TextStyle(
                    color: selected ? null : scheme.outline,
                    fontWeight:
                        selected ? FontWeight.w600 : FontWeight.w400,
                  ),
                  selected: selected,
                  selectedColor: color.withValues(alpha: 0.22),
                  side: BorderSide(
                    color: selected
                        ? color
                        : scheme.outline.withValues(alpha: 0.4),
                    width: selected ? 1.4 : 1,
                  ),
                  onSelected: (on) => service.updateConfig((c) => on
                      ? c.categories.add(category)
                      : c.categories.remove(category)),
                );
              }),
          ],
        ),
        const SizedBox(height: 8),
        Text(
          selectedCount == 0
              ? 'No categories selected — select at least one to run'
              : selectedCount == total
                  ? 'All $total categories selected'
                  : '$selectedCount of $total categories selected',
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: selectedCount == 0 ? scheme.error : scheme.outline),
        ),
      ],
    );
  }
}

class _BudgetSlider extends StatelessWidget {
  const _BudgetSlider({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.defaultValue,
    required this.onChanged,
  });

  final String label;
  final int value;
  final int min;
  final int max;
  final int defaultValue;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Row(
      children: [
        SizedBox(
          width: 120,
          child: Text(label, style: Theme.of(context).textTheme.bodyMedium),
        ),
        Expanded(
          child: Slider(
            value: value.toDouble().clamp(min.toDouble(), max.toDouble()),
            min: min.toDouble(),
            max: max.toDouble(),
            divisions: (max - min) ~/ 50,
            label: '$value ms',
            onChanged: (v) => onChanged((v / 50).round() * 50),
          ),
        ),
        SizedBox(
          width: 76,
          child: Text(
            '$value ms${value == defaultValue ? '' : ' *'}',
            style: Theme.of(context)
                .textTheme
                .bodySmall
                ?.copyWith(color: scheme.outline),
            textAlign: TextAlign.right,
          ),
        ),
      ],
    );
  }
}
