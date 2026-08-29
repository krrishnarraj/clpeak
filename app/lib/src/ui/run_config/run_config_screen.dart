import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/catalog.dart';
import '../../model/result_model.dart';
import '../../model/run_config.dart';
import '../../services/benchmark_service.dart';
import '../../theme/clpeak_theme.dart';
import '../common/kit.dart';

/// Custom run configuration: devices, categories (chips only — individual
/// tests are intentionally not exposed), and the two time budgets.
class RunConfigScreen extends StatelessWidget {
  const RunConfigScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final service = context.watch<BenchmarkService>();
    final config = service.config;
    final catalog = service.catalog;
    final canRun = config.hasSelection && config.categories.isNotEmpty;

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: 'Custom run',
              subtitle: 'devices · categories · budgets',
              onBack: () => Navigator.of(context).pop(),
            ),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 32),
                children: [
                  const CSection(label: 'Devices'),
                  const SizedBox(height: 10),
                  for (final backend in catalog.usable) ...[
                    _BackendSelector(backend: backend),
                    const SizedBox(height: 10),
                  ],
                  const SizedBox(height: 16),
                  const CSection(label: 'Test categories'),
                  const SizedBox(height: 10),
                  const _CategoryChips(),
                  const SizedBox(height: 26),
                  const CSection(label: 'Time budgets'),
                  const SizedBox(height: 10),
                  Text(
                    'Per-test measurement window. Longer budgets steady the '
                    'numbers; shorter budgets finish faster.',
                    style: t.body,
                  ),
                  const SizedBox(height: 14),
                  CPanel(
                    child: Column(
                      children: [
                        _BudgetSlider(
                          label: 'GPU backends',
                          value: config.maxTimeMs,
                          min: 100,
                          max: 2000,
                          defaultValue: kDefaultMaxTimeMs,
                          onChanged: (v) =>
                              service.updateConfig((c) => c.maxTimeMs = v),
                        ),
                        Container(height: 1, color: t.line),
                        _BudgetSlider(
                          label: 'CPU backend',
                          value: config.maxTimeCpuMs,
                          min: 250,
                          max: 5000,
                          defaultValue: kDefaultMaxTimeCpuMs,
                          onChanged: (v) =>
                              service.updateConfig((c) => c.maxTimeCpuMs = v),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            // Docked so the action never scrolls out of reach, but a
            // full-width block rather than a corner pill: parked next to the
            // tab strip's own bottom rule, a small right-aligned button reads
            // as chrome and gets skipped over.
            Container(
              decoration: BoxDecoration(
                color: t.panel,
                border: Border(top: BorderSide(color: t.line)),
              ),
              padding: const EdgeInsets.fromLTRB(20, 12, 20, 14),
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        canRun
                            ? 'Ready'
                            : 'Select at least one device and category',
                        textAlign: TextAlign.center,
                        style:
                            t.micro.copyWith(color: canRun ? t.dim : t.danger),
                      ),
                      const SizedBox(height: 10),
                      CButton(
                        label: 'Run',
                        icon: Icons.play_arrow,
                        kind: CButtonKind.primary,
                        stretch: true,
                        onPressed: canRun
                            ? () {
                                Navigator.of(context).pop();
                                service.start();
                              }
                            : null,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _BackendSelector extends StatelessWidget {
  const _BackendSelector({required this.backend});

  final CatalogBackend backend;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final service = context.watch<BenchmarkService>();
    final config = service.config;

    final refs = <(DeviceRef, CatalogDevice)>[
      for (final p in backend.platforms)
        for (final d in p.devices)
          ((platformIndex: p.index, deviceIndex: d.index), d)
    ];
    final selectedCount =
        refs.where((r) => config.isDeviceSelected(backend.name, r.$1)).length;
    final allSelected = selectedCount == refs.length && refs.isNotEmpty;

    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          CPanelHead(
            title: backend.name,
            icon: ClpeakTheme.backendIcon(backend.name),
            trailing: CSwitch(
              value: selectedCount > 0,
              onChanged: (on) => service.updateConfig((c) {
                for (final (ref, _) in refs) {
                  c.toggleDevice(backend.name, ref, on);
                }
              }),
            ),
          ),
          if (refs.length > 1 || !allSelected)
            for (var i = 0; i < refs.length; i++)
              CRow(
                rule: i != refs.length - 1,
                onTap: () => service.updateConfig((c) => c.toggleDevice(
                    backend.name,
                    refs[i].$1,
                    !config.isDeviceSelected(backend.name, refs[i].$1))),
                child: Row(
                  children: [
                    CCheckbox(
                      value: config.isDeviceSelected(backend.name, refs[i].$1),
                      onChanged: (on) => service.updateConfig((c) =>
                          c.toggleDevice(backend.name, refs[i].$1, on)),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Wraps: you pick a device by its full name, so
                          // nothing here is allowed to hide the tail of one.
                          Text(refs[i].$2.name, style: t.mono),
                          if (refs[i].$2.type.isNotEmpty) ...[
                            const SizedBox(height: 3),
                            Text(refs[i].$2.type, style: t.monoSmallDim),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
              ),
        ],
      ),
    );
  }
}

class _CategoryChips extends StatelessWidget {
  const _CategoryChips();

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final service = context.watch<BenchmarkService>();
    final config = service.config;
    final selectedCount = config.categories.length;
    final total = BenchCategory.selectable.length;
    final brightness = Theme.of(context).brightness;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final category in BenchCategory.selectable)
              CChip(
                label: category.label,
                selected: config.categories.contains(category),
                color: ClpeakTheme.categoryColor(category,
                    brightness: brightness),
                onTap: () => service.updateConfig((c) =>
                    c.categories.contains(category)
                        ? c.categories.remove(category)
                        : c.categories.add(category)),
              ),
          ],
        ),
        const SizedBox(height: 10),
        Text(
          selectedCount == 0
              ? 'No categories selected — select at least one to run'
              : selectedCount == total
                  ? 'All $total categories selected'
                  : '$selectedCount of $total categories selected',
          style: t.micro.copyWith(
              color: selectedCount == 0 ? t.danger : t.dim),
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
    final t = CP.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
      child: Row(
        children: [
          SizedBox(
            width: 108,
            child: Text(label.toUpperCase(), style: t.micro),
          ),
          Expanded(
            // Material's Slider carries the drag logic; the theme strips it
            // back to a flat rule and a square handle.
            child: SliderTheme(
              data: SliderThemeData(
                trackHeight: 3,
                activeTrackColor: t.text,
                inactiveTrackColor: t.isDark ? t.hover : t.line,
                thumbColor: t.text,
                overlayColor: Colors.transparent,
                trackShape: const RectangularSliderTrackShape(),
                thumbShape: const _SquareThumb(),
                overlayShape: SliderComponentShape.noOverlay,
                showValueIndicator: ShowValueIndicator.never,
                padding: EdgeInsets.zero,
              ),
              child: Slider(
                value: value.toDouble().clamp(min.toDouble(), max.toDouble()),
                min: min.toDouble(),
                max: max.toDouble(),
                divisions: (max - min) ~/ 50,
                onChanged: (v) => onChanged((v / 50).round() * 50),
              ),
            ),
          ),
          const SizedBox(width: 14),
          SizedBox(
            width: 72,
            child: Text(
              '$value ms${value == defaultValue ? '' : ' *'}',
              style: t.monoSmall,
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}

/// A square handle in place of Material's circle.
class _SquareThumb extends SliderComponentShape {
  const _SquareThumb();

  static const _size = Size(6, 16);

  @override
  Size getPreferredSize(bool isEnabled, bool isDiscrete) => _size;

  @override
  void paint(
    PaintingContext context,
    Offset center, {
    required Animation<double> activationAnimation,
    required Animation<double> enableAnimation,
    required bool isDiscrete,
    required TextPainter labelPainter,
    required RenderBox parentBox,
    required SliderThemeData sliderTheme,
    required TextDirection textDirection,
    required double value,
    required double textScaleFactor,
    required Size sizeWithOverflow,
  }) {
    context.canvas.drawRect(
      Rect.fromCenter(
          center: center, width: _size.width, height: _size.height),
      Paint()..color = sliderTheme.thumbColor ?? const Color(0xFFFFFFFF),
    );
  }
}
