import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../model/result_model.dart';
import '../../model/run_config.dart';
import '../../services/benchmark_service.dart';
import '../../services/settings_service.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/format.dart';
import '../common/kit.dart';
import '../dashboard/processor_groups.dart';

/// Custom run configuration: devices, categories (chips only — individual
/// tests are intentionally not exposed), and the time budgets.
class RunConfigScreen extends StatelessWidget {
  const RunConfigScreen({super.key});

  /// What its route is named, so the sidebar's Customize can tell the
  /// screen is open already.
  static const routeName = '/custom-run';

  /// The screen's route, as the dashboard and the sidebar push it.
  static Route<void> route() => MaterialPageRoute<void>(
        settings: const RouteSettings(name: routeName),
        builder: (_) => const RunConfigScreen(),
      );

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final service = context.watch<BenchmarkService>();
    final config = service.config;
    final catalog = service.catalog;
    final ready = service.catalogReady;
    final canRun =
        ready && config.hasSelection && config.categories.isNotEmpty;
    final wide = cIsWide(context);
    final side = wide ? 34.0 : 16.0;
    final groups = ready ? groupDevices(catalog) : const <ProcessorGroup>[];
    final total = groups.fold(0, (n, g) => n + g.devices.length);
    final selected = groups.fold(
        0,
        (n, g) =>
            n +
            g.devices
                .where((d) => config.isDeviceSelected(d.backend.name, d.ref))
                .length);

    // A form, so no wider than one on desktop.
    Widget column(Widget child) => Align(
          alignment: Alignment.topLeft,
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 780),
            child: child,
          ),
        );

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            CHeader(
              title: 'Custom run',
              subtitle: 'Devices, test categories and time budgets',
              onBack: () => Navigator.of(context).pop(),
            ),
            Expanded(
              child: ListView(
                padding: EdgeInsets.fromLTRB(
                    side, wide ? 14 : 12, side, wide ? 36 : 28),
                children: [
                  column(CSection(
                    label: 'Devices',
                    trailing: ready && total > 0
                        ? '$selected of $total selected'
                        : null,
                  )),
                  SizedBox(height: wide ? 12 : 10),
                  if (!ready)
                    column(const CEmptyBox(text: 'Detecting devices…'))
                  else ...[
                    column(_BackendToggles(service: service)),
                    const SizedBox(height: 16),
                    column(ProcessorGrid(
                      minWidth: 240,
                      children: [
                        for (final group in groups)
                          _GroupCard(
                              group: group,
                              service: service,
                              compact: !wide),
                      ],
                    )),
                  ],
                  SizedBox(height: wide ? 30 : 24),
                  column(const CSection(label: 'Test categories')),
                  SizedBox(height: wide ? 12 : 10),
                  column(_CategoryChips(service: service)),
                  SizedBox(height: wide ? 30 : 24),
                  column(const CSection(label: 'Time budgets')),
                  SizedBox(height: wide ? 12 : 10),
                  column(Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 6),
                    child: Text(
                      'Per-test measurement window. Longer budgets steady the '
                      'numbers; shorter budgets finish faster. The GPU budget '
                      'is also the longest a GPU is kept busy in one go, which '
                      'keeps the driver from resetting it. The ML budget is the '
                      'longest one ONNX Runtime, LiteRT or Core ML run may keep '
                      'a GPU busy; their size sweeps stop before that.',
                      style: t.prose,
                    ),
                  )),
                  const SizedBox(height: 14),
                  column(CPanel(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        _BudgetRow(
                          label: 'GPU backends',
                          value: config.maxTimeMs,
                          min: 100,
                          max: 2000,
                          defaultValue: kDefaultMaxTimeMs,
                          onChanged: (v) =>
                              service.updateConfig((c) => c.maxTimeMs = v),
                        ),
                        _BudgetRow(
                          label: 'ML on GPU',
                          value: config.maxTimeMlGpuMs,
                          min: 100,
                          max: 2000,
                          defaultValue: kDefaultMaxTimeMlGpuMs,
                          onChanged: (v) => service
                              .updateConfig((c) => c.maxTimeMlGpuMs = v),
                        ),
                        _BudgetRow(
                          label: 'CPU backend',
                          value: config.maxTimeCpuMs,
                          min: 250,
                          max: 5000,
                          defaultValue: kDefaultMaxTimeCpuMs,
                          rule: false,
                          onChanged: (v) =>
                              service.updateConfig((c) => c.maxTimeCpuMs = v),
                        ),
                      ],
                    ),
                  )),
                ],
              ),
            ),
            // Docked so the action never scrolls out of reach, and a
            // full-width pill rather than a small one in a corner, which
            // reads as chrome and gets skipped over.
            Container(
              decoration: BoxDecoration(
                color: t.ground,
                border: Border(top: BorderSide(color: t.line)),
              ),
              padding: EdgeInsets.fromLTRB(side, 12, side, 14),
              child: Align(
                alignment: wide ? Alignment.centerLeft : Alignment.center,
                child: ConstrainedBox(
                  constraints: BoxConstraints(maxWidth: wide ? 780 : 520),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (!canRun) ...[
                        Text(
                          !ready
                              ? 'Detecting devices…'
                              : 'Select at least one device and category',
                          textAlign: TextAlign.center,
                          style: t.caption.copyWith(color: t.danger),
                        ),
                        const SizedBox(height: 8),
                      ],
                      // Not "Run benchmark": on desktop the sidebar's pill
                      // of that name, a full run, is beside this one.
                      CButton(
                        label: ready ? 'Start custom run' : 'Detecting…',
                        glyph: CGlyphs.play,
                        kind: CButtonKind.primary,
                        stretch: true,
                        onPressed: canRun
                            ? () {
                                final verbose =
                                    context.read<SettingsService>().verbose;
                                // Closed at once, not popped: a page
                                // sliding away is frames drawn while the
                                // run starts on the same GPU.
                                final route = ModalRoute.of(context);
                                if (route != null && !route.isFirst) {
                                  Navigator.of(context).removeRoute(route);
                                }
                                service.start(verbose: verbose);
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

/// A chip per backend, on while any of its devices is: a tap turns all of
/// them off, or -- when none is on -- all of them on.
class _BackendToggles extends StatelessWidget {
  const _BackendToggles({required this.service});

  final BenchmarkService service;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final config = service.config;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final backend in service.catalog.usable)
              Builder(builder: (context) {
                final refs = <DeviceRef>[
                  for (final p in backend.platforms)
                    for (final d in p.devices)
                      (platformIndex: p.index, deviceIndex: d.index),
                ];
                final on = refs
                    .where((r) => config.isDeviceSelected(backend.name, r))
                    .length;
                return CChip(
                  label: backendName(backend.name),
                  selected: on > 0,
                  count: refs.length > 1 ? '$on/${refs.length}' : null,
                  onTap: () => service.updateConfig((c) {
                    for (final ref in refs) {
                      c.toggleDevice(backend.name, ref, on == 0);
                    }
                  }),
                );
              }),
          ],
        ),
        const SizedBox(height: 10),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 6),
          child: Text(
              "A backend's devices together, or one by one on its processor's "
              'card below.',
              style: t.caption),
        ),
      ],
    );
  }
}

/// A processor's card, each of its devices a row to turn on or off.
class _GroupCard extends StatelessWidget {
  const _GroupCard(
      {required this.group, required this.service, required this.compact});

  final ProcessorGroup group;
  final BenchmarkService service;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final h = t.kind(group.kind);
    final config = service.config;
    return CKindCard(
      kind: group.kind,
      label: group.label,
      title: group.name,
      compact: compact,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (group.facts.isNotEmpty) ...[
            Text(group.facts,
                style: t.caption.copyWith(
                    fontSize: compact ? 12.5 : 13, color: h.tintSub)),
            const SizedBox(height: 8),
          ],
          for (final d in group.devices)
            _DeviceCheck(
              hue: h,
              name: backendName(d.backend.name),
              detail: deviceDetail(d, group, specs: false),
              value: config.isDeviceSelected(d.backend.name, d.ref),
              onChanged: (on) => service.updateConfig(
                  (c) => c.toggleDevice(d.backend.name, d.ref, on)),
            ),
        ],
      ),
    );
  }
}

/// A device's row on its processor's card: a check box, the backend and
/// what tells the device apart.  The whole row is the control.
class _DeviceCheck extends StatelessWidget {
  const _DeviceCheck({
    required this.hue,
    required this.name,
    required this.detail,
    required this.value,
    required this.onChanged,
  });

  final KindHue hue;
  final String name;
  final String detail;
  final bool value;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: () => onChanged(!value),
      button: false,
      checked: value,
      insetFocus: true,
      focusRadius: BorderRadius.circular(CP.rHighlight),
      builder: (context, hovered, pressed) => Container(
        decoration: BoxDecoration(
          border: Border(top: BorderSide(color: hue.rule)),
        ),
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          children: [
            // The kit's check box, drawn here so the row is one control
            // rather than a box inside a button.
            SizedBox.square(
              dimension: 44,
              child: Align(
                alignment: Alignment.centerLeft,
                child: Container(
                  width: 22,
                  height: 22,
                  alignment: Alignment.center,
                  decoration: BoxDecoration(
                    color: value
                        ? t.ink
                        : (hovered || pressed ? hue.rule : Colors.transparent),
                    border:
                        Border.all(color: value ? t.ink : hue.tintSub, width: 2),
                    borderRadius: BorderRadius.circular(7),
                  ),
                  child: value
                      ? CGlyph(CGlyphs.check,
                          size: 16, color: t.inkOn, stroke: 3)
                      : null,
                ),
              ),
            ),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(name,
                        style: t.title
                            .copyWith(fontSize: 14.5, color: hue.tintInk)),
                    // Wraps: you pick a device by its full name, so nothing
                    // here is allowed to hide the tail of one.
                    if (detail.isNotEmpty)
                      Text(detail,
                          style: t.small
                              .copyWith(fontSize: 12.5, color: hue.tintSub)),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CategoryChips extends StatelessWidget {
  const _CategoryChips({required this.service});

  final BenchmarkService service;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final config = service.config;
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
              CChip(
                label: category.label,
                selected: config.categories.contains(category),
                onTap: () => service.updateConfig((c) =>
                    c.categories.contains(category)
                        ? c.categories.remove(category)
                        : c.categories.add(category)),
              ),
          ],
        ),
        const SizedBox(height: 10),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 6),
          child: Text(
            selectedCount == 0
                ? 'No categories selected — select at least one to run'
                : selectedCount == total
                    ? 'All $total categories selected'
                    : '$selectedCount of $total categories selected',
            style: t.caption
                .copyWith(color: selectedCount == 0 ? t.danger : t.muted),
          ),
        ),
      ],
    );
  }
}

/// One budget: its name and default, a slider in 50 ms steps, the value.
class _BudgetRow extends StatelessWidget {
  const _BudgetRow({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.defaultValue,
    required this.onChanged,
    this.rule = true,
  });

  final String label;
  final int value;
  final int min;
  final int max;
  final int defaultValue;
  final ValueChanged<int> onChanged;
  final bool rule;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final name = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: t.title.copyWith(fontSize: wide ? 15 : 14.5)),
        Text('Default $defaultValue ms', style: t.caption),
      ],
    );
    final reading = Text(
      '$value ms',
      textAlign: TextAlign.right,
      style: t.figure(17,
          weight: FontWeight.w700,
          color: value == defaultValue ? t.muted : t.ink),
    );
    // Material's Slider carries the drag and keyboard logic; the theme
    // draws it as the kit would: a round ink knob on a pill track, with no
    // overlay or value bubble.
    final slider = SliderTheme(
      data: SliderThemeData(
        trackHeight: 6,
        activeTrackColor: t.ink,
        inactiveTrackColor: t.surface2,
        thumbColor: t.ink,
        overlayColor: Colors.transparent,
        trackShape: const RoundedRectSliderTrackShape(),
        thumbShape: const RoundSliderThumbShape(
            enabledThumbRadius: 11, elevation: 0, pressedElevation: 0),
        overlayShape: SliderComponentShape.noOverlay,
        tickMarkShape: SliderTickMarkShape.noTickMark,
        showValueIndicator: ShowValueIndicator.never,
        padding: const EdgeInsets.symmetric(horizontal: 11),
      ),
      child: SizedBox(
        height: 44,
        child: Slider(
          value: value.toDouble().clamp(min.toDouble(), max.toDouble()),
          min: min.toDouble(),
          max: max.toDouble(),
          divisions: (max - min) ~/ 50,
          semanticFormatterCallback: (v) => '${(v / 50).round() * 50} ms',
          onChanged: (v) => onChanged((v / 50).round() * 50),
        ),
      ),
    );

    return Container(
      decoration: BoxDecoration(
        border: rule ? Border(bottom: BorderSide(color: t.line)) : null,
      ),
      padding: wide
          ? const EdgeInsets.fromLTRB(20, 12, 20, 12)
          : const EdgeInsets.fromLTRB(18, 12, 18, 10),
      child: wide
          ? Row(
              children: [
                SizedBox(width: 150, child: name),
                const SizedBox(width: 16),
                Expanded(child: slider),
                const SizedBox(width: 16),
                SizedBox(width: 84, child: reading),
              ],
            )
          : Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [Expanded(child: name), reading],
                ),
                slider,
              ],
            ),
    );
  }
}
