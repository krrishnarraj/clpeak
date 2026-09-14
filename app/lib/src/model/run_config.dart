import 'catalog.dart';
import 'result_model.dart';

/// Time-budget presets.  Custom keeps whatever the user configured.
enum RunPreset { full, custom }

/// A backend's CLI flag is its catalog name in lower case: `--no-<flag>`
/// skips it and `--device <flag>:<index>` narrows it (mirrors the backend
/// table in src/common/options.cpp).  Only backend/device/category/time
/// flags are ever emitted -- never individual test flags, so test churn in
/// the core needs no app changes.
String _backendFlag(String backendName) => backendName.toLowerCase();

/// A device reference within a backend: platform index (OpenCL) + device
/// index.  For single-platform backends platformIndex is the synthetic 0.
typedef DeviceRef = ({int platformIndex, int deviceIndex});

const int kDefaultMaxTimeMs = 500;
const int kDefaultMaxTimeCpuMs = 2000;

/// User-selected run configuration → CLI argv.
class RunConfig {
  RunConfig({
    Map<String, Set<DeviceRef>>? selectedDevices,
    Set<BenchCategory>? categories,
    this.maxTimeMs = kDefaultMaxTimeMs,
    this.maxTimeCpuMs = kDefaultMaxTimeCpuMs,
  })  : selectedDevices = selectedDevices ?? {},
        categories = categories ?? BenchCategory.selectable.toSet();

  /// Selected devices per backend name.  A backend absent from the map (or
  /// mapped to an empty set) is skipped.
  final Map<String, Set<DeviceRef>> selectedDevices;

  /// Enabled categories.  All selected = default (no flags emitted).
  final Set<BenchCategory> categories;

  int maxTimeMs;
  int maxTimeCpuMs;

  /// Select every device of every usable backend.
  factory RunConfig.allDevices(BackendCatalog catalog,
      {int maxTimeMs = kDefaultMaxTimeMs,
      int maxTimeCpuMs = kDefaultMaxTimeCpuMs}) {
    final selected = <String, Set<DeviceRef>>{};
    for (final b in catalog.usable) {
      final refs = <DeviceRef>{};
      for (final p in b.platforms) {
        for (final d in p.devices) {
          refs.add((platformIndex: p.index, deviceIndex: d.index));
        }
      }
      if (refs.isNotEmpty) selected[b.name] = refs;
    }
    return RunConfig(
        selectedDevices: selected,
        maxTimeMs: maxTimeMs,
        maxTimeCpuMs: maxTimeCpuMs);
  }

  factory RunConfig.preset(RunPreset preset, BackendCatalog catalog) =>
      RunConfig.allDevices(catalog);

  bool get hasSelection => selectedDevices.values.any((s) => s.isNotEmpty);

  bool isDeviceSelected(String backend, DeviceRef ref) =>
      selectedDevices[backend]?.contains(ref) ?? false;

  void toggleDevice(String backend, DeviceRef ref, bool on) {
    final set = selectedDevices.putIfAbsent(backend, () => {});
    on ? set.add(ref) : set.remove(ref);
  }

  /// Build the clpeak_launch argv (without the program name).
  ///
  /// Semantics match the retired mobile apps: a fully-selected backend emits
  /// no device flags (native runs all), a deselected backend emits
  /// `--no-<backend>`, and a partial selection emits index lists.
  List<String> toArgs(BackendCatalog catalog) {
    final args = <String>[];

    // Every partial selection goes into one --device list.  Device indices
    // are per backend and already unique across platforms (OpenCL numbers
    // its devices consecutively), so `backend:index` names a device exactly.
    final deviceItems = <String>[];
    for (final backend in catalog.usable) {
      final flag = _backendFlag(backend.name);
      final selected = selectedDevices[backend.name] ?? const <DeviceRef>{};

      if (selected.isEmpty) {
        args.add('--no-$flag');
        continue;
      }

      final all = <DeviceRef>{
        for (final p in backend.platforms)
          for (final d in p.devices)
            (platformIndex: p.index, deviceIndex: d.index)
      };
      if (selected.containsAll(all)) continue; // full selection: no flags

      final devices = selected.map((r) => r.deviceIndex).toSet().toList()
        ..sort();
      deviceItems.addAll(devices.map((d) => '$flag:$d'));
    }
    if (deviceItems.isNotEmpty) {
      args.addAll(['--device', deviceItems.join(',')]);
    }

    // Categories: all selected = default; otherwise positive flags flip the
    // parser into allow-list mode.
    if (categories.length < BenchCategory.selectable.length) {
      for (final c in categories) {
        args.add('--${c.flag}');
      }
    }

    if (maxTimeMs != kDefaultMaxTimeMs) {
      args.addAll(['--max-time', '$maxTimeMs']);
    }
    if (maxTimeCpuMs != kDefaultMaxTimeCpuMs) {
      args.addAll(['--max-time-cpu', '$maxTimeCpuMs']);
    }

    return args;
  }
}
