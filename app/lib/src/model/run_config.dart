import 'catalog.dart';
import 'result_model.dart';

/// Time-budget presets.  Custom keeps whatever the user configured.
enum RunPreset { full, custom }

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
  /// Only backend/device/category/time flags are ever emitted -- never
  /// individual test flags, so test churn in the core needs no app changes.
  /// Devices: the whole catalog selected means no flags (native runs
  /// everything); anything less is one `--devices` list naming every selected
  /// device as `<flag>:<index>`, which is exactly the set that runs -- a
  /// backend with nothing on the list is skipped natively, so there is no
  /// separate `--no-<backend>`.
  List<String> toArgs(BackendCatalog catalog) {
    final args = <String>[];

    final deviceItems = <String>[];
    var complete = true;
    for (final backend in catalog.usable) {
      final selected = selectedDevices[backend.name] ?? const <DeviceRef>{};
      final all = <DeviceRef>{
        for (final p in backend.platforms)
          for (final d in p.devices)
            (platformIndex: p.index, deviceIndex: d.index)
      };
      if (!selected.containsAll(all)) complete = false;

      // Device indices are per backend and unique across platforms (OpenCL
      // numbers its devices consecutively), so flag:index names one device.
      final devices = selected.map((r) => r.deviceIndex).toSet().toList()
        ..sort();
      deviceItems.addAll(devices.map((d) => '${backend.flag}:$d'));
    }
    if (!complete) {
      if (deviceItems.isNotEmpty) {
        args.addAll(['--devices', deviceItems.join(',')]);
      } else {
        // Nothing selected.  Callers gate on hasSelection, but an empty
        // --devices list cannot say "run nothing", so say it per backend.
        for (final backend in catalog.usable) {
          args.add('--no-${backend.flag}');
        }
      }
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
