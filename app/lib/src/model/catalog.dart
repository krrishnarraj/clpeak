/// Parsed device catalog — the inventoryToJson() document from
/// xpupeak_copy_backend_catalog_json (src/common/inventory.cpp).  The same
/// `backends` array a verbose run document embeds as its `inventory`, and
/// beside it the hardware the devices are views of: `packages` and
/// `processors`, linked from the listing as a run links its devices, in the
/// result document's own shape (docs/format-v4.md, Hardware) -- so a run
/// screen can group the devices by processor before anything has run.
library;

import 'hardware.dart';

class CatalogDevice {
  const CatalogDevice({
    required this.index,
    required this.name,
    required this.type,
    required this.driver,
    required this.api,
    required this.origin,
    required this.computeUnits,
    required this.clockMHz,
    required this.globalMemBytes,
    required this.fp16,
    required this.fp64,
    this.processor = '',
  });

  final int index;
  final String name;
  final String type; // "GPU" / "CPU" / "Discrete GPU" / ...
  final String driver;
  final String api;
  final String origin; // "EP plugin (…)" for ONNX plugin EPs; empty otherwise
  final int computeUnits;
  final int clockMHz;
  final int globalMemBytes;
  final bool fp16;
  final bool fp64;

  /// The id of the processor it is a view of (`gpu0`), as
  /// [BackendCatalog.processors] lists it; empty from a native library
  /// older than the catalog's hardware.  `--devices` takes the id too, for
  /// every device on that processor.
  final String processor;

  factory CatalogDevice.fromJson(Map<String, dynamic> m) => CatalogDevice(
        index: (m['index'] as num?)?.toInt() ?? -1,
        name: m['name'] as String? ?? '',
        type: m['type'] as String? ?? '',
        driver: m['driver'] as String? ?? '',
        api: m['api'] as String? ?? '',
        origin: m['origin'] as String? ?? '',
        computeUnits: (m['compute_units'] as num?)?.toInt() ?? 0,
        clockMHz: (m['clock_mhz'] as num?)?.toInt() ?? 0,
        globalMemBytes: (m['global_mem_bytes'] as num?)?.toInt() ?? 0,
        fp16: m['fp16'] as bool? ?? false,
        fp64: m['fp64'] as bool? ?? false,
        processor: m['processor'] is String ? m['processor'] as String : '',
      );
}

class CatalogPlatform {
  const CatalogPlatform({
    required this.index,
    required this.name,
    required this.devices,
  });

  final int index;
  final String name;
  final List<CatalogDevice> devices;

  factory CatalogPlatform.fromJson(Map<String, dynamic> m) => CatalogPlatform(
        index: (m['index'] as num?)?.toInt() ?? -1,
        name: m['name'] as String? ?? '',
        devices: [
          for (final d in (m['devices'] as List? ?? const []))
            CatalogDevice.fromJson(d as Map<String, dynamic>)
        ],
      );
}

class CatalogBackend {
  const CatalogBackend({
    required this.name,
    required this.flag,
    required this.available,
    required this.info,
    required this.reason,
    required this.platforms,
  });

  final String name; // "OpenCL" / "Vulkan" / "CUDA" / ... / "CPU"

  /// The backend's command-line name: `--devices <flag>:<index>` names one
  /// of its devices.  Authored natively (the backend table in
  /// src/common/options.cpp), never derived here.
  final String flag;
  final bool available;

  /// A backend-level fact worth showing: the ONNX Runtime version, the OS
  /// release Core ML comes with.  Empty when there is none.
  final String info;

  /// Why `available` is false.  Empty when it is true.
  final String reason;
  final List<CatalogPlatform> platforms;

  bool get hasDevices => platforms.any((p) => p.devices.isNotEmpty);
  int get deviceCount =>
      platforms.fold(0, (n, p) => n + p.devices.length);

  factory CatalogBackend.fromJson(Map<String, dynamic> m) => CatalogBackend(
        name: m['name'] as String? ?? '',
        flag: m['flag'] as String? ?? '',
        available: m['available'] as bool? ?? false,
        info: m['info'] as String? ?? '',
        reason: m['reason'] as String? ?? '',
        platforms: [
          for (final p in (m['platforms'] as List? ?? const []))
            CatalogPlatform.fromJson(p as Map<String, dynamic>)
        ],
      );
}

/// One device of the catalog with its backend: what `--devices` names as
/// `<backend.flag>:<device.index>`.
class CatalogEntry {
  const CatalogEntry(this.backend, this.device);

  final CatalogBackend backend;
  final CatalogDevice device;

  /// The token `--devices` takes for it: `metal:0`.
  String get token => '${backend.flag}:${device.index}';
}

class BackendCatalog {
  const BackendCatalog(this.backends,
      {this.packages = const [], this.processors = const []});

  final List<CatalogBackend> backends;

  /// The chips and cards the listed devices sit in, the host's first
  /// (`pkg0`); empty from a native library older than the catalog's
  /// hardware.
  final List<PackageInfo> packages;

  /// The processors the listed devices are views of, in the order a run
  /// numbers them: `cpu0`, `gpu0`, `npu0`.  A processor no listed device
  /// reaches is here too (the host CPU, when no CPU backend is built in).
  final List<ProcessorInfo> processors;

  /// Backends that actually enumerated at least one device.
  List<CatalogBackend> get usable =>
      backends.where((b) => b.available && b.hasDevices).toList();

  ProcessorInfo? processorById(String id) {
    for (final p in processors) {
      if (p.id == id) return p;
    }
    return null;
  }

  PackageInfo? packageById(String id) {
    for (final p in packages) {
      if (p.id == id) return p;
    }
    return null;
  }

  /// Every listed device on processor [id], in run order (by backend, then
  /// index): what `--devices <id>` runs.
  List<CatalogEntry> devicesOf(String id) => [
        for (final b in backends)
          for (final p in b.platforms)
            for (final d in p.devices)
              if (d.processor == id) CatalogEntry(b, d),
      ];

  factory BackendCatalog.fromJson(Map<String, dynamic> m) => BackendCatalog(
        [
          for (final b in (m['backends'] as List? ?? const []))
            CatalogBackend.fromJson(b as Map<String, dynamic>)
        ],
        packages: [
          for (final p in (m['packages'] is List ? m['packages'] as List : const []))
            PackageInfo.fromJson(p),
        ],
        processors: [
          for (final p
              in (m['processors'] is List ? m['processors'] as List : const []))
            ProcessorInfo.fromJson(p),
        ],
      );
}
