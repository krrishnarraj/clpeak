/// Parsed device catalog — the inventoryToJson() document from
/// clpeak_copy_backend_catalog_json (src/common/inventory.cpp).
library;

class CatalogDevice {
  const CatalogDevice({
    required this.index,
    required this.name,
    required this.type,
    required this.driver,
    required this.api,
    required this.computeUnits,
    required this.clockMHz,
    required this.globalMemBytes,
    required this.fp16,
    required this.fp64,
  });

  final int index;
  final String name;
  final String type; // "GPU" / "CPU" / "Discrete GPU" / ...
  final String driver;
  final String api;
  final int computeUnits;
  final int clockMHz;
  final int globalMemBytes;
  final bool fp16;
  final bool fp64;

  factory CatalogDevice.fromJson(Map<String, dynamic> m) => CatalogDevice(
        index: (m['index'] as num?)?.toInt() ?? -1,
        name: m['name'] as String? ?? '',
        type: m['type'] as String? ?? '',
        driver: m['driver'] as String? ?? '',
        api: m['api'] as String? ?? '',
        computeUnits: (m['computeUnits'] as num?)?.toInt() ?? 0,
        clockMHz: (m['clockMHz'] as num?)?.toInt() ?? 0,
        globalMemBytes: (m['globalMemBytes'] as num?)?.toInt() ?? 0,
        fp16: m['fp16'] as bool? ?? false,
        fp64: m['fp64'] as bool? ?? false,
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
    required this.available,
    required this.platforms,
  });

  final String name; // "OpenCL" / "Vulkan" / "CUDA" / ... / "CPU"
  final bool available;
  final List<CatalogPlatform> platforms;

  bool get hasDevices => platforms.any((p) => p.devices.isNotEmpty);
  int get deviceCount =>
      platforms.fold(0, (n, p) => n + p.devices.length);

  factory CatalogBackend.fromJson(Map<String, dynamic> m) => CatalogBackend(
        name: m['name'] as String? ?? '',
        available: m['available'] as bool? ?? false,
        platforms: [
          for (final p in (m['platforms'] as List? ?? const []))
            CatalogPlatform.fromJson(p as Map<String, dynamic>)
        ],
      );
}

class BackendCatalog {
  const BackendCatalog(this.backends);

  final List<CatalogBackend> backends;

  /// Backends that actually enumerated at least one device.
  List<CatalogBackend> get usable =>
      backends.where((b) => b.available && b.hasDevices).toList();

  factory BackendCatalog.fromJson(Map<String, dynamic> m) => BackendCatalog([
        for (final b in (m['backends'] as List? ?? const []))
          CatalogBackend.fromJson(b as Map<String, dynamic>)
      ]);
}
