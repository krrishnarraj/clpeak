/// What a run happened on, as the document records it (docs/format-v4.md:
/// System, Conditions, Hardware, Device): the machine, what it was doing,
/// its packages and the processors in them, and each device's own view --
/// its API's facts, the evidence that linked it and, for an ML runtime,
/// where it ran.
///
/// Read-only mirrors of include/common/run_document.h.  Every string is kept
/// as the file has it, a newer build's words included; nothing here rewrites
/// a document (a document is edited as JSON, see document_edit.dart).  A
/// field of the wrong type reads as absent rather than failing the file: a
/// result is editable JSON, and one bad value should not hide the rest.
library;

String _str(Map m, String key) {
  final v = m[key];
  return v is String ? v.trim() : '';
}

int _int(Map m, String key, [int absent = 0]) {
  final v = m[key];
  return v is num ? v.toInt() : absent;
}

bool? _bool(Map m, String key) {
  final v = m[key];
  return v is bool ? v : null;
}

Map _map(Object? v) => v is Map ? v : const {};

List _list(Object? v) => v is List ? v : const [];

/// The machine a run happened on: the document's `system`.  Everything is
/// optional, since which of it a platform can answer varies.
class SystemInfo {
  const SystemInfo({
    this.formFactor = '',
    this.formFactorDetail = '',
    this.formFactorFrom = '',
    this.isVirtual = false,
    this.translated = false,
    this.manufacturer = '',
    this.model = '',
    this.name = '',
    this.family = '',
    this.board = '',
    this.soc = '',
    this.osName = '',
    this.osVersion = '',
    this.osBuild = '',
    this.osKernel = '',
    this.osDistribution = '',
    this.arch = '',
    this.memoryBytes = 0,
  });

  /// `mobile`, `laptop`, `desktop`, `server` -- or empty where the file
  /// does not say; [effectiveFormFactor] is what a reader takes then.
  final String formFactor;

  /// `phone`, `tablet` or `embedded`, when known.
  final String formFactorDetail;

  /// What [formFactor] was read from.
  final String formFactorFrom;

  final bool isVirtual;

  /// An x86 binary under Rosetta or Prism.
  final bool translated;

  final String manufacturer;

  /// The model id: `MacBookPro18,3`, `SM-S921B`, a DMI product name.
  final String model;

  /// The marketing name, only as the OS gives it.  Where it gives none the
  /// app shows the `system.name` annotation or the model id
  /// ([RunMeta.systemName]).
  final String name;

  final String family;
  final String board;

  /// "Apple M1 Pro"; the CPU brand where there is no SoC name.
  final String soc;

  final String osName;
  final String osVersion;
  final String osBuild;
  final String osKernel;
  final String osDistribution;

  /// The binary's architecture: `arm64`, `x86_64`.
  final String arch;
  final int memoryBytes;

  /// The form factor a reader takes: absent means desktop.
  String get effectiveFormFactor =>
      formFactor.isEmpty ? 'desktop' : formFactor;

  factory SystemInfo.fromJson(Object? json) {
    final m = _map(json);
    final os = _map(m['os']);
    return SystemInfo(
      formFactor: _str(m, 'form_factor'),
      formFactorDetail: _str(m, 'form_factor_detail'),
      formFactorFrom: _str(m, 'form_factor_from'),
      isVirtual: _bool(m, 'virtual') ?? false,
      translated: _bool(m, 'translated') ?? false,
      manufacturer: _str(m, 'manufacturer'),
      model: _str(m, 'model'),
      name: _str(m, 'name'),
      family: _str(m, 'family'),
      board: _str(m, 'board'),
      soc: _str(m, 'soc'),
      osName: _str(os, 'name'),
      osVersion: _str(os, 'version'),
      osBuild: _str(os, 'build'),
      osKernel: _str(os, 'kernel'),
      osDistribution: _str(os, 'distribution'),
      arch: _str(m, 'arch'),
      memoryBytes: _int(m, 'memory_bytes'),
    );
  }
}

/// What the machine was doing, which changes every number: the document's
/// `conditions`.  Each field is empty (null) when not known.
class RunConditions {
  const RunConditions({
    this.powerSource = '',
    this.lowPowerMode,
    this.powerProfile = '',
    this.cpuGovernor = '',
    this.thermalStart = '',
    this.thermalEnd = '',
  });

  /// `ac` or `battery`.
  final String powerSource;
  final bool? lowPowerMode;

  /// The OS's own name for its power profile.
  final String powerProfile;
  final String cpuGovernor;

  /// The OS's thermal state at the start and end of the run, in its words.
  final String thermalStart;
  final String thermalEnd;

  bool get isEmpty =>
      powerSource.isEmpty &&
      lowPowerMode == null &&
      powerProfile.isEmpty &&
      cpuGovernor.isEmpty &&
      thermalStart.isEmpty &&
      thermalEnd.isEmpty;

  factory RunConditions.fromJson(Object? json) {
    final m = _map(json);
    return RunConditions(
      powerSource: _str(m, 'power_source'),
      lowPowerMode: _bool(m, 'low_power_mode'),
      powerProfile: _str(m, 'power_profile'),
      cpuGovernor: _str(m, 'cpu_governor'),
      thermalStart: _str(m, 'thermal_start'),
      thermalEnd: _str(m, 'thermal_end'),
    );
  }
}

/// One backend compiled into the binary that wrote the file.
class BuildBackend {
  const BuildBackend({required this.name, this.kind = '', this.portable = false});

  /// As `devices[].backend` names it.
  final String name;

  /// `gpu`, `cpu` or `ai`.
  final String kind;
  final bool portable;
}

/// One chip or card: an SoC, a host CPU package, a discrete GPU.
class PackageInfo {
  const PackageInfo({
    required this.id,
    this.kind = '',
    this.vendor = '',
    this.name = '',
  });

  /// `pkg0`, `pkg1`, …
  final String id;

  /// `soc`, `cpu` or `discrete`.
  final String kind;
  final String vendor;
  final String name;

  factory PackageInfo.fromJson(Object? json) {
    final m = _map(json);
    return PackageInfo(
      id: _str(m, 'id'),
      kind: _str(m, 'kind'),
      vendor: _str(m, 'vendor'),
      name: _str(m, 'name'),
    );
  }
}

/// A cache, of a whole processor or of one of its clusters.
class CacheInfo {
  const CacheInfo({
    required this.level,
    this.sizeBytes = 0,
    this.instances = 0,
    this.lineBytes = 0,
  });

  /// `l1d`, `l1i`, `l2`, `l3`, `slc`.
  final String level;

  /// The size of one.
  final int sizeBytes;
  final int instances;
  final int lineBytes;

  factory CacheInfo.fromJson(Object? json) {
    final m = _map(json);
    return CacheInfo(
      level: _str(m, 'level'),
      sizeBytes: _int(m, 'size_bytes'),
      instances: _int(m, 'instances'),
      lineBytes: _int(m, 'line_bytes'),
    );
  }
}

/// A group of like cores: Apple's P and E clusters, big and LITTLE.
class CpuCluster {
  const CpuCluster({
    this.kind = '',
    this.cores = 0,
    this.threadsPerCore = 0,
    this.maxMhz = 0,
    this.microarch = '',
    this.midr = '',
    this.caches = const [],
  });

  /// `prime`, `performance`, `efficiency`, `low_power`.
  final String kind;
  final int cores;
  final int threadsPerCore;
  final int maxMhz;
  final String microarch;
  final String midr;
  final List<CacheInfo> caches;

  factory CpuCluster.fromJson(Object? json) {
    final m = _map(json);
    return CpuCluster(
      kind: _str(m, 'kind'),
      cores: _int(m, 'cores'),
      threadsPerCore: _int(m, 'threads_per_core'),
      maxMhz: _int(m, 'max_mhz'),
      microarch: _str(m, 'microarch'),
      midr: _str(m, 'midr'),
      caches: [for (final c in _list(m['caches'])) CacheInfo.fromJson(c)],
    );
  }
}

/// A CPU's matrix engine: Intel's AMX tiles, Apple's AMX and SME units.
class CpuEngine {
  const CpuEngine({
    this.kind = '',
    this.name = '',
    this.scope = '',
    this.reachedBy = '',
  });

  /// `matrix`.
  final String kind;
  final String name;

  /// `core` or `cluster`: a per-cluster engine's all-core number does not
  /// scale with cores.
  final String scope;

  /// `instructions` or `library`.
  final String reachedBy;

  factory CpuEngine.fromJson(Object? json) {
    final m = _map(json);
    return CpuEngine(
      kind: _str(m, 'kind'),
      name: _str(m, 'name'),
      scope: _str(m, 'scope'),
      reachedBy: _str(m, 'reached_by'),
    );
  }
}

/// A CPU's facts, as written flat on its processor (or in a CPU device's
/// `facts`).
class CpuFacts {
  const CpuFacts({
    this.isa = '',
    this.cores = 0,
    this.threads = 0,
    this.sockets = 0,
    this.cpuidVendor = '',
    this.cpuidFamily = -1,
    this.cpuidModel = -1,
    this.cpuidStepping = -1,
    this.clusters = const [],
    this.caches = const [],
    this.engines = const [],
    this.features = const [],
  });

  /// `arm64`, `x86_64`.
  final String isa;
  final int cores;
  final int threads;

  /// A multi-socket CPU is one processor.
  final int sockets;

  final String cpuidVendor;

  /// -1 where absent.
  final int cpuidFamily;
  final int cpuidModel;
  final int cpuidStepping;

  final List<CpuCluster> clusters;

  /// The caches the whole processor shares.
  final List<CacheInfo> caches;
  final List<CpuEngine> engines;

  /// One normalised list: `neon`, `dotprod`, `avx512_vnni`, `amx_tile`, …
  final List<String> features;

  bool get isEmpty =>
      isa.isEmpty &&
      cores == 0 &&
      threads == 0 &&
      sockets == 0 &&
      cpuidVendor.isEmpty &&
      cpuidFamily < 0 &&
      clusters.isEmpty &&
      caches.isEmpty &&
      engines.isEmpty &&
      features.isEmpty;

  factory CpuFacts.fromJson(Map m) {
    final cpuid = _map(m['cpuid']);
    return CpuFacts(
      isa: _str(m, 'isa'),
      cores: _int(m, 'cores'),
      threads: _int(m, 'threads'),
      sockets: _int(m, 'sockets'),
      cpuidVendor: _str(cpuid, 'vendor'),
      cpuidFamily: _int(cpuid, 'family', -1),
      cpuidModel: _int(cpuid, 'model', -1),
      cpuidStepping: _int(cpuid, 'stepping', -1),
      clusters: [for (final c in _list(m['clusters'])) CpuCluster.fromJson(c)],
      caches: [for (final c in _list(m['caches'])) CacheInfo.fromJson(c)],
      engines: [for (final e in _list(m['engines'])) CpuEngine.fromJson(e)],
      features: [
        for (final f in _list(m['features']))
          if (f is String) f,
      ],
    );
  }
}

/// What an API reported about a device, typed -- on a processor, merged
/// from the most native backend that ran; on a device, its own API's.
class HardwareFacts {
  const HardwareFacts({
    this.vendorId = '',
    this.deviceId = '',
    this.arch = '',
    this.integrated,
    this.computeUnits = 0,
    this.computeUnit = '',
    this.clockMhz = 0,
    this.memoryBytes = 0,
    this.cpu = const CpuFacts(),
  });

  /// The PCI ids, lower-case hex: a model's ids, not a unit's.
  final String vendorId;
  final String deviceId;

  /// `sm_120`, `gfx1201`, `Apple7`.
  final String arch;
  final bool? integrated;
  final int computeUnits;

  /// What one compute unit is called: `SM`, `CU`, `Xe-core`, `core`.
  final String computeUnit;
  final int clockMhz;
  final int memoryBytes;
  final CpuFacts cpu;

  bool get isEmpty =>
      vendorId.isEmpty &&
      deviceId.isEmpty &&
      arch.isEmpty &&
      integrated == null &&
      computeUnits == 0 &&
      computeUnit.isEmpty &&
      clockMhz == 0 &&
      memoryBytes == 0 &&
      cpu.isEmpty;

  /// From an object holding the facts flat: a processor, or a device's
  /// `facts`.
  factory HardwareFacts.fromJson(Object? json) {
    final m = _map(json);
    return HardwareFacts(
      vendorId: _str(m, 'vendor_id'),
      deviceId: _str(m, 'device_id'),
      arch: _str(m, 'arch'),
      integrated: _bool(m, 'integrated'),
      computeUnits: _int(m, 'compute_units'),
      computeUnit: _str(m, 'compute_unit'),
      clockMhz: _int(m, 'clock_mhz'),
      memoryBytes: _int(m, 'memory_bytes'),
      cpu: CpuFacts.fromJson(m),
    );
  }
}

/// One CPU, GPU or NPU in a package, with its run-local id: every backend's
/// view of it is a device that names it.
class ProcessorInfo {
  const ProcessorInfo({
    required this.id,
    this.kind = '',
    this.package = '',
    this.vendor = '',
    this.vendorName = '',
    this.name = '',
    this.facts = const HardwareFacts(),
  });

  /// `cpu0`, `gpu0`, `npu0`.
  final String id;

  /// `cpu`, `gpu`, `npu`, `accelerator` -- or a newer build's word.
  final String kind;

  /// Its package's id.
  final String package;

  /// A vendor from the vocabulary, and the vendor's own string.
  final String vendor;
  final String vendorName;
  final String name;
  final HardwareFacts facts;

  factory ProcessorInfo.fromJson(Object? json) {
    final m = _map(json);
    return ProcessorInfo(
      id: _str(m, 'id'),
      kind: _str(m, 'kind'),
      package: _str(m, 'package'),
      vendor: _str(m, 'vendor'),
      vendorName: _str(m, 'vendor_name'),
      name: _str(m, 'name'),
      facts: HardwareFacts.fromJson(m),
    );
  }
}

/// An ML runtime's view of where a device ran: ONNX Runtime's execution
/// provider, LiteRT's accelerator or Core ML's compute device, and the
/// runtime's version.
class RuntimeInfo {
  const RuntimeInfo({this.provider = '', this.accelerator = '', this.version = ''});

  final String provider;
  final String accelerator;
  final String version;

  bool get isEmpty => provider.isEmpty && accelerator.isEmpty && version.isEmpty;

  factory RuntimeInfo.fromJson(Object? json) {
    final m = _map(json);
    return RuntimeInfo(
      provider: _str(m, 'provider'),
      accelerator: _str(m, 'accelerator'),
      version: _str(m, 'version'),
    );
  }
}

/// A device's linking evidence as the file has it: a token per strong
/// identifier (any string value is one), and the weaker evidence as it is.
/// Never an identifier itself.
Map<String, Object> evidenceFromJson(Object? json) => {
      for (final e in _map(json).entries)
        if (e.key is String && e.value != null) e.key as String: e.value as Object,
    };
