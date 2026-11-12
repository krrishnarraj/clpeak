import 'run_document.dart';

/// One processor as the history index remembers it: enough to name a run's
/// hardware in a list without opening the file.
class ProcessorSummary {
  const ProcessorSummary({required this.id, this.kind = '', this.name = ''});

  /// `cpu0`, `gpu0`, `npu0`.
  final String id;
  final String kind;
  final String name;

  Map<String, dynamic> toJson() => {
        'id': id,
        if (kind.isNotEmpty) 'kind': kind,
        if (name.isNotEmpty) 'name': name,
      };

  factory ProcessorSummary.fromJson(Object? m) => m is Map
      ? ProcessorSummary(
          id: m['id'] is String ? m['id'] as String : '',
          kind: m['kind'] is String ? m['kind'] as String : '',
          name: m['name'] is String ? m['name'] as String : '',
        )
      : const ProcessorSummary(id: '');
}

/// One row of the history index (`<base>/runs/index.json`, where `<base>` is
/// RunHistoryStore.baseDirectory()).
class RunSummary {
  const RunSummary({
    required this.id,
    required this.startedAt,
    required this.durationMs,
    required this.devices,
    required this.backends,
    required this.cancelled,
    required this.fileName,
    this.name = '',
    this.generator = const GeneratorInfo(),
    this.hostOs = '',
    this.hostOsVersion = '',
    this.hostArch = '',
    this.systemName = '',
    this.formFactor = '',
    this.formFactorDetail = '',
    this.processors = const [],
  });

  final String id; // also the XML base name
  final DateTime startedAt;
  final int durationMs;
  final List<String> devices;
  final List<String> backends;
  final bool cancelled;
  final String fileName;

  /// Optional user-given name (rename action in History).
  final String name;

  /// Where the run was produced — persisted for history list preview so an
  /// imported run from another machine is distinguishable without opening it.
  final GeneratorInfo generator;
  final String hostOs;
  final String hostOsVersion;
  final String hostArch;

  /// The machine, as [RunMeta.systemName] names it: the `system.name`
  /// annotation its Details gave it, else the OS's name, the model id, …
  final String systemName;

  /// As the file says it; empty where it does not.
  final String formFactor;
  final String formFactorDetail;

  /// The hardware the run's devices were linked to, in the file's order.
  final List<ProcessorSummary> processors;

  /// Title shown in lists/headers: the user's name when set, otherwise the
  /// run's timestamp id (`20260813_130317`) — the device list is identical
  /// across every run on one machine, so it can't tell two runs apart.
  String get displayTitle =>
      name.isNotEmpty ? name : (id.isNotEmpty ? id : devices.join(', '));

  RunSummary withName(String newName) => _copy(name: newName);

  RunSummary _copy({String? name}) => RunSummary(
        id: id,
        startedAt: startedAt,
        durationMs: durationMs,
        devices: devices,
        backends: backends,
        cancelled: cancelled,
        fileName: fileName,
        name: name ?? this.name,
        generator: generator,
        hostOs: hostOs,
        hostOsVersion: hostOsVersion,
        hostArch: hostArch,
        systemName: systemName,
        formFactor: formFactor,
        formFactorDetail: formFactorDetail,
        processors: processors,
      );

  Map<String, dynamic> toJson() => {
        'id': id,
        'startedAt': startedAt.toIso8601String(),
        'durationMs': durationMs,
        'devices': devices,
        'backends': backends,
        'cancelled': cancelled,
        'fileName': fileName,
        if (name.isNotEmpty) 'name': name,
        if (!generator.isEmpty) 'generator': generator.toJson(),
        if (hostOs.isNotEmpty) 'hostOs': hostOs,
        if (hostOsVersion.isNotEmpty) 'hostOsVersion': hostOsVersion,
        if (hostArch.isNotEmpty) 'hostArch': hostArch,
        if (systemName.isNotEmpty) 'systemName': systemName,
        if (formFactor.isNotEmpty) 'formFactor': formFactor,
        if (formFactorDetail.isNotEmpty) 'formFactorDetail': formFactorDetail,
        if (processors.isNotEmpty)
          'processors': [for (final p in processors) p.toJson()],
      };

  factory RunSummary.fromJson(Map<String, dynamic> m) => RunSummary(
        id: m['id'] as String? ?? '',
        startedAt:
            DateTime.tryParse(m['startedAt'] as String? ?? '') ?? DateTime(0),
        durationMs: (m['durationMs'] as num?)?.toInt() ?? 0,
        devices: [...(m['devices'] as List? ?? const []).cast<String>()],
        backends: [...(m['backends'] as List? ?? const []).cast<String>()],
        cancelled: m['cancelled'] as bool? ?? false,
        fileName: m['fileName'] as String? ?? '',
        name: m['name'] as String? ?? '',
        generator: GeneratorInfo.fromJson(m['generator']),
        hostOs: m['hostOs'] as String? ?? '',
        hostOsVersion: m['hostOsVersion'] as String? ?? '',
        hostArch: m['hostArch'] as String? ?? '',
        systemName: m['systemName'] as String? ?? '',
        formFactor: m['formFactor'] as String? ?? '',
        formFactorDetail: m['formFactorDetail'] as String? ?? '',
        processors: [
          for (final p in (m['processors'] is List
              ? m['processors'] as List
              : const []))
            ProcessorSummary.fromJson(p),
        ],
      );

  /// Summarize a finished (or loaded) document.
  factory RunSummary.fromDocument({
    required String id,
    required String fileName,
    required RunDocument doc,
    required DateTime startedAt,
    required int durationMs,
    required bool cancelled,
  }) {
    final devices = <String>{};
    final backends = <String>{};
    for (final run in doc.runs) {
      devices.add(run.device);
      backends.add(run.backend);
    }
    final meta = doc.meta;
    return RunSummary(
      id: id,
      startedAt: startedAt,
      durationMs: durationMs,
      devices: devices.toList(),
      backends: backends.toList(),
      cancelled: cancelled,
      fileName: fileName,
      generator: meta?.generator ?? const GeneratorInfo(),
      hostOs: meta?.osName ?? '',
      hostOsVersion: meta?.osVersion ?? '',
      hostArch: meta?.arch ?? '',
      systemName: meta?.systemName ?? '',
      formFactor: meta?.systemInfo.formFactor ?? '',
      formFactorDetail: meta?.systemInfo.formFactorDetail ?? '',
      processors: [
        for (final p in doc.processors)
          ProcessorSummary(id: p.id, kind: p.kind, name: p.name),
      ],
    );
  }
}
