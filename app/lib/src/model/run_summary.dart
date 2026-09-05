import 'run_document.dart';

/// One row of the history index (documents/clpeak/runs/index.json).
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
    this.clpeakVersion = '',
    this.hostOs = '',
    this.hostOsVersion = '',
    this.hostArch = '',
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
  final String clpeakVersion;
  final String hostOs;
  final String hostOsVersion;
  final String hostArch;

  /// Title shown in lists/headers: the user's name when set, otherwise the
  /// run's timestamp id (`20260813_130317`) — the device list is identical
  /// across every run on one machine, so it can't tell two runs apart.
  String get displayTitle =>
      name.isNotEmpty ? name : (id.isNotEmpty ? id : devices.join(', '));

  RunSummary withName(String newName) => RunSummary(
        id: id,
        startedAt: startedAt,
        durationMs: durationMs,
        devices: devices,
        backends: backends,
        cancelled: cancelled,
        fileName: fileName,
        name: newName,
        clpeakVersion: clpeakVersion,
        hostOs: hostOs,
        hostOsVersion: hostOsVersion,
        hostArch: hostArch,
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
        if (clpeakVersion.isNotEmpty) 'clpeakVersion': clpeakVersion,
        if (hostOs.isNotEmpty) 'hostOs': hostOs,
        if (hostOsVersion.isNotEmpty) 'hostOsVersion': hostOsVersion,
        if (hostArch.isNotEmpty) 'hostArch': hostArch,
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
        clpeakVersion: m['clpeakVersion'] as String? ?? '',
        hostOs: m['hostOs'] as String? ?? '',
        hostOsVersion: m['hostOsVersion'] as String? ?? '',
        hostArch: m['hostArch'] as String? ?? '',
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
      clpeakVersion: meta?.clpeakVersion ?? '',
      hostOs: (meta?.host['os']?.toString() ?? '').trim(),
      hostOsVersion: (meta?.host['os_version']?.toString() ?? '').trim(),
      hostArch: (meta?.host['arch']?.toString() ?? '').trim(),
    );
  }
}
