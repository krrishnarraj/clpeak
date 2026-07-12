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

  /// Title shown in lists/headers: the user's name when set, otherwise the
  /// device list.
  String get displayTitle => name.isNotEmpty ? name : devices.join(', ');

  RunSummary withName(String newName) => RunSummary(
        id: id,
        startedAt: startedAt,
        durationMs: durationMs,
        devices: devices,
        backends: backends,
        cancelled: cancelled,
        fileName: fileName,
        name: newName,
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
    return RunSummary(
      id: id,
      startedAt: startedAt,
      durationMs: durationMs,
      devices: devices.toList(),
      backends: backends.toList(),
      cancelled: cancelled,
      fileName: fileName,
    );
  }
}
