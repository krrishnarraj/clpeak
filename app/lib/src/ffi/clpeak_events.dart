import '../model/result_model.dart';

/// Decoded native run events — a 1:1 mirror of the JSON documents emitted by
/// LoggerFfi / clpeak_launch (see src/ffi/clpeak_ffi.h for the schema).
sealed class ClpeakEvent {
  const ClpeakEvent();

  static ClpeakEvent fromJson(Map<String, dynamic> m) {
    String s(String key) => m[key] as String? ?? '';
    int deviceIndex() => (m['device_index'] as num?)?.toInt() ?? -1;
    switch (m['t'] as String? ?? '') {
      case 'backend_begin':
        return BackendBeginEvent(s('backend'));
      case 'device':
        return DeviceEvent(
          backend: s('backend'),
          platform: s('platform'),
          device: s('device'),
          driver: s('driver'),
          deviceIndex: deviceIndex(),
          type: s('type'),
          platformIndex: (m['platform_index'] as num?)?.toInt() ?? -1,
          props: [
            for (final p in (m['props'] as List? ?? const []))
              (key: (p['k'] as String? ?? ''), value: (p['v'] as String? ?? ''))
          ],
        );
      case 'test_begin':
        return TestBeginEvent(
          backend: s('backend'),
          platform: s('platform'),
          device: s('device'),
          driver: s('driver'),
          deviceIndex: deviceIndex(),
          header: TestHeader.fromEvent(m),
        );
      case 'metric':
        return MetricEvent(
          backend: s('backend'),
          platform: s('platform'),
          device: s('device'),
          driver: s('driver'),
          deviceIndex: deviceIndex(),
          testKey: s('variant').isEmpty
              ? s('test')
              : '${s('test')}@${s('variant')}',
          metric: MetricResult.fromJson(m),
        );
      case 'test_skipped':
        return TestSkippedEvent(
          backend: s('backend'),
          platform: s('platform'),
          device: s('device'),
          driver: s('driver'),
          deviceIndex: deviceIndex(),
          header: TestHeader.fromEvent(m),
          metricNames: [
            for (final n in (m['metrics'] as List? ?? const []))
              n as String? ?? ''
          ],
          status: ResultStatus.fromString(s('status')),
          reason: s('reason'),
        );
      case 'test_end':
        return const TestEndEvent();
      case 'device_end':
        return const DeviceEndEvent();
      case 'backend_end':
        return const BackendEndEvent();
      case 'note':
        return NoteEvent(s('message'), backend: s('backend'), device: s('device'));
      case 'done':
        return DoneEvent(
          status: (m['status'] as num?)?.toInt() ?? 0,
          cancelled: m['cancelled'] as bool? ?? false,
        );
      default:
        return NoteEvent('unknown event: $m');
    }
  }
}

class BackendBeginEvent extends ClpeakEvent {
  const BackendBeginEvent(this.backend);
  final String backend;
}

class DeviceEvent extends ClpeakEvent {
  const DeviceEvent({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
    required this.deviceIndex,
    required this.type,
    required this.platformIndex,
    required this.props,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;
  final int deviceIndex;
  final String type; // "gpu" | "cpu" | "accelerator" | "unknown"
  final int platformIndex;
  final List<({String key, String value})> props;
}

/// A test opened.  Carries the whole resolved header — title, shape,
/// direction, unit — so the test's row exists before its first reading
/// and nothing has to be back-filled off the readings later.
class TestBeginEvent extends ClpeakEvent {
  const TestBeginEvent({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
    required this.deviceIndex,
    required this.header,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;
  final int deviceIndex;
  final TestHeader header;
}

class MetricEvent extends ClpeakEvent {
  const MetricEvent({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
    required this.deviceIndex,
    required this.testKey,
    required this.metric,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;
  final int deviceIndex;

  /// Which open test this reading belongs to — `id` or `id@variant`, matching
  /// TestResult.key.
  final String testKey;

  final MetricResult metric;
}

/// A whole test that could not run.  Unlike a per-reading skip it names every
/// reading that would have been taken, so the unavailable section can list
/// them instead of showing one nameless placeholder.
class TestSkippedEvent extends ClpeakEvent {
  const TestSkippedEvent({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
    required this.deviceIndex,
    required this.header,
    required this.metricNames,
    required this.status,
    required this.reason,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;
  final int deviceIndex;
  final TestHeader header;
  final List<String> metricNames;
  final ResultStatus status;
  final String reason;

  /// One reading per named metric, all carrying the same status and reason.
  /// Mirrors what the native side records, so a live run and a reopened file
  /// show the same rows.
  List<MetricResult> toMetrics() => [
        for (final name in metricNames)
          MetricResult(id: name, status: status, reason: reason),
      ];
}

class TestEndEvent extends ClpeakEvent {
  const TestEndEvent();
}

class DeviceEndEvent extends ClpeakEvent {
  const DeviceEndEvent();
}

class BackendEndEvent extends ClpeakEvent {
  const BackendEndEvent();
}

class NoteEvent extends ClpeakEvent {
  const NoteEvent(this.message, {this.backend = '', this.device = ''});
  final String message;
  final String backend;
  final String device;
}

class DoneEvent extends ClpeakEvent {
  const DoneEvent({required this.status, required this.cancelled});
  final int status;
  final bool cancelled;
}
