import '../model/result_entry.dart';

/// Decoded native run events — a 1:1 mirror of the JSON documents emitted by
/// LoggerFfi / clpeak_launch (see src/ffi/clpeak_ffi.h for the schema).
sealed class ClpeakEvent {
  const ClpeakEvent();

  static ClpeakEvent fromJson(Map<String, dynamic> m) {
    String s(String key) => m[key] as String? ?? '';
    switch (m['t'] as String? ?? '') {
      case 'backend_begin':
        return BackendBeginEvent(s('backend'));
      case 'device':
        return DeviceEvent(
          backend: s('backend'),
          platform: s('platform'),
          device: s('device'),
          driver: s('driver'),
          platformIndex: (m['platform_index'] as num?)?.toInt() ?? -1,
          deviceIndex: (m['device_index'] as num?)?.toInt() ?? -1,
          props: [
            for (final p in (m['props'] as List? ?? const []))
              (key: (p['k'] as String? ?? ''), value: (p['v'] as String? ?? ''))
          ],
        );
      case 'test_begin':
        return TestBeginEvent(
          backend: s('backend'),
          device: s('device'),
          test: s('test'),
          display: s('display'),
          unit: s('unit'),
          category: BenchCategory.fromTag(s('category')),
        );
      case 'metric':
        return MetricEvent(ResultEntry(
          backend: s('backend'),
          platform: s('platform'),
          device: s('device'),
          driver: s('driver'),
          category: s('category'),
          test: s('test'),
          display: s('display'),
          metric: s('metric'),
          unit: s('unit'),
          status: ResultStatus.fromString(s('status')),
          value: (m['value'] as num?)?.toDouble() ?? 0,
          reason: s('reason'),
        ));
      case 'test_skipped':
        return TestSkippedEvent(
          backend: s('backend'),
          platform: s('platform'),
          device: s('device'),
          driver: s('driver'),
          test: s('test'),
          display: s('display'),
          unit: s('unit'),
          category: s('category'),
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
        return NoteEvent(s('message'));
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
    required this.platformIndex,
    required this.deviceIndex,
    required this.props,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;
  final int platformIndex;
  final int deviceIndex;
  final List<({String key, String value})> props;
}

class TestBeginEvent extends ClpeakEvent {
  const TestBeginEvent({
    required this.backend,
    required this.device,
    required this.test,
    required this.display,
    required this.unit,
    required this.category,
  });

  final String backend;
  final String device;
  final String test;
  final String display;
  final String unit;
  final BenchCategory category;
}

class MetricEvent extends ClpeakEvent {
  const MetricEvent(this.entry);
  final ResultEntry entry;
}

class TestSkippedEvent extends ClpeakEvent {
  const TestSkippedEvent({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
    required this.test,
    required this.display,
    required this.unit,
    required this.category,
    required this.status,
    required this.reason,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;
  final String test;
  final String display;
  final String unit;
  final String category;
  final ResultStatus status;
  final String reason;

  /// As an unsupported ResultEntry (one representative row).
  ResultEntry toEntry() => ResultEntry(
        backend: backend,
        platform: platform,
        device: device,
        driver: driver,
        category: category,
        test: test,
        display: display,
        metric: '',
        unit: unit,
        status: status,
        reason: reason,
        value: 0,
      );
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
  const NoteEvent(this.message);
  final String message;
}

class DoneEvent extends ClpeakEvent {
  const DoneEvent({required this.status, required this.cancelled});
  final int status;
  final bool cancelled;
}
