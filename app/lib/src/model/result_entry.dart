/// Mirrors the native ResultEntry (include/common/result_store.h) — one
/// benchmark measurement, fully qualified by its provenance.
library;

enum ResultStatus {
  ok,
  unsupported,
  skipped,
  error;

  static ResultStatus fromString(String s) => switch (s) {
        'ok' => ResultStatus.ok,
        'unsupported' => ResultStatus.unsupported,
        'skipped' => ResultStatus.skipped,
        _ => ResultStatus.error,
      };
}

/// The six canonical categories (stable vocabulary; tests within them churn).
enum BenchCategory {
  fpCompute('fp_compute', 'FP Compute'),
  intCompute('int_compute', 'Integer'),
  crypto('crypto', 'Crypto'),
  string('string', 'String'),
  bandwidth('bandwidth', 'Bandwidth'),
  latency('latency', 'Latency'),
  unknown('unknown', 'Other');

  const BenchCategory(this.tag, this.label);

  /// Canonical lower-snake tag used by the native side and CLI flags.
  final String tag;
  final String label;

  static BenchCategory fromTag(String tag) => BenchCategory.values.firstWhere(
      (c) => c.tag == tag,
      orElse: () => BenchCategory.unknown);

  /// CLI flag name, e.g. "fp-compute" → --fp-compute / --no-fp-compute.
  String get flag => tag.replaceAll('_', '-');

  /// The user-selectable categories (excludes the `unknown` sentinel).
  static List<BenchCategory> get selectable =>
      values.where((c) => c != unknown).toList();
}

class ResultEntry {
  const ResultEntry({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
    required this.category,
    required this.test,
    required this.metric,
    required this.unit,
    required this.status,
    required this.value,
    required this.reason,
    this.display = '',
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;
  final String category; // canonical tag, e.g. "fp_compute"
  final String test; // canonical tag, e.g. "single_precision_compute"
  final String metric; // variant, e.g. "float ST"
  final String unit; // gflops | tflops | gops | tops | gbps | us | ns
  final ResultStatus status;
  final double value; // meaningful only when status == ok
  final String reason; // populated only when status != ok

  /// Human-readable test name. Present on live-run entries (from the event
  /// stream); empty for entries loaded from a saved file.
  final String display;

  BenchCategory get benchCategory => BenchCategory.fromTag(category);

  /// Provenance key matching the native ResultEntry::key() (driver excluded).
  String get key => '$backend/$platform/$device/$category/$test/$metric';

  /// One entry of the saveJson `entries` array.
  factory ResultEntry.fromJson(Map<String, dynamic> m) => ResultEntry(
        backend: m['backend'] as String? ?? '',
        platform: m['platform'] as String? ?? '',
        device: m['device'] as String? ?? '',
        driver: m['driver'] as String? ?? '',
        category: m['category'] as String? ?? '',
        test: m['test'] as String? ?? '',
        metric: m['metric'] as String? ?? '',
        unit: m['unit'] as String? ?? '',
        status: m.containsKey('value')
            ? ResultStatus.ok
            : ResultStatus.fromString(m['status'] as String? ?? 'error'),
        value: (m['value'] as num?)?.toDouble() ?? 0,
        reason: m['reason'] as String? ?? '',
      );
}
