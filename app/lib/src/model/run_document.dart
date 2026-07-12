import 'result_entry.dart';

/// Auto-scaled display value, e.g. 12500 gflops → "12.5 TFLOPS".
({String value, String unit}) formatMetric(double value, String unit) {
  String fmt(double v) => v >= 100
      ? v.toStringAsFixed(0)
      : v >= 10
          ? v.toStringAsFixed(1)
          : v.toStringAsFixed(2);
  switch (unit) {
    case 'gflops':
      if (value >= 1000) return (value: fmt(value / 1000), unit: 'TFLOPS');
      return (value: fmt(value), unit: 'GFLOPS');
    case 'tflops':
      return (value: fmt(value), unit: 'TFLOPS');
    case 'gops':
      if (value >= 1000) return (value: fmt(value / 1000), unit: 'TOPS');
      return (value: fmt(value), unit: 'GOPS');
    case 'tops':
      return (value: fmt(value), unit: 'TOPS');
    case 'gbps':
      return (value: fmt(value), unit: 'GB/s');
    case 'us':
      return (value: fmt(value), unit: 'µs');
    case 'ns':
      return (value: fmt(value), unit: 'ns');
    default:
      return (value: fmt(value), unit: unit);
  }
}

/// For latency units lower is better — bars and "peak" picking invert.
bool isLowerBetter(String unit) => unit == 'us' || unit == 'ns';

/// One test's rows on one device (all metric variants).
class TestResult {
  TestResult({required this.test, required this.display, required this.unit});

  final String test;
  String display; // human-readable; falls back to the tag for loaded files
  final String unit;
  final List<ResultEntry> metrics = [];

  /// Rows that produced a measurement.
  List<ResultEntry> get okMetrics =>
      metrics.where((m) => m.status == ResultStatus.ok).toList();

  bool get allSkipped => okMetrics.isEmpty;

  /// The reason shown for a fully-unsupported test.
  String get skipReason =>
      metrics.isEmpty ? '' : metrics.first.reason;

  /// Best value: max, or min for latency units.
  double get peakValue {
    final ok = okMetrics;
    if (ok.isEmpty) return 0;
    return isLowerBetter(unit)
        ? ok.map((m) => m.value).reduce((a, b) => a < b ? a : b)
        : ok.map((m) => m.value).reduce((a, b) => a > b ? a : b);
  }

  /// Largest value, used to normalize the per-metric bars.
  double get maxValue {
    final ok = okMetrics;
    if (ok.isEmpty) return 0;
    return ok.map((m) => m.value).reduce((a, b) => a > b ? a : b);
  }
}

class CategoryGroup {
  CategoryGroup(this.category);

  final BenchCategory category;
  final List<TestResult> tests = [];

  List<TestResult> get supported =>
      tests.where((t) => !t.allSkipped).toList();
  List<TestResult> get unsupported =>
      tests.where((t) => t.allSkipped).toList();
}

/// One backend/device run (one `<run>` block in the XML).
class DeviceRun {
  DeviceRun({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;

  /// Device props from the live event stream (empty for loaded files).
  List<({String key, String value})> props = [];

  final List<CategoryGroup> categories = [];

  String get key => '$backend|$platform|$device|$driver';

  CategoryGroup _category(BenchCategory c) =>
      categories.firstWhere((g) => g.category == c, orElse: () {
        final g = CategoryGroup(c);
        categories.add(g);
        return g;
      });

  TestResult _test(BenchCategory c, String test, String display, String unit) {
    final group = _category(c);
    for (final t in group.tests) {
      if (t.test == test) {
        if (t.display.isEmpty || t.display == t.test) {
          if (display.isNotEmpty) t.display = display;
        }
        return t;
      }
    }
    final t = TestResult(
        test: test, display: display.isEmpty ? test : display, unit: unit);
    group.tests.add(t);
    return t;
  }

  void addEntry(ResultEntry e) {
    _test(e.benchCategory, e.test, e.display, e.unit).metrics.add(e);
  }
}

/// A whole benchmark session: one or more device runs, in emission order.
class RunDocument {
  RunDocument();

  final List<DeviceRun> runs = [];

  bool get isEmpty => runs.isEmpty;

  DeviceRun runFor(String backend, String platform, String device,
      String driver) {
    final key = '$backend|$platform|$device|$driver';
    for (final r in runs) {
      if (r.key == key) return r;
    }
    final r = DeviceRun(
        backend: backend, platform: platform, device: device, driver: driver);
    runs.add(r);
    return r;
  }

  void addEntry(ResultEntry e) {
    runFor(e.backend, e.platform, e.device, e.driver).addEntry(e);
  }

  /// Build from a loaded saveJson document (history viewing).
  factory RunDocument.fromEntriesJson(Map<String, dynamic> doc) {
    final out = RunDocument();
    for (final e in (doc['entries'] as List? ?? const [])) {
      out.addEntry(ResultEntry.fromJson(e as Map<String, dynamic>));
    }
    return out;
  }
}
