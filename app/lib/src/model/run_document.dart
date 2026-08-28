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

/// For latency units — and for `ppm`, the ONNX numeric-error unit, where the
/// reading is a distance from the right answer — lower is better; bars and
/// "peak" picking invert.
bool isLowerBetter(String unit) =>
    unit == 'us' || unit == 'ns' || unit == 'ppm';

/// One test's rows on one device (all metric variants).
class TestResult {
  TestResult({required this.test, required this.display, required this.unit});

  final String test;
  String display; // human-readable; falls back to the tag for loaded files
  final String unit;
  final List<ResultEntry> metrics = [];

  /// What the test measures, shown behind the info affordance on its row.
  /// Every row of the test repeats it; empty for tests with none authored.
  /// A reading's own note is not here — it stays on its
  /// [ResultEntry.metricDescription], where the breakdown row reads it.
  String description = '';

  bool get hasInfo => description.isNotEmpty;

  /// Whether any reading is documented — the breakdown reserves its glyph
  /// column per test, so the meters of undocumented rows stay aligned with
  /// the documented ones.
  bool get hasMetricNotes =>
      metrics.any((m) => m.metricDescription.isNotEmpty);

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

  /// Device props — from the live event stream during a run, and from the
  /// file's `devices` block when a saved run is reopened.
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
    final t = _test(e.benchCategory, e.test, e.display, e.unit);
    t.metrics.add(e);
    // The test description rides the rows, not the `test_begin` event: rows
    // are the only source a reopened file has, and taking it off the event
    // would mean materializing a TestResult before its first measurement,
    // which `CategoryGroup` reads as fully-skipped and lists under "not
    // supported" until a row lands.  First non-empty wins, like `display`.
    if (t.description.isEmpty && e.description.isNotEmpty) {
      t.description = e.description;
    }
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
    // Applied after the entries so run order follows the measurements, and so
    // a device block with no rows doesn't create an empty run.  Absent for
    // CSV and for files saved before device metadata was persisted.
    for (final d in (doc['devices'] as List? ?? const [])) {
      final m = d as Map<String, dynamic>;
      final key = [
        m['backend'] ?? '',
        m['platform'] ?? '',
        m['device'] ?? '',
        m['driver'] ?? '',
      ].join('|');
      for (final run in out.runs) {
        if (run.key != key) continue;
        run.props = [
          for (final p in (m['props'] as List? ?? const []))
            (
              key: (p as Map<String, dynamic>)['k'] as String? ?? '',
              value: p['v'] as String? ?? '',
            ),
        ];
        break;
      }
    }
    return out;
  }
}
