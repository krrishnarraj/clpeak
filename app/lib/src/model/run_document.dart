import 'result_model.dart';

export 'result_model.dart' show formatValue;

/// One test's readings on one device.
///
/// Built once from a [TestHeader] — off `test_begin` during a live run, off
/// the saved test object when a run is reopened — so both paths produce the
/// same object and the UI has one thing to render.
class TestResult {
  TestResult(this.header);

  final TestHeader header;
  final List<MetricResult> metrics = [];

  String get id => header.id;
  String get key => header.key;
  String get title => header.title;
  String get variant => header.variant;
  String get axis => header.axis;
  String get description => header.description;
  TestShape get shape => header.shape;
  Units get units => header.units;

  /// Title plus the runtime qualifier that distinguishes two variants of the
  /// same test — a CPU test measured on SSE2 and on AVX2 is two rows, and
  /// without this they read as duplicates.
  String get displayTitle =>
      variant.isEmpty ? title : '$title  ·  $variant';

  bool get hasInfo => description.isNotEmpty;

  /// Whether any reading is documented — the breakdown reserves its glyph
  /// column per test, so the meters of undocumented rows stay aligned with
  /// the documented ones.
  bool get hasMetricNotes => metrics.any((m) => m.description.isNotEmpty);

  /// Readings that produced a measurement.
  List<MetricResult> get okMetrics => metrics.where((m) => m.isOk).toList();

  /// Readings that did not.  They are listed in the unavailable section
  /// rather than inline, so the table above holds only measurements.
  List<MetricResult> get unavailableMetrics =>
      metrics.where((m) => !m.isOk).toList();

  bool get allSkipped => okMetrics.isEmpty;

  /// The reason shown for a fully-unavailable test.
  String get skipReason => metrics.isEmpty ? '' : metrics.first.reason;

  /// Unit and direction for one reading, honouring its overrides.
  Units unitsOf(MetricResult m) => m.units ?? units;
  Direction directionOf(MetricResult m) => m.direction ?? header.direction;

  /// Whether one number can stand for the whole test.
  ///
  /// True for a homogeneous test, whose readings are variants of one
  /// measurement — and for any test down to a single reading, where a
  /// one-row table would say nothing the row does not.
  bool get collapsible =>
      shape == TestShape.homogeneous || okMetrics.length <= 1;

  /// The test's answer: best of the readings, by its own direction.  Only
  /// meaningful when [collapsible]; a heterogeneous test has no single
  /// answer, which is the whole reason for the distinction.
  double get peakValue {
    final ok = okMetrics;
    if (ok.isEmpty) return 0;
    return header.direction == Direction.lowerIsBetter
        ? ok.map((m) => m.value).reduce((a, b) => a < b ? a : b)
        : ok.map((m) => m.value).reduce((a, b) => a > b ? a : b);
  }

  /// How full to draw one reading's meter, relative to the best reading of
  /// the test.  Direction-aware: on a latency test the *fastest* reading is
  /// the full bar, where a plain value/max would have filled the slowest.
  double barFraction(MetricResult m) {
    if (!m.isOk) return 0;
    final ok = okMetrics.map((x) => x.value).where((v) => v > 0).toList();
    if (ok.isEmpty) return 0;
    if (directionOf(m) == Direction.lowerIsBetter) {
      final best = ok.reduce((a, b) => a < b ? a : b);
      if (m.value <= 0) return 0;
      return (best / m.value).clamp(0.0, 1.0);
    }
    final best = ok.reduce((a, b) => a > b ? a : b);
    if (best <= 0) return 0;
    return (m.value / best).clamp(0.0, 1.0);
  }
}

class CategoryGroup {
  CategoryGroup(this.category);

  final BenchCategory category;
  final List<TestResult> tests = [];

  List<TestResult> get supported => tests.where((t) => !t.allSkipped).toList();
  List<TestResult> get unsupported => tests.where((t) => t.allSkipped).toList();
}

/// One reading that could not be taken, with enough breadcrumbs to name it in
/// the unavailable section.
class UnavailableItem {
  const UnavailableItem({
    required this.test,
    required this.label,
    required this.status,
    required this.reason,
    required this.description,
  });

  final TestResult test;

  /// Empty when the whole test is unavailable — the test's own name says it,
  /// and repeating every reading under it would bury the one fact that
  /// matters.
  final String label;

  final ResultStatus status;
  final String reason;
  final String description;

  /// "MPS GEMM peak › bf16" for one missing reading, or just the test's name
  /// when the whole test is unavailable.
  String get title =>
      label.isEmpty ? test.displayTitle : '${test.displayTitle} › $label';
}

/// One backend/device run (one `devices[]` entry in the file).
class DeviceRun {
  DeviceRun({
    required this.backend,
    required this.platform,
    required this.device,
    required this.driver,
    this.index = -1,
  });

  final String backend;
  final String platform;
  final String device;
  final String driver;

  /// Enumeration index within the backend, or -1 where the backend reports
  /// none.  Part of the identity, because a name is not one: MoltenVK exposes
  /// the same GPU twice, and a multi-GPU box has N identical cards.
  final int index;

  /// Device props — from the live event stream during a run, and from the
  /// file's device object when a saved run is reopened.
  List<({String key, String value})> props = [];

  final List<CategoryGroup> categories = [];

  /// Driver is metadata, not identity — a saved run stays comparable across a
  /// driver update — but the index is, so two same-named devices stay two rows
  /// instead of folding into one test with two of every reading.
  String get key => '$backend|$platform|$device|#$index';

  CategoryGroup _category(BenchCategory c) =>
      categories.firstWhere((g) => g.category == c, orElse: () {
        final g = CategoryGroup(c);
        categories.add(g);
        return g;
      });

  /// Find or create the test this header describes.  A header that names an
  /// already-open test reopens it — the native side does that to append
  /// readings measured in a later category phase, and the two halves belong
  /// in one row.
  TestResult openTest(TestHeader header) {
    final group = _category(header.category);
    for (final t in group.tests) {
      if (t.key == header.key) return t;
    }
    final t = TestResult(header);
    group.tests.add(t);
    return t;
  }

  TestResult? findTest(String testKey) {
    for (final group in categories) {
      for (final t in group.tests) {
        if (t.key == testKey) return t;
      }
    }
    return null;
  }

  /// Everything that could not be measured on this device, in category order:
  /// whole tests first, then the individual readings missing from tests that
  /// otherwise succeeded.  Both belong in the same section — a reader wants
  /// one answer to "what did this device not do?".
  List<UnavailableItem> get unavailable {
    final out = <UnavailableItem>[];
    for (final g in categories) {
      for (final t in g.tests) {
        if (t.allSkipped) {
          final first = t.metrics.isEmpty ? null : t.metrics.first;
          out.add(UnavailableItem(
            test: t,
            label: '',
            status: first?.status ?? ResultStatus.unsupported,
            reason: first?.reason ?? '',
            description: t.description,
          ));
          continue;
        }
        for (final m in t.unavailableMetrics) {
          out.add(UnavailableItem(
            test: t,
            label: m.displayLabel,
            status: m.status,
            reason: m.reason,
            description: m.description,
          ));
        }
      }
    }
    return out;
  }
}

/// A note the run emitted outside any reading — a missing library, a driver
/// warning.  Usually the only record of *why* something is absent.
class RunNote {
  const RunNote({this.backend = '', this.device = '', required this.message});

  final String backend;
  final String device;
  final String message;
}

/// How the run was invoked and what it ran on.  Present for a saved run;
/// filled in only at the end of a live one, so null while it is in flight.
class RunMeta {
  const RunMeta({
    this.clpeakVersion = '',
    this.generatedAt = '',
    this.durationSeconds = 0,
    this.cancelled = false,
    this.host = const {},
  });

  final String clpeakVersion;
  final String generatedAt;
  final double durationSeconds;

  /// A cancelled run is a partial one; without this every test it never
  /// reached would read as hardware that lacks the feature.
  final bool cancelled;

  /// Free-form host facts (os, os_version, arch, cpu, logical_cores,
  /// memory_bytes) — shown as-is rather than modelled field by field, since
  /// which of them a platform can answer varies.
  final Map<String, dynamic> host;

  factory RunMeta.fromJson(Map<String, dynamic> m) => RunMeta(
        clpeakVersion: m['clpeak_version'] as String? ?? '',
        generatedAt: m['generated_at'] as String? ?? '',
        durationSeconds: (m['duration_s'] as num?)?.toDouble() ?? 0,
        cancelled: m['cancelled'] as bool? ?? false,
        host: (m['host'] as Map<String, dynamic>?) ?? const {},
      );
}

/// A whole benchmark session: one or more device runs, in emission order.
class RunDocument {
  RunDocument();

  final List<DeviceRun> runs = [];
  final List<RunNote> notes = [];
  RunMeta? meta;

  bool get isEmpty => runs.isEmpty;

  DeviceRun runFor(String backend, String platform, String device,
      String driver, [int index = -1]) {
    final key = '$backend|$platform|$device|#$index';
    for (final r in runs) {
      if (r.key == key) return r;
    }
    final r = DeviceRun(
        backend: backend,
        platform: platform,
        device: device,
        driver: driver,
        index: index);
    runs.add(r);
    return r;
  }

  /// Build from a saved clpeak run document (docs/format-v3.md).  A direct
  /// structural read: the file is already the shape the UI renders, so nothing
  /// is regrouped and no test-level field has to be recovered from its rows.
  factory RunDocument.fromJson(Map<String, dynamic> doc) {
    final out = RunDocument();
    out.meta = RunMeta.fromJson(doc);

    for (final d in (doc['devices'] as List? ?? const [])) {
      final dm = d as Map<String, dynamic>;
      final run = out.runFor(
        dm['backend'] as String? ?? '',
        dm['platform'] as String? ?? dm['backend'] as String? ?? '',
        dm['name'] as String? ?? '',
        dm['driver'] as String? ?? '',
        (dm['device_index'] as num?)?.toInt() ?? -1,
      );
      run.props = [
        for (final p in (dm['properties'] as List? ?? const []))
          (
            key: (p as Map<String, dynamic>)['key'] as String? ?? '',
            value: p['value'] as String? ?? '',
          ),
      ];
      for (final t in (dm['tests'] as List? ?? const [])) {
        final tm = t as Map<String, dynamic>;
        final test = run.openTest(TestHeader.fromJson(tm));
        for (final m in (tm['metrics'] as List? ?? const [])) {
          test.metrics.add(MetricResult.fromJson(m as Map<String, dynamic>));
        }
      }
    }

    for (final n in (doc['notes'] as List? ?? const [])) {
      final nm = n as Map<String, dynamic>;
      out.notes.add(RunNote(
        backend: nm['backend'] as String? ?? '',
        device: nm['device'] as String? ?? '',
        message: nm['message'] as String? ?? '',
      ));
    }

    return out;
  }
}
