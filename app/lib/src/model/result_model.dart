/// Mirrors the native result model (include/common/run_document.h) — the
/// vocabulary shared by the live event stream and the saved JSON document.
library;

import 'dart:math' as math;

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

  /// Shown beside a reading in the unavailable section.
  String get label => switch (this) {
        ResultStatus.ok => 'ok',
        ResultStatus.unsupported => 'unsupported',
        ResultStatus.skipped => 'skipped',
        ResultStatus.error => 'error',
      };
}

/// The canonical categories (stable vocabulary; tests within them churn).
enum BenchCategory {
  fpCompute('fp_compute', 'FP Compute'),
  intCompute('int_compute', 'Integer Compute'),
  crypto('crypto', 'Crypto'),
  string('string', 'String'),
  bandwidth('bandwidth', 'Bandwidth'),
  latency('latency', 'Latency'),
  ai('ai', 'AI'),
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

/// Whether a test's readings are comparable to one another.
///
/// `homogeneous` readings are interchangeable variants of one measurement
/// (float / float2 / float4), so the best of them is the test's answer and the
/// row may collapse to that number.  `heterogeneous` readings each measure
/// something different (cuBLASLt's nine datatypes, L1/L2/L3/DRAM latency, h2d
/// vs d2h), so there is no single answer and every reading is shown.
///
/// Authored natively at the `beginTest()` call site — it cannot be inferred
/// from the readings or the unit.  Heterogeneous is the default, so a test
/// nobody has classified yet is verbose rather than wrong.
enum TestShape {
  homogeneous,
  heterogeneous;

  static TestShape fromString(String s) =>
      s == 'homogeneous' ? TestShape.homogeneous : TestShape.heterogeneous;
}

/// Which way is an improvement.  Throughput rises, latency and numeric error
/// fall; the native unit table decides, and a test may override it.
enum Direction {
  higherIsBetter,
  lowerIsBetter;

  static Direction fromString(String s) => s == 'lower_is_better'
      ? Direction.lowerIsBetter
      : Direction.higherIsBetter;
}

/// What a reading measures.  Only used to decide whether a magnitude has an
/// SI ladder to slide along: a ratio or a bare count does not.
enum Quantity {
  flops('flops'),
  ops('ops'),
  bytesPerSecond('bytes_per_second'),
  seconds('seconds'),
  itemsPerSecond('items_per_second'),
  ratio('ratio'),
  count('count'),
  unknown('unknown');

  const Quantity(this.tag);
  final String tag;

  static Quantity fromTag(String tag) => Quantity.values
      .firstWhere((q) => q.tag == tag, orElse: () => Quantity.unknown);

  bool get scalable => switch (this) {
        Quantity.ratio || Quantity.count || Quantity.unknown => false,
        _ => true,
      };
}

/// How to print a reading, and how to compare it to one measured elsewhere.
///
/// `symbol` is the unit as authored ("TFLOPS", "GB/s", "µs") and `scale`
/// multiplies a value into that quantity's SI base unit.  clpeak reports
/// GFLOPS in one test and TFLOPS in the next, so a bare number means nothing
/// on its own — normalizing through `scale` is what lets one formatter serve
/// every test instead of a switch over unit strings.
class Units {
  const Units({
    required this.symbol,
    required this.quantity,
    required this.scale,
  });

  final String symbol;
  final Quantity quantity;
  final double scale;

  static const empty =
      Units(symbol: '', quantity: Quantity.unknown, scale: 1);

  /// From either an event or a saved test/metric object — the field names are
  /// the same in both.
  factory Units.fromJson(Map<String, dynamic> m) => Units(
        symbol: m['unit'] as String? ?? '',
        quantity: Quantity.fromTag(m['quantity'] as String? ?? ''),
        scale: (m['scale'] as num?)?.toDouble() ?? 1,
      );
}

/// SI prefixes, smallest first.  A unit symbol is `<prefix><base>`, so
/// rescaling is a matter of swapping the prefix — which works for GFLOPS →
/// TFLOPS, GB/s → TB/s, GTexel/s → TTexel/s and ns → µs alike, without a
/// table of every unit clpeak might one day report.
const List<(int, String)> _siPrefixes = [
  (-9, 'n'),
  (-6, 'µ'),
  (-3, 'm'),
  (0, ''),
  (3, 'k'),
  (6, 'M'),
  (9, 'G'),
  (12, 'T'),
  (15, 'P'),
];

/// Strip a leading SI prefix off a unit symbol: "TFLOPS" → (12, "FLOPS").
(int, String) _splitPrefix(String symbol) {
  for (final (exp, prefix) in _siPrefixes) {
    if (prefix.isEmpty) continue;
    if (symbol.length > prefix.length && symbol.startsWith(prefix)) {
      return (exp, symbol.substring(prefix.length));
    }
  }
  return (0, symbol);
}

String _prefixFor(int exp) =>
    _siPrefixes.firstWhere((p) => p.$1 == exp, orElse: () => (0, '')).$2;

/// Auto-scaled display value, e.g. 4476 GFLOPS → "4.48 TFLOPS", 1.5e-7 s →
/// "150 ns".  The same ladder as `formatScaledValue()` in
/// `src/common/units.cpp`, which is what the CLI's value column prints —
/// one algorithm, two presenters — and replaces the hard-coded per-unit
/// switch this used to be.
({String value, String unit}) formatValue(double value, Units units) {
  String fmt(double v) => v.abs() >= 100
      ? v.toStringAsFixed(0)
      : v.abs() >= 10
          ? v.toStringAsFixed(1)
          : v.toStringAsFixed(2);

  // Nothing to slide along: a ratio is a ratio, and a unit we don't recognise
  // is printed exactly as it was measured rather than guessed at.
  if (!units.quantity.scalable || value == 0 || !value.isFinite) {
    return (value: fmt(value), unit: units.symbol);
  }

  final (baseExp, base) = _splitPrefix(units.symbol);
  final si = value * math.pow(10, baseExp).toDouble();

  // Land the mantissa in [1, 1000) by picking the largest ladder step at or
  // below the value's own magnitude.
  var exp = ((math.log(si.abs()) / math.ln10) / 3).floor() * 3;
  final lo = _siPrefixes.first.$1;
  final hi = _siPrefixes.last.$1;
  exp = exp.clamp(lo, hi);

  final mantissa = si / math.pow(10, exp).toDouble();
  return (value: fmt(mantissa), unit: '${_prefixFor(exp)}$base');
}

/// One reading.
class MetricResult {
  const MetricResult({
    required this.id,
    this.label = '',
    this.status = ResultStatus.ok,
    this.value = 0,
    this.reason = '',
    this.description = '',
    this.units,
    this.direction,
  });

  /// Stable slug within the test, e.g. "fp8_e4m3".
  final String id;

  /// Display form; empty means the id is the label.
  final String label;

  final ResultStatus status;
  final double value; // meaningful only when status == ok
  final String reason; // populated only when status != ok

  /// What this one reading means, authored natively at its `emit()`/`skip()`
  /// call.  Empty for undocumented readings, which show no info affordance.
  final String description;

  /// Set only when this reading overrides its test's — the case that lets one
  /// test carry both TFLOPS and TOPS readings.
  final Units? units;
  final Direction? direction;

  bool get isOk => status == ResultStatus.ok;
  String get displayLabel => label.isEmpty ? id : label;

  /// One entry of a saved test's `metrics` array, or one `metric` event.
  factory MetricResult.fromJson(Map<String, dynamic> m) => MetricResult(
        // The file calls it "id"; the live event calls it "metric", since the
        // event also carries the test's id.
        id: (m['id'] ?? m['metric']) as String? ?? '',
        label: m['label'] as String? ?? '',
        // A reading with a value succeeded — the writer omits the obvious
        // "status": "ok" on every healthy row.
        status: m.containsKey('value')
            ? ResultStatus.ok
            : ResultStatus.fromString(m['status'] as String? ?? 'error'),
        value: (m['value'] as num?)?.toDouble() ?? 0,
        reason: m['reason'] as String? ?? '',
        // The file calls it "description"; the event calls it "minfo", to keep
        // it apart from the test's own "desc" on the same document.
        description:
            (m['description'] ?? m['minfo']) as String? ?? '',
        units: m.containsKey('unit') ? Units.fromJson(m) : null,
        direction: m.containsKey('direction')
            ? Direction.fromString(m['direction'] as String? ?? '')
            : null,
      );
}

/// Everything about a test except its readings — the part that arrives once,
/// on `test_begin` or as the head of a saved test object.
class TestHeader {
  const TestHeader({
    required this.id,
    required this.title,
    this.variant = '',
    this.axis = '',
    this.description = '',
    this.category = BenchCategory.unknown,
    this.shape = TestShape.heterogeneous,
    this.direction = Direction.higherIsBetter,
    this.units = Units.empty,
  });

  final String id;
  final String title;

  /// Runtime qualifier that is not part of the test's identity — the CPU
  /// backend's detected ISA, a GPU arch.  Two variants of one test are two
  /// rows, distinguished by this.
  final String variant;

  /// What varies from one reading to the next ("data type", "cache level").
  /// Heads the readings of a heterogeneous test.
  final String axis;

  final String description;
  final BenchCategory category;
  final TestShape shape;
  final Direction direction;
  final Units units;

  /// Identity within a device; mirrors the native TestResult::key().
  String get key => variant.isEmpty ? id : '$id@$variant';

  /// From a saved test object.
  factory TestHeader.fromJson(Map<String, dynamic> m) => TestHeader(
        id: m['id'] as String? ?? '',
        title: m['title'] as String? ?? m['id'] as String? ?? '',
        variant: m['variant'] as String? ?? '',
        axis: m['axis'] as String? ?? '',
        description: m['description'] as String? ?? '',
        category: BenchCategory.fromTag(m['category'] as String? ?? ''),
        shape: TestShape.fromString(m['shape'] as String? ?? ''),
        direction: Direction.fromString(m['direction'] as String? ?? ''),
        units: Units.fromJson(m),
      );

  /// From a `test_begin` / `test_skipped` event, which names the test `test`
  /// and its documentation `desc`.
  factory TestHeader.fromEvent(Map<String, dynamic> m) => TestHeader(
        id: m['test'] as String? ?? '',
        title: m['title'] as String? ?? m['test'] as String? ?? '',
        variant: m['variant'] as String? ?? '',
        axis: m['axis'] as String? ?? '',
        description: m['desc'] as String? ?? '',
        category: BenchCategory.fromTag(m['category'] as String? ?? ''),
        shape: TestShape.fromString(m['shape'] as String? ?? ''),
        direction: Direction.fromString(m['direction'] as String? ?? ''),
        units: Units.fromJson(m),
      );
}
