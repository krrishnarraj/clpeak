/// Two runs side by side (docs/format-v4.md, Two runs side by side): the
/// app's compare, as data for its screens to lay out.
///
/// What a compare says follows from what its two sides share:
///
/// - **One processor model, seen by the same backends** -- one machine
///   before and after a driver update: reading by reading, each device both
///   runs saw on it named by backend, device name and that name's
///   occurrence, then test key and reading id ([PairCompare.devices]).
/// - **One processor, its backends against each other** -- one run:
///   [BackendCompare].
/// - **Two processors** -- two runs, or two in one run: the measure keys
///   both have, each side's best with the backend that reached it, which
///   side won and by what factor, then the keys of one side alone; the
///   headlines first ([PairCompare]).  And the same through each backend
///   both were seen through, each side's best of that backend alone: an
///   M1 Pro's GPU and an RTX 5060 through Vulkan, then through OpenCL
///   ([PairCompare.backends]).
///
/// Processors pair by kind, then by model ([processorModel]), and a person
/// may pair them otherwise ([RunCompare.withPair]).  Where the two values
/// set side by side are readings of one backend's test at two revisions, a
/// compare says the method changed instead of giving a ratio.
///
/// Pure Dart over two [RunDocument]s from anywhere -- two runs from history,
/// a file imported from another machine, one file twice -- built on the
/// measure engine (measures.dart), whose keys and bests it sets side by
/// side, so a compare elsewhere that follows docs/format-v4.md gives the
/// same answers.  Numbers, sides, winners, factors and the reading's words
/// ([describeReading]); the words a screen puts around them are its own.
library;

import 'headlines.dart';
import 'measures.dart';
import 'result_model.dart';
import 'run_document.dart';
import 'run_summary.dart';
import 'vocabulary.dart';

/// The two sides of a compare: A, the run it was opened from or picked
/// first, and B.
enum CompareSide { a, b }

/// How a compare reads, from what its two sides share.
enum CompareMode {
  /// One processor model on both sides, and devices both runs saw on it:
  /// reading by reading ([PairCompare.devices]), as well as by measure.
  readings,

  /// One processor of one run, its backends against each other
  /// ([BackendCompare]).
  backends,

  /// Two processors: their measure keys side by side ([PairCompare.both]).
  processors,
}

/// Why two processors are side by side.
enum PairedBy {
  /// One kind, one model ([processorModel]).
  model,

  /// One kind, two models: the first of that kind on each side still free.
  kind,

  /// A person paired them ([RunCompare.withPair]), whatever their kinds.
  person,
}

// ── The two sides ──────────────────────────────────────────────────────────

/// One side of a compare: a run's document, what the measure engine made of
/// it, and its history row where it came from history.
class CompareRun {
  CompareRun(this.document, {this.summary})
      : measures = RunMeasures.of(document),
        _occurrence = _occurrencesOf(document);

  final RunDocument document;

  /// The run as History lists it (its name, its date); null for a document
  /// from anywhere else.
  final RunSummary? summary;

  final RunMeasures measures;

  final Map<DeviceRun, int> _occurrence;

  /// The class of each dim key the file declares (`dim_keys`).
  Map<String, String> get declaredDims => document.meta?.dimKeys ?? const {};

  /// [device]'s place among this run's devices of its backend and name,
  /// from 0 in file order.  With the backend and the name, it names a
  /// device across two runs, where an index would not: a second card shifts
  /// every index after it, and MoltenVK lists one GPU twice.
  int occurrenceOf(DeviceRun device) => _occurrence[device] ?? 0;

  static Map<DeviceRun, int> _occurrencesOf(RunDocument doc) {
    final seen = <String, int>{};
    final out = <DeviceRun, int>{};
    for (final r in doc.runs) {
      final name = '${r.backend}\u0000${r.device}';
      final n = seen[name] ?? 0;
      out[r] = n;
      seen[name] = n + 1;
    }
    return out;
  }
}

/// One side's value in a compare, with where it came from: its run,
/// backend, device, test, reading and the test's revision.
class CompareValue {
  /// A reading the measure engine keyed.
  CompareValue.of(this.run, MeasuredReading r)
      : device = r.device,
        test = r.test,
        metric = r.metric,
        measured = r;

  /// Any reading of a device of [run], keyed where it has a family.
  CompareValue.reading(this.run, this.device, this.test, this.metric)
      : measured = measureReading(device, test, metric,
            declared: run.declaredDims);

  final CompareRun run;
  final DeviceRun device;
  final TestResult test;
  final MetricResult metric;

  /// The reading as the measure engine keyed it; null for one with no
  /// family, which only reading by reading compares.
  final MeasuredReading? measured;

  double get value => metric.value;
  String get backend => device.backend;
  int get revision => test.header.revision;
  Units get units => test.unitsOf(metric);
  Direction get direction => metric.direction ?? test.header.direction;

  /// Its measure key, where it has one.
  MeasureKey? get key => measured?.key;

  /// What it measured, in a person's words ([describeReading]); a reading
  /// with no family by its own label.
  List<String> get label => measured == null
      ? [metric.displayLabel]
      : describeReading(measured!);
}

// ── What two values say ────────────────────────────────────────────────────

/// The factor under which --compare words two values "within 0.05%" rather
/// than naming a winner: one that would print as 1.000x (`verdictText`,
/// src/common/logger_text.cpp).
const _sameBelow = 1.0005;

/// Two values of one measure, side by side.
class Verdict {
  const Verdict._(
      {this.ratio, this.winner, this.factor, this.methodChanged = false});

  /// [a] (A's value) against [b] (B's), where [direction] says which way is
  /// better.  With [methodChanged] the two are not compared at all.
  factory Verdict.of(double a, double b, Direction direction,
      {bool methodChanged = false}) {
    if (methodChanged) return const Verdict._(methodChanged: true);
    if (a == b) return const Verdict._(ratio: 1, factor: 1);
    final higher = direction != Direction.lowerIsBetter;
    final winner =
        (higher ? a > b : a < b) ? CompareSide.a : CompareSide.b;
    if (a <= 0 || b <= 0) return Verdict._(winner: winner);
    return Verdict._(
      ratio: higher ? a / b : b / a,
      winner: winner,
      factor: a > b ? a / b : b / a,
    );
  }

  /// A's value against B's, turned by direction so that above 1 means A is
  /// better: A's over B's where higher is better, B's over A's where lower
  /// is.  Exactly 1 for equal values.  Null where the method changed, and
  /// where a value is not above zero, so that no ratio means anything.
  final double? ratio;

  /// The better side; null for a tie, and where the method changed.
  final CompareSide? winner;

  /// How many times better [winner] is: the larger value over the smaller,
  /// at least 1, and 1 for a tie.  Null where [ratio] is.
  final double? factor;

  /// The two are readings of one backend's test at two revisions
  /// (docs/format-v4.md, Revisions): measured two ways, so not compared.
  final bool methodChanged;

  bool get isTie => !methodChanged && winner == null;

  /// The two read the same, as --compare words them: a tie ("the same"),
  /// or a factor under 1.0005, which the CLI words "within 0.05%" rather
  /// than a winner its digits cannot show.  A screen marks neither side
  /// better then.  False where the method changed, and where a value is
  /// not above zero: a winner, and no factor.
  bool get readsSame {
    final f = factor;
    return isTie || (f != null && f < _sameBelow);
  }
}

/// The order the measure engine breaks ties in ([readingOrder]), over two
/// runs' values: by backend, device index, test key, then reading id.
int _valueOrder(CompareValue a, CompareValue b) {
  var c = a.backend.compareTo(b.backend);
  if (c != 0) return c;
  c = a.device.index.compareTo(b.device.index);
  if (c != 0) return c;
  c = a.test.key.compareTo(b.test.key);
  if (c != 0) return c;
  return a.metric.id.compareTo(b.metric.id);
}

/// Readings of one backend's test -- one backend, one test id, whatever the
/// variant -- at two revisions.
bool _methodChanged(CompareValue a, CompareValue b) =>
    a.backend == b.backend &&
    a.test.id == b.test.id &&
    a.revision != b.revision;

/// A's value against B's.  Two values of one measure disagree on direction
/// only where a file contradicts itself; then the first in the tie order
/// says, so swapping the sides only swaps the answer.
Verdict _judge(CompareValue a, CompareValue b) {
  final direction = a.direction == b.direction || _valueOrder(a, b) <= 0
      ? a.direction
      : b.direction;
  return Verdict.of(a.value, b.value, direction,
      methodChanged: _methodChanged(a, b));
}

// ── Pairing ────────────────────────────────────────────────────────────────

/// What two processors of two runs must share to be one part
/// (docs/format-v4.md, Models), as one string.  [g]'s kind, then:
///
/// - a CPU: its vendor, its name and the cores of each cluster, fastest
///   first (`8+2`; else its core count: an M1 Pro is `8+2` or `6+2`), and,
///   where the file has them, CPUID's family and model or the clusters'
///   MIDRs, which tell parts with one generic name apart;
/// - an integrated GPU: its package's name and its compute units;
/// - any other GPU: its PCI vendor and device ids, else its vendor and name;
/// - an NPU, or a processor of any other kind: its vendor, its name and its
///   package's name.
///
/// A device linked to no processor (a file older than linking) is a part of
/// its own: its type, then its API's PCI ids, else its name.  Names are
/// compared as the APIs spell them.
String processorModel(CompareRun run, ProcessorMeasures g) {
  final p = g.processor;
  if (p == null) {
    if (g.devices.isEmpty) return g.kind;
    final d = g.devices.first;
    final f = d.facts;
    return [
      g.kind,
      if (f.vendorId.isNotEmpty && f.deviceId.isNotEmpty)
        'pci:${f.vendorId}:${f.deviceId}'
      else
        d.device,
    ].join('|');
  }
  final f = p.facts;
  final package = run.document.packageById(p.package)?.name ?? '';
  switch (p.kind) {
    case 'cpu':
      final c = f.cpu;
      final cores = c.clusters.isNotEmpty
          ? c.clusters.map((k) => '${k.cores}').join('+')
          : (c.cores > 0 ? '${c.cores}' : '');
      final midrs = [
        for (final k in c.clusters)
          if (k.midr.isNotEmpty) k.midr,
      ];
      return [
        'cpu', p.vendor, p.name, cores,
        if (c.cpuidFamily >= 0) 'cpuid:${c.cpuidFamily}:${c.cpuidModel}',
        if (midrs.isNotEmpty) 'midr:${midrs.join('+')}',
      ].join('|');
    case 'gpu':
      if (f.integrated == true) {
        return [
          'gpu', 'integrated', package.isNotEmpty ? package : p.name,
          '${f.computeUnits}',
        ].join('|');
      }
      if (f.vendorId.isNotEmpty && f.deviceId.isNotEmpty) {
        return 'gpu|pci:${f.vendorId}:${f.deviceId}';
      }
      return ['gpu', p.vendor, p.name].join('|');
    default:
      return [p.kind, p.vendor, p.name, package].join('|');
  }
}

/// Two processors set side by side: one of A's and one of B's.
class ProcessorPair {
  const ProcessorPair(this.a, this.b, {this.by = PairedBy.person});

  final ProcessorMeasures a;
  final ProcessorMeasures b;
  final PairedBy by;
}

/// A's processors paired with B's, by kind, then by model.  Each of A's, in
/// A's order, takes the first of B's of its kind and model still free; then
/// the rest pair by kind alone, in order, A's first with B's first.  What
/// is left is unpaired.  Every group the measure engine made takes part: a
/// processor nothing measured, and a device linked to none (a part of its
/// own, by its type).  Pairs come in A's order.
List<ProcessorPair> pairProcessors(CompareRun a, CompareRun b) {
  final free = [...b.measures.groups];
  final modelOf = {for (final g in free) g: processorModel(b, g)};
  final paired = <ProcessorMeasures, ProcessorPair>{};
  for (final ga in a.measures.groups) {
    final model = processorModel(a, ga);
    final i =
        free.indexWhere((gb) => gb.kind == ga.kind && modelOf[gb] == model);
    if (i >= 0) {
      paired[ga] = ProcessorPair(ga, free.removeAt(i), by: PairedBy.model);
    }
  }
  for (final ga in a.measures.groups) {
    if (paired.containsKey(ga)) continue;
    final i = free.indexWhere((gb) => gb.kind == ga.kind);
    if (i >= 0) {
      paired[ga] = ProcessorPair(ga, free.removeAt(i), by: PairedBy.kind);
    }
  }
  return [for (final ga in a.measures.groups) ?paired[ga]];
}

// ── Rows ───────────────────────────────────────────────────────────────────

/// A run of rows of one category: how a compare's lists are grouped.
class CategoryRows<T> {
  const CategoryRows(this.category, this.rows);

  final BenchCategory category;
  final List<T> rows;
}

/// [rows], already in category order, grouped by category.
List<CategoryRows<T>> _grouped<T>(
    List<T> rows, BenchCategory Function(T) categoryOf) {
  final out = <CategoryRows<T>>[];
  for (final r in rows) {
    final c = categoryOf(r);
    if (out.isEmpty || out.last.category != c) out.add(CategoryRows(c, []));
    out.last.rows.add(r);
  }
  return out;
}

/// A key's category: its family's ([familyCategory]), or for a family this
/// build does not know, that of the test that measured it -- the first, in
/// the tie order, of [values] that says.
BenchCategory _categoryOf(MeasureKey key, List<CompareValue> values) {
  final tag = familyCategory(key.family);
  if (tag.isNotEmpty) return BenchCategory.fromTag(tag);
  for (final v in [...values]..sort(_valueOrder)) {
    final c = v.test.header.category;
    if (c != BenchCategory.unknown) return c;
  }
  return BenchCategory.unknown;
}

final _familyRanks = {
  for (final (i, f) in families.keys.indexed) f: i,
};

final _quantityRanks = {
  for (final (i, q)
      in Quantity.values.where((q) => q != Quantity.unknown).indexed)
    q.tag: i,
};

/// The order a compare lists measure keys in, within a category: by family
/// (the vocabulary's order, which is the `Benchmark` enum's; a family this
/// build does not know after them, by name), then by quantity (likewise),
/// then by identity dims -- the data type first, then the other keys by
/// name, a key one does not have before one it has, values in their list's
/// order (data types, a word key's words) or by number -- then by the key's
/// text.  A presentation order, the same for both sides of a compare.
int measureKeyOrder(MeasureKey a, MeasureKey b) {
  int rank(Map<String, int> ranks, String s) => ranks[s] ?? ranks.length;
  var c = rank(_familyRanks, a.family).compareTo(rank(_familyRanks, b.family));
  if (c != 0) return c;
  c = a.family.compareTo(b.family);
  if (c != 0) return c;
  c = rank(_quantityRanks, a.quantity)
      .compareTo(rank(_quantityRanks, b.quantity));
  if (c != 0) return c;
  c = a.quantity.compareTo(b.quantity);
  if (c != 0) return c;
  c = _identityOrder(a.identity, b.identity);
  if (c != 0) return c;
  return a.id.compareTo(b.id);
}

int _identityOrder(Map<String, String> a, Map<String, String> b) {
  final keys = {...a.keys, ...b.keys}.toList()
    ..sort((x, y) {
      final c = (x == 'dtype' ? 0 : 1).compareTo(y == 'dtype' ? 0 : 1);
      return c != 0 ? c : x.compareTo(y);
    });
  for (final k in keys) {
    final va = a[k], vb = b[k];
    if (va == vb) continue;
    if (va == null) return -1;
    if (vb == null) return 1;
    final c = _dimValueOrder(k, va, vb);
    if (c != 0) return c;
  }
  return 0;
}

int _dimValueOrder(String key, String a, String b) {
  final info = dimKeys[key];
  int inList(List<String> list) {
    final ia = list.indexOf(a), ib = list.indexOf(b);
    if (ia >= 0 && ib >= 0) return ia.compareTo(ib);
    if (ia >= 0) return -1;
    if (ib >= 0) return 1;
    return a.compareTo(b);
  }

  switch (info?.kind) {
    case DimKind.dataType:
      return inList(dataTypes);
    case DimKind.word:
      return inList(info!.words);
    default:
      final na = int.tryParse(a), nb = int.tryParse(b);
      if (na != null && nb != null) return na.compareTo(nb);
      return a.compareTo(b);
  }
}

int _rowOrder(BenchCategory ca, MeasureKey ka, BenchCategory cb, MeasureKey kb) {
  final c = ca.index.compareTo(cb.index);
  return c != 0 ? c : measureKeyOrder(ka, kb);
}

int _bestFirst(MeasuredReading a, MeasuredReading b) =>
    isBetter(a, b) ? -1 : (isBetter(b, a) ? 1 : 0);

/// One side's best of one measure key on one processor, with each backend's
/// best -- or, through one backend, that backend's.
class MeasureRow {
  MeasureRow._(this.key, this.best, this.backends, this.category);

  factory MeasureRow.of(CompareRun run, MeasureBest m) {
    final best = CompareValue.of(run, m.best);
    return MeasureRow._(
      m.key,
      best,
      [
        for (final r in m.byBackend.values.toList()..sort(_bestFirst))
          CompareValue.of(run, r),
      ],
      _categoryOf(m.key, [best]),
    );
  }

  /// [m] through [backend] alone, which measured it: that backend's best,
  /// the row's one backend.
  factory MeasureRow._through(CompareRun run, MeasureBest m, String backend) {
    final best = CompareValue.of(run, m.byBackend[backend]!);
    return MeasureRow._(m.key, best, [best], _categoryOf(m.key, [best]));
  }

  final MeasureKey key;

  /// The best over every backend: the processor's peak for the key.
  /// Through one backend ([SharedBackendCompare]), that backend's best.
  final CompareValue best;

  /// Each backend's best, best first, so [best]'s own backend leads;
  /// through one backend, that one.
  final List<CompareValue> backends;

  final BenchCategory category;

  List<String> get label => best.label;
}

/// One measure key both processors have: each side's best, with the
/// backend that reached it, and what the two say side by side.
class KeyCompare {
  KeyCompare._(this.key, this.a, this.b, this.verdict, this.category);

  final MeasureKey key;
  final MeasureRow a;
  final MeasureRow b;

  /// A's best against B's.
  final Verdict verdict;

  final BenchCategory category;

  /// What the key measured, in a person's words: the same on both sides,
  /// but for a reading with no coordinates, named by its own label; then the
  /// first in the tie order's.
  List<String> get label =>
      (_valueOrder(a.best, b.best) <= 0 ? a.best : b.best).label;
}

/// A headline on both processors: each side's value, the best of the keys
/// it selects, and each backend's best of them -- or, through one backend,
/// that backend's best of them alone.
class HeadlineCompare {
  HeadlineCompare._(this.headline, this.a, this.b, this.aBackends,
      this.bBackends, this.verdict);

  /// [h] from each side's value on its processor ([ha], [hb]): over every
  /// backend, or through [backend] alone (`HeadlineValue.byBackend`).
  factory HeadlineCompare._of(Headline h, CompareRun runA, HeadlineValue? ha,
      CompareRun runB, HeadlineValue? hb, {String? backend}) {
    MeasuredReading? best(HeadlineValue? v) =>
        backend == null ? v?.best : v?.byBackend[backend];
    List<MeasuredReading> ranked(HeadlineValue? v) => backend == null
        ? v?.ranked ?? const []
        : [?v?.byBackend[backend]];
    final ra = best(ha), rb = best(hb);
    final a = ra == null ? null : CompareValue.of(runA, ra);
    final b = rb == null ? null : CompareValue.of(runB, rb);
    return HeadlineCompare._(
      h,
      a,
      b,
      [for (final r in ranked(ha)) CompareValue.of(runA, r)],
      [for (final r in ranked(hb)) CompareValue.of(runB, r)],
      a != null && b != null ? _judge(a, b) : null,
    );
  }

  final Headline headline;

  /// Each side's value; null where that processor has none (a GPU's memory
  /// bandwidth against an NPU that no backend measures it on), or none
  /// through the one backend.
  final CompareValue? a;
  final CompareValue? b;

  /// Each backend's best, best first; through one backend, that one.
  final List<CompareValue> aBackends;
  final List<CompareValue> bBackends;

  /// A's value against B's; null unless both have one.  The two may be
  /// readings of different keys: a headline is the best of every key it
  /// selects (a GPU's float peak in NVFP4 against another's in FP16).
  final Verdict? verdict;
}

/// One reading both devices measured, reading by reading.
class ReadingCompare {
  ReadingCompare._(this.a, this.b, this.verdict);

  final CompareValue a;
  final CompareValue b;
  final Verdict verdict;

  /// The test's key (`id@variant`) and the reading's id, which name it on
  /// both sides.
  String get testKey => a.test.key;
  String get readingId => a.metric.id;
}

/// One device both runs saw on a processor of one model: one backend, one
/// device name, the same occurrence of it.  Reading by reading.
class DeviceCompare {
  DeviceCompare._(this.a, this.b, this.occurrence, this.readings, this.onlyA,
      this.onlyB);

  factory DeviceCompare._of(
      CompareRun runA, DeviceRun a, CompareRun runB, DeviceRun b) {
    final readings = <ReadingCompare>[];
    final onlyA = <CompareValue>[];
    final onlyB = <CompareValue>[];
    for (final ta in a.tests) {
      final tb = b.findTest(ta.key);
      for (final ma in ta.metrics) {
        if (!_measured(ma)) continue;
        final va = CompareValue.reading(runA, a, ta, ma);
        final mb = tb == null ? null : _metric(tb, ma.id);
        if (mb != null && _measured(mb)) {
          final vb = CompareValue.reading(runB, b, tb!, mb);
          readings.add(ReadingCompare._(va, vb, _judge(va, vb)));
        } else {
          onlyA.add(va);
        }
      }
    }
    for (final tb in b.tests) {
      final ta = a.findTest(tb.key);
      for (final mb in tb.metrics) {
        if (!_measured(mb)) continue;
        final ma = ta == null ? null : _metric(ta, mb.id);
        if (ma == null || !_measured(ma)) {
          onlyB.add(CompareValue.reading(runB, b, tb, mb));
        }
      }
    }
    return DeviceCompare._(
        a, b, runA.occurrenceOf(a), readings, onlyA, onlyB);
  }

  final DeviceRun a;
  final DeviceRun b;

  /// Which of its run's devices of this backend and name it is, from 0.
  final int occurrence;

  /// The readings both measured, in A's order.
  final List<ReadingCompare> readings;

  /// The readings only one side measured: a test or a reading the other
  /// does not have, or did not measure.
  final List<CompareValue> onlyA;
  final List<CompareValue> onlyB;

  String get backend => a.backend;
  String get name => a.device;

  static bool _measured(MetricResult m) => m.isOk && m.value.isFinite;

  static MetricResult? _metric(TestResult t, String id) {
    for (final m in t.metrics) {
      if (m.id == id) return m;
    }
    return null;
  }
}

// ── Two processors ─────────────────────────────────────────────────────────

/// The backends of [devices], in their order.
List<String> _backendsOf(Iterable<DeviceRun> devices) =>
    [...{for (final d in devices) d.backend}];

/// [catalog]'s headlines either processor has -- over every backend, or
/// through [backend] alone -- in the catalog's order.
List<HeadlineCompare> _headlineCompares(HeadlineCatalog catalog,
    CompareRun runA, ProcessorMeasures a, CompareRun runB, ProcessorMeasures b,
    {String? backend}) {
  bool has(HeadlineValue? v) =>
      v != null && (backend == null || v.byBackend.containsKey(backend));
  final out = <HeadlineCompare>[];
  for (final h in catalog.headlines) {
    final ha = a.headline(h);
    final hb = b.headline(h);
    if (!has(ha) && !has(hb)) continue;
    out.add(HeadlineCompare._of(h, runA, ha, runB, hb, backend: backend));
  }
  return out;
}

/// [a]'s measure keys against [b]'s -- each side's best over every backend,
/// or through [backend] alone -- as the keys both have, then those of one
/// side alone, each by category, then [measureKeyOrder].
({List<KeyCompare> both, List<MeasureRow> onlyA, List<MeasureRow> onlyB})
    _keyCompares(CompareRun runA, ProcessorMeasures a, CompareRun runB,
        ProcessorMeasures b,
        {String? backend}) {
  bool has(MeasureBest? m) =>
      m != null && (backend == null || m.byBackend.containsKey(backend));
  MeasureRow row(CompareRun run, MeasureBest m) => backend == null
      ? MeasureRow.of(run, m)
      : MeasureRow._through(run, m, backend);
  final both = <KeyCompare>[];
  final onlyA = <MeasureRow>[];
  final onlyB = <MeasureRow>[];
  for (final m in a.byKey.values) {
    if (!has(m)) continue;
    final rowA = row(runA, m);
    final mb = b.byKey[m.key.id];
    if (mb == null || !has(mb)) {
      onlyA.add(rowA);
      continue;
    }
    final rowB = row(runB, mb);
    both.add(KeyCompare._(m.key, rowA, rowB, _judge(rowA.best, rowB.best),
        _categoryOf(m.key, [rowA.best, rowB.best])));
  }
  for (final m in b.byKey.values) {
    if (has(m) && !has(a.byKey[m.key.id])) onlyB.add(row(runB, m));
  }
  both.sort((x, y) => _rowOrder(x.category, x.key, y.category, y.key));
  for (final rows in [onlyA, onlyB]) {
    rows.sort((x, y) => _rowOrder(x.category, x.key, y.category, y.key));
  }
  return (both: both, onlyA: onlyA, onlyB: onlyB);
}

/// Two processors side by side: one of A's, one of B's.
class PairCompare {
  PairCompare._({
    required this.runA,
    required this.a,
    required this.runB,
    required this.b,
    required this.pairedBy,
    required this.modelA,
    required this.modelB,
    required this.headlines,
    required this.both,
    required this.onlyA,
    required this.onlyB,
    required this.devices,
    required this.devicesOnlyA,
    required this.devicesOnlyB,
    required this.backends,
    required this.backendsOnlyA,
    required this.backendsOnlyB,
  });

  /// [a], a processor of [runA], against [b], one of [runB] -- the same run
  /// twice for two processors of one run -- with [catalog]'s headlines.
  factory PairCompare.of(CompareRun runA, ProcessorMeasures a,
      CompareRun runB, ProcessorMeasures b,
      {required HeadlineCatalog catalog, PairedBy by = PairedBy.person}) {
    final rows = _keyCompares(runA, a, runB, b);

    // Each backend both were seen through, side by side through it alone.
    final backendsA = _backendsOf(a.devices);
    final backendsB = _backendsOf(b.devices);

    // One model on both sides: the devices both runs saw on it compare
    // reading by reading too, named by backend, device name and occurrence.
    final modelA = processorModel(runA, a);
    final modelB = processorModel(runB, b);
    final devices = <DeviceCompare>[];
    final devicesOnlyA = <DeviceRun>[];
    final devicesOnlyB = <DeviceRun>[];
    if (modelA == modelB) {
      String name(CompareRun run, DeviceRun d) =>
          '${d.backend}\u0000${d.device}\u0000${run.occurrenceOf(d)}';
      final rest = {for (final d in b.devices) name(runB, d): d};
      for (final d in a.devices) {
        final other = rest.remove(name(runA, d));
        if (other == null) {
          devicesOnlyA.add(d);
        } else {
          devices.add(DeviceCompare._of(runA, d, runB, other));
        }
      }
      devicesOnlyB.addAll(rest.values);
    }

    return PairCompare._(
      runA: runA,
      a: a,
      runB: runB,
      b: b,
      pairedBy: by,
      modelA: modelA,
      modelB: modelB,
      headlines: _headlineCompares(catalog, runA, a, runB, b),
      both: rows.both,
      onlyA: rows.onlyA,
      onlyB: rows.onlyB,
      devices: devices,
      devicesOnlyA: devicesOnlyA,
      devicesOnlyB: devicesOnlyB,
      backends: [
        for (final name in backendsA)
          if (backendsB.contains(name))
            SharedBackendCompare._of(name, runA, a, runB, b, catalog),
      ],
      backendsOnlyA: [
        for (final name in backendsA)
          if (!backendsB.contains(name)) name,
      ],
      backendsOnlyB: [
        for (final name in backendsB)
          if (!backendsA.contains(name)) name,
      ],
    );
  }

  final CompareRun runA;
  final ProcessorMeasures a;
  final CompareRun runB;
  final ProcessorMeasures b;
  final PairedBy pairedBy;

  /// Each side's [processorModel].
  final String modelA;
  final String modelB;

  bool get sameModel => modelA == modelB;

  /// [CompareMode.readings] where the two are one model and both runs saw
  /// a device on it; else [CompareMode.processors].
  CompareMode get mode =>
      devices.isEmpty ? CompareMode.processors : CompareMode.readings;

  /// The catalog's headlines either side has, in the catalog's order.
  final List<HeadlineCompare> headlines;

  /// The measure keys both have, by category, then [measureKeyOrder].
  final List<KeyCompare> both;

  /// The keys of one side alone, in the same order.
  final List<MeasureRow> onlyA;
  final List<MeasureRow> onlyB;

  /// Where the two are one model: each device both runs saw on it, in A's
  /// order, reading by reading.  Empty for two models.
  final List<DeviceCompare> devices;

  /// Where the two are one model: the devices of one side the other did not
  /// see (a backend only one run had).  Empty for two models.
  final List<DeviceRun> devicesOnlyA;
  final List<DeviceRun> devicesOnlyB;

  /// Each backend both processors were seen through -- a device of it on
  /// each -- in A's run order, the two set side by side through it alone.
  final List<SharedBackendCompare> backends;

  /// The backends of one side's processor alone: A's in A's run order, B's
  /// in B's.
  final List<String> backendsOnlyA;
  final List<String> backendsOnlyB;

  List<CategoryRows<KeyCompare>> get bothByCategory =>
      _grouped(both, (r) => r.category);
  List<CategoryRows<MeasureRow>> get onlyAByCategory =>
      _grouped(onlyA, (r) => r.category);
  List<CategoryRows<MeasureRow>> get onlyBByCategory =>
      _grouped(onlyB, (r) => r.category);
}

/// Two processors side by side through one backend both were seen through:
/// each side's best of that backend alone, where [PairCompare.both] takes
/// each side's best of every backend.  An M1 Pro's GPU and an RTX 5060
/// through Vulkan: the M1 Pro's shader peak there, though Metal reaches
/// more, against the RTX's cooperative matrices, though CUDA reaches more.
///
/// The pair's rules, over the measure engine's bests of one backend
/// (`MeasureBest.byBackend`, `HeadlineValue.byBackend`): both values are
/// that backend's, so its test at two revisions reads "method changed".
class SharedBackendCompare {
  SharedBackendCompare._(this.backend, this.devicesA, this.devicesB,
      this.headlines, this.both, this.onlyA, this.onlyB);

  factory SharedBackendCompare._of(String backend, CompareRun runA,
      ProcessorMeasures a, CompareRun runB, ProcessorMeasures b,
      HeadlineCatalog catalog) {
    final rows = _keyCompares(runA, a, runB, b, backend: backend);
    return SharedBackendCompare._(
      backend,
      [for (final d in a.devices) if (d.backend == backend) d],
      [for (final d in b.devices) if (d.backend == backend) d],
      _headlineCompares(catalog, runA, a, runB, b, backend: backend),
      rows.both,
      rows.onlyA,
      rows.onlyB,
    );
  }

  final String backend;

  /// Its devices on each side's processor, in run order: one, or more where
  /// it sees the processor more than once (two OpenCL platforms on one
  /// CPU), whose best is the backend's.
  final List<DeviceRun> devicesA;
  final List<DeviceRun> devicesB;

  /// The catalog's headlines either side has through it, in the catalog's
  /// order: each side's value its best of the keys the headline selects.
  final List<HeadlineCompare> headlines;

  /// The measure keys both measured through it, by category, then
  /// [measureKeyOrder]: each side's row this backend's best of the key.
  final List<KeyCompare> both;

  /// The keys one side alone measured through it, in the same order --
  /// whatever the other measured through another backend.
  final List<MeasureRow> onlyA;
  final List<MeasureRow> onlyB;

  List<CategoryRows<KeyCompare>> get bothByCategory =>
      _grouped(both, (r) => r.category);
  List<CategoryRows<MeasureRow>> get onlyAByCategory =>
      _grouped(onlyA, (r) => r.category);
  List<CategoryRows<MeasureRow>> get onlyBByCategory =>
      _grouped(onlyB, (r) => r.category);
}

/// Two runs side by side: their processors paired, and each pair compared.
class RunCompare {
  RunCompare._({
    required this.a,
    required this.b,
    required this.catalog,
    required this.pairs,
    required this.unpairedA,
    required this.unpairedB,
    required this.backends,
    required this.backendsOnlyA,
    required this.backendsOnlyB,
    required this.unpairedDevicesA,
    required this.unpairedDevicesB,
  });

  /// [a] against [b] -- two runs, or one twice -- with [catalog]'s
  /// headlines.  The processors pair as [pairProcessors] pairs them, or as
  /// [pairs] says: each of A's and B's groups (`measures.groups`) in at most
  /// one pair, whatever their kinds.
  factory RunCompare.of(CompareRun a, CompareRun b,
      {required HeadlineCatalog catalog, List<ProcessorPair>? pairs}) {
    final pairing = pairs ?? pairProcessors(a, b);
    final orderA = {for (final (i, g) in a.measures.groups.indexed) g: i};
    final orderB = {for (final (i, g) in b.measures.groups.indexed) g: i};
    final usedA = <ProcessorMeasures>{};
    final usedB = <ProcessorMeasures>{};
    for (final p in pairing) {
      if (!orderA.containsKey(p.a) || !orderB.containsKey(p.b)) {
        throw ArgumentError('a pair holds a processor of neither run');
      }
      if (!usedA.add(p.a) || !usedB.add(p.b)) {
        throw ArgumentError('a processor is in two pairs');
      }
    }
    final sorted = [...pairing]
      ..sort((x, y) => orderA[x.a]!.compareTo(orderA[y.a]!));
    final compares = [
      for (final p in sorted)
        PairCompare.of(a, p.a, b, p.b, catalog: catalog, by: p.by),
    ];

    // Every device of a backend both runs have is set side by side through
    // it by a pair, or unpaired.
    final backendsA = _backendsOf(a.document.runs);
    final backendsB = _backendsOf(b.document.runs);
    final pairedA = <DeviceRun>{
      for (final p in compares)
        for (final x in p.backends) ...x.devicesA,
    };
    final pairedB = <DeviceRun>{
      for (final p in compares)
        for (final x in p.backends) ...x.devicesB,
    };
    return RunCompare._(
      a: a,
      b: b,
      catalog: catalog,
      pairs: compares,
      unpairedA: [
        for (final g in a.measures.groups)
          if (!usedA.contains(g)) g,
      ],
      unpairedB: [
        for (final g in b.measures.groups)
          if (!usedB.contains(g)) g,
      ],
      backends: [
        for (final name in backendsA)
          if (backendsB.contains(name)) name,
      ],
      backendsOnlyA: [
        for (final name in backendsA)
          if (!backendsB.contains(name)) name,
      ],
      backendsOnlyB: [
        for (final name in backendsB)
          if (!backendsA.contains(name)) name,
      ],
      unpairedDevicesA: [
        for (final d in a.document.runs)
          if (backendsB.contains(d.backend) && !pairedA.contains(d)) d,
      ],
      unpairedDevicesB: [
        for (final d in b.document.runs)
          if (backendsA.contains(d.backend) && !pairedB.contains(d)) d,
      ],
    );
  }

  final CompareRun a;
  final CompareRun b;
  final HeadlineCatalog catalog;

  /// One per pair, in A's processor order.
  final List<PairCompare> pairs;

  /// The processors no pair holds: A's in A's order, B's in B's.
  final List<ProcessorMeasures> unpairedA;
  final List<ProcessorMeasures> unpairedB;

  /// The backends both runs have, in A's run order: those a pair can set
  /// side by side through ([PairCompare.backends]).
  final List<String> backends;

  /// The backends of one run alone: A's in A's run order, B's in B's.
  final List<String> backendsOnlyA;
  final List<String> backendsOnlyB;

  /// The devices of a backend both runs have that no pair sets side by side
  /// through it, in their run's order: on a processor whose partner has no
  /// device of that backend -- the Threadripper's llvmpipe through Vulkan,
  /// against an M1 Pro whose CPU no Vulkan device reaches -- or on one no
  /// pair holds.
  final List<DeviceRun> unpairedDevicesA;
  final List<DeviceRun> unpairedDevicesB;

  /// The pairing, as [RunCompare.of] takes it back.
  List<ProcessorPair> get pairing =>
      [for (final p in pairs) ProcessorPair(p.a, p.b, by: p.pairedBy)];

  /// The pair holding [g], one of A's processors; null where none does.
  PairCompare? pairOf(ProcessorMeasures g) {
    for (final p in pairs) {
      if (identical(p.a, g)) return p;
    }
    return null;
  }

  /// B against A: the same pairs, each turned round, through each backend
  /// too, and each side's backends and devices with them.
  RunCompare swapped() => RunCompare.of(b, a,
      catalog: catalog,
      pairs: [
        for (final p in pairs) ProcessorPair(p.b, p.a, by: p.pairedBy),
      ]);

  /// Re-paired by a person: A's processor [fromA] with B's [toB], whatever
  /// their kinds.  Whatever either was paired with before is left unpaired.
  RunCompare withPair(ProcessorMeasures fromA, ProcessorMeasures toB) =>
      RunCompare.of(a, b, catalog: catalog, pairs: [
        for (final p in pairs)
          if (!identical(p.a, fromA) && !identical(p.b, toB))
            ProcessorPair(p.a, p.b, by: p.pairedBy),
        ProcessorPair(fromA, toB, by: PairedBy.person),
      ]);

  /// Without the pair holding A's processor [fromA]: both left unpaired.
  RunCompare withoutPair(ProcessorMeasures fromA) =>
      RunCompare.of(a, b, catalog: catalog, pairs: [
        for (final p in pairs)
          if (!identical(p.a, fromA)) ProcessorPair(p.a, p.b, by: p.pairedBy),
      ]);
}

// ── One processor's backends ───────────────────────────────────────────────

/// One measure key on one processor, against its backends.
class BackendRow {
  BackendRow._(this.key, this.best, this.byBackend, this.category);

  final MeasureKey key;

  /// The best over every backend.
  final CompareValue best;

  /// Each backend's best, in [BackendCompare.backends]' order: only those
  /// that measured the key.
  final Map<String, CompareValue> byBackend;

  final BenchCategory category;

  /// More than one backend measured it: something to set side by side.
  bool get shared => byBackend.length > 1;

  /// How many times better [best] is than [backend]'s best: 1 for the
  /// best's own backend.  Null where [backend] did not measure the key, or
  /// a value is not above zero.
  double? behind(String backend) {
    final v = byBackend[backend];
    if (v == null) return null;
    return _judge(best, v).factor;
  }

  List<String> get label => best.label;
}

/// A headline on one processor, against its backends.
class BackendHeadline {
  BackendHeadline._(this.headline, this.best, this.byBackend);

  final Headline headline;
  final CompareValue best;

  /// Each backend's best, in [BackendCompare.backends]' order.
  final Map<String, CompareValue> byBackend;
}

/// One processor of one run, its backends against each other: the RTX 5060
/// through CUDA, Vulkan, OpenCL, LiteRT and ONNX Runtime.
class BackendCompare {
  BackendCompare._(
      this.run, this.processor, this.backends, this.headlines, this.rows);

  factory BackendCompare.of(CompareRun run, ProcessorMeasures processor,
      {required HeadlineCatalog catalog}) {
    final backends = _backendsOf(processor.devices);
    Map<String, CompareValue> inOrder(Map<String, MeasuredReading> m) => {
          for (final name in backends)
            if (m[name] case final r?) name: CompareValue.of(run, r),
        };
    final rows = [
      for (final m in processor.byKey.values)
        BackendRow._(m.key, CompareValue.of(run, m.best), inOrder(m.byBackend),
            _categoryOf(m.key, [CompareValue.of(run, m.best)])),
    ]..sort((x, y) => _rowOrder(x.category, x.key, y.category, y.key));
    return BackendCompare._(
      run,
      processor,
      backends,
      [
        for (final h in catalog.headlines)
          if (processor.headline(h) case final v?)
            BackendHeadline._(h, CompareValue.of(run, v.best),
                inOrder(v.byBackend)),
      ],
      rows,
    );
  }

  /// Every processor of [run] that two or more backends measured.
  static List<BackendCompare> ofRun(CompareRun run,
          {required HeadlineCatalog catalog}) =>
      [
        for (final g in run.measures.groups)
          if ({for (final m in g.byKey.values) ...m.byBackend.keys}.length > 1)
            BackendCompare.of(run, g, catalog: catalog),
      ];

  final CompareRun run;
  final ProcessorMeasures processor;

  /// Its backends, in run order.
  final List<String> backends;

  /// The catalog's headlines it has, in the catalog's order.
  final List<BackendHeadline> headlines;

  /// Every key measured on it, by category, then [measureKeyOrder].
  final List<BackendRow> rows;

  CompareMode get mode => CompareMode.backends;

  /// The keys more than one backend measured.
  List<BackendRow> get shared => [for (final r in rows) if (r.shared) r];

  List<CategoryRows<BackendRow>> get byCategory =>
      _grouped(rows, (r) => r.category);
}
