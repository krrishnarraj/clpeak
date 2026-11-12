/// The measure engine (docs/format-v4.md, `dims`): what makes two readings
/// comparable, and each processor's best of every measure, over all its
/// backends and per backend, with the reading it came from.
///
/// Peaks are derived, never stored: a document holds readings, and this is
/// the one algorithm that turns them into views, so a newer rule applies to
/// an older file for free.  docs/format-v4.md fixes it, and an engine in the
/// CLI must agree with this one on every key and every best:
///
///   for each ok reading r on a device linked to processor P:
///       k = measureKey(r)                  # family | quantity | identity dims
///       best[P][k]            = better(best[P][k], r)       # by direction
///       best[P][k][r.backend] = better(best[P][k][r.backend], r)
///   headline h = better of best[P][k] over the keys h selects
///
/// Pure Dart over a [RunDocument]: it works on imported files and without
/// the native library, as history does.
library;

import 'headlines.dart';
import 'result_model.dart';
import 'run_document.dart';
import 'vocabulary.dart';

/// A dim's value as a measure key spells it: a string as written, a number
/// as a decimal (a whole number without a fraction, as the native writer
/// writes it), anything else as its JSON-ish text.
String dimValueText(Object v) {
  if (v is String) return v;
  if (v is int) return '$v';
  if (v is double) {
    return v.isFinite && v == v.truncateToDouble() ? '${v.toInt()}' : '$v';
  }
  return '$v';
}

/// The identity dims of [dims] that differ from their defaults, for a
/// reading of [family], sorted by key: what a measure key carries.  Variant
/// keys go, and so does every key at its default; a key this build does not
/// know stays, since it is identity (docs/format-v4.md).
///
/// [dims] are a reading's effective dims (its test's under its own), or a
/// headline's dims object, which is normalised the same way to be compared.
Map<String, String> identityDims(Map<String, Object> dims, String family) {
  final dtypeValue = dims['dtype'];
  final dtype = dtypeValue == null ? '' : dimValueText(dtypeValue);
  final keys = dims.keys.where(isIdentityDim).toList()..sort();
  final out = <String, String>{};
  for (final key in keys) {
    final text = dimValueText(dims[key]!);
    if (text == dimDefaultFor(key, family, dtype)) continue;
    out[key] = text;
  }
  return out;
}

/// What a reading measured, in terms every backend shares: its family, its
/// quantity, and its identity dims away from their defaults.  Two readings
/// with the same key -- and, from one backend's test, the same revision --
/// are comparable across backends, devices and runs.
class MeasureKey {
  MeasureKey(this.family, this.quantity, Map<String, String> identity)
      : identity = Map.unmodifiable(identity),
        id = [
          family,
          quantity,
          for (final e in identity.entries) '${e.key}=${e.value}',
        ].join('|');

  final String family;
  final String quantity;

  /// Sorted by key.
  final Map<String, String> identity;

  /// `matrix-compute|flops|dtype=fp16`,
  /// `matrix-compute|flops|dtype=nvfp4|sparsity=2:4`.
  final String id;

  @override
  bool operator ==(Object other) => other is MeasureKey && other.id == id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => id;
}

/// One ok reading, where the engine found it.
class MeasuredReading {
  MeasuredReading({
    required this.device,
    required this.test,
    required this.metric,
    required this.key,
    required this.dims,
    required this.dtype,
    required this.direction,
  });

  final DeviceRun device;
  final TestResult test;
  final MetricResult metric;
  final MeasureKey key;

  /// Its effective dims: its test's, with its own over them.
  final Map<String, Object> dims;

  /// The data type it computed on -- its dims', else its family's -- or
  /// empty where neither says.
  final String dtype;
  final Direction direction;

  double get value => metric.value;
  String get backend => device.backend;
  int get revision => test.header.revision;
  Units get units => test.unitsOf(metric);
}

/// A fixed order over the readings on one processor -- by backend, the
/// backend's device index, test key, then reading id, each by code unit --
/// unique there, since those four are a reading's identity in a run.  It
/// settles what value cannot, so a best never depends on how the readings
/// were walked, here or in any other engine.
int readingOrder(MeasuredReading a, MeasuredReading b) {
  var c = a.backend.compareTo(b.backend);
  if (c != 0) return c;
  c = a.device.index.compareTo(b.device.index);
  if (c != 0) return c;
  c = a.test.key.compareTo(b.test.key);
  if (c != 0) return c;
  return a.metric.id.compareTo(b.metric.id);
}

/// Whether [a] is a better reading than [b] of one measure: by direction --
/// that of whichever comes first in [readingOrder], since two readings of
/// one key disagree only in a file that contradicts itself -- and on a tie
/// in value, whichever comes first.
bool isBetter(MeasuredReading a, MeasuredReading b) {
  final order = readingOrder(a, b);
  if (a.value != b.value) {
    final direction = order <= 0 ? a.direction : b.direction;
    return direction == Direction.lowerIsBetter
        ? a.value < b.value
        : a.value > b.value;
  }
  return order < 0;
}

/// The key, and the reading that measured it [ok] on a device, or null when
/// it has none: a reading that did not succeed, has no family, or no finite
/// value.
MeasuredReading? measureReading(
    DeviceRun device, TestResult test, MetricResult m) {
  if (!m.isOk || !m.value.isFinite) return null;
  final family = m.benchmark.isNotEmpty ? m.benchmark : test.header.benchmark;
  if (family.isEmpty) return null;
  final dims = <String, Object>{...test.header.dims, ...m.dims};
  final dtypeValue = dims['dtype'];
  final dtype = dtypeValue != null
      ? dimValueText(dtypeValue)
      : (families[family]?.dtype ?? '');
  return MeasuredReading(
    device: device,
    test: test,
    metric: m,
    key: MeasureKey(family, test.unitsOf(m).quantityTag,
        identityDims(dims, family)),
    dims: dims,
    dtype: dtype,
    direction: m.direction ?? test.header.direction,
  );
}

/// One measure on one processor: its best reading over every backend, and
/// each backend's best.
class MeasureBest {
  MeasureBest(this.key, MeasuredReading first)
      : best = first,
        byBackend = {first.backend: first};

  final MeasureKey key;
  MeasuredReading best;

  /// By backend, in the order they first measured it.
  final Map<String, MeasuredReading> byBackend;

  void add(MeasuredReading r) {
    if (isBetter(r, best)) best = r;
    final mine = byBackend[r.backend];
    if (mine == null || isBetter(r, mine)) byBackend[r.backend] = r;
  }
}

/// A headline's value on one processor: the best reading of the keys it
/// selects, and each backend's best of them.
class HeadlineValue {
  const HeadlineValue(this.headline, this.best, this.byBackend);

  final Headline headline;
  final MeasuredReading best;
  final Map<String, MeasuredReading> byBackend;

  /// The backends' bests, best first.
  List<MeasuredReading> get ranked => byBackend.values.toList()
    ..sort((a, b) => isBetter(a, b) ? -1 : (isBetter(b, a) ? 1 : 0));
}

/// Everything measured on one processor -- or, in a file whose devices are
/// not linked to processors, on one device, which is then all the engine
/// can say is one piece of hardware.
class ProcessorMeasures {
  ProcessorMeasures._(this.id, this.processor, this.devices);

  /// The processor's id, or for an unlinked device its key.
  final String id;

  /// Null for an unlinked device.
  final ProcessorInfo? processor;

  /// Every backend's view of it, in run order.
  final List<DeviceRun> devices;

  /// By [MeasureKey.id], in the order the keys were first measured.
  final Map<String, MeasureBest> byKey = {};

  bool get linked => processor != null;

  /// `cpu`, `gpu`, `npu`, `accelerator`: the processor's kind, or an
  /// unlinked device's type.
  String get kind =>
      processor?.kind ?? (devices.isEmpty ? '' : devices.first.type);

  String get name =>
      processor?.name ?? (devices.isEmpty ? '' : devices.first.device);

  void _add(MeasuredReading r) {
    final best = byKey[r.key.id];
    if (best == null) {
      byKey[r.key.id] = MeasureBest(r.key, r);
    } else {
      best.add(r);
    }
  }

  /// [h]'s value here, or null when it does not apply to this kind of
  /// processor or nothing here measured a key it selects.
  HeadlineValue? headline(Headline h) {
    if (!h.appliesTo(kind)) return null;
    MeasuredReading? best;
    final byBackend = <String, MeasuredReading>{};
    for (final m in byKey.values) {
      if (!h.selects(m.key)) continue;
      if (best == null || isBetter(m.best, best)) best = m.best;
      for (final r in m.byBackend.values) {
        final mine = byBackend[r.backend];
        if (mine == null || isBetter(r, mine)) byBackend[r.backend] = r;
      }
    }
    return best == null ? null : HeadlineValue(h, best, byBackend);
  }
}

/// A document's measures, per processor.
class RunMeasures {
  RunMeasures._(this.groups);

  /// The document's processors in its order, each with the devices that
  /// name it -- a processor nothing measured included -- then each device
  /// linked to none, on its own.
  final List<ProcessorMeasures> groups;

  ProcessorMeasures? forProcessor(String id) {
    for (final g in groups) {
      if (g.linked && g.id == id) return g;
    }
    return null;
  }

  factory RunMeasures.of(RunDocument doc) {
    final groups = <ProcessorMeasures>[];
    final byProcessor = <String, ProcessorMeasures>{};
    for (final p in doc.processors) {
      if (byProcessor.containsKey(p.id)) continue;
      final g = ProcessorMeasures._(p.id, p, []);
      byProcessor[p.id] = g;
      groups.add(g);
    }
    for (final run in doc.runs) {
      var group = byProcessor[run.processor];
      if (group == null) {
        // Not linked, or linked to a processor the file does not list.
        group = ProcessorMeasures._(run.key, null, []);
        groups.add(group);
      }
      group.devices.add(run);
      for (final test in run.tests) {
        for (final m in test.metrics) {
          final r = measureReading(run, test, m);
          if (r != null) group._add(r);
        }
      }
    }
    return RunMeasures._(groups);
  }
}

// ── Words ──────────────────────────────────────────────────────────────────

/// A data type as a person reads it: `FP16`, `FP8 E4M3`, `NVFP4`, `INT8`.
String dataTypeLabel(String t) => t.toUpperCase().replaceAll('_', ' ');

/// What a reading measured, in a person's words: its data type, then what
/// sets it apart from its family's usual reading -- `[NVFP4, 2:4 structured
/// sparsity]`, `[FP16, FP16 accumulate]`, `[triad]`.  A reading with no
/// coordinates at all, from a backend that does not author them yet, is
/// named by its own label instead.
List<String> describeReading(MeasuredReading r) {
  final parts = <String>[if (r.dtype.isNotEmpty) dataTypeLabel(r.dtype)];
  final identity = r.key.identity;
  const lead = ['weights', 'kv', 'acc', 'quant', 'sparsity', 'scope'];
  for (final key in [
    ...lead.where(identity.containsKey),
    ...identity.keys.where((k) => !lead.contains(k)),
  ]) {
    final word = _dimWords(key, identity[key]!);
    if (word.isNotEmpty) parts.add(word);
  }
  if (r.dims.isEmpty && !(families[r.key.family]?.hasDtype ?? false)) {
    return [r.metric.displayLabel];
  }
  return parts;
}

String _dimWords(String key, String v) => switch (key) {
      'dtype' => '',
      'weights' => '${dataTypeLabel(v)} weights',
      'kv' => '${dataTypeLabel(v)} KV cache',
      'acc' => '${dataTypeLabel(v)} accumulate',
      'quant' => switch (v) {
          'qdq' => 'QDQ',
          'lut' => 'lookup-table weights',
          'block' => 'block-scaled',
          'channel' => 'per-channel scales',
          _ => v,
        },
      'sparsity' => v == '2:4' ? '2:4 structured sparsity' : '$v sparsity',
      'scope' => v == 'core' ? 'one core' : v,
      'level' => v.toUpperCase(),
      'direction' => switch (v) {
          'h2d' => 'host to device',
          'd2h' => 'device to host',
          'roundtrip' => 'round trip',
          _ => v,
        },
      'memory' => '$v memory',
      'context' => '$v tokens',
      'in_flight' => '$v in flight',
      'threads_per_core' => '$v threads per core',
      'contention' => v == 'all' ? 'contended' : v,
      'working_set' => _bytes(v),
      'op' || 'phase' || 'layer' || 'pattern' || 'pixel_format' =>
        v.replaceAll('_', ' '),
      'workload' => v.replaceAll('_', ' '),
      _ => '${key.replaceAll('_', ' ')} $v',
    };

String _bytes(String v) {
  final n = int.tryParse(v);
  if (n == null) return v;
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  var size = n.toDouble();
  var i = 0;
  while (size >= 1024 && i < units.length - 1) {
    size /= 1024;
    i++;
  }
  final text = size == size.truncateToDouble()
      ? '${size.toInt()}'
      : size.toStringAsFixed(1);
  return '$text ${units[i]}';
}
