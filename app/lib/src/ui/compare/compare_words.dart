/// The words History and the compare screen put around the model's: how a
/// person names a run, a family and this machine, and the
/// compare's one reading of the two runs it was given ([isSameHardware]).
/// Everything a compare decides -- pairing, winners, ratios, revisions --
/// is lib/src/model/compare.dart's; this only names things.
library;

import 'dart:io' show Platform;

import '../../model/compare.dart';
import '../../model/measures.dart';
import '../../model/run_document.dart';
import '../../model/run_summary.dart';
import '../common/format.dart';

/// Two runs of the same hardware: every pair is one model compared reading
/// by reading ([PairCompare.mode]), and no processor of either run is left
/// unpaired.  Anything else -- two machines, or one machine with a
/// processor the other run did not see -- reads as different hardware.
bool isSameHardware(RunCompare c) =>
    c.pairs.isNotEmpty &&
    c.unpairedA.isEmpty &&
    c.unpairedB.isEmpty &&
    c.pairs.every((p) => p.mode == CompareMode.readings);

/// This machine, as its owner calls it ("this Mac", "this PC"), as the
/// sidebar's recent runs word it; [capital] for a chip or a tag.
String thisMachine({bool capital = false}) {
  final w = Platform.isMacOS
      ? 'this Mac'
      : Platform.isWindows
          ? 'this PC'
          : Platform.isLinux
              ? 'this computer'
              : 'this device';
  return capital ? '${w[0].toUpperCase()}${w.substring(1)}' : w;
}

/// A saved run's name: the one its user gave it, else its machine's, else
/// its timestamp id -- as the sidebar's recent runs name it.
String runName(RunSummary run) => run.name.isNotEmpty
    ? run.name
    : (run.systemName.isNotEmpty ? run.systemName : run.displayTitle);

/// One side of a compare by name: its History row's, else its file's.
String sideName(CompareRun side) {
  final s = side.summary;
  if (s != null) return runName(s);
  final name = side.document.meta?.systemName ?? '';
  return name.isNotEmpty ? name : 'Run';
}

/// When a side ran, from its History row, else from its file; null where
/// neither says.
DateTime? sideStarted(CompareRun side) =>
    side.summary?.startedAt ??
    DateTime.tryParse(side.document.meta?.generatedAt ?? '')?.toLocal();

const _months = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', //
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

const monthNames = [
  'January', 'February', 'March', 'April', 'May', 'June', //
  'July', 'August', 'September', 'October', 'November', 'December',
];

/// `19 Nov`.
String dayMonth(DateTime t) => '${t.day} ${_months[t.month - 1]}';

/// `04:43`.
String clock(DateTime t) =>
    '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}';

/// `19 Nov 2026, 04:43`.
String dayAndTime(DateTime t) => '${formatDay(t)}, ${clock(t)}';

/// What tells two runs of one machine apart in a column head and a fold:
/// their days (`19 Nov`), or on one day their times (`19 Nov, 04:43`).
(String, String) sideWhen(CompareRun a, CompareRun b) {
  final ta = sideStarted(a), tb = sideStarted(b);
  if (ta == null || tb == null) return (sideName(a), sideName(b));
  final oneDay =
      ta.year == tb.year && ta.month == tb.month && ta.day == tb.day;
  return oneDay
      ? ('${dayMonth(ta)}, ${clock(ta)}', '${dayMonth(tb)}, ${clock(tb)}')
      : (dayMonth(ta), dayMonth(tb));
}

/// A family (a Benchmark, by its flag name) as a person reads it; a family
/// this build does not know, as the file spells it.
///
/// The model names a measure by its family and [describeReading]'s words,
/// and has no titles of its own for families, so these are the screen's
/// until the vocabulary carries them (`families` in vocabulary.dart).
String familyTitle(String family) => _familyTitles[family] ?? family;

const _familyTitles = <String, String>{
  'single-precision-compute': 'Single-precision compute',
  'half-precision-compute': 'Half-precision compute',
  'double-precision-compute': 'Double-precision compute',
  'mixed-precision-compute': 'Mixed-precision compute',
  'bfloat16-compute': 'BF16 compute',
  'integer-compute': 'Integer compute',
  'integer-compute-fast': 'Integer compute, fast 24-bit',
  'integer-compute-char': 'Integer compute, 8-bit',
  'integer-compute-short': 'Integer compute, 16-bit',
  'int8-dot-product-compute': 'INT8 dot-product compute',
  'int16-dot-product-compute': 'INT16 dot-product compute',
  'fp8-dot-product-compute': 'FP8 dot-product compute',
  'divide-sqrt-compute': 'Divide and square root',
  'integer-divide-compute': 'Integer divide',
  'matrix-compute': 'Matrix engine',
  'gemm': 'Matrix multiply',
  'attention': 'Attention',
  'convolution': 'Convolution',
  'numeric-error': 'Numeric error',
  'smt-scaling': 'SMT scaling',
  'aes': 'AES-128 encrypt',
  'sha256': 'SHA-256',
  'sha512': 'SHA-512',
  'crc32c': 'CRC32-C',
  'string-scan': 'String scan',
  'utf8-validate': 'UTF-8 validate',
  'global-memory-bandwidth': 'Memory bandwidth',
  'local-memory-bandwidth': 'Local memory bandwidth',
  'image-memory-bandwidth': 'Image memory bandwidth',
  'transfer-bandwidth': 'Transfer bandwidth',
  'tensor-bandwidth': 'Resident-weight bandwidth',
  'activation': 'Activation throughput',
  'cache-bandwidth': 'Cache bandwidth',
  'texture-sample': 'Texture sampling',
  'kernel-launch-latency': 'Launch latency',
  'memory-latency': 'Memory latency',
  'atomics': 'Atomic latency',
  'branch-penalty': 'Branch mispredict',
  'store-forward': 'Store-to-load forwarding',
  'transformer-block': 'Transformer block',
};

/// A headline tile's caption: the data type the value was reached at, and
/// the backend that reached it (`FP16 · Metal`).  A value with no data type
/// (a bandwidth) says what it measured instead (`copy · CPU`).
String headCaption(CompareValue v) {
  final dtype = v.measured?.dtype ?? '';
  final what = dtype.isNotEmpty ? dataTypeLabel(dtype) : wordsOf(v);
  return [what, backendName(v.backend)].where((s) => s.isNotEmpty).join(' · ');
}

/// Two runs of one machine: whether a headline's two values are readings of
/// one measure key.  A headline is the best of every key it selects, so the
/// two runs' values may be different measures (one run's INT8 peak against
/// the other's INT16), which is then no compare of like with like.  Two
/// machines' headlines are set side by side whatever their keys: that is
/// what a headline is for.
bool likeForLike(HeadlineCompare h) {
  final a = h.a?.key, b = h.b?.key;
  return a != null && b != null && a.id == b.id;
}

/// Why [likeForLike] is false, in a person's words: `INT8 against INT16`,
/// by data type where the two differ in it, else by all they measured.
String unlikeWhy(CompareValue a, CompareValue b) {
  final da = a.measured?.dtype ?? '', db = b.measured?.dtype ?? '';
  if (da.isNotEmpty && db.isNotEmpty && da != db) {
    return '${dataTypeLabel(da)} against ${dataTypeLabel(db)}';
  }
  return '${wordsOf(a)} against ${wordsOf(b)}';
}

/// Why two processors are side by side, as the processors strip says it.
String pairedByText(PairedBy by) => switch (by) {
      PairedBy.model => 'same model',
      PairedBy.kind => 'paired by kind',
      PairedBy.person => 'paired by you',
    };

/// The kind of processor [device] of [run] is a view of: its processor's,
/// else the device's own type.
String deviceKind(CompareRun run, DeviceRun device) {
  for (final g in run.measures.groups) {
    if (g.devices.contains(device)) return g.kind;
  }
  return device.type;
}

/// `November 2026`, a History month's heading.
String monthTitle(DateTime t) => '${monthNames[t.month - 1]} ${t.year}';

/// A value and its unit as the results print them: `4.48` `TFLOPS`.
({String value, String unit}) valueText(CompareValue v) =>
    formatValue(v.value, v.units);

/// What a value measured, in the model's words: `FP16 · 2:4 structured
/// sparsity`.
String wordsOf(CompareValue v) => v.label.join(' · ');

/// Where a value came from: `CUDA · NVIDIA GeForce RTX 5060 · Tensor
/// cores (WMMA / mma.sync) · nvfp4 2:4`.
String sourceOf(CompareValue v) => [
      backendName(v.backend),
      v.device.device,
      v.test.header.title.isNotEmpty ? v.test.header.title : v.test.key,
      v.metric.id,
    ].where((s) => s.isNotEmpty).join(' · ');

/// A processor kind as a label says it: `CPU`, `GPU`, `NPU`.
String kindLabel(String kind) => kind.isEmpty ? 'DEVICE' : kind.toUpperCase();

/// `1 reading`, `12 readings`.
String plural(int n, String one, [String? many]) =>
    '$n ${n == 1 ? one : (many ?? '${one}s')}';

/// What a run is, in a run card's second line for two runs of one machine:
/// how long it took, how much it ran, and whether it was cut short.
String runNote(RunDocument doc) {
  final meta = doc.meta;
  final parts = <String>[];
  final seconds = meta?.durationSeconds ?? 0;
  if (seconds > 0) {
    parts.add(formatDuration(Duration(milliseconds: (seconds * 1000).round())));
  }
  final tests = [for (final d in doc.runs) d.tests.length];
  if (tests.isNotEmpty) {
    final most = tests.reduce((a, b) => a > b ? a : b);
    final all = tests.every((n) => n == most);
    parts.add(all
        ? '${plural(most, 'test')} per device'
        : 'up to ${plural(most, 'test')} per device');
  }
  if (meta?.cancelled ?? false) parts.add('cancelled');
  return parts.join(' · ');
}

/// The machine's OS and architecture, for a run card: `macOS 27.2 · arm64`.
String systemLine(RunDocument doc) {
  final s = doc.meta?.systemInfo;
  if (s == null) return '';
  final os = s.osDistribution.isNotEmpty
      ? s.osDistribution
      : [s.osName, s.osVersion].where((v) => v.isNotEmpty).join(' ');
  return [os, s.arch].where((v) => v.isNotEmpty).join(' · ');
}
