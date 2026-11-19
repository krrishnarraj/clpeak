// The compare (lib/src/model/compare.dart) over the two hand-written v4
// fixtures in test/fixtures/ -- the M1 Pro laptop (soc_laptop) and sf1-ub's
// Threadripper with an RTX 5060 (desktop_two_packages) -- against answers
// worked out by hand from their readings: the pairing, each side's best,
// the winner and its factor, "method changed", and files brought in through
// History's import.
//
// Pure Dart: the documents are read with dart:convert, as history reads them.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:xpupeak/src/model/compare.dart';
import 'package:xpupeak/src/model/headlines.dart';
import 'package:xpupeak/src/model/measures.dart';
import 'package:xpupeak/src/model/result_model.dart';
import 'package:xpupeak/src/model/run_document.dart';
import 'package:xpupeak/src/model/vocabulary.dart';
import 'package:xpupeak/src/services/compare_loading.dart';
import 'package:xpupeak/src/services/run_history_store.dart';

String _text(String name) =>
    File('test/fixtures/$name.xpupeak.json').readAsStringSync();

CompareRun _run(String text) =>
    CompareRun(RunDocument.fromJson(jsonDecode(text) as Map<String, dynamic>));

final _laptop = _text('soc_laptop');
final _desktop = _text('desktop_two_packages');

final _catalog = HeadlineCatalog.fromJson(
    jsonDecode(File('assets/measures.json').readAsStringSync())
        as Map<String, dynamic>);

RunCompare _compare(String a, String b) =>
    RunCompare.of(_run(a), _run(b), catalog: _catalog);

PairCompare _pair(RunCompare c, String processor) =>
    c.pairs.firstWhere((p) => p.a.id == processor);

KeyCompare _row(PairCompare p, String key) =>
    p.both.firstWhere((r) => r.key.id == key);

SharedBackendCompare _through(PairCompare p, String backend) =>
    p.backends.firstWhere((x) => x.backend == backend);

/// Where a value came from, as `backend:index test "reading" rN`.
String _from(CompareValue v) =>
    '${v.backend}:${v.device.index} ${v.test.key} "${v.metric.id}" r${v.revision}';

/// A device, as `backend:index name`.
String _device(DeviceRun d) => '${d.backend}:${d.index} ${d.device}';

CompareSide? _flip(CompareSide? side) => switch (side) {
      CompareSide.a => CompareSide.b,
      CompareSide.b => CompareSide.a,
      null => null,
    };

/// [turned] is [v] with the sides swapped.
void _expectTurned(Verdict turned, Verdict v) {
  expect(turned.methodChanged, v.methodChanged);
  expect(turned.winner, _flip(v.winner));
  expect(turned.factor, v.factor);
  if (v.ratio != null) expect(turned.ratio! * v.ratio!, closeTo(1, 1e-12));
}

/// Every device of a backend both runs have is in one place: a pair's
/// compare through that backend, on its own side, or unpaired.
void _expectEveryDeviceOnce(RunCompare c) {
  for (final (run, through, unpaired) in [
    (
      c.a,
      [for (final p in c.pairs) for (final x in p.backends) ...x.devicesA],
      c.unpairedDevicesA,
    ),
    (
      c.b,
      [for (final p in c.pairs) for (final x in p.backends) ...x.devicesB],
      c.unpairedDevicesB,
    ),
  ]) {
    final shared = [
      for (final d in run.document.runs)
        if (c.backends.contains(d.backend)) d,
    ];
    expect(through.length + unpaired.length, shared.length);
    expect({...through, ...unpaired}, {...shared});
  }
}

/// Every verdict a compare holds: headlines, keys and readings, by
/// processor and through each backend.
Iterable<Verdict> _verdicts(RunCompare c) sync* {
  for (final p in c.pairs) {
    for (final h in p.headlines) {
      if (h.verdict != null) yield h.verdict!;
    }
    for (final r in p.both) {
      yield r.verdict;
    }
    for (final x in p.backends) {
      for (final h in x.headlines) {
        if (h.verdict != null) yield h.verdict!;
      }
      for (final r in x.both) {
        yield r.verdict;
      }
    }
    for (final d in p.devices) {
      for (final r in d.readings) {
        yield r.verdict;
      }
    }
  }
}

/// Every verdict of [c] is a tie -- and there is at least one.
void _expectAllTies(RunCompare c) {
  final all = _verdicts(c).toList();
  expect(all, isNotEmpty);
  expect(all.where((v) => !v.isTie), isEmpty);
}

/// A run of one GPU with one kernel-launch reading.
String _oneLatency(double value,
        {String direction = 'lower_is_better', int revision = 1}) =>
    jsonEncode({
      'schema': 'xpupeak/run',
      'format_version': 4,
      'processors': [
        {'id': 'gpu0', 'kind': 'gpu', 'name': 'GPU'},
      ],
      'devices': [
        {
          'backend': 'Vulkan',
          'name': 'GPU',
          'device_index': 0,
          'type': 'gpu',
          'processor': 'gpu0',
          'tests': [
            {
              'id': 'kernel_launch_latency',
              'benchmark': 'kernel-launch-latency',
              if (revision > 1) 'revision': revision,
              'category': 'latency',
              'direction': direction,
              'quantity': 'seconds',
              'unit': 's',
              'metrics': [
                {'id': 'dispatch', 'dims': {'op': 'dispatch'}, 'value': value},
              ],
            },
          ],
        },
      ],
    });

void main() {
  group('pairing', () {
    test('the M1 Pro against sf1-ub: by kind, the Neural Engine alone', () {
      final c = _compare(_laptop, _desktop);
      expect([for (final p in c.pairs) '${p.a.id}-${p.b.id}:${p.pairedBy.name}'],
          ['cpu0-cpu0:kind', 'gpu0-gpu0:kind']);
      expect([for (final g in c.unpairedA) g.id], ['npu0']);
      expect(c.unpairedB, isEmpty);
      // Two models: the keys compare, the devices do not.
      for (final p in c.pairs) {
        expect(p.sameModel, isFalse);
        expect(p.mode, CompareMode.processors);
        expect(p.devices, isEmpty);
      }
    });

    test('a model is read from the facts docs/format-v4.md names', () {
      final laptop = _run(_laptop);
      final desktop = _run(_desktop);
      String model(CompareRun run, String id) =>
          processorModel(run, run.measures.forProcessor(id)!);
      // A CPU by its cores per cluster, and CPUID where the file has it.
      expect(model(laptop, 'cpu0'), 'cpu|apple|Apple M1 Pro|8+2');
      expect(model(desktop, 'cpu0'),
          'cpu|amd|AMD Ryzen Threadripper PRO 3955WX 16-Cores|16|cpuid:23:49');
      // An integrated GPU by its package and cores; a card by its PCI ids.
      expect(model(laptop, 'gpu0'), 'gpu|integrated|Apple M1 Pro|16');
      expect(model(desktop, 'gpu0'), 'gpu|pci:10de:2d05');
      expect(model(laptop, 'npu0'),
          'npu|apple|Apple Neural Engine|Apple M1 Pro');
    });

    test('two runs of one machine pair by model', () {
      final c = _compare(_desktop, _desktop);
      expect([for (final p in c.pairs) '${p.a.id}-${p.b.id}:${p.pairedBy.name}'],
          ['cpu0-cpu0:model', 'gpu0-gpu0:model']);
    });

    test('a person re-pairs, whatever the kinds', () {
      final c = _compare(_laptop, _desktop);
      final npu = c.a.measures.forProcessor('npu0')!;
      final rtx = c.b.measures.forProcessor('gpu0')!;
      final r = c.withPair(npu, rtx);
      // In A's order; what the two were paired with is left alone.
      expect([for (final p in r.pairs) '${p.a.id}-${p.b.id}:${p.pairedBy.name}'],
          ['cpu0-cpu0:kind', 'npu0-gpu0:person']);
      expect([for (final g in r.unpairedA) g.id], ['gpu0']);
      expect(r.unpairedB, isEmpty);
      // The Neural Engine's fp16 GEMM through Core ML against the RTX's
      // through cuBLASLt.
      final gemm = _row(r.pairOf(npu)!, 'gemm|flops|dtype=fp16');
      expect(_from(gemm.a.best), 'CoreML:0 coreml_gemm "fp16" r1');
      expect(_from(gemm.b.best), 'CUDA:0 cublas_gemm "fp16" r1');
      expect(gemm.verdict.winner, CompareSide.b);
      expect(gemm.verdict.factor, closeTo(4.41e13 / 1.1e13, 1e-9));

      final apart = r.withoutPair(npu);
      expect([for (final p in apart.pairs) p.a.id], ['cpu0']);
      expect([for (final g in apart.unpairedA) g.id], ['gpu0', 'npu0']);
      expect([for (final g in apart.unpairedB) g.id], ['gpu0']);

      // The pairing goes back in as it came out.
      final again = RunCompare.of(c.a, c.b, catalog: _catalog, pairs: r.pairing);
      expect([for (final p in again.pairs) '${p.a.id}-${p.b.id}:${p.pairedBy.name}'],
          ['cpu0-cpu0:kind', 'npu0-gpu0:person']);
    });

    test('a pairing must hold each run\'s own processors, once each', () {
      final c = _compare(_laptop, _desktop);
      final cpuA = c.a.measures.forProcessor('cpu0')!;
      final cpuB = c.b.measures.forProcessor('cpu0')!;
      final gpuB = c.b.measures.forProcessor('gpu0')!;
      expect(
          () => RunCompare.of(c.a, c.b,
              catalog: _catalog, pairs: [ProcessorPair(cpuB, cpuA)]),
          throwsArgumentError);
      expect(
          () => RunCompare.of(c.a, c.b, catalog: _catalog, pairs: [
                ProcessorPair(cpuA, cpuB),
                ProcessorPair(cpuA, gpuB),
              ]),
          throwsArgumentError);
    });

    test('swapping the sides turns every answer round', () {
      final c = _compare(_laptop, _desktop);
      final s = c.swapped();
      expect(identical(s.a, c.b), isTrue);
      expect([for (final p in s.pairs) '${p.a.id}-${p.b.id}'],
          ['cpu0-cpu0', 'gpu0-gpu0']);
      // The desktop has no NPU: the laptop's is B's, unpaired.
      expect(s.unpairedA, isEmpty);
      expect([for (final g in s.unpairedB) g.id], ['npu0']);
      for (final (i, p) in c.pairs.indexed) {
        final q = s.pairs[i];
        expect([for (final r in q.both) r.key.id], [for (final r in p.both) r.key.id]);
        expect([for (final r in q.onlyA) r.key.id], [for (final r in p.onlyB) r.key.id]);
        for (final (j, r) in p.both.indexed) {
          final t = q.both[j].verdict;
          expect(t.winner, _flip(r.verdict.winner));
          expect(t.factor, r.verdict.factor);
          expect(t.ratio! * r.verdict.ratio!, closeTo(1, 1e-12));
        }
      }
    });
  });

  group('two processors', () {
    test('the GPUs: headlines first, then by category and family', () {
      final gpu = _pair(_compare(_laptop, _desktop), 'gpu0');
      expect([for (final h in gpu.headlines) h.headline.id],
          ['peak-float', 'peak-integer', 'peak-memory-bandwidth']);
      // Each side's float peak, whatever key reached it: Metal's fp16 ALUs
      // against the RTX's sparse NVFP4 tensor cores.
      final float = gpu.headlines.first;
      expect(_from(float.a!), 'Metal:0 half_precision_compute "half4" r1');
      expect(_from(float.b!), 'CUDA:0 wmma "nvfp4 2:4" r1');
      expect(float.b!.label, ['NVFP4', '2:4 structured sparsity']);
      expect(float.verdict!.winner, CompareSide.b);
      expect(float.verdict!.factor, closeTo(6.33e14 / 5.3e12, 1e-9));
      expect([for (final v in float.bBackends) v.backend],
          ['CUDA', 'Vulkan', 'ONNX', 'LiteRT', 'OpenCL']);

      expect([
        for (final g in gpu.bothByCategory)
          '${g.category.tag}: ${[for (final r in g.rows) r.key.id].join(', ')}'
      ], [
        'compute: single-precision-compute|flops, half-precision-compute|flops, '
            'matrix-compute|flops|dtype=fp16, '
            'matrix-compute|flops|acc=fp16|dtype=fp16, '
            'gemm|flops|dtype=fp32, gemm|flops|dtype=fp16',
        'bandwidth: global-memory-bandwidth|bytes_per_second',
        'latency: kernel-launch-latency|seconds|op=dispatch, '
            'kernel-launch-latency|seconds|op=roundtrip',
      ]);
      expect([for (final r in gpu.onlyA) r.key.id],
          ['integer-compute|ops', 'matrix-compute|flops|dtype=fp32']);
      expect([for (final r in gpu.onlyB) r.key.id], [
        'int8-dot-product-compute|ops',
        'matrix-compute|flops|dtype=tf32',
        'matrix-compute|flops|dtype=bf16',
        'matrix-compute|flops|dtype=fp8_e4m3',
        'matrix-compute|flops|dtype=fp8_e4m3|sparsity=2:4',
        'matrix-compute|flops|dtype=nvfp4',
        'matrix-compute|flops|dtype=nvfp4|sparsity=2:4',
        'matrix-compute|ops|dtype=int8',
        'matrix-compute|ops|dtype=int8|sparsity=2:4',
        'gemm|ops|dtype=int8',
        'gemm|ops|dtype=int8|quant=qdq',
        'transfer-bandwidth|bytes_per_second|direction=h2d|memory=pinned',
        'transfer-bandwidth|bytes_per_second|direction=d2h|memory=pinned',
      ]);
    });

    test('each value says where it came from', () {
      final c = _compare(_laptop, _desktop);
      final fp16 = _row(_pair(c, 'gpu0'), 'matrix-compute|flops|dtype=fp16');
      // The run, backend, device, test, reading and revision of each side.
      expect(identical(fp16.a.best.run, c.a), isTrue);
      expect(identical(fp16.b.best.run, c.b), isTrue);
      expect(fp16.a.best.device.device, 'Apple M1 Pro');
      expect(_from(fp16.a.best), 'Metal:0 simdgroup_matrix "fp16" r2');
      expect(_from(fp16.b.best), 'CUDA:0 wmma "fp16" r1');
      // Two backends' tests at two revisions still compare: only one
      // backend's test at two revisions is a changed method.
      expect(fp16.verdict.methodChanged, isFalse);
      expect(fp16.verdict.winner, CompareSide.b);
      expect(fp16.verdict.factor, closeTo(4.77e13 / 4.1e12, 1e-9));
      expect(fp16.label, ['FP16']);
      // Every backend's best on each side, best first.
      expect([for (final v in fp16.a.backends) '${v.backend} ${v.value}'],
          ['Metal 4100000000000.0', 'Vulkan 3600000000000.0']);
      expect([for (final v in fp16.b.backends) v.backend], ['CUDA', 'Vulkan']);
      expect(fp16.category, BenchCategory.compute);
    });

    test('the ratio is turned by direction so that above 1 is better', () {
      final cpu = _pair(_compare(_laptop, _desktop), 'cpu0');
      // Higher is better: the Threadripper's AVX2 all-core fp32 against the
      // M1 Pro's NEON.
      final sp = _row(cpu, 'single-precision-compute|flops');
      expect(_from(sp.a.best), 'CPU:0 single_precision_compute@NEON "float MT" r1');
      expect(_from(sp.b.best),
          'CPU:0 single_precision_compute@AVX2+FMA "float MT" r1');
      expect(sp.verdict.winner, CompareSide.b);
      expect(sp.verdict.ratio, closeTo(8.6e11 / 2.05e12, 1e-12));
      expect(sp.verdict.factor, closeTo(2.05e12 / 8.6e11, 1e-12));
      // Lower is better: DRAM latency goes to the Threadripper, 91 ns
      // against 96, and L1 to the M1 Pro, 0.94 ns against 1.
      final dram = _row(cpu, 'memory-latency|seconds|level=dram');
      expect(dram.verdict.winner, CompareSide.b);
      expect(dram.verdict.ratio, closeTo(9.1e-8 / 9.6e-8, 1e-12));
      expect(dram.verdict.ratio, lessThan(1));
      expect(dram.verdict.factor, closeTo(9.6e-8 / 9.1e-8, 1e-12));
      final l1 = _row(cpu, 'memory-latency|seconds|level=l1');
      expect(l1.verdict.winner, CompareSide.a);
      expect(l1.verdict.ratio, closeTo(1e-9 / 9.4e-10, 1e-12));
      expect(l1.verdict.ratio, greaterThan(1));
      // The GPUs' launch latency: CUDA's 2.4 µs against Metal's 5.2.
      final launch = _row(_pair(_compare(_laptop, _desktop), 'gpu0'),
          'kernel-launch-latency|seconds|op=dispatch');
      expect(launch.verdict.winner, CompareSide.b);
      expect(launch.verdict.factor, closeTo(5.2e-6 / 2.4e-6, 1e-12));
    });

    test('a lower-is-better reading wins by being lower', () {
      RunCompare c(String a, String b) => _compare(a, b);
      final slow = _oneLatency(2e-6), fast = _oneLatency(1e-6);
      final v = c(slow, fast).pairs.single.both.single.verdict;
      expect(v.winner, CompareSide.b);
      expect(v.factor, closeTo(2, 1e-12));
      expect(v.ratio, closeTo(0.5, 1e-12));
      expect(c(fast, slow).pairs.single.both.single.verdict.winner,
          CompareSide.a);
      // The same numbers where higher is better go the other way.
      final h = c(_oneLatency(2e-6, direction: 'higher_is_better'),
              _oneLatency(1e-6, direction: 'higher_is_better'))
          .pairs
          .single
          .both
          .single
          .verdict;
      expect(h.winner, CompareSide.a);
      expect(h.ratio, closeTo(2, 1e-12));
      // A value not above zero still has a winner, and no ratio.
      final zero = c(_oneLatency(0), fast).pairs.single.both.single.verdict;
      expect(zero.winner, CompareSide.a);
      expect(zero.ratio, isNull);
      expect(zero.factor, isNull);
    });

    test('within 0.05% reads the same, as --compare words it', () {
      Verdict v(double a, double b, [Direction d = Direction.higherIsBetter]) =>
          Verdict.of(a, b, d);
      // A factor under 1.0005 would print as 1.000x: --compare says "within
      // 0.05%", and a screen marks neither side better, though one is.
      final near = v(1.0004, 1);
      expect(near.winner, CompareSide.a);
      expect(near.isTie, isFalse);
      expect(near.readsSame, isTrue);
      expect(v(1, 1.0004).readsSame, isTrue);
      final faster = v(1, 1.0004, Direction.lowerIsBetter);
      expect(faster.winner, CompareSide.a);
      expect(faster.readsSame, isTrue);
      // 1.0005 itself, and anything wider, names its winner.
      expect(v(1.0005, 1).readsSame, isFalse);
      expect(v(1.0006, 1).readsSame, isFalse);
      // A tie reads the same; a changed method, and a value not above zero,
      // do not.
      expect(v(2, 2).readsSame, isTrue);
      expect(
          Verdict.of(1, 1, Direction.higherIsBetter, methodChanged: true)
              .readsSame,
          isFalse);
      expect(v(0, 1e-9).readsSame, isFalse);
      // On the verdicts a screen reads: one launch 0.04% slower, by
      // processor, through its backend and reading by reading.
      final p = _compare(_oneLatency(1e-6), _oneLatency(1.0004e-6)).pairs.single;
      for (final verdict in [
        p.both.single.verdict,
        p.backends.single.both.single.verdict,
        p.devices.single.readings.single.verdict,
      ]) {
        expect(verdict.winner, CompareSide.a);
        expect(verdict.readsSame, isTrue);
      }
    });

    test('two processors of one run', () {
      final laptop = _run(_laptop);
      final p = PairCompare.of(laptop, laptop.measures.forProcessor('cpu0')!,
          laptop, laptop.measures.forProcessor('gpu0')!,
          catalog: _catalog);
      expect(p.mode, CompareMode.processors);
      final sp = _row(p, 'single-precision-compute|flops');
      expect(_from(sp.a.best), 'CPU:0 single_precision_compute@NEON "float MT" r1');
      expect(_from(sp.b.best), 'Metal:0 single_precision_compute "float4" r1');
      expect(sp.verdict.winner, CompareSide.b);
      expect(sp.verdict.factor, closeTo(5.21e12 / 8.6e11, 1e-12));
    });
  });

  group('reading by reading', () {
    /// A run of [cards], each a GPU by name and PCI device id, every one
    /// seen through Vulkan in that order -- and [twice] listed a second time
    /// by the same backend, as MoltenVK lists one GPU.
    String gpus(List<(String, String)> cards, {bool twice = false}) {
      Map<String, dynamic> device(String name, int index, String gpu,
              double value) =>
          {
            'backend': 'Vulkan',
            'name': name,
            'device_index': index,
            'type': 'gpu',
            'processor': gpu,
            'tests': [
              {
                'id': 'single_precision_compute',
                'benchmark': 'single-precision-compute',
                'category': 'compute',
                'direction': 'higher_is_better',
                'quantity': 'flops',
                'unit': 'FLOPS',
                'metrics': [
                  {'id': 'float4', 'dims': {'width': 4}, 'value': value},
                ],
              },
            ],
          };
      return jsonEncode({
        'schema': 'xpupeak/run',
        'format_version': 4,
        'processors': [
          for (final (i, (name, id)) in cards.indexed)
            {
              'id': 'gpu$i',
              'kind': 'gpu',
              'name': name,
              'vendor_id': id.split(':').first,
              'device_id': id.split(':').last,
              'integrated': false,
            },
        ],
        'devices': [
          for (final (i, (name, _)) in cards.indexed)
            device(name, i, 'gpu$i', 1e13 * (i + 1)),
          if (twice) device(cards.last.$1, cards.length, 'gpu${cards.length - 1}', 5e12),
        ],
      });
    }

    test('a device is named by backend, name and occurrence, not index', () {
      // The RTX alone, then with an A380 enumerated before it: its index
      // moves from 0 to 1, and it is still the same device.
      const rtx = ('NVIDIA GeForce RTX 5060', '10de:2d05');
      const a380 = ('Intel(R) Arc(TM) A380 Graphics', '8086:56a5');
      final c = _compare(gpus([rtx]), gpus([a380, rtx]));
      // The card pairs by model, ahead of the first GPU in order.
      expect([for (final p in c.pairs) '${p.a.id}-${p.b.id}:${p.pairedBy.name}'],
          ['gpu0-gpu1:model']);
      expect([for (final g in c.unpairedB) g.name], [a380.$1]);
      final d = c.pairs.single.devices.single;
      expect('${d.backend} ${d.name} #${d.occurrence}',
          'Vulkan NVIDIA GeForce RTX 5060 #0');
      expect([d.a.index, d.b.index], [0, 1]);
      expect(d.readings.single.readingId, 'float4');

      // One GPU listed twice by one backend: two occurrences of one name.
      final twice = gpus([rtx], twice: true);
      final p = _compare(twice, twice).pairs.single;
      expect([for (final x in p.devices) '${x.name} #${x.occurrence} ${x.b.index}'],
          ['${rtx.$1} #0 0', '${rtx.$1} #1 1']);
      expect([for (final x in p.devices) x.readings.single.verdict.isTie],
          [true, true]);
    });
  });

  group('one processor\'s backends', () {
    test('the RTX 5060 through five backends, the best marked', () {
      final desktop = _run(_desktop);
      final b = BackendCompare.of(
          desktop, desktop.measures.forProcessor('gpu0')!,
          catalog: _catalog);
      expect(b.mode, CompareMode.backends);
      expect(b.backends, ['CUDA', 'Vulkan', 'OpenCL', 'LiteRT', 'ONNX']);
      final float = b.headlines.first;
      expect(float.headline.id, 'peak-float');
      expect(_from(float.best), 'CUDA:0 wmma "nvfp4 2:4" r1');
      expect({for (final e in float.byBackend.entries) e.key: e.value.value}, {
        'CUDA': 6.33e14,
        'Vulkan': 9.38e13,
        'OpenCL': 1.88e13,
        'LiteRT': 2.9e13,
        'ONNX': 4.2e13,
      });

      BackendRow row(String key) => b.rows.firstWhere((r) => r.key.id == key);
      final sp = row('single-precision-compute|flops');
      expect(sp.best.backend, 'CUDA');
      expect(sp.byBackend.keys, ['CUDA', 'Vulkan', 'OpenCL']);
      expect(sp.behind('CUDA'), 1);
      expect(sp.behind('OpenCL'), closeTo(1.92e13 / 1.88e13, 1e-12));
      expect(sp.behind('LiteRT'), isNull);
      final gemm = row('gemm|flops|dtype=fp16');
      expect({for (final e in gemm.byBackend.entries) e.key: e.value.value},
          {'CUDA': 4.41e13, 'LiteRT': 2.9e13, 'ONNX': 4.2e13});
      // A key one backend alone measured is listed, not set side by side.
      expect(row('matrix-compute|flops|dtype=nvfp4|sparsity=2:4').shared,
          isFalse);
      expect([for (final r in b.shared) r.key.id], [
        'single-precision-compute|flops',
        'matrix-compute|flops|dtype=fp16',
        'matrix-compute|flops|acc=fp16|dtype=fp16',
        'matrix-compute|ops|dtype=int8',
        'gemm|flops|dtype=fp32',
        'gemm|flops|dtype=fp16',
        'global-memory-bandwidth|bytes_per_second',
      ]);
      expect([for (final g in b.byCategory) g.category],
          [BenchCategory.compute, BenchCategory.bandwidth, BenchCategory.latency]);

      // Both of the desktop's processors were measured by several backends.
      expect([for (final x in BackendCompare.ofRun(desktop, catalog: _catalog)) x.processor.id],
          ['cpu0', 'gpu0']);
    });
  });

  group('by backend', () {
    test('the GPUs through each backend both have, its best alone', () {
      final c = _compare(_laptop, _desktop);
      expect(c.backends, ['Vulkan', 'OpenCL', 'CPU', 'LiteRT', 'ONNX']);
      final gpu = _pair(c, 'gpu0');
      expect([for (final x in gpu.backends) x.backend],
          ['Vulkan', 'OpenCL', 'LiteRT']);

      final vulkan = _through(gpu, 'Vulkan');
      expect([for (final d in vulkan.devicesA) _device(d)],
          ['Vulkan:0 Apple M1 Pro']);
      expect([for (final d in vulkan.devicesB) _device(d)],
          ['Vulkan:0 NVIDIA GeForce RTX 5060']);
      // The float peak through Vulkan: the M1 Pro's shader ALUs, though
      // Metal's half4 is its peak by processor, against the RTX's
      // fp16-accumulate cooperative matrices, though CUDA's NVFP4 is its.
      expect([for (final h in vulkan.headlines) h.headline.id],
          ['peak-float', 'peak-integer', 'peak-memory-bandwidth']);
      final float = vulkan.headlines.first;
      expect(_from(float.a!), 'Vulkan:0 single_precision_compute "float4" r1');
      expect(_from(float.b!), 'Vulkan:0 coopmat "fp16 f16acc" r1');
      expect([for (final v in [...float.aBackends, ...float.bBackends]) v.backend],
          ['Vulkan', 'Vulkan']);
      expect(float.verdict!.winner, CompareSide.b);
      expect(float.verdict!.factor, closeTo(9.38e13 / 5.05e12, 1e-9));
      expect(_from(gpu.headlines.first.a!),
          'Metal:0 half_precision_compute "half4" r1');
      // No integer rate on the M1 Pro through Vulkan: B's alone, no verdict.
      final integer = vulkan.headlines[1];
      expect(integer.a, isNull);
      expect(_from(integer.b!), 'Vulkan:0 coopmat "int8" r1');
      expect(integer.verdict, isNull);
      expect(vulkan.headlines[2].verdict!.factor,
          closeTo(4.13e11 / 1.97e11, 1e-12));

      // The keys both measured through it, in the pair's order.
      expect([
        for (final g in vulkan.bothByCategory)
          '${g.category.tag}: ${[for (final r in g.rows) r.key.id].join(', ')}'
      ], [
        'compute: single-precision-compute|flops, '
            'matrix-compute|flops|dtype=fp16, '
            'matrix-compute|flops|acc=fp16|dtype=fp16',
        'bandwidth: global-memory-bandwidth|bytes_per_second',
      ]);
      final fp16 = vulkan.both[1];
      expect(_from(fp16.a.best), 'Vulkan:0 coopmat "fp16" r1');
      expect(_from(fp16.b.best), 'Vulkan:0 coopmat "fp16" r1');
      expect([for (final v in [...fp16.a.backends, ...fp16.b.backends]) v.backend],
          ['Vulkan', 'Vulkan']);
      expect(fp16.verdict.winner, CompareSide.b);
      expect(fp16.verdict.factor, closeTo(4.71e13 / 3.6e12, 1e-12));
      expect(fp16.category, BenchCategory.compute);
      // By processor the same key is Metal's against CUDA's.
      final whole = _row(gpu, 'matrix-compute|flops|dtype=fp16');
      expect(_from(whole.a.best), 'Metal:0 simdgroup_matrix "fp16" r2');
      expect(_from(whole.b.best), 'CUDA:0 wmma "fp16" r1');
      expect(vulkan.onlyA, isEmpty);
      expect([for (final r in vulkan.onlyB) r.key.id],
          ['matrix-compute|ops|dtype=int8']);

      // Through OpenCL the integer headline is two keys: the M1 Pro's int32
      // vectors against the RTX's int8 dot products.
      final opencl = _through(gpu, 'OpenCL');
      expect([for (final r in opencl.both) r.key.id], [
        'single-precision-compute|flops',
        'global-memory-bandwidth|bytes_per_second',
      ]);
      expect([for (final r in opencl.onlyA) r.key.id], ['integer-compute|ops']);
      expect([for (final r in opencl.onlyB) r.key.id],
          ['int8-dot-product-compute|ops']);
      final openclInteger = opencl.headlines[1];
      expect(_from(openclInteger.a!), 'OpenCL:0 integer_compute "int4" r1');
      expect(_from(openclInteger.b!),
          'OpenCL:0 integer_compute_int8_dp "int8_dp4" r1');
      expect(openclInteger.verdict!.factor, closeTo(7.4e13 / 1.31e12, 1e-9));

      // A headline neither side reached through a backend is left out:
      // LiteRT measured GEMMs alone.
      final litert = _through(gpu, 'LiteRT');
      expect([for (final h in litert.headlines) h.headline.id], ['peak-float']);
      expect([for (final r in litert.both) r.key.id], ['gemm|flops|dtype=fp16']);
      expect([for (final r in litert.onlyB) r.key.id], ['gemm|flops|dtype=fp32']);
    });

    test('the CPUs through the CPU backend and ONNX Runtime', () {
      final cpu = _pair(_compare(_laptop, _desktop), 'cpu0');
      expect([for (final x in cpu.backends) x.backend], ['CPU', 'ONNX']);
      final native = _through(cpu, 'CPU');
      expect([for (final r in native.both) r.key.id], [
        'single-precision-compute|flops',
        'single-precision-compute|flops|scope=core',
        'double-precision-compute|flops',
        'int8-dot-product-compute|ops',
        'int8-dot-product-compute|ops|scope=core',
        'global-memory-bandwidth|bytes_per_second|op=read',
        'global-memory-bandwidth|bytes_per_second|op=copy',
        'global-memory-bandwidth|bytes_per_second|op=triad',
        'memory-latency|seconds|level=l1',
        'memory-latency|seconds|level=dram',
      ]);
      // Accelerate's GEMMs are the M1 Pro's alone through the CPU backend,
      // though by processor both have fp32: the Threadripper's through
      // ONNX Runtime.
      expect([for (final r in native.onlyA) r.key.id], [
        'double-precision-compute|flops|scope=core',
        'gemm|flops|dtype=fp32',
        'gemm|flops|dtype=fp16',
        'aes|bytes_per_second',
        'aes|bytes_per_second|scope=core',
        'global-memory-bandwidth|bytes_per_second|op=read|scope=core',
        'memory-latency|seconds|level=l2',
        'memory-latency|seconds|level=slc',
        'memory-latency|seconds|in_flight=8|level=dram',
      ]);
      expect(_from(_row(cpu, 'gemm|flops|dtype=fp32').b.best),
          'ONNX:1 onnx_gemm "fp32" r1');
      expect([for (final r in native.onlyB) r.key.id], [
        'int16-dot-product-compute|ops',
        'sha256|bytes_per_second',
        'memory-latency|seconds|level=l3',
      ]);
      expect([for (final g in native.onlyAByCategory) g.category], [
        BenchCategory.compute,
        BenchCategory.crypto,
        BenchCategory.bandwidth,
        BenchCategory.latency,
      ]);
      // Accelerate's fp32 GEMM against the Threadripper's AVX2 all-core rate.
      final float = native.headlines.first;
      expect(_from(float.a!), 'CPU:0 accelerate_gemm "fp32" r1');
      expect(_from(float.b!),
          'CPU:0 single_precision_compute@AVX2+FMA "float MT" r1');
      expect(float.verdict!.winner, CompareSide.a);
      expect(float.verdict!.factor, closeTo(2.3e12 / 2.05e12, 1e-12));
      // Lower is better through one backend too: DRAM to the Threadripper.
      final dram = native.both.last;
      expect(dram.verdict.winner, CompareSide.b);
      expect(dram.verdict.factor, closeTo(9.6e-8 / 9.1e-8, 1e-12));

      final onnx = _through(cpu, 'ONNX');
      expect([for (final d in [...onnx.devicesA, ...onnx.devicesB]) _device(d)],
          ['ONNX:1 CPU', 'ONNX:1 CPU']);
      expect([
        for (final h in onnx.headlines) '${h.headline.id} ${h.a!.value} ${h.b!.value}'
      ], [
        'peak-float 640000000000.0 1620000000000.0',
        'peak-integer 1900000000000.0 4400000000000.0',
      ]);
      expect(onnx.onlyA, isEmpty);
      expect(onnx.onlyB, isEmpty);
    });

    test('a backend one side alone has', () {
      final c = _compare(_laptop, _desktop);
      expect(c.backendsOnlyA, ['Metal', 'CoreML']);
      expect(c.backendsOnlyB, ['CUDA']);
      // ONNX Runtime reached the RTX and not the M1 Pro's GPU; Vulkan and
      // OpenCL reached the Threadripper and no M1 Pro CPU.
      final gpu = _pair(c, 'gpu0');
      expect(gpu.backendsOnlyA, ['Metal', 'CoreML']);
      expect(gpu.backendsOnlyB, ['CUDA', 'ONNX']);
      final cpu = _pair(c, 'cpu0');
      expect(cpu.backendsOnlyA, ['CoreML']);
      expect(cpu.backendsOnlyB, ['Vulkan', 'OpenCL']);
      // What one backend alone measured compares by processor only: the
      // RTX's sparse NVFP4, through CUDA.
      const nvfp4 = 'matrix-compute|flops|dtype=nvfp4|sparsity=2:4';
      expect([for (final r in gpu.onlyB) r.key.id], contains(nvfp4));
      for (final x in gpu.backends) {
        expect([
          for (final r in x.both) r.key.id,
          for (final r in [...x.onlyA, ...x.onlyB]) r.key.id,
        ], isNot(contains(nvfp4)));
      }
    });

    test('one backend\'s test at two revisions reads "method changed"', () {
      // The RTX's cooperative-matrix test, re-measured another way.
      final revised = _desktop.replaceFirst(
          '"title": "Cooperative-matrix peak",',
          '"title": "Cooperative-matrix peak",\n          "revision": 2,');
      expect(revised, isNot(_desktop));
      final gpu = _pair(_compare(_laptop, revised), 'gpu0');
      final vulkan = _through(gpu, 'Vulkan');
      // Both sides' matrix rows through Vulkan are its coopmat test, at r1
      // and r2: no ratio.
      expect([
        for (final r in vulkan.both)
          if (r.verdict.methodChanged) r.key.id,
      ], [
        'matrix-compute|flops|dtype=fp16',
        'matrix-compute|flops|acc=fp16|dtype=fp16',
      ]);
      final fp16 = vulkan.both[1];
      expect(_from(fp16.b.best), 'Vulkan:0 coopmat "fp16" r2');
      expect(fp16.verdict.ratio, isNull);
      expect(fp16.verdict.winner, isNull);
      expect(fp16.verdict.factor, isNull);
      expect(fp16.verdict.readsSame, isFalse);
      // Two tests of one backend still compare: the float headline is the
      // M1 Pro's shader peak against the RTX's coopmat.
      final float = vulkan.headlines.first;
      expect(_from(float.b!), 'Vulkan:0 coopmat "fp16 f16acc" r2');
      expect(float.verdict!.methodChanged, isFalse);
      expect(float.verdict!.winner, CompareSide.b);
      // By processor the key is Metal's against CUDA's, which compare.
      final whole = _row(gpu, 'matrix-compute|flops|dtype=fp16');
      expect(whole.verdict.methodChanged, isFalse);
      expect(whole.verdict.winner, CompareSide.b);
      expect(_through(gpu, 'OpenCL').both.where((r) => r.verdict.methodChanged),
          isEmpty);
    });

    test('a shared backend\'s devices no pair sets side by side', () {
      final c = _compare(_laptop, _desktop);
      // The Neural Engine pairs with nothing, so its ONNX Runtime device has
      // no partner; its Core ML one is of a backend the desktop lacks.
      expect([for (final d in c.unpairedDevicesA) _device(d)],
          ['ONNX:0 Apple CoreML (Neural Engine)']);
      // The Threadripper through Vulkan and OpenCL, which reach no M1 Pro
      // CPU, and the RTX through ONNX Runtime, which reaches no M1 Pro GPU.
      expect([for (final d in c.unpairedDevicesB) _device(d)], [
        'Vulkan:1 llvmpipe (LLVM 19.1.1, 256 bits)',
        'OpenCL:1 pthread-AMD Ryzen Threadripper PRO 3955WX 16-Cores',
        'ONNX:0 NVIDIA CUDA (NVIDIA GeForce RTX 5060)',
      ]);
      _expectEveryDeviceOnce(c);

      // Paired with the RTX, the Neural Engine meets it through ONNX
      // Runtime, and the M1 Pro's GPU, now unpaired, leaves its devices
      // without a partner.
      final npu = c.a.measures.forProcessor('npu0')!;
      final r = c.withPair(npu, c.b.measures.forProcessor('gpu0')!);
      final onnx = _through(r.pairOf(npu)!, 'ONNX');
      expect([for (final d in [...onnx.devicesA, ...onnx.devicesB]) _device(d)], [
        'ONNX:0 Apple CoreML (Neural Engine)',
        'ONNX:0 NVIDIA CUDA (NVIDIA GeForce RTX 5060)',
      ]);
      expect([for (final x in onnx.both) x.key.id],
          ['gemm|flops|dtype=fp16', 'gemm|ops|dtype=int8|quant=qdq']);
      expect([for (final d in r.unpairedDevicesA) _device(d)], [
        'Vulkan:0 Apple M1 Pro',
        'OpenCL:0 Apple M1 Pro',
        'LiteRT:0 GPU (Metal)',
      ]);
      expect([for (final d in r.unpairedDevicesB) _device(d)], [
        'Vulkan:0 NVIDIA GeForce RTX 5060',
        'Vulkan:1 llvmpipe (LLVM 19.1.1, 256 bits)',
        'OpenCL:0 NVIDIA GeForce RTX 5060',
        'OpenCL:1 pthread-AMD Ryzen Threadripper PRO 3955WX 16-Cores',
        'LiteRT:0 WebGPU (NVIDIA GeForce RTX 5060)',
      ]);
      _expectEveryDeviceOnce(r);
    });

    test('swapping the sides turns every backend round', () {
      final c = _compare(_laptop, _desktop);
      final s = c.swapped();
      // In the desktop's run order now, which here is the laptop's.
      expect(s.backends, ['Vulkan', 'OpenCL', 'CPU', 'LiteRT', 'ONNX']);
      expect(s.backendsOnlyA, c.backendsOnlyB);
      expect(s.backendsOnlyB, c.backendsOnlyA);
      expect(s.unpairedDevicesA, c.unpairedDevicesB);
      expect(s.unpairedDevicesB, c.unpairedDevicesA);
      _expectEveryDeviceOnce(s);
      for (final p in c.pairs) {
        final q = s.pairs.firstWhere((q) => identical(q.a, p.b));
        expect(q.backendsOnlyA, p.backendsOnlyB);
        expect(q.backendsOnlyB, p.backendsOnlyA);
        expect([for (final x in q.backends) x.backend],
            [for (final x in p.backends) x.backend]);
        for (final x in p.backends) {
          final y = _through(q, x.backend);
          expect(y.devicesA, x.devicesB);
          expect(y.devicesB, x.devicesA);
          expect([for (final r in y.both) r.key.id], [for (final r in x.both) r.key.id]);
          expect([for (final r in y.onlyA) r.key.id], [for (final r in x.onlyB) r.key.id]);
          expect([for (final r in y.onlyB) r.key.id], [for (final r in x.onlyA) r.key.id]);
          for (final (j, r) in x.both.indexed) {
            _expectTurned(y.both[j].verdict, r.verdict);
          }
          expect([for (final h in y.headlines) h.headline.id],
              [for (final h in x.headlines) h.headline.id]);
          for (final (j, h) in x.headlines.indexed) {
            final g = y.headlines[j];
            expect(g.a?.value, h.b?.value);
            expect(g.b?.value, h.a?.value);
            if (h.verdict != null) _expectTurned(g.verdict!, h.verdict!);
          }
        }
      }
    });
  });

  group('a file against itself', () {
    for (final name in ['soc_laptop', 'desktop_two_packages']) {
      test('$name reads 1.00x everywhere, and nothing on one side', () {
        final text = _text(name);
        // The same file, read twice, and one run on both sides.
        final once = _run(text);
        for (final c in [
          _compare(text, text),
          RunCompare.of(once, once, catalog: _catalog),
        ]) {
          expect(c.unpairedA, isEmpty);
          expect(c.unpairedB, isEmpty);
          expect(c.backendsOnlyA, isEmpty);
          expect(c.backendsOnlyB, isEmpty);
          expect(c.unpairedDevicesA, isEmpty);
          expect(c.unpairedDevicesB, isEmpty);
          expect(c.pairs, hasLength(c.a.measures.groups.length));
          for (final p in c.pairs) {
            expect(p.pairedBy, PairedBy.model);
            expect(p.mode, CompareMode.readings);
            expect(p.onlyA, isEmpty);
            expect(p.onlyB, isEmpty);
            expect(p.devicesOnlyA, isEmpty);
            expect(p.devicesOnlyB, isEmpty);
            expect(p.devices, hasLength(p.a.devices.length));
            for (final d in p.devices) {
              expect(d.onlyA, isEmpty);
              expect(d.onlyB, isEmpty);
            }
            expect(p.backendsOnlyA, isEmpty);
            expect(p.backendsOnlyB, isEmpty);
            for (final x in p.backends) {
              expect([for (final d in x.devicesA) _device(d)],
                  [for (final d in x.devicesB) _device(d)]);
              expect(x.onlyA, isEmpty);
              expect(x.onlyB, isEmpty);
            }
          }
          final all = _verdicts(c).toList();
          expect(all, isNotEmpty);
          for (final v in all) {
            expect(v.methodChanged, isFalse);
            expect(v.isTie, isTrue);
            expect(v.ratio, 1.0);
            expect(v.factor, 1.0);
          }
        }
      });
    }
  });

  group('revisions', () {
    test('a test at another revision reads "method changed"', () {
      // The RTX's tensor-core test, re-measured another way.
      final revised = _desktop.replaceFirst(
          '"title": "Tensor cores (WMMA / mma.sync)",',
          '"title": "Tensor cores (WMMA / mma.sync)",\n          "revision": 2,');
      expect(revised, isNot(_desktop));
      final c = _compare(_desktop, revised);
      final gpu = _pair(c, 'gpu0');

      // Every key whose best is that test, on both sides: no ratio.
      final changed = [
        for (final r in gpu.both)
          if (r.verdict.methodChanged) r.key.id,
      ];
      expect(changed, [
        for (final r in gpu.both)
          if (r.a.best.test.id == 'wmma') r.key.id,
      ]);
      expect(changed, hasLength(10));
      final nvfp4 = _row(gpu, 'matrix-compute|flops|dtype=nvfp4|sparsity=2:4');
      expect(nvfp4.verdict.methodChanged, isTrue);
      expect(nvfp4.verdict.ratio, isNull);
      expect(nvfp4.verdict.winner, isNull);
      expect(nvfp4.verdict.factor, isNull);
      expect(nvfp4.verdict.isTie, isFalse);
      // The rest are unchanged.
      expect(_row(gpu, 'single-precision-compute|flops').verdict.isTie, isTrue);
      expect(_row(gpu, 'gemm|flops|dtype=fp16').verdict.isTie, isTrue);
      expect(
          [for (final h in gpu.headlines) '${h.headline.id} ${h.verdict!.methodChanged}'],
          ['peak-float true', 'peak-integer true', 'peak-memory-bandwidth false']);

      // Reading by reading too: the CUDA device's ten tensor-core readings.
      final cuda = gpu.devices.firstWhere((d) => d.backend == 'CUDA');
      expect([
        for (final r in cuda.readings)
          if (r.verdict.methodChanged) r.readingId,
      ], [
        'fp16', 'fp16 f16acc', 'bf16', 'tf32', 'fp8_e4m3', 'fp8_e4m3 2:4', //
        'nvfp4', 'nvfp4 2:4', 'int8', 'int8 2:4',
      ]);
      expect(cuda.readings.where((r) => r.verdict.isTie), hasLength(12));
      // The CPU's tests did not change.
      _expectAllTies(
          RunCompare.of(c.a, c.b, catalog: _catalog, pairs: [c.pairing.first]));
    });

    test('a reading compares by revision too', () {
      final v = _compare(_oneLatency(2e-6), _oneLatency(1e-6, revision: 2))
          .pairs
          .single;
      expect(v.both.single.verdict.methodChanged, isTrue);
      expect(v.devices.single.readings.single.verdict.methodChanged, isTrue);
    });
  });

  group('imported files compare like any other', () {
    late Directory dir;
    late RunHistoryStore store;

    setUp(() {
      dir = Directory.systemTemp.createTempSync('xpupeak_compare');
      store = RunHistoryStore(directoryOverride: dir);
    });
    tearDown(() => dir.deleteSync(recursive: true));

    Future<CompareRun> imported(String text, String fileName) async {
      final summary = await store.importContent(text, fileName: fileName);
      final run = await store.loadForCompare(summary);
      expect(run, isNotNull);
      expect(run!.summary!.fileName, summary.fileName);
      return run;
    }

    test('a file from another machine', () async {
      // The M1 Pro's history, and sf1-ub's run brought in from elsewhere.
      final laptop = await store.importContent(_laptop, fileName: 'm1.json');
      final desktop =
          await store.importContent(_desktop, fileName: 'sf1-ub.xpupeak.json');
      final c = (await store.compareRuns(laptop, desktop, catalog: _catalog))!;
      final direct = _compare(_laptop, _desktop);
      expect([for (final p in c.pairs) '${p.a.id}-${p.b.id}:${p.pairedBy.name}'],
          [for (final p in direct.pairs) '${p.a.id}-${p.b.id}:${p.pairedBy.name}']);
      expect([for (final g in c.unpairedA) g.id], ['npu0']);
      for (final (i, p) in c.pairs.indexed) {
        final q = direct.pairs[i];
        expect([for (final r in p.both) '${r.key.id} ${r.verdict.factor}'],
            [for (final r in q.both) '${r.key.id} ${r.verdict.factor}']);
        expect([for (final r in p.onlyB) r.key.id], [for (final r in q.onlyB) r.key.id]);
      }
      // The same row twice is one run against itself.
      final self = (await store.compareRuns(desktop, desktop, catalog: _catalog))!;
      _expectAllTies(self);
    });

    test('a dim key this build does not know, declared in dim_keys', () async {
      String withLanes(String declared) => _laptop
          .replaceFirst('"dims": {"dtype": "fp16"},\n              "value": 4.1e+12',
              '"dims": {"dtype": "fp16", "lanes": 32},\n              "value": 4.1e+12')
          .replaceFirst('  "devices": [',
              '  "dim_keys": {"dtype": "identity", "lanes": "$declared"},\n  "devices": [');
      final mine = _run(_laptop);

      // Variant, as its file says: the reading keeps its key.
      final variant = await imported(withLanes('variant'), 'lanes.xpupeak.json');
      final v = _pair(RunCompare.of(mine, variant, catalog: _catalog), 'gpu0');
      expect(v.onlyA, isEmpty);
      expect(v.onlyB, isEmpty);
      expect(_row(v, 'matrix-compute|flops|dtype=fp16').verdict.isTie, isTrue);

      // Identity: a measure of its own, which only B has.
      final identity =
          await imported(withLanes('identity'), 'lanes2.xpupeak.json');
      final i = _pair(RunCompare.of(mine, identity, catalog: _catalog), 'gpu0');
      expect([for (final r in i.onlyB) r.key.id],
          ['matrix-compute|flops|dtype=fp16|lanes=32']);
      final fp16 = _row(i, 'matrix-compute|flops|dtype=fp16');
      expect(_from(fp16.b.best), 'Vulkan:0 coopmat "fp16" r1');
      expect(fp16.verdict.winner, CompareSide.a);
      expect(fp16.verdict.factor, closeTo(4.1e12 / 3.6e12, 1e-12));
      // Reading by reading, it is still Metal's fp16 reading.
      final metal = i.devices.firstWhere((d) => d.backend == 'Metal');
      expect(metal.readings.firstWhere((r) => r.testKey == 'simdgroup_matrix' && r.readingId == 'fp16').verdict.isTie,
          isTrue);
    });

    test('a file from a newer minor', () async {
      final newer = _laptop
          .replaceFirst('"format_minor": 0', '"format_minor": 3')
          .replaceFirst('  "packages": [',
              '  "power_trace": {"samples": [1.5, 2.25], "unit": "W"},\n  "packages": [')
          .replaceFirst('"link": "registry",',
              '"link": "registry",\n      "future_device_key": "kept",')
          .replaceFirst('"dims": {"dtype": "fp16"},\n              "value": 4.1e+12',
              '"dims": {"dtype": "fp16"},\n              "value": 4.1e+12,\n'
                  '              "ext": {"cycles": 81920}');
      final run = await imported(newer, 'newer.xpupeak.json');
      expect(run.document.meta!.formatMinor, 3);
      final c = RunCompare.of(_run(_laptop), run, catalog: _catalog);
      expect(c.unpairedA, isEmpty);
      expect(c.unpairedB, isEmpty);
      for (final p in c.pairs) {
        expect(p.pairedBy, PairedBy.model);
        expect(p.onlyA, isEmpty);
        expect(p.onlyB, isEmpty);
      }
      _expectAllTies(c);
    });

    test('a file with no processors', () async {
      final json = jsonDecode(_laptop) as Map<String, dynamic>;
      json.remove('packages');
      json.remove('processors');
      for (final d in (json['devices'] as List).cast<Map<String, dynamic>>()) {
        d.remove('processor');
        d.remove('link');
      }
      final unlinked = await imported(jsonEncode(json), 'unlinked.xpupeak.json');
      // Each device is a part of its own.
      expect(unlinked.measures.groups, hasLength(10));
      expect(unlinked.measures.groups.every((g) => !g.linked), isTrue);

      // Against itself, device by device.
      final self = RunCompare.of(unlinked, unlinked, catalog: _catalog);
      expect(self.pairs, hasLength(10));
      expect(self.pairs.every((p) => p.pairedBy == PairedBy.model), isTrue);
      _expectAllTies(self);

      // Against the file that links them: the first device of each kind
      // pairs with that kind's processor, and the rest are listed.
      final c = RunCompare.of(unlinked, _run(_laptop), catalog: _catalog);
      expect([
        for (final p in c.pairs)
          '${p.a.devices.single.backend}:${p.a.devices.single.index}-${p.b.id}:${p.pairedBy.name}'
      ], ['Metal:0-gpu0:kind', 'CPU:0-cpu0:kind', 'CoreML:0-npu0:kind']);
      expect([for (final g in c.unpairedA) '${g.devices.single.backend}:${g.devices.single.index}'],
          ['Vulkan:0', 'OpenCL:0', 'CoreML:1', 'CoreML:2', 'LiteRT:0', 'ONNX:0', 'ONNX:1']);
      expect(c.unpairedB, isEmpty);
      final metal = c.pairs.first;
      expect(_row(metal, 'single-precision-compute|flops').verdict.isTie, isTrue);
      expect([for (final r in metal.onlyA) r.key.id], isEmpty);
      expect([for (final r in metal.onlyB) r.key.id],
          contains('matrix-compute|flops|acc=fp16|dtype=fp16'));
    });
  });

  group('words and order', () {
    test('every family has its category, as --help lists it', () {
      for (final f in families.keys) {
        expect(BenchCategory.fromTag(familyCategory(f)),
            isNot(BenchCategory.unknown), reason: f);
      }
      expect(familyCategory('gemm'), 'compute');
      expect(familyCategory('transfer-bandwidth'), 'bandwidth');
      expect(familyCategory('kernel-launch-latency'), 'latency');
      expect(familyCategory('transformer-block'), 'ai');
      expect(familyCategory('a-newer-family'), '');
    });

    test('a family this build does not know takes its test\'s category', () {
      final text = _oneLatency(1e-6)
          .replaceFirst('"benchmark":"kernel-launch-latency"', '"benchmark":"a-newer-family"')
          .replaceFirst('"category":"latency"', '"category":"bandwidth"');
      final row = _compare(text, text).pairs.single.both.single;
      expect(row.key.family, 'a-newer-family');
      expect(row.category, BenchCategory.bandwidth);
    });

    test('keys sort by family, quantity, then data type and the rest', () {
      MeasureKey k(String family, String quantity, Map<String, String> dims) =>
          MeasureKey(family, quantity, dims);
      final keys = [
        k('memory-latency', 'seconds', {'in_flight': '8', 'level': 'dram'}),
        k('memory-latency', 'seconds', {'level': 'dram'}),
        k('memory-latency', 'seconds', {'level': 'l1'}),
        k('a-newer-family', 'flops', {}),
        k('matrix-compute', 'ops', {'dtype': 'int8'}),
        k('matrix-compute', 'flops', {'dtype': 'bf16'}),
        k('matrix-compute', 'flops', {'acc': 'fp16', 'dtype': 'fp16'}),
        k('matrix-compute', 'flops', {'dtype': 'fp16'}),
        k('single-precision-compute', 'flops', {'scope': 'core'}),
        k('single-precision-compute', 'flops', {}),
        k('transformer-block', 'flops', {'context': '2048', 'dtype': 'fp16'}),
        k('transformer-block', 'flops', {'context': '512', 'dtype': 'fp16'}),
      ]..sort(measureKeyOrder);
      expect([for (final x in keys) x.id], [
        'single-precision-compute|flops',
        'single-precision-compute|flops|scope=core',
        'matrix-compute|flops|dtype=fp16',
        'matrix-compute|flops|acc=fp16|dtype=fp16',
        'matrix-compute|flops|dtype=bf16',
        'matrix-compute|ops|dtype=int8',
        'memory-latency|seconds|level=l1',
        'memory-latency|seconds|level=dram',
        'memory-latency|seconds|in_flight=8|level=dram',
        'transformer-block|flops|context=512|dtype=fp16',
        'transformer-block|flops|context=2048|dtype=fp16',
        'a-newer-family|flops',
      ]);
    });
  });
}
