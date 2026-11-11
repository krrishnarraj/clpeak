// The measure engine (lib/src/model/measures.dart) over the two hand-written
// v4 fixtures in test/fixtures/ -- an SoC laptop and a desktop with two
// packages -- against the keys and headlines worked out by hand from their
// readings, and the Dart model's read of everything the spec added.
//
// Pure Dart: the documents are read with dart:convert, as history reads them.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:xpupeak/src/model/headlines.dart';
import 'package:xpupeak/src/model/measures.dart';
import 'package:xpupeak/src/model/result_model.dart';
import 'package:xpupeak/src/model/run_document.dart';

RunDocument _fixture(String name) => RunDocument.fromJson(
    jsonDecode(File('test/fixtures/$name.xpupeak.json').readAsStringSync())
        as Map<String, dynamic>);

HeadlineCatalog _catalog() => HeadlineCatalog.fromJson(
    jsonDecode(File('assets/measures.json').readAsStringSync())
        as Map<String, dynamic>);

/// One reading's measure key, found by where it is in the document.
String _keyOf(RunDocument doc, String backend, String testKey, String metric,
    {int index = 0}) {
  final run =
      doc.runs.firstWhere((r) => r.backend == backend && r.index == index);
  final test = run.findTest(testKey)!;
  final m = test.metrics.firstWhere((m) => m.id == metric);
  return measureReading(run, test, m)!.key.id;
}

/// A headline's value on a processor, as (value, backend, test key, reading).
(double, String, String, String)? _headline(
    RunMeasures measures, String processor, String headline) {
  final h = _catalog().headlines.firstWhere((h) => h.id == headline);
  final v = measures.forProcessor(processor)!.headline(h);
  if (v == null) return null;
  return (v.best.value, v.best.backend, v.best.test.key, v.best.metric.id);
}

void main() {
  group('the fixtures read as the spec writes them', () {
    test('an SoC laptop: one package, three processors', () {
      final doc = _fixture('soc_laptop');
      final meta = doc.meta!;
      expect(meta.formatMinor, 0);
      expect(meta.runId, '6f1c2e8a-0d4b-4a59-9a77-3b1f0c2d9e10');
      expect(meta.standard, isTrue);
      expect(meta.buildConfig, 'Release');
      expect(meta.buildBackends.map((b) => '${b.name}:${b.kind}'), [
        'Metal:gpu', 'Vulkan:gpu', 'OpenCL:gpu', 'CPU:cpu', //
        'CoreML:ai', 'LiteRT:ai', 'ONNX:ai',
      ]);
      expect(meta.buildBackends.where((b) => b.portable).map((b) => b.name),
          ['Vulkan', 'OpenCL', 'ONNX']);

      final s = meta.systemInfo;
      expect(s.name, 'MacBook Pro (14-inch, 2021)');
      expect(s.model, 'MacBookPro18,3');
      expect(s.formFactor, 'laptop');
      expect(s.formFactorFrom, 'battery');
      expect(s.osName, 'macOS');
      expect(s.osVersion, '27.2');
      expect(s.memoryBytes, 34359738368);
      // The OS's name, the annotation naming nothing here.
      expect(meta.systemName, 'MacBook Pro (14-inch, 2021)');
      expect(meta.conditions.powerSource, 'ac');
      expect(meta.conditions.lowPowerMode, isFalse);
      expect(meta.conditions.thermalEnd, 'fair');
      expect(meta.annotations, {'cooling': 'stock'});

      expect(doc.packages.single.kind, 'soc');
      expect(doc.processors.map((p) => '${p.id}:${p.kind}:${p.package}'),
          ['cpu0:cpu:pkg0', 'gpu0:gpu:pkg0', 'npu0:npu:pkg0']);
      final cpu = doc.processorById('cpu0')!.facts.cpu;
      expect(cpu.cores, 10);
      expect(cpu.clusters.map((c) => '${c.kind}:${c.cores}:${c.maxMhz}'),
          ['performance:8:3228', 'efficiency:2:2064']);
      expect(cpu.clusters.first.caches.last.sizeBytes, 12582912);
      expect(cpu.caches.single.level, 'slc');
      expect(cpu.engines.single.scope, 'cluster');
      expect(cpu.features, contains('dotprod'));
      final gpu = doc.processorById('gpu0')!.facts;
      expect(gpu.arch, 'Apple7');
      expect(gpu.integrated, isTrue);
      expect(gpu.computeUnits, 16);
      expect(gpu.computeUnit, 'core');

      // Every backend's view of the GPU names it.
      expect(doc.devicesOf('gpu0').map((r) => '${r.backend}:${r.link}'), [
        'Metal:registry', 'Vulkan:sole', 'OpenCL:sole', 'CoreML:registry', //
        'LiteRT:sole',
      ]);
      final metal = doc.runs.first;
      expect(metal.type, 'gpu');
      expect(metal.evidence, {'registry': '9f2c41d0a7e3b5c8'});
      expect(metal.facts.vendorId, '106b');
      final onnx = doc.devicesOf('npu0').last;
      expect(onnx.runtime.provider, 'CoreMLExecutionProvider');
      expect(onnx.runtime.version, '1.31.0');
      expect(onnx.evidence, {'soc': 'npu'});

      final simd = metal.findTest('simdgroup_matrix')!;
      expect(simd.header.benchmark, 'matrix-compute');
      expect(simd.header.revision, 2);
      expect(simd.metrics.first.dims, {'dtype': 'fp16'});
      final sp = metal.findTest('single_precision_compute')!;
      expect(sp.header.revision, 1);
      expect(sp.header.dims, {'dtype': 'fp32'});
      expect(sp.metrics.last.dims, {'width': 4});
    });

    test('a desktop: a CPU package and a discrete card', () {
      final doc = _fixture('desktop_two_packages');
      final meta = doc.meta!;
      expect(doc.packages.map((p) => '${p.id}:${p.kind}'),
          ['pkg0:cpu', 'pkg1:discrete']);
      expect(doc.processorById('gpu0')!.package, 'pkg1');
      expect(doc.processorById('gpu0')!.facts.memoryBytes, 8150581248);
      final cpu = doc.processorById('cpu0')!.facts.cpu;
      expect(cpu.threads, 32);
      expect(cpu.cpuidFamily, 23);
      expect(cpu.clusters.single.kind, isEmpty);
      // No name from the OS (an OEM placeholder): the user's wins, and
      // without it the board would.
      expect(meta.systemInfo.name, isEmpty);
      expect(meta.systemInfo.board, 'Pro WS WRX80E-SAGE SE WIFI');
      expect(meta.systemName, 'Threadripper workstation');
      expect(meta.conditions.cpuGovernor, 'schedutil');
      expect(meta.systemInfo.osDistribution, 'Ubuntu 24.04.3 LTS');

      // The Threadripper's four views and the RTX's five.
      expect(doc.devicesOf('cpu0').map((r) => r.backend),
          ['Vulkan', 'OpenCL', 'CPU', 'ONNX']);
      expect(doc.devicesOf('gpu0').map((r) => r.backend),
          ['CUDA', 'Vulkan', 'OpenCL', 'LiteRT', 'ONNX']);

      // A reading measured under another family than its test's.
      final dot = doc.runs
          .firstWhere((r) => r.backend == 'CPU')
          .findTest('integer_dot_product@AVX2')!;
      expect(dot.header.benchmark, 'int8-dot-product-compute');
      expect(dot.metrics.last.benchmark, 'int16-dot-product-compute');
    });
  });

  group('measure keys', () {
    final laptop = _fixture('soc_laptop');
    final desktop = _fixture('desktop_two_packages');

    test('drop variant dims and defaults, keep the rest sorted', () {
      // Width is variant, fp32 the family's own type.
      expect(_keyOf(laptop, 'Metal', 'single_precision_compute', 'float4'),
          'single-precision-compute|flops');
      // An ISA is variant too; one core is not the default scope.
      expect(
          _keyOf(laptop, 'CPU', 'single_precision_compute@NEON', 'float ST'),
          'single-precision-compute|flops|scope=core');
      expect(_keyOf(laptop, 'CPU', 'single_precision_compute@NEON', 'float MT'),
          'single-precision-compute|flops');
      // A matrix family has no type of its own; fp32 is fp16's usual
      // accumulator, so only an fp16 one is keyed.
      expect(_keyOf(laptop, 'Metal', 'simdgroup_matrix', 'fp16'),
          'matrix-compute|flops|dtype=fp16');
      expect(_keyOf(laptop, 'Vulkan', 'coopmat', 'fp16'),
          'matrix-compute|flops|dtype=fp16');
      expect(_keyOf(laptop, 'Vulkan', 'coopmat', 'fp16 f16acc'),
          'matrix-compute|flops|acc=fp16|dtype=fp16');
      expect(_keyOf(laptop, 'CoreML', 'coreml_gemm', 'int8_weight'),
          'gemm|flops|dtype=fp16|weights=int8');
      // The reading's own quantity, over its test's.
      expect(_keyOf(laptop, 'ONNX', 'onnx_gemm', 'int8_qdq'),
          'gemm|ops|dtype=int8|quant=qdq');
      expect(
          _keyOf(laptop, 'CPU', 'integer_dot_product@NEON', 'int8_dp MT'),
          'int8-dot-product-compute|ops');
      expect(_keyOf(laptop, 'CPU', 'memory_latency', 'DRAM x8'),
          'memory-latency|seconds|in_flight=8|level=dram');
      expect(_keyOf(laptop, 'CoreML', 'coreml_block_prefill', 'fp16_prefill_s512'),
          'transformer-block|flops|context=512|dtype=fp16|phase=prefill');

      expect(_keyOf(desktop, 'CUDA', 'wmma', 'nvfp4 2:4'),
          'matrix-compute|flops|dtype=nvfp4|sparsity=2:4');
      expect(_keyOf(desktop, 'CUDA', 'transfer_bandwidth', 'h2d'),
          'transfer-bandwidth|bytes_per_second|direction=h2d|memory=pinned');
      // Its own family, whose type is int16.
      expect(_keyOf(desktop, 'CPU', 'integer_dot_product@AVX2', 'int16_dp MT'),
          'int16-dot-product-compute|ops');
    });

    test('a dim key this build does not know is identity', () {
      final run = DeviceRun(
          backend: 'X', platform: 'X', device: 'D', driver: '', index: 0);
      const header = TestHeader(
          id: 't',
          title: 't',
          benchmark: 'matrix-compute',
          units: Units(symbol: 'FLOPS', quantity: Quantity.flops));
      final test = run.openTest(header);
      const m = MetricResult(
          id: 'fp16', value: 1, dims: {'dtype': 'fp16', 'lanes': 32});
      expect(measureReading(run, test, m)!.key.id,
          'matrix-compute|flops|dtype=fp16|lanes=32');
    });

    test('a reading without a family or a value has no key', () {
      final run = DeviceRun(
          backend: 'X', platform: 'X', device: 'D', driver: '', index: 0);
      final test = run.openTest(const TestHeader(id: 't', title: 't'));
      expect(measureReading(run, test, const MetricResult(id: 'm', value: 1)),
          isNull);
      final known = run.openTest(
          const TestHeader(id: 'u', title: 'u', benchmark: 'gemm'));
      expect(
          measureReading(
              run,
              known,
              const MetricResult(id: 'm', status: ResultStatus.unsupported)),
          isNull);
    });
  });

  group('best of every backend, per processor and key', () {
    test('one GPU seen through two backends', () {
      final m = RunMeasures.of(_fixture('soc_laptop'));
      final fp16 =
          m.forProcessor('gpu0')!.byKey['matrix-compute|flops|dtype=fp16']!;
      expect(fp16.best.backend, 'Metal');
      expect(fp16.best.value, 4.1e12);
      expect({for (final e in fp16.byBackend.entries) e.key: e.value.value},
          {'Metal': 4.1e12, 'Vulkan': 3.6e12});
    });

    test('a CPU seen through four backends, two ISAs of one test included', () {
      final m = RunMeasures.of(_fixture('desktop_two_packages'));
      final cpu0 = m.forProcessor('cpu0')!;
      final sp = cpu0.byKey['single-precision-compute|flops']!;
      expect(sp.best.backend, 'CPU');
      expect(sp.best.test.key, 'single_precision_compute@AVX2+FMA');
      expect({for (final e in sp.byBackend.entries) e.key: e.value.value},
          {'Vulkan': 4.6e11, 'OpenCL': 1.21e12, 'CPU': 2.05e12});
      // Measured on one processor, never merged with another's.
      expect(m.forProcessor('gpu0')!.byKey['single-precision-compute|flops']!
          .best.value, 1.92e13);
    });

    test('a tie in value goes by backend, device, test and reading', () {
      // Not by the order the readings were walked, which another engine --
      // or a model that groups tests by category -- may not share.
      final doc = RunDocument();
      for (final backend in ['Vulkan', 'Metal']) {
        doc.runFor(backend, backend, 'GPU', '', 0)
          ..processor = 'gpu0'
          ..openTest(const TestHeader(
                  id: 'sp',
                  title: 'sp',
                  benchmark: 'single-precision-compute',
                  units: Units(symbol: 'FLOPS', quantity: Quantity.flops)))
              .metrics
              .add(const MetricResult(id: 'float4', value: 5e12));
      }
      doc.processors.add(const ProcessorInfo(id: 'gpu0', kind: 'gpu'));
      final best = RunMeasures.of(doc)
          .forProcessor('gpu0')!
          .byKey['single-precision-compute|flops']!
          .best;
      expect(best.backend, 'Metal');
    });

    test('a file nothing linked is measured device by device', () {
      final doc = _fixture('soc_laptop');
      doc.processors.clear();
      final m = RunMeasures.of(doc);
      expect(m.groups, hasLength(doc.runs.length));
      expect(m.groups.every((g) => !g.linked), isTrue);
      expect(m.groups.first.name, 'Apple M1 Pro');
      expect(m.groups.first.kind, 'gpu');
    });
  });

  group('headlines worked out by hand', () {
    test('the catalog is the three flagship numbers', () {
      final c = _catalog();
      expect(c.version, 2);
      expect(c.headlines.map((h) => h.id),
          ['peak-float', 'peak-integer', 'peak-memory-bandwidth']);
      expect(c.headlines.map((h) => h.quantity),
          ['flops', 'ops', 'bytes_per_second']);
    });

    test('the SoC laptop', () {
      final m = RunMeasures.of(_fixture('soc_laptop'));
      // The CPU's best float is its AMX, through Accelerate; its integer,
      // the NEON SDOT rows on every core; its bandwidth, STREAM's read.
      expect(_headline(m, 'cpu0', 'peak-float'),
          (2.3e12, 'CPU', 'accelerate_gemm', 'fp32'));
      expect(_headline(m, 'cpu0', 'peak-integer'),
          (3.4e12, 'CPU', 'integer_dot_product@NEON', 'int8_dp MT'));
      expect(_headline(m, 'cpu0', 'peak-memory-bandwidth'),
          (1.18e11, 'CPU', 'global_memory_bandwidth', 'read'));
      // The GPU's: fp16 ALUs on Metal, int32 on OpenCL, and Vulkan's read.
      expect(_headline(m, 'gpu0', 'peak-float'),
          (5.3e12, 'Metal', 'half_precision_compute', 'half4'));
      expect(_headline(m, 'gpu0', 'peak-integer'),
          (1.31e12, 'OpenCL', 'integer_compute', 'int4'));
      expect(_headline(m, 'gpu0', 'peak-memory-bandwidth'),
          (1.97e11, 'Vulkan', 'global_memory_bandwidth', 'float4'));
      // The Neural Engine: fp16 through Core ML, int8 through ONNX's
      // CoreML provider -- Core ML's own int8 row did not run -- and no
      // memory bandwidth, which no NPU backend measures.
      expect(_headline(m, 'npu0', 'peak-float'),
          (1.1e13, 'CoreML', 'coreml_gemm', 'fp16'));
      expect(_headline(m, 'npu0', 'peak-integer'),
          (6.2e12, 'ONNX', 'onnx_gemm', 'int8_qdq'));
      expect(_headline(m, 'npu0', 'peak-memory-bandwidth'), isNull);
    });

    test('the desktop', () {
      final m = RunMeasures.of(_fixture('desktop_two_packages'));
      expect(_headline(m, 'cpu0', 'peak-float'),
          (2.05e12, 'CPU', 'single_precision_compute@AVX2+FMA', 'float MT'));
      // ONNX Runtime's int8 GEMM beats the CPU backend's own dot products.
      expect(_headline(m, 'cpu0', 'peak-integer'),
          (4.4e12, 'ONNX', 'onnx_gemm', 'int8_qdq'));
      expect(_headline(m, 'cpu0', 'peak-memory-bandwidth'),
          (1.13e11, 'CPU', 'global_memory_bandwidth', 'read'));
      expect(_headline(m, 'gpu0', 'peak-float'),
          (6.33e14, 'CUDA', 'wmma', 'nvfp4 2:4'));
      expect(_headline(m, 'gpu0', 'peak-integer'),
          (3.16e14, 'CUDA', 'wmma', 'int8 2:4'));
      expect(_headline(m, 'gpu0', 'peak-memory-bandwidth'),
          (4.21e11, 'CUDA', 'global_memory_bandwidth', 'float4'));
    });

    test('each backend\'s best, best first, in a person\'s words', () {
      final m = RunMeasures.of(_fixture('desktop_two_packages'));
      final float = _catalog().headlines.first;
      final v = m.forProcessor('gpu0')!.headline(float)!;
      expect(describeReading(v.best), ['NVFP4', '2:4 structured sparsity']);
      expect(
          [
            for (final r in v.ranked)
              '${r.backend} ${r.value} ${describeReading(r).join(', ')}'
          ],
          [
            'CUDA 633000000000000.0 NVFP4, 2:4 structured sparsity',
            'Vulkan 93800000000000.0 FP16, FP16 accumulate',
            'ONNX 42000000000000.0 FP16',
            'LiteRT 29000000000000.0 FP16',
            'OpenCL 18800000000000.0 FP32',
          ]);
      final integer = m.forProcessor('cpu0')!.headline(_catalog().headlines[1])!;
      expect(describeReading(integer.best), ['INT8', 'QDQ']);
      final bandwidth =
          m.forProcessor('cpu0')!.headline(_catalog().headlines[2])!;
      expect(describeReading(bandwidth.best), ['read']);
    });
  });

  group('a headline\'s dims', () {
    final key = MeasureKey('matrix-compute', 'flops', {'dtype': 'fp16'});
    final f16acc = MeasureKey(
        'matrix-compute', 'flops', {'acc': 'fp16', 'dtype': 'fp16'});

    test('select with their defaults filled in', () {
      for (final dims in [
        {'dtype': 'fp16'},
        // The same measure, its default spelled out.
        {'dtype': 'fp16', 'acc': 'fp32', 'sparsity': 'dense'},
      ]) {
        final h = Headline(
            id: 'h', title: 'h', quantity: 'flops', dims: [dims]);
        expect(h.selects(key), isTrue, reason: '$dims');
        expect(h.selects(f16acc), isFalse, reason: '$dims');
      }
    });

    test('absent, select a key of any dims in their families', () {
      const h = Headline(
          id: 'h', title: 'h', quantity: 'flops', families: {'gemm'});
      expect(h.selects(key), isFalse);
      expect(h.selects(MeasureKey('gemm', 'flops', {'dtype': 'bf16'})), isTrue);
      expect(h.selects(MeasureKey('gemm', 'ops', {'dtype': 'int8'})), isFalse);
    });
  });
}
