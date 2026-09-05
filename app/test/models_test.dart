import 'package:clpeak/src/ffi/clpeak_events.dart';
import 'package:clpeak/src/model/catalog.dart';
import 'package:clpeak/src/model/result_model.dart';
import 'package:clpeak/src/model/run_config.dart';
import 'package:clpeak/src/model/run_document.dart';
import 'package:clpeak/src/model/run_summary.dart';
import 'package:flutter_test/flutter_test.dart';

BackendCatalog _catalog() => BackendCatalog.fromJson({
      'backends': [
        {
          'name': 'Metal',
          'available': true,
          'platforms': [
            {
              'index': 0,
              'name': 'Metal',
              'devices': [
                {'index': 0, 'name': 'Apple M1 Pro', 'type': 'GPU'}
              ]
            }
          ]
        },
        {
          'name': 'OpenCL',
          'available': true,
          'platforms': [
            {
              'index': 0,
              'name': 'Apple',
              'devices': [
                {'index': 0, 'name': 'M1 Pro CPU', 'type': 'CPU'},
                {'index': 1, 'name': 'M1 Pro GPU', 'type': 'GPU'},
              ]
            }
          ]
        },
        {
          'name': 'CPU',
          'available': true,
          'platforms': [
            {
              'index': 0,
              'name': 'CPU',
              'devices': [
                {'index': 0, 'name': 'Apple M1 Pro', 'type': 'CPU'}
              ]
            }
          ]
        },
        {
          'name': 'ONNX',
          'available': true,
          'platforms': [
            {
              'index': 0,
              'name': 'ONNX Runtime 1.20.1',
              'devices': [
                {'index': 0, 'name': 'CPUExecutionProvider', 'type': 'CPU'},
                {'index': 1, 'name': 'CoreMLExecutionProvider', 'type': 'NPU'},
              ]
            }
          ]
        },
        {'name': 'Vulkan', 'available': false, 'platforms': []},
      ]
    });

void main() {
  group('RunConfig.toArgs', () {
    test('full selection emits no flags', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      expect(config.toArgs(catalog), isEmpty);
    });

    test('deselected backend emits --no-<backend>', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      config.selectedDevices.remove('Metal');
      expect(config.toArgs(catalog), ['--no-metal']);
    });

    test('partial OpenCL selection emits platform+device lists', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      config.toggleDevice(
          'OpenCL', (platformIndex: 0, deviceIndex: 0), false);
      final args = config.toArgs(catalog);
      expect(args, ['--cl-platform', '0', '--cl-device', '1']);
    });

    test('deselected ONNX backend emits --no-onnx', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      config.selectedDevices.remove('ONNX');
      expect(config.toArgs(catalog), ['--no-onnx']);
    });

    test('partial ONNX selection emits an EP index list', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      config.toggleDevice('ONNX', (platformIndex: 0, deviceIndex: 0), false);
      expect(config.toArgs(catalog), ['--onnx-device', '1']);
    });

    test('category subset flips to allow-list flags', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      config.categories
        ..clear()
        ..addAll({BenchCategory.compute, BenchCategory.bandwidth});
      final args = config.toArgs(catalog);
      expect(args, containsAll(['--compute', '--bandwidth']));
      expect(args, isNot(contains('--crypto')));
    });

    test('non-default budgets emit time flags', () {
      final catalog = _catalog();
      final config =
          RunConfig.allDevices(catalog, maxTimeMs: 200, maxTimeCpuMs: 500);
      final args = config.toArgs(catalog);
      expect(args, containsAllInOrder(['--max-time', '200']));
      expect(args, containsAllInOrder(['--max-time-cpu', '500']));
    });

    test('defaults emit no time flags', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      expect(config.toArgs(catalog).where((a) => a.startsWith('--max-time')),
          isEmpty);
    });
  });

  group('event decoding', () {
    Map<String, dynamic> testHeader({
      String test = 'single_precision_compute',
      String title = 'Single-precision compute',
      String category = 'compute',
      String shape = 'homogeneous',
      String unit = 'FLOPS',
      String quantity = 'flops',
      String direction = 'higher_is_better',
      String variant = '',
      String axis = '',
      String desc = '',
    }) =>
        {
          'test': test,
          'title': title,
          'variant': variant,
          'axis': axis,
          'category': category,
          'shape': shape,
          'direction': direction,
          'quantity': quantity,
          'unit': unit,
          'desc': desc,
        };

    test('test_begin carries the resolved header', () {
      final e = ClpeakEvent.fromJson({
        't': 'test_begin',
        'backend': 'CPU',
        'platform': 'CPU',
        'device': 'M1',
        'driver': '',
        ...testHeader(
          test: 'memory_latency',
          title: 'Memory latency (pointer-chase)',
          category: 'latency',
          shape: 'heterogeneous',
          unit: 's',
          quantity: 'seconds',
          direction: 'lower_is_better',
          axis: 'cache level',
          desc: 'How long the core waits for one memory read.',
        ),
        'reopened': false,
      });
      final h = (e as TestBeginEvent).header;
      expect(h.id, 'memory_latency');
      expect(h.title, 'Memory latency (pointer-chase)');
      expect(h.shape, TestShape.heterogeneous);
      expect(h.direction, Direction.lowerIsBetter);
      expect(h.axis, 'cache level');
      expect(h.units.quantity, Quantity.seconds);
      expect(h.description, 'How long the core waits for one memory read.');
    });

    test('an undocumented test decodes with empty documentation', () {
      final e = ClpeakEvent.fromJson({
        't': 'test_begin',
        'backend': 'CPU',
        'device': 'M1',
        ...testHeader(test: 'atomics', title: 'Atomic fetch-add latency'),
      });
      expect((e as TestBeginEvent).header.description, isEmpty);
    });

    test('metric event carries the reading and its note', () {
      final e = ClpeakEvent.fromJson({
        't': 'metric',
        'backend': 'CPU',
        'platform': 'CPU',
        'device': 'M1',
        'driver': '',
        'test': 'memory_latency',
        'variant': '',
        'metric': 'DRAM x8',
        'label': '',
        'value': 16.15,
        'minfo': 'Eight independent chases at once.',
      });
      final m = e as MetricEvent;
      expect(m.testKey, 'memory_latency');
      expect(m.metric.value, 16.15);
      expect(m.metric.status, ResultStatus.ok);
      expect(m.metric.description, 'Eight independent chases at once.');
    });

    test('a reading with a variant keys off id@variant', () {
      final e = ClpeakEvent.fromJson({
        't': 'metric',
        'backend': 'CPU',
        'test': 'single_precision_compute',
        'variant': 'AVX2+FMA',
        'metric': 'float MT',
        'value': 900.0,
      });
      expect((e as MetricEvent).testKey, 'single_precision_compute@AVX2+FMA');
    });

    test('a reading that overrides its unit keeps its own', () {
      final e = ClpeakEvent.fromJson({
        't': 'metric',
        'backend': 'CUDA',
        'test': 'cublas_gemm',
        'metric': 'int8',
        'value': 74.2,
        'unit': 'TOPS',
        'quantity': 'ops',
      });
      final m = (e as MetricEvent).metric;
      expect(m.units, isNotNull);
      expect(m.units!.symbol, 'TOPS');
      expect(m.units!.quantity, Quantity.ops);
    });

    test('a failed reading carries status and reason, not a value', () {
      final e = ClpeakEvent.fromJson({
        't': 'metric',
        'backend': 'Metal',
        'test': 'mps_gemm',
        'metric': 'bf16',
        'status': 'unsupported',
        'reason': 'requires M3+',
      });
      final m = (e as MetricEvent).metric;
      expect(m.status, ResultStatus.unsupported);
      expect(m.isOk, isFalse);
      expect(m.reason, 'requires M3+');
    });

    test('done event', () {
      final e = ClpeakEvent.fromJson(
          {'t': 'done', 'status': -2, 'cancelled': true});
      expect(e, isA<DoneEvent>());
      expect((e as DoneEvent).cancelled, isTrue);
    });

    test('test_skipped names every reading it stands in for', () {
      final e = ClpeakEvent.fromJson({
        't': 'test_skipped',
        'backend': 'Metal',
        'platform': 'Metal',
        'device': 'M1',
        'driver': 'macOS',
        ...testHeader(
            test: 'coopmat',
            title: 'Cooperative matrix',
            unit: 'FLOPS'),
        'metrics': ['fp16', 'bf16'],
        'status': 'unsupported',
        'reason': 'requires M3+',
      });
      final s = e as TestSkippedEvent;
      final metrics = s.toMetrics();
      expect(metrics.map((m) => m.id), ['fp16', 'bf16']);
      expect(metrics.every((m) => m.status == ResultStatus.unsupported), isTrue);
      expect(metrics.first.reason, 'requires M3+');
    });
  });

  group('RunDocument', () {
    TestHeader header({
      String id = 'single_precision_compute',
      String title = 'Single-precision compute',
      BenchCategory category = BenchCategory.compute,
      TestShape shape = TestShape.homogeneous,
      Direction direction = Direction.higherIsBetter,
      String variant = '',
      String axis = '',
      String description = '',
      Units units = const Units(
          symbol: 'FLOPS', quantity: Quantity.flops),
    }) =>
        TestHeader(
          id: id,
          title: title,
          category: category,
          shape: shape,
          direction: direction,
          variant: variant,
          axis: axis,
          description: description,
          units: units,
        );

    const nsUnits =
        Units(symbol: 's', quantity: Quantity.seconds);

    test('groups by run, category, test', () {
      final doc = RunDocument();
      final metal = doc.runFor('Metal', 'Metal', 'M1', 'd');
      metal.openTest(header())
        ..metrics.add(const MetricResult(id: 'float', value: 100))
        ..metrics.add(const MetricResult(id: 'float2', value: 120));
      metal
          .openTest(header(
              id: 'global_memory_bandwidth',
              title: 'Global memory bandwidth',
              category: BenchCategory.bandwidth,
              units: const Units(
                  symbol: 'B/s',
                  quantity: Quantity.bytesPerSecond)))
          .metrics
          .add(const MetricResult(id: 'float', value: 200));
      doc
          .runFor('CPU', 'CPU', 'M1', 'd')
          .openTest(header())
          .metrics
          .add(const MetricResult(id: 'float', value: 50));

      expect(doc.runs, hasLength(2));
      final run = doc.runs.first;
      expect(run.categories, hasLength(2));
      final fp = run.categories.first;
      expect(fp.category, BenchCategory.compute);
      expect(fp.tests.single.metrics, hasLength(2));
      expect(fp.tests.single.peakValue, 120);
    });

    test('a reopened test appends rather than duplicating', () {
      final doc = RunDocument();
      final run = doc.runFor('CUDA', 'CUDA', 'RTX', 'd');
      run
          .openTest(header(id: 'cublas_gemm', title: 'cuBLASLt GEMM peak'))
          .metrics
          .add(const MetricResult(id: 'fp32', value: 14.87));
      // The int phase reopens the same test with the same header.
      run
          .openTest(header(id: 'cublas_gemm', title: 'cuBLASLt GEMM peak'))
          .metrics
          .add(const MetricResult(id: 'int8', value: 74.2));

      expect(run.categories.single.tests, hasLength(1));
      expect(run.categories.single.tests.single.metrics, hasLength(2));
    });

    test('variants of one test are separate rows', () {
      final doc = RunDocument();
      final run = doc.runFor('CPU', 'CPU', 'M1', 'd');
      run.openTest(header(variant: 'SSE2'));
      run.openTest(header(variant: 'AVX2+FMA'));
      expect(run.categories.single.tests, hasLength(2));
      expect(run.categories.single.tests.map((t) => t.key), [
        'single_precision_compute@SSE2',
        'single_precision_compute@AVX2+FMA',
      ]);
      expect(run.categories.single.tests.first.displayTitle,
          contains('SSE2'));
    });

    test('a heterogeneous test never collapses', () {
      final doc = RunDocument();
      final t = doc.runFor('CUDA', 'CUDA', 'RTX', 'd').openTest(header(
          id: 'cublas_gemm',
          title: 'cuBLASLt GEMM peak',
          shape: TestShape.heterogeneous,
          axis: 'data type'));
      t.metrics.addAll(const [
        MetricResult(id: 'fp32', value: 14.87),
        MetricResult(id: 'nvf4_e2m1', value: 300.43),
      ]);
      expect(t.collapsible, isFalse);
    });

    test('a single-reading test collapses whatever its shape', () {
      final doc = RunDocument();
      final t = doc.runFor('Metal', 'Metal', 'M1', 'd').openTest(header(
          id: 'mps-attention',
          title: 'MPS attention',
          shape: TestShape.heterogeneous));
      t.metrics.add(const MetricResult(id: 'fp16', value: 2.9));
      expect(t.collapsible, isTrue);
      expect(t.peakValue, 2.9);
    });

    test('documentation rides the header and the readings', () {
      final doc = RunDocument();
      final t = doc.runFor('CPU', 'CPU', 'M1', 'd').openTest(header(
            id: 'memory_latency',
            title: 'Memory latency (pointer-chase)',
            category: BenchCategory.latency,
            shape: TestShape.heterogeneous,
            direction: Direction.lowerIsBetter,
            units: nsUnits,
            description: 'What the wait for one memory read costs.',
          ));
      t.metrics.addAll(const [
        MetricResult(
            id: 'L1', value: 1.26, description: 'The small cache inside the core.'),
        MetricResult(
            id: 'DRAM x8',
            value: 16.15,
            description: 'Eight independent chases at once.'),
        MetricResult(id: 'TLB miss', value: 16.28),
      ]);

      expect(t.hasInfo, isTrue);
      expect(t.hasMetricNotes, isTrue);
      expect(
          t.metrics
              .where((m) => m.description.isNotEmpty)
              .map((m) => m.id),
          ['L1', 'DRAM x8']);
    });

    test('an undocumented test offers no info', () {
      final doc = RunDocument();
      final t = doc.runFor('Metal', 'Metal', 'M1', 'd').openTest(header());
      t.metrics.add(const MetricResult(id: 'float', value: 100));
      expect(t.hasInfo, isFalse);
      expect(t.hasMetricNotes, isFalse);
    });

    test('lower-is-better picks the minimum; bars stay magnitude', () {
      final doc = RunDocument();
      final t = doc.runFor('Metal', 'Metal', 'M1', 'd').openTest(header(
          id: 'kernel_launch_latency',
          title: 'Kernel launch latency',
          category: BenchCategory.latency,
          shape: TestShape.homogeneous,
          direction: Direction.lowerIsBetter,
          units: const Units(
              symbol: 's', quantity: Quantity.seconds)));
      t.metrics.addAll(const [
        MetricResult(id: 'dispatch', value: 5.2),
        MetricResult(id: 'roundtrip', value: 188.0),
      ]);
      expect(t.peakValue, 5.2);
      // The meter is a picture of the number beside it, so the larger reading
      // draws the longer bar even though it is the worse one.  Direction
      // decides the collapsed value above, not the bars.
      expect(t.barFraction(t.metrics.first), closeTo(5.2 / 188.0, 1e-9));
      expect(t.barFraction(t.metrics.last), 1.0);
    });

    test('all-skipped tests partition into unsupported', () {
      final doc = RunDocument();
      final run = doc.runFor('Metal', 'Metal', 'M1', 'd');
      run.openTest(header()).metrics.add(const MetricResult(id: 'float', value: 100));
      run
          .openTest(header(
              id: 'double_precision_compute', title: 'Double-precision compute'))
          .metrics
          .add(const MetricResult(
              id: 'double',
              status: ResultStatus.unsupported,
              reason: 'nope'));
      final group = run.categories.single;
      expect(group.supported, hasLength(1));
      expect(group.unsupported, hasLength(1));
      expect(group.unsupported.single.skipReason, 'nope');
    });

    test('unavailable collects whole tests and individual readings', () {
      final doc = RunDocument();
      final run = doc.runFor('CUDA', 'CUDA', 'RTX', 'd');
      // A test that measured most of its readings, missing one.
      run.openTest(header(
          id: 'cublas_gemm',
          title: 'cuBLASLt GEMM peak',
          shape: TestShape.heterogeneous))
        ..metrics.add(const MetricResult(id: 'fp32', value: 14.87))
        ..metrics.add(const MetricResult(
            id: 'mxf4_e2m1',
            status: ResultStatus.unsupported,
            reason: 'requires Blackwell'));
      // A test that measured nothing at all.
      run.openTest(header(id: 'wmma_fp64', title: 'WMMA fp64')).metrics.add(
          const MetricResult(
              id: 'wmma_fp64',
              status: ResultStatus.unsupported,
              reason: 'no fp64 tensor cores'));

      final items = run.unavailable;
      expect(items, hasLength(2));
      // The whole-test one is named by the test alone.
      expect(items.map((i) => i.title),
          ['cuBLASLt GEMM peak › mxf4_e2m1', 'WMMA fp64']);
      // The measured test still shows up top with only its measurement.
      final gemm = run.categories.single.tests.first;
      expect(gemm.okMetrics.map((m) => m.id), ['fp32']);
    });

    test('builds from a saved document', () {
      final doc = RunDocument.fromJson({
        'format_version': 3,
        'clpeak_version': '3.0.0',
        'generated_at': '2026-08-29T14:03:11Z',
        'duration_s': 12.5,
        'cancelled': true,
        'host': {'os': 'Macintosh', 'cpu': 'Apple M1 Pro'},
        'notes': [
          {'backend': 'ONNX', 'message': 'QNN EP not found'}
        ],
        'devices': [
          {
            'backend': 'CPU',
            'platform': 'CPU',
            'name': 'X',
            'driver': '',
            'type': 'cpu',
            'properties': [
              {'key': 'Cores', 'value': '10'},
              {'key': 'RAM', 'value': '32.0 GB'},
            ],
            'tests': [
              {
                'id': 'single_precision_compute',
                'title': 'Single-precision compute',
                'variant': 'NEON',
                'category': 'compute',
                'shape': 'homogeneous',
                'direction': 'higher_is_better',
                'quantity': 'flops',
                'unit': 'FLOPS',
                'description': 'Peak arithmetic speed.',
                'metrics': [
                  {
                    'id': 'float ST',
                    'value': 251.0,
                    'description': 'One core.'
                  },
                ],
              },
              {
                'id': 'cpu_matrix_fp',
                'title': 'CPU matrix engine',
                'category': 'compute',
                'shape': 'heterogeneous',
                'direction': 'higher_is_better',
                'quantity': 'flops',
                'unit': 'FLOPS',
                'metrics': [
                  {'id': 'bf16', 'status': 'unsupported', 'reason': 'no AMX'},
                ],
              },
            ],
          },
        ],
      });

      final run = doc.runs.single;
      expect(doc.meta!.cancelled, isTrue);
      expect(doc.meta!.host['cpu'], 'Apple M1 Pro');
      expect(doc.notes.single.message, 'QNN EP not found');
      expect(run.props.map((p) => p.key), ['Cores', 'RAM']);

      final group = run.categories.single;
      expect(group.supported, hasLength(1));
      expect(group.unsupported, hasLength(1));

      final sp = group.supported.single;
      // The variant is metadata now, not part of the name -- and shows beside
      // it rather than being un-slugged out of the tag.
      expect(sp.id, 'single_precision_compute');
      expect(sp.variant, 'NEON');
      expect(sp.displayTitle, contains('NEON'));
      expect(sp.description, 'Peak arithmetic speed.');
      expect(sp.metrics.single.description, 'One core.');
      expect(sp.units.quantity, Quantity.flops);
    });

    test('two devices of the same name stay two runs', () {
      // MoltenVK exposes one GPU twice, and a multi-GPU box has N identical
      // cards.  Keyed on the name alone they folded into one run, and every
      // test came out with two of every reading.
      final doc = RunDocument();
      doc.runFor('Vulkan', 'Vulkan', 'Apple M1 Pro', '26.1.99', 0)
          .openTest(header())
          .metrics
          .add(const MetricResult(id: 'float', value: 100));
      doc.runFor('Vulkan', 'Vulkan', 'Apple M1 Pro', '26.1.99', 1)
          .openTest(header())
          .metrics
          .add(const MetricResult(id: 'float', value: 90));

      expect(doc.runs, hasLength(2));
      expect(doc.runs.map((r) => r.index), [0, 1]);
      for (final r in doc.runs) {
        expect(r.categories.single.tests.single.metrics, hasLength(1));
      }
    });

    test('a driver update does not split a run', () {
      // Driver is metadata, not identity: a baseline stays comparable across
      // one, which is exactly the comparison people want.
      final doc = RunDocument();
      doc.runFor('CUDA', 'CUDA', 'RTX 5060', '580.65', 0);
      doc.runFor('CUDA', 'CUDA', 'RTX 5060', '581.00', 0);
      expect(doc.runs, hasLength(1));
    });

    test('a document with no devices loads empty', () {
      final doc = RunDocument.fromJson({'format_version': 3});
      expect(doc.isEmpty, isTrue);
    });
  });

  group('RunSummary', () {
    test('rename round-trips through json', () {
      final doc = RunDocument();
      doc
          .runFor('CPU', 'CPU', 'M1', '')
          .openTest(const TestHeader(id: 't', title: 't'))
          .metrics
          .add(const MetricResult(id: 'm', value: 1));
      final summary = RunSummary.fromDocument(
        id: '20260712_094501',
        fileName: '20260712_094501.clpeak.json',
        doc: doc,
        startedAt: DateTime(2026, 7, 12, 9, 45, 1),
        durationMs: 100,
        cancelled: false,
      );
      // Unnamed runs are titled by their timestamp id, not by the device.
      expect(summary.displayTitle, '20260712_094501');
      final named = summary.withName('after undervolt');
      expect(named.displayTitle, 'after undervolt');
      final roundTripped = RunSummary.fromJson(named.toJson());
      expect(roundTripped.name, 'after undervolt');
      expect(roundTripped.devices, ['M1']);
      // Clearing the name falls back to the timestamp id.
      expect(named.withName('').displayTitle, '20260712_094501');
    });
  });

  group('formatValue', () {
    const flops = Units(symbol: 'FLOPS', quantity: Quantity.flops);
    const s = Units(symbol: 's', quantity: Quantity.seconds);
    const bps = Units(symbol: 'B/s', quantity: Quantity.bytesPerSecond);
    const ppm = Units(symbol: 'ppm', quantity: Quantity.ratio);

    test('slides the SI prefix instead of switching on the unit', () {
      expect(formatValue(12.5e12, flops), (value: '12.5', unit: 'TFLOPS'));
      expect(formatValue(950e9, flops), (value: '950', unit: 'GFLOPS'));
      expect(formatValue(5.2083e-6, s), (value: '5.21', unit: 'µs'));
      // Sub-microsecond latency reads in nanoseconds, not "0.12 µs".
      expect(formatValue(0.125e-6, s), (value: '125', unit: 'ns'));
      expect(formatValue(183.4e9, bps), (value: '183', unit: 'GB/s'));
    });

    test('a quantity with no ladder prints as measured', () {
      expect(formatValue(4.5, ppm), (value: '4.50', unit: 'ppm'));
    });

    test('zero and unknown units survive', () {
      expect(formatValue(0, flops), (value: '0.00', unit: 'FLOPS'));
      expect(formatValue(3, Units.empty), (value: '3.00', unit: ''));
    });
  });
}
