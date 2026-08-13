import 'package:clpeak/src/ffi/clpeak_events.dart';
import 'package:clpeak/src/model/catalog.dart';
import 'package:clpeak/src/model/result_entry.dart';
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

    test('category subset flips to allow-list flags', () {
      final catalog = _catalog();
      final config = RunConfig.allDevices(catalog);
      config.categories
        ..clear()
        ..addAll({BenchCategory.fpCompute, BenchCategory.bandwidth});
      final args = config.toArgs(catalog);
      expect(args, containsAll(['--fp-compute', '--bandwidth']));
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
    test('metric event carries a full entry', () {
      final e = ClpeakEvent.fromJson({
        't': 'metric',
        'backend': 'CPU',
        'platform': 'CPU',
        'device': 'M1',
        'driver': '',
        'category': 'fp_compute',
        'test': 'single_precision_compute',
        'display': 'Single-precision compute',
        'metric': 'float MT',
        'unit': 'gflops',
        'value': 4480.5,
        'status': 'ok',
        'reason': '',
        'sub': false,
      });
      expect(e, isA<MetricEvent>());
      final m = (e as MetricEvent).entry;
      expect(m.value, 4480.5);
      expect(m.benchCategory, BenchCategory.fpCompute);
      expect(m.status, ResultStatus.ok);
      expect(m.key, 'CPU/CPU/M1/fp_compute/single_precision_compute/float MT');
    });

    test('done event', () {
      final e = ClpeakEvent.fromJson(
          {'t': 'done', 'status': -2, 'cancelled': true});
      expect(e, isA<DoneEvent>());
      expect((e as DoneEvent).cancelled, isTrue);
    });

    test('test_skipped maps to an unsupported entry', () {
      final e = ClpeakEvent.fromJson({
        't': 'test_skipped',
        'backend': 'Metal',
        'platform': 'Metal',
        'device': 'M1',
        'driver': 'macOS',
        'test': 'simdgroup_matrix_bf16',
        'display': 'Simdgroup matrix bf16',
        'unit': 'tflops',
        'category': 'fp_compute',
        'status': 'unsupported',
        'reason': 'requires M3+',
      });
      final entry = (e as TestSkippedEvent).toEntry();
      expect(entry.status, ResultStatus.unsupported);
      expect(entry.benchCategory, BenchCategory.fpCompute);
      expect(entry.reason, 'requires M3+');
    });
  });

  group('RunDocument', () {
    ResultEntry entry({
      String backend = 'Metal',
      String device = 'M1',
      String category = 'fp_compute',
      String test = 'single_precision_compute',
      String metric = 'float',
      String unit = 'gflops',
      double value = 100,
      ResultStatus status = ResultStatus.ok,
    }) =>
        ResultEntry(
          backend: backend,
          platform: backend,
          device: device,
          driver: 'd',
          category: category,
          test: test,
          display: test,
          metric: metric,
          unit: unit,
          status: status,
          value: value,
          reason: status == ResultStatus.ok ? '' : 'nope',
        );

    test('groups by run, category, test', () {
      final doc = RunDocument();
      doc.addEntry(entry(metric: 'float', value: 100));
      doc.addEntry(entry(metric: 'float2', value: 120));
      doc.addEntry(entry(
          category: 'bandwidth',
          test: 'global_memory_bandwidth',
          unit: 'gbps',
          metric: 'float',
          value: 200));
      doc.addEntry(entry(backend: 'CPU', device: 'M1', value: 50));

      expect(doc.runs, hasLength(2));
      final metal = doc.runs.first;
      expect(metal.categories, hasLength(2));
      final fp = metal.categories.first;
      expect(fp.category, BenchCategory.fpCompute);
      expect(fp.tests.single.metrics, hasLength(2));
      expect(fp.tests.single.peakValue, 120);
    });

    test('latency picks minimum as peak', () {
      final doc = RunDocument();
      doc.addEntry(entry(
          category: 'latency',
          test: 'kernel_launch_latency',
          unit: 'us',
          metric: 'dispatch',
          value: 5.2));
      doc.addEntry(entry(
          category: 'latency',
          test: 'kernel_launch_latency',
          unit: 'us',
          metric: 'roundtrip',
          value: 188.0));
      final t = doc.runs.single.categories.single.tests.single;
      expect(t.peakValue, 5.2);
    });

    test('all-skipped tests partition into unsupported', () {
      final doc = RunDocument();
      doc.addEntry(entry(value: 100));
      doc.addEntry(entry(
          test: 'double_precision_compute',
          metric: 'double',
          status: ResultStatus.unsupported,
          value: 0));
      final group = doc.runs.single.categories.single;
      expect(group.supported, hasLength(1));
      expect(group.unsupported, hasLength(1));
      expect(group.unsupported.single.skipReason, 'nope');
    });

    test('builds from a loaded saveJson document', () {
      final doc = RunDocument.fromEntriesJson({
        'format_version': 2,
        'entries': [
          {
            'backend': 'CPU',
            'platform': 'CPU',
            'device': 'X',
            'driver': '',
            'category': 'fp_compute',
            'test': 'single_precision_compute',
            'display': 'Single-precision compute (NEON)',
            'metric': 'float ST',
            'unit': 'gflops',
            'value': 251.0,
          },
          {
            'backend': 'CPU',
            'platform': 'CPU',
            'device': 'X',
            'driver': '',
            'category': 'fp_compute',
            'test': 'amx',
            'metric': 'bf16',
            'unit': 'tflops',
            'status': 'unsupported',
            'reason': 'no AMX',
          },
        ]
      });
      final run = doc.runs.single;
      final group = run.categories.single;
      expect(group.supported, hasLength(1));
      expect(group.unsupported, hasLength(1));
      // Saved files carry the human-readable name; the tag is only a fallback
      // (it cannot be un-slugged back into "… (NEON)").
      expect(group.supported.single.display, 'Single-precision compute (NEON)');
      // The row without one still renders as something.
      expect(group.unsupported.single.display, 'amx');
    });

    test('attaches device props from a loaded document', () {
      Map<String, dynamic> entry(String device) => {
            'backend': 'CPU',
            'platform': 'CPU',
            'device': device,
            'driver': '',
            'category': 'fp_compute',
            'test': 't',
            'metric': 'm',
            'unit': 'gflops',
            'value': 1.0,
          };
      final doc = RunDocument.fromEntriesJson({
        'format_version': 2,
        'devices': [
          {
            'backend': 'CPU',
            'platform': 'CPU',
            'device': 'X',
            'driver': '',
            'props': [
              {'k': 'Cores', 'v': '10'},
              {'k': 'RAM', 'v': '32.0 GB'},
            ],
          },
          // No matching run: must not invent an empty one.
          {
            'backend': 'CUDA',
            'platform': 'CUDA',
            'device': 'Absent',
            'driver': '',
            'props': [
              {'k': 'SMs', 'v': '84'},
            ],
          },
        ],
        'entries': [entry('X'), entry('Y')],
      });
      expect(doc.runs, hasLength(2));
      expect(doc.runs.first.props.map((p) => p.key), ['Cores', 'RAM']);
      expect(doc.runs.first.props.last.value, '32.0 GB');
      // Runs the devices block doesn't mention keep empty props.
      expect(doc.runs.last.props, isEmpty);
    });

    test('a document without a devices block still loads', () {
      final doc = RunDocument.fromEntriesJson({
        'format_version': 2,
        'entries': [
          {
            'backend': 'CPU',
            'platform': 'CPU',
            'device': 'X',
            'driver': '',
            'category': 'fp_compute',
            'test': 't',
            'metric': 'm',
            'unit': 'gflops',
            'value': 1.0,
          },
        ],
      });
      expect(doc.runs.single.props, isEmpty);
    });
  });

  group('RunSummary', () {
    test('rename round-trips through json', () {
      final doc = RunDocument()
        ..addEntry(ResultEntry(
          backend: 'CPU',
          platform: 'CPU',
          device: 'M1',
          driver: '',
          category: 'fp_compute',
          test: 't',
          metric: 'm',
          unit: 'gflops',
          status: ResultStatus.ok,
          value: 1,
          reason: '',
        ));
      final summary = RunSummary.fromDocument(
        id: '20260712_094501',
        fileName: '20260712_094501.xml',
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

  group('formatMetric', () {
    test('scales gflops to TFLOPS', () {
      expect(formatMetric(12500, 'gflops'),
          (value: '12.5', unit: 'TFLOPS'));
      expect(formatMetric(950, 'gflops'), (value: '950', unit: 'GFLOPS'));
      expect(formatMetric(5.2083, 'us'), (value: '5.21', unit: 'µs'));
    });
  });
}
