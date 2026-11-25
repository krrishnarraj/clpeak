// The info affordance on the results view, at both levels: a glyph on a test's
// row for what the test measures, and one on each documented reading in the
// breakdown for what that reading means — each opening only its own text.
// And what the breakdown marks: the best reading of a homogeneous test, by
// its direction, and nothing on a heterogeneous one.
//
// Pure Dart — ResultsBody takes a RunDocument, so no native bridge is needed.
import 'package:xpupeak/src/model/result_model.dart';
import 'package:xpupeak/src/model/run_document.dart';
import 'package:xpupeak/src/theme/xpupeak_theme.dart';
import 'package:xpupeak/src/ui/common/kit.dart';
import 'package:xpupeak/src/ui/results/results_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

const _ns = Units(symbol: 's', quantity: Quantity.seconds);

/// Every info glyph on screen.
final _info =
    find.byWidgetPredicate((w) => w is CGlyph && w.path == CGlyphs.about);

TestHeader _header({
  required String id,
  required String title,
  String description = '',
  BenchCategory category = BenchCategory.latency,
  TestShape shape = TestShape.homogeneous,
  Units units = _ns,
}) =>
    TestHeader(
      id: id,
      title: title,
      description: description,
      category: category,
      shape: shape,
      direction: Direction.lowerIsBetter,
      units: units,
    );

Widget _host(RunDocument doc) => MaterialApp(
      theme: XpuPeakTheme.dark(),
      home: Scaffold(body: ResultsBody(document: doc)),
    );

void main() {
  testWidgets('a documented test offers info; an undocumented one does not',
      (tester) async {
    final doc = RunDocument();
    final run = doc.runFor('CPU', 'CPU', 'M1 Pro', '');
    run
        .openTest(_header(
          id: 'memory_latency',
          title: 'Memory latency (pointer-chase)',
          description: 'What the wait for one memory read costs.',
        ))
        .metrics
        .add(const MetricResult(
            id: 'DRAM x8',
            value: 16.15,
            description: 'Eight independent chases at once.'));
    run
        .openTest(_header(id: 'atomics', title: 'Atomic fetch-add latency'))
        .metrics
        .add(const MetricResult(id: 'uncontended ST', value: 2.2));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // Two tests on screen, one glyph — collapsed, so no reading glyphs yet.
    // A documented name carries its glyph as an inline WidgetSpan, so its
    // plain text is the name plus a placeholder rune: match by containment.
    expect(find.textContaining('Memory latency (pointer-chase)'),
        findsOneWidget);
    expect(find.text('Atomic fetch-add latency'), findsOneWidget);
    expect(_info, findsOneWidget);

    await tester.tap(_info);
    await tester.pump();

    // The test's glyph explains the test, and nothing else.
    expect(find.text('What the wait for one memory read costs.'),
        findsOneWidget);
    expect(find.text('Eight independent chases at once.'), findsNothing);

    await tester.tap(find.text('Close'));
    await tester.pump();
    expect(find.text('What the wait for one memory read costs.'), findsNothing);
  });

  testWidgets('each documented reading carries its own info', (tester) async {
    final doc = RunDocument();
    doc.runFor('CPU', 'CPU', 'M1 Pro', '').openTest(_header(
          id: 'memory_latency',
          title: 'Memory latency (pointer-chase)',
          description: 'What the wait for one memory read costs.',
        ))
      ..metrics.add(const MetricResult(
          id: 'DRAM x8',
          value: 16.15,
          description: 'Eight independent chases at once.'))
      // A second reading of the same test, undocumented.
      ..metrics.add(const MetricResult(id: 'L1', value: 1.26));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // Open the readings: the test's glyph, plus one for the documented
    // reading and none for the other.
    await tester.tap(find.textContaining('Memory latency (pointer-chase)'));
    await tester.pump();
    expect(_info, findsNWidgets(2));

    // The reading's own glyph is the second one — the test's comes first.
    await tester.tap(_info.last);
    await tester.pump();

    expect(find.text('Eight independent chases at once.'), findsOneWidget);
    // Titled by the reading, not the test.
    expect(find.text('DRAM x8'), findsOneWidget);
  });

  testWidgets('a heterogeneous test shows every reading without a headline',
      (tester) async {
    final doc = RunDocument();
    doc.runFor('CUDA', 'CUDA', 'RTX 5060', '').openTest(const TestHeader(
          id: 'cublas_gemm',
          title: 'cuBLASLt GEMM peak',
          category: BenchCategory.compute,
          shape: TestShape.heterogeneous,
          axis: 'data type',
          units: Units(
              symbol: 'FLOPS', quantity: Quantity.flops),
        ))
      ..metrics.add(const MetricResult(id: 'fp32', value: 14.87e12))
      ..metrics.add(const MetricResult(id: 'nvf4_e2m1', value: 300.43e12));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // No tap needed: both readings are already on screen, headed by what
    // varies across them.
    expect(find.text('DATA TYPE'), findsOneWidget);
    expect(find.text('fp32'), findsOneWidget);
    expect(find.text('nvf4_e2m1'), findsOneWidget);
    // The largest reading is shown as one reading among several, never as the
    // test's own number — it appears once, in the table, not in the header —
    // and nothing marks it best: the readings measure different things.
    expect(find.text('300 TFLOPS'), findsOneWidget);
    expect(find.text('Best'), findsNothing);
  });

  testWidgets('a homogeneous test marks its best reading, by its direction',
      (tester) async {
    // Lower is better here: the best reading is the smallest, and it is the
    // one the row collapses to.
    final doc = RunDocument();
    doc.runFor('Metal', 'Metal', 'M1 Pro', '').openTest(_header(
          id: 'kernel_launch_latency',
          title: 'Kernel launch latency',
        ))
      ..metrics.add(const MetricResult(id: 'dispatch', value: 5.24e-6))
      ..metrics.add(const MetricResult(id: 'roundtrip', value: 184.0e-6));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // Collapsed to the best reading, with the count of readings beside it.
    expect(find.text('5.24 µs'), findsOneWidget);
    expect(find.text('2 results'), findsOneWidget);
    expect(find.text('Best'), findsNothing);

    await tester.tap(find.text('Kernel launch latency'));
    await tester.pump();

    // Open: both readings, the best one marked, on its own row.
    expect(find.text('roundtrip'), findsOneWidget);
    expect(find.text('Best'), findsOneWidget);
    expect(tester.getCenter(find.text('Best')).dy,
        closeTo(tester.getCenter(find.text('dispatch')).dy, 4));
    // Nothing but numbers: no bar beside them repeats what they say.
    expect(find.byType(CMeter), findsNothing);
  });

  testWidgets('Expand all and Collapse all hold every test', (tester) async {
    final doc = RunDocument();
    final run = doc.runFor('Metal', 'Metal', 'M1 Pro', '');
    for (final id in ['a', 'b']) {
      run.openTest(_header(id: id, title: 'Test $id'))
        ..metrics.add(MetricResult(id: '$id-1', value: 1e-6))
        ..metrics.add(MetricResult(id: '$id-2', value: 2e-6));
    }
    await tester.pumpWidget(_host(doc));
    await tester.pump();
    expect(find.text('a-2'), findsNothing);

    await tester.tap(find.text('Expand all'));
    await tester.pump();
    expect(find.text('a-2'), findsOneWidget);
    expect(find.text('b-2'), findsOneWidget);

    // A tap on one test still opens or closes it alone.
    await tester.tap(find.text('Test a'));
    await tester.pump();
    expect(find.text('a-2'), findsNothing);
    expect(find.text('b-2'), findsOneWidget);

    await tester.tap(find.text('Collapse all'));
    await tester.pump();
    expect(find.text('a-2'), findsNothing);
    expect(find.text('b-2'), findsNothing);
  });

  testWidgets('an unavailable test can still be explained', (tester) async {
    final doc = RunDocument();
    doc
        .runFor('CPU', 'CPU', 'M1 Pro', '')
        .openTest(const TestHeader(
          id: 'cpu_matrix_fp',
          title: 'Matrix engine (AMX)',
          category: BenchCategory.compute,
          description: 'Throughput of the dedicated matrix-multiply unit.',
          units: Units(
              symbol: 'FLOPS', quantity: Quantity.flops),
        ))
        .metrics
        .add(const MetricResult(
            id: 'bf16',
            status: ResultStatus.unsupported,
            reason: 'no AMX on this CPU'));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // Listed at the foot, with its reason, and its info a tap away.
    expect(find.text('Not available on this device'), findsOneWidget);
    expect(find.text('1 test'), findsOneWidget);
    expect(find.text('no AMX on this CPU'), findsOneWidget);
    await tester.tap(_info);
    await tester.pump();

    expect(find.text('Throughput of the dedicated matrix-multiply unit.'),
        findsOneWidget);
  });

  testWidgets('one missing reading leaves the measured table alone',
      (tester) async {
    final doc = RunDocument();
    doc.runFor('Metal', 'Metal', 'M1 Pro', '').openTest(const TestHeader(
          id: 'mps_gemm',
          title: 'MPS GEMM peak',
          category: BenchCategory.compute,
          shape: TestShape.heterogeneous,
          axis: 'data type',
          units: Units(
              symbol: 'FLOPS', quantity: Quantity.flops),
        ))
      ..metrics.add(const MetricResult(id: 'fp32', value: 4.06))
      ..metrics.add(const MetricResult(id: 'fp16', value: 3.95))
      ..metrics.add(const MetricResult(
          id: 'bf16',
          status: ResultStatus.unsupported,
          reason: 'requires M3+'));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // The measured readings are in the table; the missing one is not, and
    // the table counts only what was measured.
    expect(find.text('fp32'), findsOneWidget);
    expect(find.text('fp16'), findsOneWidget);
    expect(find.text('bf16'), findsNothing);
    expect(find.text('2 results'), findsOneWidget);

    // It is listed once, at the foot, by its test.
    expect(find.text('1 reading'), findsOneWidget);
    expect(find.textContaining('MPS GEMM peak › bf16'), findsOneWidget);
    expect(find.text('requires M3+'), findsOneWidget);
  });
}
