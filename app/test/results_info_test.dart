// The info affordance on the results view, at both levels: a glyph on a test's
// row for what the test measures, and one on each documented reading in the
// breakdown for what that reading means — each opening only its own text.
//
// Pure Dart — ResultsBody takes a RunDocument, so no native bridge is needed.
import 'package:clpeak/src/model/result_model.dart';
import 'package:clpeak/src/model/run_document.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/results/results_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

const _ns = Units(symbol: 's', quantity: Quantity.seconds);

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
      theme: ClpeakTheme.dark(),
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
    expect(find.byIcon(Icons.info_outline), findsOneWidget);

    await tester.tap(find.byIcon(Icons.info_outline));
    await tester.pump();

    // The test's glyph explains the test, and nothing else.
    expect(find.text('What the wait for one memory read costs.'),
        findsOneWidget);
    expect(find.text('Eight independent chases at once.'), findsNothing);

    await tester.tap(find.text('CLOSE'));
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

    // Expand the breakdown: the test's glyph, plus one for the documented
    // reading and none for the other.
    await tester.tap(find.textContaining('Memory latency (pointer-chase)'));
    await tester.pump();
    expect(find.byIcon(Icons.info_outline), findsNWidgets(2));

    // The reading's own glyph is the second one — the test's comes first.
    await tester.tap(find.byIcon(Icons.info_outline).last);
    await tester.pump();

    expect(find.text('Eight independent chases at once.'), findsOneWidget);
    // Titled by the reading, not the test.
    expect(find.text('DRAM X8'), findsOneWidget);
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
    // test's own number — it appears once, in the table, not in the header.
    expect(find.text('300 TFLOPS'), findsOneWidget);
  });

  testWidgets('meters are magnitude, even where lower is better',
      (tester) async {
    // A bar beside a number is read as that number's size.  Scaling it by
    // which reading is best instead drew the fastest time as the longest bar,
    // which looked simply wrong.
    final doc = RunDocument();
    doc.runFor('Metal', 'Metal', 'M1 Pro', '').openTest(_header(
          id: 'kernel_launch_latency',
          title: 'Kernel launch latency',
          shape: TestShape.heterogeneous,
          units: const Units(
              symbol: 's', quantity: Quantity.seconds),
        ))
      ..metrics.add(const MetricResult(id: 'dispatch', value: 5.24))
      ..metrics.add(const MetricResult(id: 'roundtrip', value: 184.0));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // Nothing tells the reader to invert the bars, because they are not.
    expect(find.textContaining('LOWER IS BETTER'), findsNothing);

    final t = doc.runs.single.categories.single.tests.single;
    expect(t.barFraction(t.metrics.first), closeTo(5.24 / 184.0, 1e-9));
    expect(t.barFraction(t.metrics.last), 1.0);
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

    // The unavailable panel starts collapsed; open it, then the test's info.
    await tester.tap(find.text('NOT AVAILABLE ON THIS DEVICE (1)'));
    await tester.pump();
    await tester.tap(find.byIcon(Icons.info_outline));
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

    // The measured readings are in the table; the missing one is not.
    expect(find.text('fp32'), findsOneWidget);
    expect(find.text('fp16'), findsOneWidget);
    expect(find.text('bf16'), findsNothing);

    await tester.tap(find.text('NOT AVAILABLE ON THIS DEVICE (1)'));
    await tester.pump();
    expect(find.textContaining('MPS GEMM peak › bf16'), findsOneWidget);
    expect(find.text('requires M3+'), findsOneWidget);
  });
}
