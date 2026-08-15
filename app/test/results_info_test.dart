// The info affordance on the results view, at both levels: a glyph on a test's
// row for what the test measures, and one on each documented reading in the
// breakdown for what that reading means — each opening only its own text.
//
// Pure Dart — ResultsBody takes a RunDocument, so no native bridge is needed.
import 'package:clpeak/src/model/result_entry.dart';
import 'package:clpeak/src/model/run_document.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/results/results_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

ResultEntry _entry({
  required String test,
  required String display,
  required String metric,
  required double value,
  String description = '',
  String metricDescription = '',
}) =>
    ResultEntry(
      backend: 'CPU',
      platform: 'CPU',
      device: 'M1 Pro',
      driver: '',
      category: 'latency',
      test: test,
      display: display,
      metric: metric,
      unit: 'ns',
      status: ResultStatus.ok,
      value: value,
      reason: '',
      description: description,
      metricDescription: metricDescription,
    );

Widget _host(RunDocument doc) => MaterialApp(
      theme: ClpeakTheme.dark(),
      home: Scaffold(body: ResultsBody(document: doc)),
    );

void main() {
  testWidgets('a documented test offers info; an undocumented one does not',
      (tester) async {
    final doc = RunDocument();
    doc.addEntry(_entry(
      test: 'memory_latency',
      display: 'Memory latency (pointer-chase)',
      metric: 'DRAM x8',
      value: 16.15,
      description: 'What the wait for one memory read costs.',
      metricDescription: 'Eight independent chases at once.',
    ));
    doc.addEntry(_entry(
      test: 'atomics',
      display: 'Atomic fetch-add latency',
      metric: 'uncontended ST',
      value: 2.2,
    ));

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
    doc.addEntry(_entry(
      test: 'memory_latency',
      display: 'Memory latency (pointer-chase)',
      metric: 'DRAM x8',
      value: 16.15,
      description: 'What the wait for one memory read costs.',
      metricDescription: 'Eight independent chases at once.',
    ));
    // A second reading of the same test, undocumented.
    doc.addEntry(_entry(
      test: 'memory_latency',
      display: 'Memory latency (pointer-chase)',
      metric: 'L1',
      value: 1.26,
      description: 'What the wait for one memory read costs.',
    ));

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

  testWidgets('an unsupported test can still be explained', (tester) async {
    final doc = RunDocument();
    doc.addEntry(ResultEntry(
      backend: 'CPU',
      platform: 'CPU',
      device: 'M1 Pro',
      driver: '',
      category: 'fp_compute',
      test: 'amx',
      display: 'Matrix engine (AMX)',
      metric: 'bf16',
      unit: 'tflops',
      status: ResultStatus.unsupported,
      value: 0,
      reason: 'no AMX on this CPU',
      description: 'Throughput of the dedicated matrix-multiply unit.',
    ));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // The unsupported panel starts collapsed; open it, then the test's info.
    await tester.tap(find.text('NOT SUPPORTED ON THIS DEVICE (1)'));
    await tester.pump();
    await tester.tap(find.byIcon(Icons.info_outline));
    await tester.pump();

    expect(find.text('Throughput of the dedicated matrix-multiply unit.'),
        findsOneWidget);
  });
}
