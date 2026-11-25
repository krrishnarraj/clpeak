// The results screen's Summary and header on a saved format 4 run (the SoC
// laptop fixture): a kind card per processor with its flagship numbers, each
// linking to the backend that reached it and opening every backend's best;
// a way into every backend's tab; and a header that names the run and has
// one action, Share.  A live run, which has no file yet, has only its
// backends' tabs.
//
// Pure Dart — ResultsBody takes a RunDocument, so no native bridge is needed.
import 'dart:convert';
import 'dart:io';

import 'package:xpupeak/src/model/headlines.dart';
import 'package:xpupeak/src/model/result_model.dart';
import 'package:xpupeak/src/model/run_document.dart';
import 'package:xpupeak/src/model/run_summary.dart';
import 'package:xpupeak/src/theme/xpupeak_theme.dart';
import 'package:xpupeak/src/ui/results/results_body.dart';
import 'package:xpupeak/src/ui/results/results_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

RunDocument _laptop() => RunDocument.fromJson(jsonDecode(
        File('test/fixtures/soc_laptop.xpupeak.json').readAsStringSync())
    as Map<String, dynamic>);

final _catalog = HeadlineCatalog.fromJson(
    jsonDecode(File('assets/measures.json').readAsStringSync())
        as Map<String, dynamic>);

Widget _host(Widget child, {bool dark = false}) => MaterialApp(
      theme: dark ? XpuPeakTheme.dark() : XpuPeakTheme.light(),
      home: Scaffold(body: child),
    );

/// A desktop window, or a phone.
void _size(WidgetTester tester, {required bool wide}) {
  tester.view.physicalSize =
      wide ? const Size(2560, 2000) : const Size(1170, 2532);
  tester.view.devicePixelRatio = wide ? 2.0 : 3.0;
  addTearDown(tester.view.reset);
}

void main() {
  for (final wide in [true, false]) {
    final where = wide ? 'desktop' : 'phone';

    testWidgets('the summary is a kind card per processor ($where)',
        (tester) async {
      _size(tester, wide: wide);
      await tester.pumpWidget(
          _host(ResultsBody(document: _laptop(), catalog: _catalog)));
      await tester.pump();
      expect(tester.takeException(), isNull);

      // Summary first, then a tab per backend with its test count.
      expect(find.byKey(const ValueKey('results-tab:Summary')), findsOneWidget);
      expect(find.byKey(const ValueKey('results-tab:CoreML')), findsOneWidget);

      // Each processor: its kind and cores, its name, and three numbers.
      for (final label in ['CPU · 10 CORES', 'GPU · 16 CORES', 'NPU · 16 CORES']) {
        expect(find.text(label), findsOneWidget);
      }
      expect(find.text('Apple Neural Engine'), findsOneWidget);
      expect(find.text(wide ? 'Floating point' : 'Float'), findsNWidgets(3));
      expect(find.text(wide ? 'Memory bandwidth' : 'Memory'), findsNWidgets(3));
      // Where a processor measured none, it says so.
      expect(find.text('Not measured'), findsOneWidget);
      // The GPU's best floating-point number, and who reached it.
      expect(find.text('via Metal'), findsOneWidget);

      // Then a way into every backend -- below the cards, which on a phone
      // fill the screen; the list builds a row as it is reached.
      await tester.scrollUntilVisible(find.text('6 tests on 3 devices'), 300,
          scrollable: find
              .descendant(
                  of: find.byType(ListView), matching: find.byType(Scrollable))
              .first);
      expect(find.text('Explore each backend'), findsOneWidget);
    });

    testWidgets('"via" opens that backend on that device ($where)',
        (tester) async {
      _size(tester, wide: wide);
      await tester.pumpWidget(
          _host(ResultsBody(document: _laptop(), catalog: _catalog)));
      await tester.pump();

      await tester.tap(find.text('via CoreML'));
      await tester.pump();
      await tester.pump();

      // Core ML's tab, on the Neural Engine its best was measured on.
      expect(find.text('6 tests on 3 devices'), findsOneWidget);
      expect(find.text('NPU · COREML'), findsOneWidget);
      expect(find.text('Expand all'), findsOneWidget);
      expect(find.text('Collapse all'), findsOneWidget);

      // The tab on show is on screen, even past the edge of a phone's
      // sideways strip.
      final tab =
          tester.getRect(find.byKey(const ValueKey('results-tab:CoreML')));
      final width = tester.view.physicalSize.width / tester.view.devicePixelRatio;
      expect(tab.left, greaterThanOrEqualTo(0));
      expect(tab.right, lessThanOrEqualTo(width));
    });
  }

  testWidgets("a flagship number opens every backend's best", (tester) async {
    _size(tester, wide: true);
    await tester.pumpWidget(
        _host(ResultsBody(document: _laptop(), catalog: _catalog)));
    await tester.pump();

    // The CPU's integer number.
    await tester.tap(find.text('3.40  TOPS'));
    await tester.pump();
    expect(find.text('Integer · Apple M1 Pro'), findsOneWidget);
    expect(find.text("EACH BACKEND'S BEST"), findsOneWidget);
    await tester.tap(find.text('Close'));
    await tester.pump();
    expect(find.text("EACH BACKEND'S BEST"), findsNothing);
  });

  testWidgets('a backend tile opens its tab', (tester) async {
    _size(tester, wide: true);
    await tester.pumpWidget(
        _host(ResultsBody(document: _laptop(), catalog: _catalog)));
    await tester.pump();

    await tester.tap(find.text('7 tests on 1 device'));
    await tester.pump();
    expect(find.text('CPU · CPU'), findsOneWidget);
    expect(find.text('Explore each backend'), findsNothing);
  });

  testWidgets('the header names the run and has one action, Share',
      (tester) async {
    _size(tester, wide: true);
    final doc = _laptop();
    final summary = RunSummary.fromDocument(
      id: '20261020_101500',
      fileName: '20261020_101500.xpupeak.json',
      doc: doc,
      startedAt: DateTime(2026, 10, 20, 10, 15),
      durationMs: 148000,
      cancelled: false,
    );
    await tester.pumpWidget(MaterialApp(
      theme: XpuPeakTheme.light(),
      home: SavedResultsScreen(document: doc, summary: summary),
    ));
    await tester.pump();
    expect(tester.takeException(), isNull);

    // The machine's name, its model year below it with the OS and the day.
    expect(find.text('MacBook Pro'), findsOneWidget);
    expect(find.text('14-inch, 2021 · macOS 27.2 · 20 Oct 2026, 10:15'),
        findsOneWidget);
    expect(find.text('Share'), findsOneWidget);
    expect(find.byTooltip('Export JSON'), findsNothing);
  });

  testWidgets('a run the user named keeps its name', (tester) async {
    _size(tester, wide: false);
    final doc = _laptop();
    final summary = RunSummary.fromDocument(
      id: '20261020_101500',
      fileName: '20261020_101500.xpupeak.json',
      doc: doc,
      startedAt: DateTime(2026, 10, 20, 10, 15),
      durationMs: 148000,
      cancelled: false,
    ).withName('Before the update');
    await tester.pumpWidget(MaterialApp(
      theme: XpuPeakTheme.dark(),
      home: SavedResultsScreen(document: doc, summary: summary),
    ));
    await tester.pump();
    expect(tester.takeException(), isNull);
    expect(find.text('Before the update'), findsOneWidget);
    expect(
        find.text(
            'MacBook Pro (14-inch, 2021) · macOS 27.2 · 20 Oct 2026, 10:15'),
        findsOneWidget);
  });

  testWidgets('a live run has only its backends, following the one in flight',
      (tester) async {
    final doc = RunDocument();
    for (final backend in ['Metal', 'CPU']) {
      doc
          .runFor(backend, backend, 'M1 Pro', '')
          .openTest(const TestHeader(
            id: 'global_memory_bandwidth',
            title: 'Global memory bandwidth',
            category: BenchCategory.bandwidth,
            shape: TestShape.homogeneous,
            units: Units(symbol: 'B/s', quantity: Quantity.bytesPerSecond),
          ))
          .metrics
          .add(const MetricResult(id: 'float', value: 1.8e11));
    }
    await tester.pumpWidget(_host(ResultsBody(document: doc, catalog: _catalog),
        dark: true));
    await tester.pump();

    expect(find.byKey(const ValueKey('results-tab:Summary')), findsNothing);
    expect(find.byKey(const ValueKey('results-tab:Metal')), findsOneWidget);
    // The device in flight is the last one to report.
    expect(find.text('DEVICE · CPU'), findsOneWidget);
  });
}
