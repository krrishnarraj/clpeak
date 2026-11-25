// The compare screen (lib/src/ui/compare/) over the two hand-written v4
// fixtures in test/fixtures/: the M1 Pro laptop against sf1-ub's
// Threadripper with an RTX 5060 (different hardware), and the laptop
// against itself with some readings moved (the same hardware).  What is
// checked is what the screen lays out of the model: the mode, the
// processors strip and its unpaired NPU, the by-backend tabs, swap, and a
// same-hardware compare's tabs and chips with no derived numbers anywhere.
//
// Pure Dart: the documents are read with dart:convert, as history reads them.
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:xpupeak/src/model/compare.dart';
import 'package:xpupeak/src/model/headlines.dart';
import 'package:xpupeak/src/model/run_document.dart';
import 'package:xpupeak/src/theme/xpupeak_theme.dart';
import 'package:xpupeak/src/ui/common/kit.dart';
import 'package:xpupeak/src/ui/compare/compare_screen.dart';
import 'package:xpupeak/src/ui/compare/compare_words.dart';

Map<String, dynamic> _json(String name) =>
    jsonDecode(File('test/fixtures/$name.xpupeak.json').readAsStringSync())
        as Map<String, dynamic>;

CompareRun _run(Map<String, dynamic> json) =>
    CompareRun(RunDocument.fromJson(json));

final _catalog = HeadlineCatalog.fromJson(
    jsonDecode(File('assets/measures.json').readAsStringSync())
        as Map<String, dynamic>);

/// [json] with every reading of its first test with readings multiplied by
/// [factor]: the same machine, measured again.
Map<String, dynamic> _moved(Map<String, dynamic> json, double factor) {
  final copy = jsonDecode(jsonEncode(json)) as Map<String, dynamic>;
  copy['generated_at'] = '2026-11-13T09:30:00Z';
  for (final d in copy['devices'] as List) {
    for (final t in (d as Map)['tests'] as List) {
      final metrics = (t as Map)['metrics'] as List? ?? const [];
      if (metrics.isEmpty) continue;
      for (final m in metrics) {
        final v = (m as Map)['value'];
        if (v is num) m['value'] = v * factor;
      }
      return copy;
    }
  }
  return copy;
}

Future<void> _pump(WidgetTester tester, RunCompare c,
    {Size size = const Size(1280, 5000)}) async {
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    theme: XpuPeakTheme.light(),
    home: CompareScreen(compare: c),
  ));
  await tester.pump();
}

/// Every text on screen, rich text included.
Iterable<String> _texts(WidgetTester tester) => [
      for (final w in tester.widgetList<RichText>(find.byType(RichText)))
        w.text.toPlainText(),
    ];

void main() {
  final laptop = _json('soc_laptop');
  final desktop = _json('desktop_two_packages');

  testWidgets('two machines: processors paired, the NPU alone, by backend',
      (tester) async {
    final c = RunCompare.of(_run(laptop), _run(desktop), catalog: _catalog);
    expect(isSameHardware(c), isFalse);
    await _pump(tester, c);

    expect(find.text('Different hardware'), findsOneWidget);
    expect(find.text('Processors'), findsOneWidget);
    expect(find.text('Headlines'), findsOneWidget);
    // The Neural Engine has nothing to pair with on the Threadripper.
    expect(find.text('Nothing in B to pair with'), findsOneWidget);
    expect(find.text('unpaired'), findsOneWidget);
    expect(find.text('Measured on both'), findsOneWidget);

    // By backend: a tab per backend both runs have, and what one alone has.
    await tester.tap(find.text('By backend'));
    await tester.pump();
    expect(find.text('Backend by backend'), findsOneWidget);
    for (final b in c.backends) {
      expect(find.byKey(ValueKey('compare-backend:$b')), findsOneWidget);
    }
    final nameA = sideName(c.a), nameB = sideName(c.b);
    expect(
        find.text('Only on $nameA: '
            '${c.backendsOnlyA.map(backendName).join(', ')}'),
        findsOneWidget);
    expect(
        find.text('Only on $nameB: '
            '${c.backendsOnlyB.map(backendName).join(', ')}'),
        findsOneWidget);
    expect(find.text('Peaks through ${backendName(c.backends.first)}'),
        findsOneWidget);
    // The Threadripper's llvmpipe has no Vulkan partner on the M1 Pro.
    expect(find.textContaining('$nameB alone, through Vulkan: llvmpipe'),
        findsOneWidget);

    await tester.tap(find.byKey(const ValueKey('compare-backend:OpenCL')));
    await tester.pump();
    expect(find.text('Peaks through OpenCL'), findsOneWidget);
    expect(find.text('Peaks through Vulkan'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('swap turns the runs round', (tester) async {
    final c = RunCompare.of(_run(laptop), _run(desktop), catalog: _catalog);
    await _pump(tester, c);
    final nameA = sideName(c.a), nameB = sideName(c.b);
    double x(String name) => tester.getTopLeft(find.text(name).first).dx;
    expect(x(nameA), lessThan(x(nameB)));
    expect(find.text('Nothing in B to pair with'), findsOneWidget);

    await tester.tap(find.byTooltip('Swap A and B'));
    await tester.pump();
    expect(x(nameB), lessThan(x(nameA)));
    // The Neural Engine is B's now.
    expect(find.text('Nothing in A to pair with'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('one machine twice: reading by reading, no derived numbers',
      (tester) async {
    final c = RunCompare.of(_run(laptop), _run(_moved(laptop, 1.1)),
        catalog: _catalog);
    expect(isSameHardware(c), isTrue);
    await _pump(tester, c);

    expect(find.text('Same hardware'), findsOneWidget);
    expect(find.text('Reading by reading'), findsOneWidget);
    expect(find.text('Processors'), findsNothing);
    expect(find.text('By backend'), findsNothing);

    // A tab per backend, and on Core ML -- three devices, one per
    // processor -- a chip per device.
    await tester.tap(find.byKey(const ValueKey('compare-backend:CoreML')));
    await tester.pump();
    expect(find.byKey(const ValueKey('compare-device:CoreML:0')),
        findsOneWidget);
    expect(find.byKey(const ValueKey('compare-device:CoreML:2')),
        findsOneWidget);
    await tester.tap(find.byKey(const ValueKey('compare-device:CoreML:1')));
    await tester.pump();

    // The first backend's moved readings are marked, and no number is
    // derived from two: no "+x%" anywhere.
    await tester.tap(find.byKey(ValueKey('compare-backend:'
        '${c.a.document.runs.first.backend}')));
    await tester.pump();
    expect(
        find.byWidgetPredicate((w) => w is CValue && w.highlight != null),
        findsWidgets);
    final percent = RegExp(r'[+\-−]\s?\d+(\.\d+)?\s?%');
    expect(_texts(tester).where(percent.hasMatch), isEmpty);
    expect(tester.takeException(), isNull);
  });

  testWidgets('a phone lays out both modes without overflowing',
      (tester) async {
    for (final c in [
      RunCompare.of(_run(laptop), _run(desktop), catalog: _catalog),
      RunCompare.of(_run(laptop), _run(_moved(laptop, 0.9)),
          catalog: _catalog),
    ]) {
      await _pump(tester, c, size: const Size(390, 4000));
      expect(find.text('Compare'), findsOneWidget);
      if (!isSameHardware(c)) {
        await tester.tap(find.text('By backend'));
        await tester.pump();
      }
      expect(tester.takeException(), isNull);
    }
  });
}
