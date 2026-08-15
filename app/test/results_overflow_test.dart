// Long names must wrap, never overflow and never get cut: a driver-reported
// device name ("Goldfish GFXStream (llvmpipe (LLVM 21.1.4, 128 bits))") is
// wider than a phone, and its tail is what tells two such devices apart.
// Wrapping is also what puts the info glyphs at risk of going ragged, so the
// glyph column is pinned here too.
//
// Pure Dart — ResultsBody takes a RunDocument, so no native bridge is needed.
import 'package:clpeak/src/model/result_entry.dart';
import 'package:clpeak/src/model/run_document.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/results/results_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

const _longDevice = 'Goldfish GFXStream (llvmpipe (LLVM 21.1.4, 128 bits))';
const _longTest = 'Half-precision compute fp16 (vector width 8)';
const _shortTest = 'Integer compute';

ResultEntry _entry({
  String backend = 'Vulkan',
  String device = _longDevice,
  String test = 'compute_hp',
  String display = _longTest,
  // Documented by default: an undocumented test has no glyph to place.
  String description = 'How fast the device does fp16 math.',
}) =>
    ResultEntry(
      backend: backend,
      platform: backend,
      device: device,
      driver: '24.2.0',
      category: 'fp_compute',
      test: test,
      display: display,
      metric: 'float8 uncontended',
      unit: 'gflops',
      status: ResultStatus.ok,
      value: 93.1,
      reason: '',
      description: description,
      metricDescription: '',
    );

/// The phone width, where names are widest relative to the screen.
void _phone(WidgetTester tester) {
  tester.view.physicalSize = const Size(1125, 2436);
  tester.view.devicePixelRatio = 3.0;
  addTearDown(tester.view.reset);
}

Widget _host(RunDocument doc) => MaterialApp(
      theme: ClpeakTheme.dark(),
      home: Scaffold(body: ResultsBody(document: doc)),
    );

void main() {
  testWidgets('long device and test names wrap instead of overflowing',
      (tester) async {
    _phone(tester);

    final doc = RunDocument();
    // Two runs, so the device selector chips are shown — the chip is the
    // narrowest place the device name has to fit.
    doc.addEntry(_entry());
    doc.addEntry(_entry(backend: 'CPU', device: 'Apple CPU (part 0x000)'));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // A RenderFlex overflow paints an error and reports it to the binding.
    expect(tester.takeException(), isNull);

    // Present in full, on however many lines it takes.
    expect(find.text('Vulkan · $_longDevice'), findsOneWidget);
    expect(find.text(_longTest), findsOneWidget);

    // Nothing clamps them to one line.
    for (final text in [
      tester.widget<Text>(find.text('Vulkan · $_longDevice')),
      tester.widget<Text>(find.text(_longTest)),
    ]) {
      expect(text.maxLines, isNull);
      expect(text.overflow, isNot(TextOverflow.ellipsis));
    }
  });

  testWidgets('info glyphs share one column, wrapped rows included',
      (tester) async {
    _phone(tester);

    final doc = RunDocument();
    // A title too long for the width, one that fits, and one with nothing to
    // explain — the third holds the column open without filling it.
    doc.addEntry(_entry());
    doc.addEntry(_entry(test: 'compute_int', display: _shortTest));
    doc.addEntry(_entry(
        test: 'compute_dp', display: 'Double-precision', description: ''));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    // The long one did wrap — otherwise this proves nothing.
    final wrapped = tester.getRect(find.text(_longTest));
    final plain = tester.getRect(find.text(_shortTest));
    expect(wrapped.height, greaterThan(plain.height));

    // Both glyphs in the same column, each on its own row.
    final glyphs = find.byIcon(Icons.info_outline);
    expect(glyphs, findsNWidgets(2));
    final first = tester.getRect(glyphs.at(0));
    final second = tester.getRect(glyphs.at(1));
    expect(first.left, second.left);
    expect(first.center.dy, inInclusiveRange(wrapped.top, wrapped.bottom));
    expect(second.center.dy, inInclusiveRange(plain.top, plain.bottom));

    // The undocumented row keeps the column open rather than letting its
    // title run into it.
    expect(tester.getRect(find.text('Double-precision')).right,
        lessThanOrEqualTo(first.left));
  });
}
