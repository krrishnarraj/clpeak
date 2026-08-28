// Long names must wrap, never overflow and never get cut: a driver-reported
// device name ("Goldfish GFXStream (llvmpipe (LLVM 21.1.4, 128 bits))") is
// wider than a phone, and its tail is what tells two such devices apart.
//
// The info glyph on a documented name rides in the text flow rather than a
// column reserved at the name's far edge — the reserved-column version put a
// visible gap between a short name and its glyph on a wide desktop window,
// since the column tracked the full row width, not the text.
//
// Pure Dart — ResultsBody takes a RunDocument, so no native bridge is needed.
import 'package:clpeak/src/model/result_entry.dart';
import 'package:clpeak/src/model/run_document.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/results/results_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';

const _longDevice = 'Goldfish GFXStream (llvmpipe (LLVM 21.1.4, 128 bits))';
const _longTest = 'Half-precision compute fp16 (vector width 8)';
const _shortTest = 'Integer compute';

ResultEntry _entry({
  String backend = 'Vulkan',
  String device = _longDevice,
  String test = 'compute_hp',
  String display = _longTest,
  String description = '',
  String? platform,
  String driver = '24.2.0',
}) =>
    ResultEntry(
      backend: backend,
      platform: platform ?? backend,
      device: device,
      driver: driver,
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

Widget _host(RunDocument doc) => MaterialApp(
      theme: ClpeakTheme.dark(),
      home: Scaffold(body: ResultsBody(document: doc)),
    );

void main() {
  testWidgets('long device and test names wrap instead of overflowing',
      (tester) async {
    // A phone, where the names are widest relative to the screen.
    tester.view.physicalSize = const Size(1125, 2436);
    tester.view.devicePixelRatio = 3.0;
    addTearDown(tester.view.reset);

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

  testWidgets(
      'a documented name keeps its glyph tight, even in a wide window',
      (tester) async {
    // A wide desktop window with plenty of room past either name — the
    // scenario the reserved-column glyph placement got visibly wrong: a
    // Text's *layout box* fills the row regardless of name length (that's
    // what the outer Expanded is for), so comparing glyph position against
    // the box edge can't tell the two placements apart. Comparing a short
    // name's glyph against a longer one's does: inline, the glyph follows
    // the visible text, so a longer name pushes its glyph further right; a
    // glyph anchored to the row's edge would land in the same place either
    // way.
    tester.view.physicalSize = const Size(2560, 1600);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    final doc = RunDocument();
    doc.addEntry(_entry(
      test: 'compute_int',
      display: _shortTest,
      description: 'How fast the device does plain int32 math.',
    ));
    doc.addEntry(_entry(
      test: 'compute_int_dp4a',
      display: 'Integer compute int32 dot-product',
      description: 'How fast the device does int32 dot-product math.',
    ));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    expect(tester.takeException(), isNull);

    final glyphs = find.byIcon(Icons.info_outline);
    expect(glyphs, findsNWidgets(2));
    final shortGlyph = tester.getRect(glyphs.at(0));
    final longGlyph = tester.getRect(glyphs.at(1));

    expect(shortGlyph.left, lessThan(longGlyph.left));
  });

  testWidgets('the driver line uses the width the panel has, not a fixed column',
      (tester) async {
    // The header's platform/driver line used to sit in a 260px column, so a
    // driver version got ellipsized away even here, with the panel hundreds
    // of pixels wider than the text.
    tester.view.physicalSize = const Size(2560, 1600);
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    const platform = 'Intel(R) OpenCL Graphics';
    const driver = '32.0.101.8993';

    final doc = RunDocument();
    doc.addEntry(_entry(
      backend: 'OpenCL',
      device: 'Intel(R) Arc(TM) A380 Graphics',
      platform: platform,
      driver: driver,
    ));

    await tester.pumpWidget(_host(doc));
    await tester.pump();

    expect(tester.takeException(), isNull);

    final line = find.text('$platform  ·  $driver');
    expect(line, findsOneWidget);

    // Laid out at its full one-line width: nothing was cut or wrapped away.
    final paragraph = tester.renderObject<RenderParagraph>(line);
    expect(paragraph.size.width,
        greaterThanOrEqualTo(paragraph.getMaxIntrinsicWidth(double.infinity) - 0.5));
  });
}
