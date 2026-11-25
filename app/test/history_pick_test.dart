// History's pick of two runs, the one way into a compare: Compare runs
// turns the list into a pick, rows take an A or B badge, a third pick drops
// the oldest, the floating bar says what each pick is and what kind of
// compare the two make, and Compare opens it.
//
// Pure Dart + real files, no native bridge: HistoryScreen is fed a
// BenchmarkService built on stub bindings, and a temp runs directory holding
// the v4 fixtures (adopted from disk as History adopts a file it finds).
import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:xpupeak/src/ffi/xpupeak_bindings.dart';
import 'package:xpupeak/src/services/benchmark_service.dart';
import 'package:xpupeak/src/services/export_service.dart';
import 'package:xpupeak/src/services/run_history_store.dart';
import 'package:xpupeak/src/theme/xpupeak_theme.dart';
import 'package:xpupeak/src/ui/compare/compare_screen.dart';
import 'package:xpupeak/src/ui/history/history_screen.dart';

class _StubBindings implements XpuPeakBindings {
  @override
  XpuPeakLaunch get launch => (_, _, _, _) => 0;

  @override
  XpuPeakRequestCancel get requestCancel => () {};

  @override
  void setOnScreen(bool onScreen) {}

  @override
  void setPaused(bool paused) {}

  @override
  void setHostFacts(Map<String, Object> facts) {}

  @override
  String version() => 'test';

  @override
  String? takeString(Pointer<Utf8> ptr) => null;

  @override
  Map<String, dynamic> backendCatalog() => const {'backends': []};

  @override
  void setOnnxLibrary(String path) {}

  @override
  void setOnnxEpLibraries(List<OnnxEpLibrary> libs) {}

  @override
  void setOnnxWinml({required bool enabled, required String path}) {}

  @override
  OnnxStatus onnxStatus() => const OnnxStatus.unavailable('test');

  @override
  void setLitertLibrary(String path) {}

  @override
  void setLitertNpuStageDir(String dir) {}

  @override
  LitertStatus litertStatus() => const LitertStatus.unavailable('test');
}

const _laptopName = 'MacBook Pro (14-inch, 2021)';
const _desktopName = 'Threadripper workstation';

/// A run's row in the list (the bar names the picked runs too).
Finder _row(String name) => find.descendant(
    of: find.byKey(const ValueKey('history-list')), matching: find.text(name));

/// A pick in the bar: its run's name and day.
Finder _pick(String text) => find.byWidgetPredicate(
    (w) => w is Semantics && w.properties.label == text);

Future<void> _settle(WidgetTester tester, Finder until) async {
  for (var i = 0; i < 60 && until.evaluate().isEmpty; i++) {
    await tester.runAsync(
        () => Future<void>.delayed(const Duration(milliseconds: 30)));
    await tester.pump();
  }
  await tester.pump();
}

void main() {
  testWidgets('pick two runs, and Compare opens them', (tester) async {
    tester.view.physicalSize = const Size(1280, 2000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    final dir = Directory.systemTemp.createTempSync('xpupeak_hist_pick');
    addTearDown(() => dir.deleteSync(recursive: true));
    await tester.runAsync(() async {
      String fixture(String name) =>
          File('test/fixtures/$name.xpupeak.json').readAsStringSync();
      final laptop = fixture('soc_laptop');
      // The laptop measured again, a week earlier: the same hardware.
      final again = jsonDecode(laptop) as Map<String, dynamic>
        ..['generated_at'] = '2026-11-02T10:00:00Z';
      await File('${dir.path}/laptop.xpupeak.json').writeAsString(laptop);
      await File('${dir.path}/again.xpupeak.json')
          .writeAsString(jsonEncode(again));
      await File('${dir.path}/desktop.xpupeak.json')
          .writeAsString(fixture('desktop_two_packages'));
      await File('${dir.path}/origins.json').writeAsString(jsonEncode({
        'laptop': 'measured',
        'again': 'measured',
        'desktop': 'imported',
      }));
    });

    final history = RunHistoryStore(directoryOverride: dir);
    final service = BenchmarkService(_StubBindings(), history);
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          Provider.value(value: history),
          Provider(create: (_) => ExportService()),
          ChangeNotifierProvider.value(value: service),
        ],
        child: MaterialApp(
          theme: XpuPeakTheme.light(),
          home: const Scaffold(body: HistoryScreen()),
        ),
      ),
    );
    await _settle(tester, find.text(_desktopName));

    expect(find.text('November 2026'), findsOneWidget);
    expect(find.text('3 runs'), findsOneWidget);
    expect(find.text('3 saved runs · 1 imported'), findsOneWidget);
    expect(find.text('Imported'), findsWidgets);

    // The filters: the imported run alone.
    await tester.tap(find.byKey(const ValueKey('history-filter:imported')));
    await tester.pump();
    expect(find.text(_laptopName), findsNothing);
    expect(find.text(_desktopName), findsOneWidget);
    await tester.tap(find.byKey(const ValueKey('history-filter:all')));
    await tester.pump();

    // Into the pick: the header's button turns to Cancel, the bar waits.
    await tester.tap(find.text('Compare runs'));
    await tester.pump();
    expect(find.text('Cancel'), findsWidgets);
    expect(find.text('Choose run A'), findsOneWidget);
    expect(find.text('then run B'), findsOneWidget);
    expect(find.text('The first run you pick is A, the second B'),
        findsOneWidget);
    // Compare waits for two: nothing opens.
    await tester.tap(find.byKey(const ValueKey('pick-bar-compare')));
    await tester.pump();
    expect(find.byType(CompareScreen), findsNothing);

    // A first: the desktop, then B: the newer laptop run.
    await tester.tap(_row(_desktopName));
    await tester.pump();
    expect(_pick('$_desktopName, 9 Nov'), findsOneWidget);
    expect(find.text('Choose run B'), findsOneWidget);
    await tester.tap(_row(_laptopName).first);
    await _settle(
        tester,
        find.text('Different hardware: compared by processor, and by '
            'backend'));
    expect(_pick('$_laptopName, 9 Nov'), findsOneWidget);
    expect(
        find.text('Different hardware: compared by processor, and by '
            'backend'),
        findsOneWidget);

    // A third pick drops the oldest: the laptop is A now, its earlier run B,
    // and the two are one machine.
    await tester.tap(_row(_laptopName).last);
    await _settle(
        tester,
        find.text('Same hardware: compared reading by reading, backend by '
            'backend'));
    expect(_pick('$_desktopName, 9 Nov'), findsNothing);
    expect(_pick('$_laptopName, 2 Nov'), findsOneWidget);

    // Unpicking B leaves A, and B to choose.
    await tester.tap(_row(_laptopName).last);
    await tester.pump();
    expect(find.text('Choose run B'), findsOneWidget);
    await tester.tap(_row(_desktopName));
    await _settle(
        tester,
        find.text('Different hardware: compared by processor, and by '
            'backend'));

    await tester.tap(find.byKey(const ValueKey('pick-bar-compare')));
    await _settle(tester, find.byType(CompareScreen));
    await tester.pumpAndSettle();
    expect(find.byType(CompareScreen), findsOneWidget);
    expect(find.text('Different hardware'), findsOneWidget);

    // Change on run A's card: back to the pick, B kept, A to choose again.
    await tester.tap(find.text('Change').first);
    await tester.pumpAndSettle();
    expect(find.byType(CompareScreen), findsNothing);
    expect(find.text('Choose run A'), findsOneWidget);
    expect(_pick('$_desktopName, 9 Nov'), findsOneWidget);
    await tester.tap(_row(_laptopName).last);
    await _settle(tester, _pick('$_laptopName, 2 Nov'));
    await tester.tap(find.byKey(const ValueKey('pick-bar-compare')));
    await _settle(tester, find.byType(CompareScreen));
    await tester.pumpAndSettle();
    expect(find.byType(CompareScreen), findsOneWidget);

    // Back to history ends the pick.
    await tester.tap(find.text('Back to history'));
    await tester.pumpAndSettle();
    expect(find.byType(CompareScreen), findsNothing);
    expect(find.text('Compare runs'), findsOneWidget);
    expect(find.text('Choose run A'), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
