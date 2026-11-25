// A run measured in parts -- continued with --resume after it stopped
// part-way -- is marked "Resumed" in History.  The mark is the document's
// `resumed` (docs/format-v4.md, Resumed runs), carried on the index row; a
// row an older build indexed has no such key, reads false, and takes it
// from its file the next time the list is read.
//
// Pure Dart + real files, no native bridge: HistoryScreen is fed a
// BenchmarkService built on stub bindings and a temp runs directory.
import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:xpupeak/src/ffi/xpupeak_bindings.dart';
import 'package:xpupeak/src/model/run_document.dart';
import 'package:xpupeak/src/model/run_summary.dart';
import 'package:xpupeak/src/services/benchmark_service.dart';
import 'package:xpupeak/src/services/export_service.dart';
import 'package:xpupeak/src/services/run_history_store.dart';
import 'package:xpupeak/src/theme/xpupeak_theme.dart';
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

Map<String, dynamic> _fixture(String name) =>
    jsonDecode(File('test/fixtures/$name.xpupeak.json').readAsStringSync())
        as Map<String, dynamic>;

/// The laptop fixture, continued once after the app was closed part-way.
Map<String, dynamic> _resumedLaptop() => _fixture('soc_laptop')
  ..['resumed'] = [
    {
      'at': '2026-11-09T14:20:00Z',
      'elapsed_s': 1009.0,
      'after': 'USER_REQUESTED',
      'from': {
        'backend': 'Metal',
        'device': 'Apple M1 Pro',
        'device_index': 0,
        'test': 'single-precision-compute',
      },
    },
  ];

/// An index row as a build before `resumed` wrote it.
Map<String, dynamic> _oldRow(String id) => {
      'id': id,
      'startedAt': DateTime(2026, 11, 9, 14, 3).toIso8601String(),
      'durationMs': 1000,
      'devices': ['Apple M1 Pro'],
      'backends': ['Metal'],
      'cancelled': false,
      'fileName': '$id.xpupeak.json',
    };

/// Write [docs] under their ids, and index [indexed] of them as an older
/// build did; the rest are left to be adopted from disk.
Future<void> _store(Directory dir, Map<String, Map<String, dynamic>> docs,
    {List<String> indexed = const []}) async {
  for (final e in docs.entries) {
    await File('${dir.path}/${e.key}.xpupeak.json')
        .writeAsString(jsonEncode(e.value));
  }
  await File('${dir.path}/index.json').writeAsString(jsonEncode({
    'runs': [for (final id in indexed) _oldRow(id)],
  }));
}

void main() {
  test('a summary of a resumed document is resumed, and keeps it in json',
      () {
    final doc = RunDocument.fromJson(_resumedLaptop());
    final summary = RunSummary.fromDocument(
      id: 'parts',
      fileName: 'parts.xpupeak.json',
      doc: doc,
      startedAt: DateTime(2026, 11, 9),
      durationMs: 1000,
      cancelled: false,
    );
    expect(summary.resumed, isTrue);
    expect(summary.toJson()['resumed'], isTrue);
    expect(RunSummary.fromJson(summary.toJson()).resumed, isTrue);
    expect(summary.withName('after undervolt').resumed, isTrue);

    final once = RunSummary.fromDocument(
      id: 'once',
      fileName: 'once.xpupeak.json',
      doc: RunDocument.fromJson(_fixture('soc_laptop')),
      startedAt: DateTime(2026, 11, 9),
      durationMs: 1000,
      cancelled: false,
    );
    expect(once.resumed, isFalse);
    // Written only where it is true, as `cancelled` in the file is.
    expect(once.toJson().containsKey('resumed'), isFalse);
    // A row from before the key reads as a run done in one go.
    expect(RunSummary.fromJson(_oldRow('old')).resumed, isFalse);
  });

  test('list() takes resumed from the file for old rows and adopted ones',
      () async {
    final dir = Directory.systemTemp.createTempSync('xpupeak_hist_resumed');
    addTearDown(() => dir.deleteSync(recursive: true));
    await _store(
      dir,
      {
        'parts': _resumedLaptop(),
        'once': _fixture('soc_laptop'),
        'adopted': _resumedLaptop(),
      },
      indexed: ['parts', 'once'],
    );

    final runs = await RunHistoryStore(directoryOverride: dir).list();
    final byId = {for (final r in runs) r.id: r};
    expect(byId.keys, unorderedEquals(['parts', 'once', 'adopted']));
    expect(byId['parts']!.resumed, isTrue);
    expect(byId['once']!.resumed, isFalse);
    expect(byId['adopted']!.resumed, isTrue);

    // The index was rewritten with it, so the shell's rows say so too.
    final index = jsonDecode(File('${dir.path}/index.json').readAsStringSync())
        as Map<String, dynamic>;
    final resumed = {
      for (final r in (index['runs'] as List).cast<Map<String, dynamic>>())
        r['id']: r['resumed'] == true,
    };
    expect(resumed, {'parts': true, 'once': false, 'adopted': true});
  });

  testWidgets('History tags a resumed run "Resumed"', (tester) async {
    tester.view.physicalSize = const Size(1280, 2000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    final dir = Directory.systemTemp.createTempSync('xpupeak_hist_resumed');
    addTearDown(() => dir.deleteSync(recursive: true));
    await tester.runAsync(() => _store(dir, {
          'parts': _resumedLaptop(),
          'desktop': _fixture('desktop_two_packages'),
        }, indexed: ['parts']));

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
    final desktop = find.text('Threadripper workstation');
    for (var i = 0; i < 60 && desktop.evaluate().isEmpty; i++) {
      await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 30)));
      await tester.pump();
    }
    await tester.pump();

    expect(find.text('2 runs'), findsOneWidget);
    // The laptop run alone was measured in parts.
    expect(find.text('Resumed'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });
}
