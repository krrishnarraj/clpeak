// The "runs that did not finish" footer used to overflow on a phone: its
// metadata line (date, log-line count, version) sat in an inflexible Text
// next to the backend tags, so a long version string painted the yellow
// overflow stripes instead of wrapping.
//
// Pure Dart + real files, no native bridge: HistoryScreen is fed a
// BenchmarkService built on stub bindings, and a temp runs directory holding
// crash-log sidecars shaped like the ones in the report.
import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';

import 'package:clpeak/src/ffi/clpeak_bindings.dart';
import 'package:clpeak/src/services/benchmark_service.dart';
import 'package:clpeak/src/services/export_service.dart';
import 'package:clpeak/src/services/run_history_store.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/history/history_screen.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

class _StubBindings implements ClpeakBindings {
  @override
  ClpeakLaunch get launch => (_, _, _, _) => 0;

  @override
  ClpeakRequestCancel get requestCancel => () {};

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

/// One `<id>.clpeak.log` sidecar: a JSON header line plus NDJSON entries.
Future<void> _writeSidecar(
  Directory dir,
  String id, {
  String version = '3.0.0-36-gabcdef1234567890',
  bool verbose = false,
  List<Map<String, dynamic>> entries = const [],
}) async {
  final header = jsonEncode({
    'schema': 'clpeak/run-log',
    'format_version': 3,
    'generated_at': '2026-09-16T15:18:00',
    'clpeak_version': version,
    'invocation': {'verbose': verbose},
  });
  final lines = [
    header,
    for (final e in entries) jsonEncode(e),
  ];
  await File('${dir.path}/$id.clpeak.log').writeAsString('${lines.join('\n')}\n');
}

Map<String, dynamic> _entry(String backend, String message) => {
      'level': 'info',
      'backend': backend,
      'message': message,
    };

void main() {
  testWidgets('unfinished-run footers wrap on a phone instead of overflowing',
      (tester) async {
    // A phone, where the footer is widest relative to the screen.
    tester.view.physicalSize = const Size(1125, 2436);
    tester.view.devicePixelRatio = 3.0;
    addTearDown(tester.view.reset);

    final dir = Directory.systemTemp.createTempSync('clpeak_hist_overflow');
    addTearDown(() => dir.deleteSync(recursive: true));

    // Disk IO hangs in the widget-test fake-async zone; run it for real.
    await tester.runAsync(() async {
      // The report: no backends, no log lines, long version — the metadata
      // alone is wider than the tile.
      await _writeSidecar(dir, '20260916_204812');
      // Same tile shape but with backend tags, verbose and a last line,
      // which makes the footer longer still.
      await _writeSidecar(
        dir,
        '20260916_204409',
        verbose: true,
        entries: [
          _entry('OpenCL', 'probing platform 0'),
          _entry('Vulkan', 'enumerating devices'),
        ],
      );
      // A finished run with a long device name and two backends, so the
      // saved-run footer (same row pattern) is covered too.
      await File('${dir.path}/20260918_151958.clpeak.json')
          .writeAsString(jsonEncode({'format_version': 3}));
      await File('${dir.path}/index.json').writeAsString(jsonEncode({
        'runs': [
          {
            'id': '20260918_151958',
            'startedAt': DateTime(2026, 9, 18, 15, 19).toIso8601String(),
            'durationMs': 11000,
            'devices': [
              'Goldfish GFXStream (llvmpipe (LLVM 21.1.4, 128 bits))',
            ],
            'backends': ['LiteRT', 'ONNX'],
            'cancelled': true,
            'fileName': '20260918_151958.clpeak.json',
            'clpeakVersion': '3.0.0-36-gabcdef1234567890',
            'hostOs': 'Android',
            'hostOsVersion': '16',
            'hostArch': 'arm64-v8a',
          },
        ],
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
          theme: ClpeakTheme.dark(),
          home: const Scaffold(body: HistoryScreen()),
        ),
      ),
    );

    // The list loads off disk; settle until both sections render.
    for (var i = 0;
        i < 50 && find.text('DID NOT FINISH').evaluate().isEmpty;
        i++) {
      await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 50)));
      await tester.pump();
    }
    await tester.pump();

    expect(find.text('DID NOT FINISH'), findsNWidgets(2));
    expect(find.text('RUNS THAT DID NOT FINISH'), findsOneWidget);
    expect(find.text('20260918_151958'), findsOneWidget);

    // A RenderFlex overflow paints an error and reports it to the binding.
    expect(tester.takeException(), isNull);
  });
}
