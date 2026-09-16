// The diagnostic stream end to end on the Dart side: a `log` event decodes
// into the same entry the file holds, the results view shows the problems
// outright and folds the rest, a run-log sidecar left behind by a crash is
// adopted by history, and the verbose setting round-trips.
//
// Pure Dart — no native bridge needed.
import 'dart:convert';
import 'dart:io';

import 'package:clpeak/src/ffi/clpeak_events.dart';
import 'package:clpeak/src/model/run_document.dart';
import 'package:clpeak/src/services/run_history_store.dart';
import 'package:clpeak/src/services/settings_service.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/results/results_body.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

Widget _host(RunDocument doc) => MaterialApp(
      theme: ClpeakTheme.dark(),
      home: Scaffold(body: ResultsBody(document: doc)),
    );

String _header({bool verbose = false}) => jsonEncode({
      'schema': 'clpeak/run-log',
      'format_version': 3,
      'clpeak_version': '3.0.0',
      'generated_at': '2026-09-15T06:05:57Z',
      'invocation': {
        'argv': ['clpeak', if (verbose) '--verbose'],
        if (verbose) 'verbose': true,
      },
    });

String _line(String level, String message,
        {String backend = '', String source = ''}) =>
    jsonEncode({
      'elapsed_s': 1.5,
      'level': level,
      if (source.isNotEmpty) 'source': source,
      if (backend.isNotEmpty) 'backend': backend,
      'message': message,
    });

void main() {
  group('log events', () {
    test('decode into the entry the file records', () {
      final event = ClpeakEvent.fromJson({
        't': 'log',
        'backend': 'Vulkan',
        'platform': 'Vulkan',
        'device': 'Apple M1 Pro',
        'driver': '26.1.99',
        'device_index': 1,
        'level': 'warning',
        'source': 'vulkan',
        'elapsed_s': 4.25,
        'test': 'kernel_latency',
        'variant': '',
        'message': 'query timeout (VK_ERROR_DEVICE_LOST)',
      }) as LogEntryEvent;
      final e = event.entry;
      expect(e.level, LogLevel.warning);
      expect(e.source, 'vulkan');
      expect(e.elapsedSeconds, 4.25);
      expect(e.deviceIndex, 1);
      expect(e.test, 'kernel_latency');
      expect(e.scope, 'Vulkan · Apple M1 Pro · kernel_latency');
      expect(e.level.isProblem, isTrue);
    });

    test('a variant joins the test key, as on metric events', () {
      final event = ClpeakEvent.fromJson({
        't': 'log',
        'level': 'debug',
        'test': 'single_precision_compute',
        'variant': 'NEON',
        'message': 'x',
      }) as LogEntryEvent;
      expect(event.entry.test, 'single_precision_compute@NEON');
      expect(event.entry.level.isProblem, isFalse);
    });

    test('an unknown event kind becomes a warning, not a crash', () {
      final event = ClpeakEvent.fromJson({'t': 'brand_new'}) as LogEntryEvent;
      expect(event.entry.level, LogLevel.warning);
      expect(event.entry.message, contains('unknown event'));
    });
  });

  group('results diagnostics section', () {
    testWidgets('problems are shown outright; the rest is folded',
        (tester) async {
      final doc = RunDocument();
      doc.runFor('CPU', 'CPU', 'M1 Pro', '');
      doc.log.addAll(const [
        LogEntry(
            level: LogLevel.warning,
            backend: 'ONNX',
            message: 'ONNX: this onnxruntime build exposes CPU providers only'),
        LogEntry(
            level: LogLevel.debug,
            backend: 'CPU',
            message: '[cpu] STREAM array 512 MB x3'),
      ]);

      await tester.pumpWidget(_host(doc));
      await tester.pump();

      expect(find.text('DIAGNOSTICS'), findsOneWidget);
      expect(find.text('1 warning, 2 lines'), findsOneWidget);
      expect(find.textContaining('CPU providers only'), findsOneWidget);
      // Debug lines wait behind the fold.
      expect(find.textContaining('STREAM array'), findsNothing);
      expect(find.text('FULL LOG (2)'), findsOneWidget);

      await tester.tap(find.text('FULL LOG (2)'));
      await tester.pump();
      expect(find.textContaining('STREAM array'), findsOneWidget);
    });

    testWidgets('a run with no diagnostics shows no section', (tester) async {
      final doc = RunDocument();
      doc.runFor('CPU', 'CPU', 'M1 Pro', '');
      await tester.pumpWidget(_host(doc));
      await tester.pump();
      expect(find.text('DIAGNOSTICS'), findsNothing);
    });
  });

  group('crashed runs', () {
    late Directory dir;
    setUp(() {
      dir = Directory.systemTemp.createTempSync('clpeak_crashlog');
    });
    tearDown(() => dir.deleteSync(recursive: true));

    test('a sidecar without its document is a crashed run', () async {
      await File('${dir.path}/20260915_120000.clpeak.log').writeAsString([
        _header(verbose: true),
        _line('debug', 'Vulkan: working set 512 MB', backend: 'Vulkan'),
        _line('warning', 'query timeout', backend: 'Vulkan', source: 'vulkan'),
        // The line the crash cut short.
        '{"elapsed_s": 9.1, "level": "debug", "mes',
      ].join('\n'));

      final store = RunHistoryStore(directoryOverride: dir);
      final logs = await store.listCrashLogs();
      expect(logs, hasLength(1));
      final log = logs.single;
      expect(log.id, '20260915_120000');
      expect(log.fileName, '20260915_120000.clpeak.log');
      expect(log.clpeakVersion, '3.0.0');
      expect(log.verbose, isTrue);
      expect(log.entries, 2);
      expect(log.backends, ['Vulkan']);
      expect(log.lastEntry!.message, 'query timeout');
      expect(log.startedAt.toUtc().hour, 6);

      // The store lists documents only; the sidecar is not mistaken for one.
      expect(await store.list(), isEmpty);

      await store.deleteCrashLog(log);
      expect(await store.listCrashLogs(), isEmpty);
    });

    test('the in-flight run and a stale sidecar are not crashes', () async {
      await File('${dir.path}/live.clpeak.log').writeAsString(_header());
      await File('${dir.path}/done.clpeak.log').writeAsString(_header());
      await File('${dir.path}/done.clpeak.json')
          .writeAsString(jsonEncode({'format_version': 3, 'devices': []}));

      final store = RunHistoryStore(directoryOverride: dir);
      final logs = await store.listCrashLogs(inFlightId: 'live');
      expect(logs, isEmpty);
      // A sidecar whose run did finish is cleaned up; the live one is left.
      expect(File('${dir.path}/done.clpeak.log').existsSync(), isFalse);
      expect(File('${dir.path}/live.clpeak.log').existsSync(), isTrue);
    });

    test('a file that is not a run log is ignored', () async {
      await File('${dir.path}/junk.clpeak.log').writeAsString('not json');
      final store = RunHistoryStore(directoryOverride: dir);
      expect(await store.listCrashLogs(), isEmpty);
    });
  });

  group('verbose setting', () {
    test('defaults off and persists', () async {
      SharedPreferences.setMockInitialValues({});
      final settings = await SettingsService.load();
      expect(settings.verbose, isFalse);
      await settings.setVerbose(true);
      expect(settings.verbose, isTrue);
      expect((await SharedPreferences.getInstance()).getBool('verbose'),
          isTrue);

      final reloaded = await SettingsService.load();
      expect(reloaded.verbose, isTrue);
    });
  });
}
