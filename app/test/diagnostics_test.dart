// The diagnostic stream end to end on the Dart side: a `log` event decodes
// into the same entry the file holds, the results view shows the problems
// outright and folds the rest, and the verbose setting round-trips.
//
// Pure Dart — no native bridge needed.
import 'package:clpeak/src/ffi/clpeak_events.dart';
import 'package:clpeak/src/model/run_document.dart';
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
