// Integration test for the native bridge: requires a built clpeak_ffi
// library, pointed at via CLPEAK_FFI_PATH, e.g.
//   CLPEAK_FFI_PATH=$PWD/../build-gui/clpeak_ffi.framework/clpeak_ffi \
//     flutter test test/ffi_integration_test.dart
// Skipped automatically when the variable is unset (pure-Dart CI).
import 'dart:io';

import 'package:clpeak/src/ffi/clpeak_bindings.dart';
import 'package:clpeak/src/ffi/clpeak_events.dart';
import 'package:clpeak/src/ffi/clpeak_runner.dart';
import 'package:clpeak/src/model/result_entry.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  final ffiPath = Platform.environment['CLPEAK_FFI_PATH'];
  final skip = ffiPath == null || ffiPath.isEmpty
      ? 'CLPEAK_FFI_PATH not set — native bridge not available'
      : false;

  group('clpeak_ffi', () {
    late ClpeakBindings bindings;

    setUpAll(() {
      bindings = ClpeakBindings.open();
    });

    test('version and catalog', () {
      expect(bindings.version(), isNotEmpty);
      final catalog = bindings.backendCatalog();
      final backends = catalog['backends'] as List;
      expect(backends, isNotEmpty);
      expect(backends.any((b) => b['name'] == 'CPU'), isTrue);
    });

    test('cpu quick run streams events and saves xml', () async {
      final xml = File(
          '${Directory.systemTemp.path}/clpeak_dart_ffi_test.xml');
      if (xml.existsSync()) xml.deleteSync();

      final run = ClpeakRunner(bindings)
          .start(['--cpu', '-i', '1', '--xml-file', xml.path]);
      final events = await run.events.toList();
      final rc = await run.result;

      expect(rc, clpeakRunOk);
      expect(events.last, isA<DoneEvent>());
      final metrics = events.whereType<MetricEvent>().toList();
      expect(metrics, isNotEmpty);
      expect(metrics.first.entry.backend, 'CPU');
      expect(
          metrics.any((m) =>
              m.entry.status == ResultStatus.ok && m.entry.value > 0),
          isTrue);
      expect(xml.existsSync(), isTrue);

      // Round-trip the saved file through the native loader.
      final doc = bindings.loadResultFile(xml.path);
      expect(doc, isNotNull);
      expect((doc!['entries'] as List).length, metrics.length);
      xml.deleteSync();
    });

    test('bad args are rejected without side effects', () async {
      final run = ClpeakRunner(bindings).start(['--bogus-flag']);
      final events = await run.events.toList();
      expect(await run.result, clpeakRunBadArgs);
      expect(events.whereType<NoteEvent>(), isNotEmpty);
      final done = events.last as DoneEvent;
      expect(done.status, clpeakRunBadArgs);
    });

    test('cancel stops a long run early', () async {
      final run = ClpeakRunner(bindings).start(['--cpu']);
      final sw = Stopwatch()..start();
      Future.delayed(const Duration(seconds: 2), run.cancel);
      final events = await run.events.toList();
      sw.stop();
      expect(await run.result, clpeakRunCancelled);
      expect((events.last as DoneEvent).cancelled, isTrue);
      expect(sw.elapsed.inSeconds, lessThan(30));
    });
  }, skip: skip);
}
