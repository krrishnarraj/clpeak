// Integration test for the native bridge: requires a built clpeak_ffi
// library, pointed at via CLPEAK_FFI_PATH, e.g.
//   CLPEAK_FFI_PATH=$PWD/../build-gui/clpeak_ffi.framework/clpeak_ffi \
//     flutter test test/ffi_integration_test.dart
// Skipped automatically when the variable is unset (pure-Dart CI).
import 'dart:convert';
import 'dart:io';

import 'package:clpeak/src/ffi/clpeak_bindings.dart';
import 'package:clpeak/src/ffi/clpeak_events.dart';
import 'package:clpeak/src/ffi/clpeak_runner.dart';
import 'package:clpeak/src/model/result_model.dart';
import 'package:clpeak/src/model/run_document.dart';
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

    // A single `--cpu -i 1` sweep is ~35s on an M1 Pro and grows with the
    // CPU test list (cache bandwidth, DRAM chase, SMT scaling, crypto,
    // string, AMX, Apple BLAS), so it needs more than flutter_test's 30s
    // default.  Overrunning here is not a local failure: clpeak_launch
    // guards on a process-global flag, so the still-running sweep makes
    // every later launch in this file return CLPEAK_RUN_BUSY.
    test('cpu quick run streams events and saves a document',
        timeout: const Timeout(Duration(minutes: 5)), () async {
      final out = File(
          '${Directory.systemTemp.path}/clpeak_dart_ffi_test.clpeak.json');
      if (out.existsSync()) out.deleteSync();

      final run =
          ClpeakRunner(bindings).start(['--cpu', '-i', '1', '-o', out.path]);
      final events = await run.events.toList();
      final rc = await run.result;

      expect(rc, clpeakRunOk);
      expect(events.last, isA<DoneEvent>());
      final metrics = events.whereType<MetricEvent>().toList();
      expect(metrics, isNotEmpty);
      expect(metrics.first.backend, 'CPU');
      expect(metrics.any((m) => m.metric.isOk && m.metric.value > 0), isTrue);
      expect(out.existsSync(), isTrue);

      // The whole resolved header reaches the GUI live, on test_begin — shape
      // and direction included, so the row exists before its first reading.
      final latency = events
          .whereType<TestBeginEvent>()
          .firstWhere((e) => e.header.id == 'memory_latency');
      expect(latency.header.description, isNotEmpty);
      expect(latency.header.direction, Direction.lowerIsBetter);
      expect(latency.header.units.quantity, Quantity.seconds);

      // Each reading's own note rides the reading, which is also all a
      // reopened file has to go on.
      expect(
          metrics.any((m) =>
              m.testKey == 'memory_latency' &&
              m.metric.id == 'DRAM x8' &&
              m.metric.description.isNotEmpty),
          isTrue);

      // Round-trip the saved document straight through dart:convert — the GUI
      // reads history this way, with no native loader involved.
      final doc = RunDocument.fromJson(
          jsonDecode(out.readAsStringSync()) as Map<String, dynamic>);
      final reloaded = doc.runs.single.findTest('memory_latency')!;
      expect(reloaded.description, latency.header.description);
      expect(reloaded.shape, latency.header.shape);
      expect(
          reloaded.metrics
              .firstWhere((m) => m.id == 'DRAM x8')
              .description,
          isNotEmpty);
      out.deleteSync();
    });

    test('bad args are rejected without side effects', () async {
      final run = ClpeakRunner(bindings).start(['--bogus-flag']);
      final rc = await run.result;
      // A busy launch emits no events at all, so the drain below would hang
      // until this test's own timeout and blame the wrong run.  Say plainly
      // that an earlier run is still holding the process-global guard.
      expect(rc, isNot(clpeakRunBusy),
          reason: 'a previous run is still in flight (CLPEAK_RUN_BUSY) — it '
              'overran its timeout rather than this test misbehaving');
      expect(rc, clpeakRunBadArgs);
      final events = await run.events.toList();
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
