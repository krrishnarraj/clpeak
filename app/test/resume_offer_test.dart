// Whether a run that stopped part-way is offered to be continued, and how:
// the facts come off disk as History reads them -- the run's log, its
// checkpoint, the note the app left beside it, Android's exit records -- and
// ResumeOffer turns them into the dashboard's words and the continuation's
// flags.  A run stops part-way when the app dies in it, and when it is
// cancelled, which keeps its checkpoint beside the result it saved.
//
// Pure Dart — no native bridge needed.
import 'dart:convert';
import 'dart:io';

import 'package:clpeak/src/model/catalog.dart';
import 'package:clpeak/src/model/run_summary.dart';
import 'package:clpeak/src/services/process_exit.dart';
import 'package:clpeak/src/services/resume_offer.dart';
import 'package:clpeak/src/services/run_history_store.dart';
import 'package:flutter_test/flutter_test.dart';

const _id = '20261017_201306';
const _version = '3.0.2-1-gabc1234';
const _catalog = 'CPU||0:Qualcomm CPU (part 0x002)';
const _os = 'Android 17';
final _started = DateTime.utc(2026, 10, 17, 16, 13, 6);

String _header() => jsonEncode({
      'schema': 'clpeak/run-log',
      'format_version': 3,
      'clpeak_version': _version,
      'generated_at': '2026-10-17T16:13:06Z',
      'host': {'os': 'Android', 'os_version': '17'},
      'invocation': {
        'argv': ['clpeak', '-o', '$_id.clpeak.json'],
      },
    });

String _entry(double at, String message,
        {String level = 'info', String source = '', String test = ''}) =>
    jsonEncode({
      'elapsed_s': at,
      'level': level,
      if (source.isNotEmpty) 'source': source,
      'backend': 'CPU',
      'device': 'Qualcomm CPU (part 0x002)',
      'device_index': 0,
      if (test.isNotEmpty) 'test': test,
      'message': message,
    });

// The run's own lines, as the native logger writes them: structure lines,
// a reading, and the test it stopped in.
List<String> _run({bool leftScreen = false}) => [
      _entry(0.1, 'Backend: CPU'),
      _entry(0.2, '  Device 0: Qualcomm CPU (part 0x002)'),
      _entry(1.0, '    Integer divide compute u64',
          test: 'integer_divide_compute'),
      _entry(4.0, '      u64 ST : 899 MOPS', test: 'integer_divide_compute'),
      if (leftScreen)
        _entry(5.0,
            'the app left the screen: the run holds at its next reading until '
            'the app is back, and the phone may end the app meanwhile',
            level: 'warning',
            source: 'app'),
      _entry(6.0, '    Memory latency (pointer-chase)', test: 'memory_latency'),
    ];

Map<String, dynamic> _checkpoint({String version = _version}) => {
      'schema': 'clpeak/run-partial',
      'format_version': 3,
      'clpeak_version': version,
      'generated_at': '2026-10-17T16:13:06Z',
      'duration_s': 6.0,
      'host': {'os': 'Android', 'os_version': '17'},
      'invocation': {
        'argv': ['clpeak', '-o', '$_id.clpeak.json'],
      },
      'progress': {
        'backends_done': [],
        'devices': [
          {
            'backend': 'CPU',
            'device_index': 0,
            'benchmarks_done': ['integer-divide-compute'],
          },
        ],
      },
      'devices': [
        {
          'backend': 'CPU',
          'platform': 'CPU',
          'name': 'Qualcomm CPU (part 0x002)',
          'type': 'cpu',
          'device_index': 0,
          'tests': [
            {
              'id': 'integer_divide_compute',
              'title': 'Integer divide compute u64',
              'category': 'compute',
              'shape': 'homogeneous',
              'direction': 'higher_is_better',
              'quantity': 'ops',
              'unit': 'OPS',
              'metrics': [
                {'id': 'u64 ST', 'value': 899000000.0},
              ],
            },
          ],
        },
      ],
    };

ProcessExitRecord _exit(String reason, {int importance = 400, int min = 30}) =>
    ProcessExitRecord(
      runId: _id,
      at: _started.add(Duration(minutes: min)),
      reason: reason,
      importance: importance,
      rssKb: 410 << 10,
    );

void main() {
  late Directory dir;
  setUp(() => dir = Directory.systemTemp.createTempSync('clpeak_resume'));
  tearDown(() => dir.deleteSync(recursive: true));

  // A run that did not finish, as the native side and the app leave it.
  Future<CrashLog> leave({
    List<String>? lines,
    List<ProcessExitRecord> exits = const [],
    Map<String, dynamic>? checkpoint,
    bool noCheckpoint = false,
    Map<String, dynamic>? note,
    DateTime? checkpointAt,
  }) async {
    await File('${dir.path}/$_id.clpeak.log')
        .writeAsString('${[_header(), ...?lines].join('\n')}\n');
    if (!noCheckpoint) {
      final f = File('${dir.path}/$_id.clpeak.partial.json');
      await f.writeAsString(jsonEncode(checkpoint ?? _checkpoint()));
      await f.setLastModified(
          checkpointAt ?? _started.add(const Duration(minutes: 5)));
    }
    await File('${dir.path}/$_id.clpeak.resume.json')
        .writeAsString(jsonEncode(note ?? {'catalog': _catalog, 'os': _os}));
    final store =
        RunHistoryStore(directoryOverride: dir, processExits: () async => exits);
    return (await store.listCrashLogs()).single;
  }

  ResumeOffer decide(CrashLog log,
          {String version = _version,
          String os = _os,
          String? catalog = _catalog,
          DateTime? now,
          DateTime? newer}) =>
      ResumeOffer.decide(log,
          appVersion: version,
          osVersion: os,
          catalog: catalog,
          now: now ?? _started.add(const Duration(hours: 2)),
          newerRunStarted: newer);

  group('what happened', () {
    test('Android ended it off screen: continue, running its test again',
        () async {
      final log = await leave(
          lines: _run(leftScreen: true), exits: [_exit('LOW_MEMORY')]);
      final offer = decide(log);
      expect(offer.canContinue, isTrue);
      expect(offer.skipCrashed, isFalse);
      expect(offer.testsSaved, 1);
      expect(offer.headline,
          'Android closed clpeak during CPU · Memory latency (pointer-chase).');
    });

    test('a swipe from Recents is the user ending it, wherever the app was',
        () async {
      final log = await leave(lines: _run(), exits: [_exit('USER_REQUESTED')]);
      final offer = decide(log);
      expect(offer.canContinue, isTrue);
      expect(offer.skipCrashed, isFalse);
      expect(offer.headline, startsWith('You closed clpeak during CPU'));
    });

    test('a crash on screen does not run the test it died in again',
        () async {
      final log = await leave(
          lines: _run(), exits: [_exit('CRASH_NATIVE', importance: 100)]);
      final offer = decide(log);
      expect(offer.canContinue, isTrue);
      expect(offer.skipCrashed, isTrue);
      expect(offer.headline, startsWith('clpeak crashed during CPU'));
    });

    test('running out of memory on screen is not run again either', () async {
      final log = await leave(
          lines: _run(), exits: [_exit('LOW_MEMORY', importance: 100)]);
      final offer = decide(log);
      expect(offer.skipCrashed, isTrue);
      expect(offer.headline, startsWith('The phone ran out of memory'));
    });

    test('with no exit record (iOS), where the app was decides', () async {
      final off = await leave(lines: _run(leftScreen: true));
      expect(decide(off).headline, startsWith('clpeak was closed during'));
      expect(decide(off).skipCrashed, isFalse);

      final on = await leave(lines: _run());
      expect(decide(on).headline, startsWith('clpeak stopped during'));
      expect(decide(on).skipCrashed, isTrue);
    });

    test('a resumed part starts on screen, with nothing ended yet', () async {
      // Part one left the screen and was ended; part two died on screen.
      final log = await leave(lines: [
        ..._run(leftScreen: true),
        jsonEncode({
          'elapsed_s': 1800.0,
          'level': 'error',
          'source': 'android',
          'message': 'process ended: LOW_MEMORY (cached, rss 410 MB)',
        }),
        _entry(2400.0,
            'resumed the run started at 2026-10-17T16:13:06Z, which had '
            'measured 1 tests when it stopped'),
        _entry(2401.0, '    Memory latency (pointer-chase)',
            test: 'memory_latency'),
      ]);
      expect(log.diedOffScreen, isFalse);
      expect(log.processExit, isNull);
      expect(decide(log).skipCrashed, isTrue);
    });
  });

  group('when it cannot be continued', () {
    test('nothing was saved to continue from', () async {
      final offer = decide(await leave(lines: _run(), noCheckpoint: true));
      expect(offer.canContinue, isFalse);
      expect(offer.whyNot, 'it saved nothing to continue from');
    });

    test('another build of clpeak', () async {
      final offer = decide(await leave(lines: _run()), version: '3.0.3');
      expect(offer.canContinue, isFalse);
      expect(offer.whyNot, 'clpeak was updated since it started');
    });

    test('another OS release', () async {
      final offer = decide(await leave(lines: _run()), os: 'Android 18');
      expect(offer.whyNot, 'the system was updated since it started');
    });

    test('other devices or runtimes', () async {
      final offer = decide(await leave(lines: _run()),
          catalog: '$_catalog\nONNX|1.24.4|0:Qualcomm QNN (HTP)');
      expect(offer.whyNot,
          'the devices or runtimes clpeak finds have changed since');
    });

    test('devices still being detected: not yet told', () async {
      final offer = decide(await leave(lines: _run()), catalog: null);
      expect(offer.canContinue, isFalse);
      expect(offer.pending, isTrue);
      expect(offer.whyNot, isEmpty);
    });
  });

  group('the dashboard offers it more narrowly', () {
    test('not more than a day later', () async {
      final offer = decide(await leave(lines: _run()),
          now: _started.add(const Duration(hours: 26)));
      expect(offer.whyNot, 'it stopped more than a day ago');
    });

    test('not once a newer run has started', () async {
      final offer = decide(await leave(lines: _run()),
          newer: _started.add(const Duration(hours: 1)));
      expect(offer.whyNot, 'a newer run has started since');
    });

    test('History offers it all the same', () async {
      final offer = ResumeOffer.decide(await leave(lines: _run()),
          appVersion: _version, osVersion: _os, catalog: _catalog);
      expect(offer.canContinue, isTrue);
    });
  });

  group('a cancelled run', () {
    // What a cancel leaves: the result it saved, the checkpoint it kept and
    // the app's note -- the sidecar went once the result was written.
    Future<RunHistoryStore> cancelled() async {
      final result = _checkpoint()
        ..remove('progress')
        ..['schema'] = 'clpeak/run'
        ..['cancelled'] = true;
      await File('${dir.path}/$_id.clpeak.json')
          .writeAsString(jsonEncode(result));
      await File('${dir.path}/$_id.clpeak.partial.json')
          .writeAsString(jsonEncode(_checkpoint()));
      await File('${dir.path}/$_id.clpeak.resume.json')
          .writeAsString(jsonEncode({'catalog': _catalog, 'os': _os}));
      return RunHistoryStore(
          directoryOverride: dir, processExits: () async => const []);
    }

    Future<ResumeOffer> decideOn(RunHistoryStore store, RunSummary run,
            {String version = _version}) async =>
        ResumeOffer.decideCancelled(run,
            checkpoint: await store.checkpointInfo(run.id),
            state: await store.resumeState(run.id),
            appVersion: version,
            osVersion: _os,
            catalog: _catalog);

    test('is continued from its result, however long ago', () async {
      final store = await cancelled();
      final run = (await store.list()).single;
      expect(run.cancelled, isTrue);
      final offer = await decideOn(store, run);
      expect(offer.canContinue, isTrue);
      expect(offer.skipCrashed, isFalse);
      expect(offer.testsSaved, 1);
      // A saved run, not one that did not finish.
      expect(await store.listCrashLogs(), isEmpty);
      expect((await decideOn(store, run, version: '3.0.3')).whyNot,
          'clpeak was updated since it started');
    });

    test('goes whole when deleted', () async {
      final store = await cancelled();
      await store.delete((await store.list()).single);
      for (final suffix in ['.json', '.partial.json', '.resume.json']) {
        expect(File('${dir.path}/$_id.clpeak$suffix').existsSync(), isFalse);
      }
    });

    test('continued and ended in turn, it did not finish', () async {
      final store = await cancelled();
      // The continuation's sidecar, beside the result its first part saved.
      final sidecar = File('${dir.path}/$_id.clpeak.log');
      await sidecar.writeAsString('${[_header(), ..._run()].join('\n')}\n');
      expect((await store.listCrashLogs()).single.checkpoint?.tests, 1);
      // Without the checkpoint, a saved run's sidecar is stale.
      await File('${dir.path}/$_id.clpeak.partial.json').delete();
      expect(await store.listCrashLogs(), isEmpty);
      expect(sidecar.existsSync(), isFalse);
    });
  });

  test('a death is recorded on the log once, beside an earlier one', () async {
    // A resumed run that died again: Android hands back both records, the
    // first already on the log, ahead of everything the second part wrote.
    final first = _exit('LOW_MEMORY', min: 30);
    final second = _exit('CRASH_NATIVE', importance: 100, min: 50);
    await leave(lines: _run(leftScreen: true), exits: [first]);
    final store = RunHistoryStore(
        directoryOverride: dir, processExits: () async => [first, second]);
    final log = File('${dir.path}/$_id.clpeak.log');
    await log.writeAsString(
        [for (var i = 0; i < 6000; i++) _entry(1900.0 + i, 'later line')]
            .join('\n'),
        mode: FileMode.append);
    await store.listCrashLogs();
    await store.listCrashLogs();
    final text = await log.readAsString();
    expect(RegExp('LOW_MEMORY').allMatches(text), hasLength(1));
    expect(RegExp('CRASH_NATIVE').allMatches(text), hasLength(1));
  });

  test('the catalog a run notes: each usable backend and its devices', () {
    final catalog = BackendCatalog.fromJson({
      'backends': [
        {
          'name': 'CPU',
          'flag': 'cpu',
          'available': true,
          'platforms': [
            {
              'index': 0,
              'name': 'CPU',
              'devices': [
                {'index': 0, 'name': 'Qualcomm CPU (part 0x002)', 'type': 'CPU'},
              ],
            },
          ],
        },
        {
          'name': 'CUDA',
          'flag': 'cuda',
          'available': false,
          'reason': 'no driver',
          'platforms': [],
        },
      ],
    });
    // Pinned: a run noted in another format would read as "the devices
    // changed" after an update that changed it.
    expect(ResumeOffer.fingerprint(catalog), _catalog);
    // "device 3 of 9": the run's own --devices list, else every device.
    expect(ResumeOffer.plannedDevices(['clpeak', '-o', 'x'], catalog), 1);
    expect(
        ResumeOffer.plannedDevices(
            ['clpeak', '--devices', 'cpu:0,vulkan:0', '-o', 'x'], catalog),
        2);
  });
}
