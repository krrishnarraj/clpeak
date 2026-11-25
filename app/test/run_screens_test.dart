// The screens around a run -- the dashboard, the custom run, the live run --
// and Settings and About: each lays out at a desktop and a phone width in
// both themes; the sidebar's two pills do what the dashboard's buttons do;
// and the live run never animates (app/AGENTS.md, Traps).
//
// Pure Dart: the shell is pumped over stub bindings and a service pinned in
// the state each case needs.
import 'dart:ffi' hide Size;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:xpupeak/src/ffi/xpupeak_bindings.dart';
import 'package:xpupeak/src/model/catalog.dart';
import 'package:xpupeak/src/model/run_config.dart';
import 'package:xpupeak/src/model/run_document.dart';
import 'package:xpupeak/src/services/benchmark_service.dart';
import 'package:xpupeak/src/services/export_service.dart';
import 'package:xpupeak/src/services/resume_offer.dart';
import 'package:xpupeak/src/services/run_history_store.dart';
import 'package:xpupeak/src/services/settings_service.dart';
import 'package:xpupeak/src/theme/xpupeak_theme.dart';
import 'package:xpupeak/src/ui/app.dart';
import 'package:xpupeak/src/ui/dashboard/processor_groups.dart';
import 'package:xpupeak/src/ui/run_config/run_config_screen.dart';

/// An Apple M1 Pro as the native catalog lists it: three processors, each
/// reached by several backends, and a backend that found nothing.
final Map<String, dynamic> _catalogJson = {
  'backends': [
    _backend('Metal', 'metal', [
      _device(0, 'Apple M1 Pro', 'GPU', 'gpu0'),
    ]),
    _backend('Vulkan', 'vulkan', [
      _device(0, 'Apple M1 Pro', 'Integrated GPU', 'gpu0'),
    ]),
    _backend('OpenCL', 'opencl', [
      _device(0, 'Apple M1 Pro', 'GPU', 'gpu0',
          cu: 16, mhz: 1000, mem: 26800603136),
    ], platform: 'Apple'),
    _backend('CPU', 'cpu', [
      _device(0, 'Apple M1 Pro', 'CPU', 'cpu0', cu: 10, mem: 34359738368),
    ]),
    _backend('CoreML', 'coreml', [
      _device(0, 'Apple Neural Engine', 'NPU', 'npu0'),
      _device(1, 'GPU via Core ML (Apple M1 Pro)', 'GPU', 'gpu0'),
      _device(2, 'CPU via Core ML', 'CPU', 'cpu0'),
    ]),
    {
      'name': 'LiteRT',
      'flag': 'litert',
      'available': false,
      'reason': 'LiteRT library (libLiteRt) not found',
      'platforms': [],
    },
    _backend('ONNX', 'onnx', [
      _device(0, 'Apple CoreML (Neural Engine)', 'NPU', 'npu0'),
      _device(1, 'ONNX Runtime CPU', 'CPU', 'cpu0'),
    ]),
  ],
  'packages': [
    {'id': 'pkg0', 'kind': 'soc', 'vendor': 'apple', 'name': 'Apple M1 Pro'},
  ],
  'processors': [
    {
      'id': 'cpu0',
      'kind': 'cpu',
      'package': 'pkg0',
      'name': 'Apple M1 Pro',
      'isa': 'arm64',
      'cores': 10,
      'threads': 10,
      'clusters': [
        {'kind': 'performance', 'cores': 8},
        {'kind': 'efficiency', 'cores': 2},
      ],
      'engines': [
        {'kind': 'matrix', 'name': 'AMX', 'scope': 'cluster'},
      ],
    },
    {
      'id': 'gpu0',
      'kind': 'gpu',
      'package': 'pkg0',
      'name': 'Apple M1 Pro',
      'arch': 'Apple7',
      'integrated': true,
      'compute_units': 16,
      'compute_unit': 'core',
      'clock_mhz': 1000,
    },
    {
      'id': 'npu0',
      'kind': 'npu',
      'package': 'pkg0',
      'name': 'Apple Neural Engine',
      'compute_units': 16,
      'compute_unit': 'core',
    },
  ],
};

Map<String, dynamic> _backend(
        String name, String flag, List<Map<String, dynamic>> devices,
        {String? platform}) =>
    {
      'name': name,
      'flag': flag,
      'available': true,
      'platforms': [
        {'index': 0, 'name': platform ?? name, 'devices': devices},
      ],
    };

Map<String, dynamic> _device(int index, String name, String type,
        String processor,
        {int cu = 0, int mhz = 0, int mem = 0}) =>
    {
      'index': index,
      'name': name,
      'type': type,
      'processor': processor,
      if (cu > 0) 'compute_units': cu,
      if (mhz > 0) 'clock_mhz': mhz,
      if (mem > 0) 'global_mem_bytes': mem,
    };

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
  String version() => '4.0.0';

  @override
  String? takeString(Pointer<Utf8> ptr) => null;

  @override
  Map<String, dynamic> backendCatalog() => _catalogJson;

  @override
  void setOnnxLibrary(String path) {}

  @override
  void setOnnxEpLibraries(List<OnnxEpLibrary> libs) {}

  @override
  void setOnnxWinml({required bool enabled, required String path}) {}

  @override
  OnnxStatus onnxStatus() => OnnxStatus.fromJson(const {
        'available': true,
        'version': '1.31.0',
        'path': '/opt/homebrew/lib/libonnxruntime.1.31.0.dylib',
        'error': '',
        'epLibraries': [],
        'winml': {'enabled': false, 'path': '', 'error': ''},
      });

  @override
  void setLitertLibrary(String path) {}

  @override
  void setLitertNpuStageDir(String dir) {}

  @override
  LitertStatus litertStatus() =>
      const LitertStatus.unavailable('LiteRT library (libLiteRt) not found');
}

/// The service pinned in one state; what the screens call is recorded
/// rather than run.
class _PinnedService extends BenchmarkService {
  _PinnedService(RunHistoryStore history,
      {this.devicesFound = true,
      this.live = false,
      this.offer,
      RunDocument? doc})
      : _doc = doc ?? RunDocument(),
        _pinnedCatalog = BackendCatalog.fromJson(_catalogJson),
        super(_StubBindings(), history) {
    _pinnedConfig = RunConfig.allDevices(_pinnedCatalog);
  }

  /// The catalog has loaded ([catalogReady]).
  final bool devicesFound;
  bool live;
  bool cancellingRun = false;
  final ResumeOffer? offer;
  final RunDocument _doc;
  final BackendCatalog _pinnedCatalog;
  late RunConfig _pinnedConfig;

  final starts = <RunPreset?>[];
  final pauses = <bool>[];

  @override
  BackendCatalog get catalog => _pinnedCatalog;
  @override
  RunConfig get config => _pinnedConfig;
  @override
  bool get catalogReady => devicesFound;
  @override
  bool get isLoadingCatalog => false;
  @override
  BenchmarkState get state => !live
      ? BenchmarkState.idle
      : cancellingRun
          ? BenchmarkState.cancelling
          : BenchmarkState.running;
  @override
  bool get isRunning => live;
  @override
  RunDocument get document => _doc;
  @override
  ResumeOffer? get resumeOffer => offer;
  @override
  Duration get elapsed => const Duration(minutes: 7, seconds: 42);

  @override
  void updateConfig(void Function(RunConfig) mutate) {
    mutate(_pinnedConfig);
    notifyListeners();
  }

  @override
  Future<void> start({RunPreset? preset, bool verbose = false}) async {
    starts.add(preset);
  }

  @override
  void setPaused(bool pause) {
    pauses.add(pause);
    paused = pause;
    notifyListeners();
  }

  @override
  void cancel() {
    cancellingRun = true;
    notifyListeners();
  }
}

/// A run in flight on the Neural Engine, through Core ML, after the GPU
/// backends.
RunDocument _liveDocument() => RunDocument()
  ..runFor('Metal', 'Metal', 'Apple M1 Pro', '', 0)
  ..runFor('CoreML', 'Core ML', 'Apple Neural Engine', '', 0);

const _desktop = Size(1280, 860);
const _phone = Size(390, 844);

late Directory _tmp;

Future<void> _pump(WidgetTester tester, BenchmarkService service,
    {required Size size, ThemeMode theme = ThemeMode.light}) async {
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.reset);
  final settings = await SettingsService.load();
  final history = RunHistoryStore(directoryOverride: _tmp);
  await tester.pumpWidget(MultiProvider(
    providers: [
      Provider.value(value: history),
      Provider(create: (_) => ExportService()),
      ChangeNotifierProvider.value(value: settings),
      ChangeNotifierProvider<BenchmarkService>.value(value: service),
    ],
    child: MaterialApp(
      theme: XpuPeakTheme.light(),
      darkTheme: XpuPeakTheme.dark(),
      themeMode: theme,
      home: const XpuPeakShell(),
    ),
  ));
  await tester.pump();
}

/// Leave the screen, so the live run's clock stops its timer.
Future<void> _unmount(WidgetTester tester) async {
  await tester.pumpWidget(const SizedBox.shrink());
}

void main() {
  setUp(() {
    SharedPreferences.setMockInitialValues({});
    _tmp = Directory.systemTemp.createTempSync('xpupeak_screens');
  });
  tearDown(() => _tmp.deleteSync(recursive: true));

  test('the catalog groups by processor, then by kind where unlinked', () {
    final catalog = BackendCatalog.fromJson(_catalogJson);
    final groups = groupDevices(catalog);
    expect([for (final g in groups) g.label],
        ['CPU · 10 CORES', 'GPU · 16 CORES', 'NPU · 16 CORES']);
    expect(groups[0].facts, '8P + 2E · arm64 · AMX');
    expect(groups[1].facts, 'Apple7 · 1.0 GHz · integrated');
    expect(groups[2].facts, 'in Apple M1 Pro');
    expect([for (final d in groups[1].devices) d.backend.name],
        ['Metal', 'Vulkan', 'OpenCL', 'CoreML']);
    // A device's line says what its card does not.
    expect(deviceDetail(groups[1].devices[1], groups[1]), 'Integrated GPU');
    expect(deviceDetail(groups[1].devices[2], groups[1]),
        '16 CU · 1000 MHz · 25.0 GB');
    expect(deviceDetail(groups[1].devices[3], groups[1]),
        'GPU via Core ML (Apple M1 Pro)');

    // An older library lists no hardware: one card per kind, unnamed.
    final old = BackendCatalog.fromJson({'backends': _catalogJson['backends']});
    final byKind = groupDevices(old);
    expect([for (final g in byKind) g.label],
        ['GPU · 4 DEVICES', 'CPU · 3 DEVICES', 'NPU · 2 DEVICES']);
    expect(byKind.every((g) => g.name == null), isTrue);
  });

  for (final size in [_desktop, _phone]) {
    final where = size == _desktop ? 'desktop' : 'phone';
    for (final theme in [ThemeMode.light, ThemeMode.dark]) {
      testWidgets('the screens lay out on a $where, ${theme.name}',
          (tester) async {
        final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp),
            offer: ResumeOffer(
              id: '20261119_101500',
              startedAt: DateTime(2026, 11, 19, 10, 15),
              headline: 'XpuPeak stopped during Metal · Kernel latency.',
              canContinue: true,
              skipCrashed: true,
            ));
        await _pump(tester, service, size: size, theme: theme);

        // The dashboard: the offer, the launcher and a card per processor
        // (below the fold on a phone).
        Finder text(String s) => find.text(s, skipOffstage: false);
        expect(text('Did not finish'), findsOneWidget);
        expect(text('RUN BENCHMARK'), findsOneWidget);
        expect(text('THIS SYSTEM'), findsOneWidget);
        expect(text('6 backends · 9 devices'), findsOneWidget);
        expect(text('NPU · 16 CORES'), findsOneWidget);
        expect(tester.takeException(), isNull);

        // The custom run, from the dashboard's own button.
        await tester.tap(find.text('Customize').last);
        await tester.pumpAndSettle();
        expect(find.text('Custom run'), findsOneWidget);
        expect(find.text('9 of 9 selected'), findsOneWidget);
        expect(tester.takeException(), isNull);
        await tester.scrollUntilVisible(find.text('TIME BUDGETS'), 300,
            scrollable: find
                .descendant(
                    of: find.byType(RunConfigScreen),
                    matching: find.byType(Scrollable))
                .first);
        await tester.pump();
        expect(find.text('Default 500 ms'), findsOneWidget);
        expect(tester.takeException(), isNull);
        await tester.pageBack();
        await tester.pumpAndSettle();

        for (final tab in ['Settings', 'About']) {
          await tester.tap(find.text(tab).last);
          await tester.pumpAndSettle();
          expect(tester.takeException(), isNull, reason: tab);
        }
        expect(find.text('v4.0.0'), findsOneWidget);
      });
    }
  }

  testWidgets('a backend chip turns all its devices off, then on',
      (tester) async {
    final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp));
    await _pump(tester, service, size: _desktop);
    await tester.tap(find.text('Customize').first);
    await tester.pumpAndSettle();

    const coreml = 'CoreML';
    bool on(int i) => service.config
        .isDeviceSelected(coreml, (platformIndex: 0, deviceIndex: i));
    await tester.tap(find.text('3/3'));
    await tester.pump();
    expect([on(0), on(1), on(2)], [false, false, false]);
    expect(find.text('0/3'), findsOneWidget);
    await tester.tap(find.text('0/3'));
    await tester.pump();
    expect([on(0), on(1), on(2)], [true, true, true]);

    // A device's row on its processor's card turns that one off alone.
    await tester.tap(find.text('Integrated GPU'));
    await tester.pump();
    expect(
        service.config
            .isDeviceSelected('Vulkan', (platformIndex: 0, deviceIndex: 0)),
        isFalse);
    expect(find.text('8 of 9 selected'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('the custom run starts its own selection, closing at once',
      (tester) async {
    final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp));
    await _pump(tester, service, size: _phone);
    await tester.tap(find.text('Customize'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Start custom run'));
    await tester.pump();
    // The configuration as chosen, not a preset.
    expect(service.starts, [null]);
    // Gone on the next frame: no page slides away while the run starts.
    expect(find.text('Custom run'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('About lists the licenses in its own frame, Sora among them',
      (tester) async {
    // In place of the binding's own collector: only these two.
    LicenseRegistry.reset();
    LicenseRegistry.addLicense(() => Stream.fromIterable(const [
          LicenseEntryWithLineBreaks(['provider'], 'MIT License'),
          LicenseEntryWithLineBreaks(['Sora'],
              'Copyright 2019 The Sora Project Authors\n\nThis Font Software '
              'is licensed under the SIL Open Font License, Version 1.1.'),
        ]));
    final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp));
    await _pump(tester, service, size: _phone);
    await tester.tap(find.text('About').last);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Open-source licenses'));
    await tester.pumpAndSettle();

    expect(find.byType(LicensePage), findsNothing);
    expect(find.text('Licenses'), findsOneWidget);
    expect(find.text('XpuPeak v4.0.0 · 2 packages'), findsOneWidget);
    // By name, whatever the case.
    final names = [
      for (final w in tester.widgetList<Text>(find.byType(Text)))
        if (w.data == 'provider' || w.data == 'Sora') w.data,
    ];
    expect(names, ['provider', 'Sora']);

    await tester.tap(find.text('Sora'));
    await tester.pumpAndSettle();
    expect(find.text('1 license'), findsOneWidget);
    expect(find.textContaining('SIL Open Font License'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  group('the sidebar', () {
    testWidgets('waits for the devices, as the dashboard does',
        (tester) async {
      final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp),
          devicesFound: false);
      await _pump(tester, service, size: _desktop);
      expect(find.text('Detecting devices…'), findsOneWidget);
      await tester.tap(find.text('Run benchmark'));
      await tester.tap(find.text('Customize'));
      await tester.pump();
      expect(service.starts, isEmpty);
      expect(find.text('Custom run'), findsNothing);
    });

    testWidgets('runs the full run, and opens the custom one once',
        (tester) async {
      final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp));
      await _pump(tester, service, size: _desktop);
      // The sidebar's pill comes first; the dashboard's own is the other.
      await tester.tap(find.text('Run benchmark').first);
      await tester.pump();
      expect(service.starts, [RunPreset.full]);

      // From another tab: Customize opens the custom run over Benchmark.
      await tester.tap(find.text('Settings'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Customize').first);
      await tester.pumpAndSettle();
      expect(find.text('Custom run'), findsOneWidget);
      // Again, with it open: nothing more is pushed.
      await tester.tap(find.text('Customize').first);
      await tester.pumpAndSettle();
      await tester.pageBack();
      await tester.pumpAndSettle();
      expect(find.text('Custom run'), findsNothing);
      expect(find.text('THIS SYSTEM'), findsOneWidget);
    });

    testWidgets('holds both while a run is in flight', (tester) async {
      final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp),
          live: true, doc: _liveDocument());
      await _pump(tester, service, size: _desktop);
      await tester.tap(find.text('Run benchmark'));
      await tester.tap(find.text('Customize'));
      await tester.pump();
      expect(service.starts, isEmpty);
      expect(find.text('Custom run'), findsNothing);
      await _unmount(tester);
    });
  });

  group('the live run', () {
    for (final size in [_desktop, _phone]) {
      final where = size == _desktop ? 'desktop' : 'phone';
      testWidgets('never animates, on a $where', (tester) async {
        final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp),
            live: true, doc: _liveDocument())
          ..currentBackend = 'CoreML'
          ..currentTest = 'Single-precision GEMM'
          ..completedTests = 57;
        await _pump(tester, service, size: size, theme: ThemeMode.dark);
        await tester.pump(const Duration(seconds: 3));

        expect(find.text('Benchmarking'), findsOneWidget);
        expect(find.text('RUNNING · NPU'), findsOneWidget);
        expect(find.text('Single-precision GEMM'), findsOneWidget);
        // In the status card, and as the live results' device.
        expect(find.text('Apple Neural Engine'), findsNWidgets(2));
        expect(find.text('57'), findsOneWidget);
        expect(find.text('7m 42s'), findsOneWidget);

        // Nothing that animates on its own...
        for (final type in [
          CircularProgressIndicator,
          LinearProgressIndicator,
          RefreshProgressIndicator,
          AnimatedContainer,
          AnimatedOpacity,
          AnimatedSwitcher,
          AnimatedCrossFade,
          Hero,
          Tooltip,
        ]) {
          expect(find.byType(type), findsNothing, reason: '$type');
        }
        // ...and no ticker running: between the clock's ticks the screen
        // asks for no frame at all.
        expect(SchedulerBinding.instance.transientCallbackCount, 0);
        expect(tester.binding.hasScheduledFrame, isFalse);
        expect(tester.takeException(), isNull);
        await _unmount(tester);
      });
    }

    testWidgets('says each state, and Pause and Cancel reach the service',
        (tester) async {
      final service = _PinnedService(RunHistoryStore(directoryOverride: _tmp),
          live: true, doc: _liveDocument())
        ..currentBackend = 'CoreML';
      await _pump(tester, service, size: _desktop);
      expect(find.text('Preparing CoreML…'), findsOneWidget);

      await tester.tap(find.text('Pause'));
      await tester.pump();
      expect(service.pauses, [true]);
      expect(find.text('Pausing'), findsOneWidget);
      expect(find.text('Pausing — the reading in progress finishes first…'),
          findsOneWidget);

      service
        ..held = true
        ..notifyListeners();
      await tester.pump();
      expect(find.text('PAUSED'), findsOneWidget);
      expect(
          find.text('Paused — the run waits before its next reading until '
              'you resume it.'),
          findsOneWidget);
      await tester.tap(find.text('Resume'));
      await tester.pump();
      expect(service.pauses, [true, false]);

      service
        ..held = false
        ..coolingDown = true
        ..notifyListeners();
      await tester.pump();
      expect(find.text('Cooling down'), findsOneWidget);
      expect(find.text('Paused — the device is hot; resuming once it cools…'),
          findsOneWidget);

      await tester.tap(find.text('Cancel'));
      await tester.pump();
      expect(find.text('CANCELLING'), findsOneWidget);
      expect(find.text('Cancelling — finishing the current test…'),
          findsOneWidget);
      expect(find.text('Pause'), findsNothing);
      expect(tester.takeException(), isNull);
      await _unmount(tester);
    });
  });
}
