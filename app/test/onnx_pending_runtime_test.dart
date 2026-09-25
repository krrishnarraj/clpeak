// The runtime setup is fixed for a launch by the first runtime that loads:
// a change made after that is saved for the next launch.  The native status
// reports it as `pendingRuntime`, and the settings panel shows both setups --
// the one active now, and the inactive one the next launch loads.
//
// Pure Dart, no native bridge: the status JSON is decoded directly, and the
// settings screen is pumped over stub bindings that report a pending setup.
import 'dart:ffi' hide Size;
import 'dart:io';

import 'package:clpeak/src/ffi/clpeak_bindings.dart';
import 'package:clpeak/src/services/benchmark_service.dart';
import 'package:clpeak/src/services/run_history_store.dart';
import 'package:clpeak/src/services/settings_service.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/settings/settings_screen.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

const _active = r'C:\ort\dml-1.24.4\onnxruntime.dll';
const _next = r'C:\ort\ms-1.30\onnxruntime.dll';

Map<String, dynamic> _onnxJson({Map<String, dynamic>? pending}) => {
      'available': true,
      'linkedIn': false,
      'version': '1.24.4',
      'path': _active,
      'error': '',
      'epLibraries': [],
      'winml': {'enabled': false, 'path': '', 'error': ''},
      'pendingRuntime': ?pending,
    };

Map<String, dynamic> _litertJson({String? pendingPath}) => {
      'available': true,
      'version': 'ABI 1.0.0',
      'path': '/opt/litert/libLiteRt.dylib',
      'error': '',
      if (pendingPath != null) 'pendingRuntime': {'path': pendingPath},
    };

class _StubBindings implements ClpeakBindings {
  _StubBindings(this.onnx, this.litert);

  final OnnxStatus onnx;
  final LitertStatus litert;

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
  OnnxStatus onnxStatus() => onnx;

  @override
  void setLitertLibrary(String path) {}

  @override
  void setLitertNpuStageDir(String dir) {}

  @override
  LitertStatus litertStatus() => litert;
}

Future<void> _pumpSettings(
    WidgetTester tester, OnnxStatus onnx, LitertStatus litert) async {
  SharedPreferences.setMockInitialValues({'onnxLibraryPath': _next});
  final settings = await SettingsService.load();
  final service = BenchmarkService(_StubBindings(onnx, litert),
      RunHistoryStore(directoryOverride: Directory.systemTemp));
  await tester.pumpWidget(
    MultiProvider(
      providers: [
        ChangeNotifierProvider.value(value: settings),
        ChangeNotifierProvider.value(value: service),
      ],
      child: MaterialApp(
        theme: ClpeakTheme.dark(),
        home: const SettingsScreen(),
      ),
    ),
  );
  await tester.pump();
}

void main() {
  test('pendingRuntime decodes, and its absence means nothing waits', () {
    expect(OnnxStatus.fromJson(_onnxJson()).pendingRuntime, isNull);

    final s = OnnxStatus.fromJson(_onnxJson(pending: {
      'path': _next,
      'winml': {'enabled': true, 'path': r'C:\winml\x64'},
    }));
    expect(s.pendingRuntime, isNotNull);
    expect(s.pendingRuntime!.path, _next);
    expect(s.pendingRuntime!.winml, isTrue);
    expect(s.pendingRuntime!.winmlPath, r'C:\winml\x64');
    // The loaded runtime is still the one reported.
    expect(s.version, '1.24.4');
    expect(s.path, _active);

    expect(LitertStatus.fromJson(_litertJson()).pendingPath, isNull);
    // '' is a real answer: the default search, at the next launch.
    expect(LitertStatus.fromJson(_litertJson(pendingPath: '')).pendingPath, '');
  });

  testWidgets('the panel shows the active setup and the next launch\'s',
      (tester) async {
    // A narrow window: the long paths must wrap.
    tester.view.physicalSize = const Size(420, 2400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await _pumpSettings(
      tester,
      OnnxStatus.fromJson(_onnxJson(pending: {
        'path': _next,
        'winml': {'enabled': true, 'path': r'C:\winml\x64'},
      })),
      LitertStatus.fromJson(_litertJson(pendingPath: '')),
    );

    // Active: the runtime that loaded, from the file it loaded.
    expect(find.text('ONNX Runtime 1.24.4'), findsOneWidget);
    expect(find.text(_active), findsOneWidget);
    // Inactive: what the next launch loads, ONNX and LiteRT both.
    expect(find.text('Next launch'), findsNWidgets(2));
    expect(find.text(_next), findsOneWidget);
    expect(find.text(r'Windows ML on · C:\winml\x64'), findsOneWidget);
    expect(find.text('Found by name on the system paths'), findsOneWidget);
    expect(find.text('Takes effect the next time clpeak starts.'),
        findsNWidgets(2));
    // No relaunch from inside the app: the note is the whole story.
    expect(find.textContaining('RESTART'), findsNothing);
    expect(tester.takeException(), isNull);
  });

  testWidgets('no next-launch block when nothing waits', (tester) async {
    tester.view.physicalSize = const Size(1280, 2400);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await _pumpSettings(tester, OnnxStatus.fromJson(_onnxJson()),
        LitertStatus.fromJson(_litertJson()));

    expect(find.text('ONNX Runtime 1.24.4'), findsOneWidget);
    expect(find.text('Next launch'), findsNothing);
    expect(find.textContaining('Takes effect'), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
