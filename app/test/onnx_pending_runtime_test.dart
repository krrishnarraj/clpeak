// A runtime chosen while plugin providers (Windows ML's among them) are
// loaded into the current one cannot be loaded in the same process: the
// native side keeps the loaded runtime and reports the choice as
// `pendingRuntime`.  The settings panel has to say so -- before this, the
// GUI either crashed or sat on the old runtime with no explanation.
//
// Pure Dart, no native bridge: the status JSON is decoded directly, and the
// settings screen is pumped over stub bindings that report a pending choice.
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

const _reason = 'plugin libraries are loaded into ONNX Runtime 1.24.4 '
    '(OpenVINOExecutionProvider), and a provider library cannot move to '
    'another runtime inside a running process';

Map<String, dynamic> _statusJson({Map<String, dynamic>? pending}) => {
      'available': true,
      'linkedIn': false,
      'version': '1.24.4',
      'path': r'C:\ort\dml-1.24.4\onnxruntime.dll',
      'error': '',
      'epLibraries': [],
      'winml': {'enabled': true, 'path': '', 'error': ''},
      'pendingRuntime': ?pending,
    };

class _StubBindings implements ClpeakBindings {
  _StubBindings(this.status);

  final OnnxStatus status;

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
  OnnxStatus onnxStatus() => status;

  @override
  void setLitertLibrary(String path) {}

  @override
  void setLitertNpuStageDir(String dir) {}

  @override
  LitertStatus litertStatus() => const LitertStatus.unavailable('test');
}

void main() {
  test('pendingRuntime decodes, and its absence means nothing waits', () {
    expect(OnnxStatus.fromJson(_statusJson()).pendingRuntime, isNull);

    final s = OnnxStatus.fromJson(_statusJson(pending: {
      'path': r'C:\ort\ms-1.30\onnxruntime.dll',
      'reason': _reason,
    }));
    expect(s.pendingRuntime, isNotNull);
    expect(s.pendingRuntime!.path, r'C:\ort\ms-1.30\onnxruntime.dll');
    expect(s.pendingRuntime!.reason, _reason);
    // The loaded runtime is still the one reported.
    expect(s.version, '1.24.4');
  });

  testWidgets('the settings panel names the next start and why',
      (tester) async {
    // A narrow window: the long path and reason must wrap, and so must the
    // three buttons under the panel.
    tester.view.physicalSize = const Size(420, 1600);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    SharedPreferences.setMockInitialValues(
        {'onnxLibraryPath': r'C:\ort\ms-1.30\onnxruntime.dll'});
    final settings = await SettingsService.load();
    final status = OnnxStatus.fromJson(_statusJson(pending: {
      'path': r'C:\ort\ms-1.30\onnxruntime.dll',
      'reason': _reason,
    }));
    final service = BenchmarkService(_StubBindings(status),
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

    // The runtime in use stays on top; the choice is the next start's.
    expect(find.text('ONNX Runtime 1.24.4'), findsOneWidget);
    expect(find.text(r'Next start: C:\ort\ms-1.30\onnxruntime.dll'),
        findsOneWidget);
    // The native clause, printed as a sentence.
    expect(find.text('Plugin libraries are loaded into ONNX Runtime 1.24.4 '
            '(OpenVINOExecutionProvider), and a provider library cannot move '
            'to another runtime inside a running process.'),
        findsOneWidget);
    // Desktop hosts (where the tests run) can relaunch themselves.
    expect(find.text('RESTART NOW'), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('no pending row when nothing waits', (tester) async {
    tester.view.physicalSize = const Size(1280, 1600);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    SharedPreferences.setMockInitialValues({});
    final settings = await SettingsService.load();
    final service = BenchmarkService(
        _StubBindings(OnnxStatus.fromJson(_statusJson())),
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

    expect(find.textContaining('Next start'), findsNothing);
    expect(find.text('RESTART NOW'), findsNothing);
    expect(tester.takeException(), isNull);
  });
}
