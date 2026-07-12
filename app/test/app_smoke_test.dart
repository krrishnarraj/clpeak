// UI smoke test with the real native bridge: verifies the app boots against
// the actual catalog and that all three tabs render.  Requires
// CLPEAK_FFI_PATH (see ffi_integration_test.dart); skipped otherwise.
//
// The full native run round-trip (events, XML, cancel) is covered by
// ffi_integration_test.dart — NativeCallable.listener events are not
// deliverable inside the testWidgets fake-async environment, so the run
// itself is exercised there and manually via the app.
import 'dart:io';

import 'package:clpeak/src/ffi/clpeak_bindings.dart';
import 'package:clpeak/src/services/benchmark_service.dart';
import 'package:clpeak/src/services/export_service.dart';
import 'package:clpeak/src/services/run_history_store.dart';
import 'package:clpeak/src/services/settings_service.dart';
import 'package:clpeak/src/ui/app.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  final ffiPath = Platform.environment['CLPEAK_FFI_PATH'];
  // Skipped when CLPEAK_FFI_PATH is unset (pure-Dart CI).
  final skip = ffiPath == null || ffiPath.isEmpty;

  testWidgets('app boots against the real catalog; tabs render',
      (tester) async {
    SharedPreferences.setMockInitialValues({});
    final bindings = ClpeakBindings.open();
    final tmp = Directory.systemTemp.createTempSync('clpeak_smoke');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final history = RunHistoryStore(bindings, directoryOverride: tmp);
    final service = BenchmarkService(bindings, history);

    await tester.pumpWidget(MultiProvider(
      providers: [
        Provider.value(value: history),
        Provider(create: (_) => ExportService()),
        ChangeNotifierProvider(create: (_) => SettingsService()),
        ChangeNotifierProvider.value(value: service),
      ],
      child: const ClpeakApp(),
    ));
    await tester.pump();

    // Dashboard: launcher + real system summary from the native catalog.
    expect(find.text('Run benchmark'), findsOneWidget);
    expect(find.text('This system'), findsOneWidget);
    expect(service.catalog.usable, isNotEmpty);
    expect(service.version, isNotEmpty);

    // History tab: empty state (pump until the async list() resolves).
    await tester.tap(find.text('History'));
    await tester.pump();
    for (var i = 0;
        i < 50 && find.text('No saved runs yet').evaluate().isEmpty;
        i++) {
      await tester.runAsync(
          () => Future<void>.delayed(const Duration(milliseconds: 50)));
      await tester.pump();
    }
    expect(find.text('No saved runs yet'), findsOneWidget);

    // About tab: version + theme control.
    await tester.tap(find.text('About'));
    await tester.pump();
    expect(find.text('v${service.version}'), findsOneWidget);
    expect(find.text('Theme'), findsOneWidget);
  }, skip: skip);
}
