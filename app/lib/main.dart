import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'src/ffi/clpeak_bindings.dart';
import 'src/model/run_config.dart';
import 'src/services/benchmark_service.dart';
import 'src/services/export_service.dart';
import 'src/services/run_history_store.dart';
import 'src/services/settings_service.dart';
import 'src/ui/app.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  final bindings = ClpeakBindings.open();
  final history = RunHistoryStore(bindings);
  final service = BenchmarkService(bindings, history);

  // On quit during a run: cancel and let the native side save partial
  // results before the process exits.
  AppLifecycleListener(onExitRequested: service.onExitRequested);

  // Dev hook: CLPEAK_AUTORUN=quick|full starts a run at launch (used by
  // automated UI verification; harmless otherwise).
  final autorun = Platform.environment['CLPEAK_AUTORUN'];
  if (autorun == 'quick' || autorun == 'full') {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      service.start(
          preset: autorun == 'quick' ? RunPreset.quick : RunPreset.full);
    });
  }

  runApp(MultiProvider(
    providers: [
      Provider.value(value: history),
      Provider(create: (_) => ExportService()),
      ChangeNotifierProvider(create: (_) => SettingsService()),
      ChangeNotifierProvider.value(value: service),
    ],
    child: const ClpeakApp(),
  ));
}
