import 'dart:async';
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

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final bindings = ClpeakBindings.open();

  // Settings first, and specifically before BenchmarkService: its constructor
  // enumerates the device catalog, and enumeration is what loads the ONNX
  // Runtime — so a saved library path applied any later would not take effect
  // until the next launch.
  final settings = await SettingsService.load();
  bindings.setOnnxLibrary(settings.onnxLibraryPath);

  final history = RunHistoryStore();
  final service = BenchmarkService(bindings, history);

  // First frame never waits for enumeration: the device catalog (seconds
  // on NPU/GPU boxes while native viability probes compile) loads on a
  // worker isolate and the UI populates when it lands.  init() never
  // throws — failure lands in service.catalogError with history still
  // usable — so this future needs no error handler.
  unawaited(service.init());

  // On quit during a run: cancel and let the native side save partial
  // results before the process exits.
  AppLifecycleListener(onExitRequested: service.onExitRequested);

  // Dev hook: CLPEAK_AUTORUN=1 starts a run at launch (used by automated UI
  // verification; harmless otherwise).  Waits for the catalog: device
  // indices are positions in the enumerated list.
  final autorun = Platform.environment['CLPEAK_AUTORUN'];
  if (autorun != null && autorun.isNotEmpty && autorun != '0') {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await service.ready;
      service.start(preset: RunPreset.full);
    });
  }

  runApp(MultiProvider(
    providers: [
      Provider.value(value: history),
      Provider(create: (_) => ExportService()),
      ChangeNotifierProvider.value(value: settings),
      ChangeNotifierProvider.value(value: service),
    ],
    child: const ClpeakApp(),
  ));
}
