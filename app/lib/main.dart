import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import 'src/ffi/clpeak_bindings.dart';
import 'src/ffi/clpeak_engine.dart';
import 'src/model/run_config.dart';
import 'src/services/benchmark_service.dart';
import 'src/services/export_service.dart';
import 'src/services/run_history_store.dart';
import 'src/services/settings_service.dart';
import 'src/ui/app.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final bindings = ClpeakBindings.open();
  // On desktop, enumeration and runs happen in an engine process of their
  // own (ClpeakEngine explains why); on mobile, here.
  final engine = ClpeakEngine.forPlatform(bindings);

  // Settings first, and specifically before BenchmarkService's first
  // enumeration, which is what loads the ONNX Runtime -- in-process, a saved
  // library path applied any later would not take effect until the next
  // launch.
  final settings = await SettingsService.load();
  engine.setOnnxLibrary(settings.onnxLibraryPath);
  engine.setOnnxEpLibraries(settings.effectiveOnnxEpLibraries);
  engine.setOnnxWinml(
      enabled: settings.onnxWinml, path: settings.onnxWinmlPath);
  engine.setLitertLibrary(settings.litertLibraryPath);
  if (Platform.isAndroid) {
    // An APK carrying more than one vendor's NPU shims needs a directory
    // per vendor for LiteRT to choose by; the backend stages links there.
    engine.setLitertNpuStageDir(
        p.join((await getApplicationSupportDirectory()).path, 'litert-npu'));
  }

  final history = RunHistoryStore();
  final service = BenchmarkService(bindings, history, engine: engine);

  // First frame never waits for enumeration: the device catalog (seconds
  // on NPU/GPU boxes while native viability probes compile) loads off the
  // UI thread and the UI populates when it lands.  init() never
  // throws — failure lands in service.catalogError with history still
  // usable — so this future needs no error handler.
  unawaited(service.init());

  // On quit during a run: cancel and let the engine save partial results
  // before this process exits.
  AppLifecycleListener(onExitRequested: service.onExitRequested);

  // Dev hook: CLPEAK_AUTORUN=1 starts a run at launch (used by automated UI
  // verification; harmless otherwise).  Waits for the catalog: device
  // indices are positions in the enumerated list.
  final autorun = Platform.environment['CLPEAK_AUTORUN'];
  if (autorun != null && autorun.isNotEmpty && autorun != '0') {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      await service.ready;
      service.start(preset: RunPreset.full, verbose: settings.verbose);
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
