// Renders the README / docs screenshots from saved runs.
//
// Run by tools/screenshots.sh, never by the test suite: it is an integration
// test so it runs on the desktop engine, where text resolves through the
// platform's fonts exactly as in the app.  (flutter_tester's font manager
// answers every family name with its test font, which swallows the theme's
// monospace fallback walk -- CP.monoStack -- on any span that sets no family
// of its own.)  Each shot is the real app shell (ClpeakShell and the screens
// under it) fed a saved run document from results/ and a device catalog
// captured from the machine that produced it
// (tools/screenshots/m1_pro_catalog.json: the `inventory` of a --verbose run),
// over stub native bindings -- so no native library is needed, and the layout
// is the app's own.  Nothing is captured from the screen: each shot is its
// widget tree rendered to an image.
//
//   CLPEAK_SHOTS_OUT      where the raw PNGs go (required)
//   CLPEAK_SHOTS_REPO     the repository root (required)
//   CLPEAK_SHOTS_VERSION  the version label on the shots (default: 3.0.0)
import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:ui' as ui;

import 'package:clpeak/src/ffi/clpeak_bindings.dart';
import 'package:clpeak/src/model/catalog.dart';
import 'package:clpeak/src/model/run_config.dart';
import 'package:clpeak/src/model/run_document.dart';
import 'package:clpeak/src/model/run_summary.dart';
import 'package:clpeak/src/services/benchmark_service.dart';
import 'package:clpeak/src/services/export_service.dart';
import 'package:clpeak/src/services/run_history_store.dart';
import 'package:clpeak/src/services/settings_service.dart';
import 'package:clpeak/src/theme/clpeak_theme.dart';
import 'package:clpeak/src/ui/app.dart';
import 'package:clpeak/src/ui/run_config/run_config_screen.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

final _env = Platform.environment;
final _outDir = _env['CLPEAK_SHOTS_OUT'] ?? '';
final _repo = _env['CLPEAK_SHOTS_REPO'] ?? '';
final _version = _env['CLPEAK_SHOTS_VERSION'] ?? '3.0.0';

/// The desktop app's default window (app/AGENTS.md), captured at 2x.
const _desktop = Size(1280, 860);
const _pixelRatio = 2.0;

class _StubBindings implements ClpeakBindings {
  _StubBindings(this._catalog);

  final Map<String, dynamic> _catalog;

  @override
  ClpeakLaunch get launch => (_, _, _, _) => 0;

  @override
  ClpeakRequestCancel get requestCancel => () {};

  @override
  String version() => _version;

  @override
  String? takeString(Pointer<Utf8> ptr) => null;

  @override
  Map<String, dynamic> backendCatalog() => _catalog;

  @override
  void setOnnxLibrary(String path) {}

  @override
  void setOnnxEpLibraries(List<OnnxEpLibrary> libs) {}

  @override
  void setOnnxWinml({required bool enabled, required String path}) {}

  @override
  OnnxStatus onnxStatus() => const OnnxStatus.unavailable('screenshot');

  @override
  void setLitertLibrary(String path) {}

  @override
  void setLitertNpuStageDir(String dir) {}

  @override
  LitertStatus litertStatus() => const LitertStatus.unavailable('screenshot');
}

/// The service in a fixed state: a loaded catalog, and either idle (the
/// dashboard) or a finished run showing [run].  Only the getters the screens
/// read are pinned; nothing here can start a run.
class _ShotService extends BenchmarkService {
  _ShotService(super.bindings, super.history,
      {required BackendCatalog catalog, this.run})
      : _shotCatalog = catalog,
        _shotConfig = RunConfig.allDevices(catalog);

  final BackendCatalog _shotCatalog;
  final RunConfig _shotConfig;
  final ({RunDocument document, RunSummary summary})? run;

  @override
  BackendCatalog get catalog => _shotCatalog;
  @override
  RunConfig get config => _shotConfig;
  @override
  bool get catalogReady => true;
  @override
  bool get isLoadingCatalog => false;
  @override
  BenchmarkState get state =>
      run == null ? BenchmarkState.idle : BenchmarkState.finished;
  @override
  RunDocument get document => run?.document ?? RunDocument();
  @override
  RunSummary? get lastSummary => run?.summary;
}

Map<String, dynamic> _readJson(String path) =>
    jsonDecode(File('$_repo/$path').readAsStringSync()) as Map<String, dynamic>;

/// A saved run as the app holds it once it finishes.  The runs in results/
/// were produced by the release candidate; the document's version is
/// relabelled to the release the screenshots ship with.
({RunDocument document, RunSummary summary}) _run(String path, String id) {
  final json = _readJson(path)..['clpeak_version'] = _version;
  final doc = RunDocument.fromJson(json);
  final summary = RunSummary(
    id: id,
    startedAt: DateTime.parse(json['generated_at'] as String),
    durationMs: (((json['duration_s'] as num?) ?? 0) * 1000).round(),
    devices: [for (final r in doc.runs) r.device],
    backends: {for (final r in doc.runs) r.backend}.toList(),
    cancelled: false,
    fileName: '$id.clpeak.json',
    clpeakVersion: _version,
  );
  return (document: doc, summary: summary);
}

/// One screen, in one theme, written to `<out>/<name>-<theme>.png`.
Future<void> _shoot(
  WidgetTester tester, {
  required String name,
  required ThemeMode theme,
  required BenchmarkService service,
  required SettingsService settings,
  required RunHistoryStore history,
  Widget? home,
  Future<void> Function(WidgetTester tester)? arrange,
}) async {
  final boundary = GlobalKey();
  await tester.pumpWidget(RepaintBoundary(
    key: boundary,
    child: MultiProvider(
      providers: [
        Provider.value(value: history),
        Provider(create: (_) => ExportService()),
        ChangeNotifierProvider.value(value: settings),
        ChangeNotifierProvider<BenchmarkService>.value(value: service),
      ],
      // ClpeakApp, with the theme pinned instead of read from settings.
      child: MaterialApp(
        title: 'clpeak',
        debugShowCheckedModeBanner: false,
        theme: ClpeakTheme.light(),
        darkTheme: ClpeakTheme.dark(),
        themeMode: theme,
        home: home ?? const ClpeakShell(),
      ),
    ),
  ));
  await tester.pumpAndSettle();
  if (arrange != null) {
    await arrange(tester);
    await tester.pumpAndSettle();
  }

  final render =
      boundary.currentContext!.findRenderObject()! as RenderRepaintBoundary;
  final image = await render.toImage(pixelRatio: _pixelRatio);
  final data = await image.toByteData(format: ui.ImageByteFormat.png);
  image.dispose();
  File('$_outDir/$name-${theme.name}.png')
      .writeAsBytesSync(data!.buffer.asUint8List());
  // Unmount before the next shot so no state carries over.
  await tester.pumpWidget(const SizedBox.shrink());
}

/// Select a device chip in the results view by its label.
Future<void> _selectDevice(WidgetTester tester, String label) async {
  await tester.tap(find.text(label));
  await tester.pumpAndSettle();
}

/// Scroll the screen's list until the first text containing [text] sits
/// [alignment] of the way down the viewport.  Names carry their info glyph
/// inline (see app/AGENTS.md), hence textContaining.
Future<void> _scrollTo(WidgetTester tester, String text,
    {double alignment = 0.12}) async {
  // Not .first until it exists: the list builds its rows lazily.
  final target = find.textContaining(text, findRichText: true);
  final list = find.byType(Scrollable).first;
  await tester.scrollUntilVisible(target, 300, scrollable: list);
  await tester.pumpAndSettle();
  await Scrollable.ensureVisible(tester.element(target.first),
      alignment: alignment);
  await tester.pumpAndSettle();
}

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('screenshots', (tester) async {
    if (_outDir.isEmpty || _repo.isEmpty) {
      fail('set CLPEAK_SHOTS_OUT and CLPEAK_SHOTS_REPO '
          '(tools/screenshots.sh does)');
    }
    Directory(_outDir).createSync(recursive: true);
    await binding.setSurfaceSize(_desktop);
    addTearDown(() => binding.setSurfaceSize(null));

    SharedPreferences.setMockInitialValues({});
    final settings = await SettingsService.load();
    final tmp = Directory.systemTemp.createTempSync('clpeak_shots');
    addTearDown(() => tmp.deleteSync(recursive: true));
    final history = RunHistoryStore(directoryOverride: tmp);

    final catalogJson = _readJson('tools/screenshots/m1_pro_catalog.json');
    final catalog = BackendCatalog.fromJson(catalogJson);
    final bindings = _StubBindings(catalogJson);

    final m1 = _run('results/Apple/M1_Pro.json', '20260925_120924');
    final rtx =
        _run('results/Nvidia/GeForce_RTX_5060.json', '20260924_152237');
    final tr = _run('results/AMD/Ryzen_Threadripper_PRO_3955WX_16_Cores.json',
        '20260924_153931');

    BenchmarkService svc(
            [({RunDocument document, RunSummary summary})? run]) =>
        _ShotService(bindings, history, catalog: catalog, run: run);

    for (final theme in [ThemeMode.dark, ThemeMode.light]) {
      // The dashboard: every device the machine exposes, by backend.
      await _shoot(tester,
          name: 'dashboard',
          theme: theme,
          service: svc(),
          settings: settings,
          history: history);

      // Custom run: devices, categories, time budgets.
      await _shoot(tester,
          name: 'custom-run',
          theme: theme,
          service: svc(),
          settings: settings,
          history: history,
          home: const RunConfigScreen(), arrange: (tester) async {
        await _scrollTo(tester, 'TEST CATEGORIES', alignment: 0.45);
      });

      // An NPU: the Neural Engine through Core ML.
      await _shoot(tester,
          name: 'results-npu',
          theme: theme,
          service: svc(m1),
          settings: settings,
          history: history, arrange: (tester) async {
        await _selectDevice(tester, 'CoreML · Apple Neural Engine');
      });

      // A GPU: tensor cores on CUDA, one reading per data type.
      await _shoot(tester,
          name: 'results-gpu',
          theme: theme,
          service: svc(rtx),
          settings: settings,
          history: history, arrange: (tester) async {
        await _selectDevice(tester, 'CUDA · NVIDIA GeForce RTX 5060');
        await _scrollTo(tester, 'Tensor cores (WMMA / mma.sync)');
      });

      // A CPU: ISA, core and cache topology, then the per-thread rates.
      await _shoot(tester,
          name: 'results-cpu',
          theme: theme,
          service: svc(tr),
          settings: settings,
          history: history, arrange: (tester) async {
        await _selectDevice(
            tester, 'CPU · AMD Ryzen Threadripper PRO 3955WX 16-Cores');
        // The chip row first, so the selection shows above the device.
        await _scrollTo(tester, 'oneAPI · ', alignment: 0.02);
      });
    }
  });
}
