import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../ffi/clpeak_bindings.dart';
import '../../model/result_model.dart';
import '../../services/benchmark_service.dart';
import '../../services/settings_service.dart';
import '../../theme/clpeak_theme.dart';
import '../common/kit.dart';

/// Appearance, which ONNX Runtime and LiteRT libraries the two backends
/// measure, and whether runs record verbose diagnostics.
///
/// The runtime picker is the reason this screen exists.  Unlike every other
/// backend, ONNX has no single driver on a machine: NPU vendors ship their own
/// builds, and which one is loaded decides which execution providers appear at
/// all — a stock build offers CPU and nothing else.  So the library is a
/// setting, and changing it re-enumerates immediately rather than asking for a
/// restart.
///
/// The verbose switch is how a problem on a phone reaches a maintainer: with
/// it on, the saved document carries the backends' debug output, the device
/// inventory and the runtimes' own messages, and the exported file is the
/// bug report.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  OnnxStatus? _onnx;
  LitertStatus? _litert;
  BenchmarkService? _service;
  VoidCallback? _catalogListener;
  bool _lastLoading = false;
  bool _lastReady = false;

  @override
  void initState() {
    super.initState();
    final service = context.read<BenchmarkService>();
    _service = service;
    _lastLoading = service.isLoadingCatalog;
    _lastReady = service.catalogReady;
    _refreshOnnx();
    // The first catalog load lands after this screen may already be open
    // (enumeration takes seconds, longer with Windows ML installing): pick
    // up the resolved status when it does, without a toggle. Guarded to
    // catalog transitions only so live-run ticks never re-query.
    _catalogListener = () {
      if (!mounted) return;
      final s = _service;
      if (s == null || s.isRunning) return;
      if (s.isLoadingCatalog == _lastLoading &&
          s.catalogReady == _lastReady) {
        return;
      }
      _lastLoading = s.isLoadingCatalog;
      _lastReady = s.catalogReady;
      _refreshOnnx();
    };
    service.addListener(_catalogListener!);
  }

  @override
  void dispose() {
    if (_service != null && _catalogListener != null) {
      _service!.removeListener(_catalogListener!);
    }
    super.dispose();
  }

  void _refreshOnnx() {
    final service = _service ?? context.read<BenchmarkService>();
    setState(() {
      _onnx = service.onnxStatus();
      _litert = service.litertStatus();
    });
  }

  /// Apply a library choice: persist it, hand it to the native loader, then
  /// re-enumerate so the provider list on the run screen matches what was
  /// just chosen.
  Future<void> _applyLibrary(_Runtime which, String path) async {
    final settings = context.read<SettingsService>();
    final service = context.read<BenchmarkService>();
    if (which == _Runtime.onnx) {
      await settings.setOnnxLibraryPath(path);
    } else {
      await settings.setLitertLibraryPath(path);
    }
    if (!mounted) return;
    // Async since catalog loads off-thread; refresh the status after the
    // re-enumeration lands.
    if (which == _Runtime.onnx) {
      await service.setOnnxLibrary(path);
    } else {
      await service.setLitertLibrary(path);
    }
    if (!mounted) return;
    _refreshOnnx();
  }

  Future<void> _pickLibrary(_Runtime which) async {
    // No extension filter: a runtime is .so / .dylib / .dll depending on the
    // platform, and plenty of real ones are versioned
    // (libonnxruntime.so.1.27.0) where an extension filter matches nothing.
    final file = await openFile(
      acceptedTypeGroups: const [XTypeGroup(label: 'Shared library')],
    );
    if (file == null) return;
    await _applyLibrary(which, await _durablePath(which, file.path));
  }

  /// Where the chosen library should live so it is still there next launch.
  ///
  /// Android's picker goes through the storage-access framework and hands back
  /// a copy it made under `{cacheDir}/{uuid}/`, which the OS is free to evict
  /// whenever it wants the space — a runtime chosen today would simply be gone
  /// tomorrow, and the setting would look like it had forgotten itself.  So
  /// the file is copied somewhere durable inside the sandbox, which is also
  /// the only place Android will dlopen from.  One library is kept at a time;
  /// it is ~27 MB.
  ///
  /// The copy is stamped per pick: vendor builds are usually all named
  /// libonnxruntime.so, so reusing the bare basename would land every pick at
  /// the same destination path and the native loader -- which keys runtimes
  /// by path and never unmaps -- would see the same key and keep measuring
  /// the first file.  Desktop pickers return the real file, already stable
  /// and not ours to duplicate.
  Future<String> _durablePath(_Runtime which, String picked) async {
    if (!Platform.isAndroid) return picked;
    final dir = Directory(
        p.join((await getApplicationSupportDirectory()).path, which.dirName));
    if (dir.existsSync()) dir.deleteSync(recursive: true);
    dir.createSync(recursive: true);
    final stamp = DateTime.now().millisecondsSinceEpoch;
    final dest = p.join(dir.path, '${stamp}_${p.basename(picked)}');
    await File(picked).copy(dest);
    return dest;
  }

  /// Back to the bundled/system runtime, and drop the imported copy with it —
  /// leaving 27 MB stranded in the sandbox would be its own small bug.
  Future<void> _resetLibrary(_Runtime which) async {
    await _applyLibrary(which, '');
    if (!Platform.isAndroid) return;
    final dir = Directory(
        p.join((await getApplicationSupportDirectory()).path, which.dirName));
    if (dir.existsSync()) dir.deleteSync(recursive: true);
  }

  /// Plugin execution providers are a desktop matter: a phone's plugin, if
  /// any, is bundled with the app and registered without asking.
  static bool get _desktop =>
      Platform.isWindows || Platform.isLinux || Platform.isMacOS;

  /// Persist the plugin library set, hand it to the native side and
  /// re-enumerate, then read back how each registered.
  Future<void> _applyEpLibraries(List<OnnxEpLibrary> libs) async {
    final settings = context.read<SettingsService>();
    final service = context.read<BenchmarkService>();
    await settings.setOnnxEpLibraries(libs);
    if (!mounted) return;
    await service.setOnnxEpLibraries(settings.effectiveOnnxEpLibraries);
    if (!mounted) return;
    _refreshOnnx();
  }

  /// Pick a plugin library, then ask for the registration name the provider
  /// expects -- guessed from the file name, which for every plugin shipped so
  /// far is `onnxruntime_providers_<short>`; Qualcomm's QNN insists on
  /// "QNNExecutionProvider", and the rest follow the same pattern.
  Future<void> _addEpLibrary() async {
    final file = await openFile(
      acceptedTypeGroups: const [XTypeGroup(label: 'Shared library')],
    );
    if (file == null || !mounted) return;
    final name = await _askRegistrationName(file.path);
    if (name == null || name.isEmpty || !mounted) return;
    final settings = context.read<SettingsService>();
    final libs = [
      for (final l in settings.onnxEpLibraries)
        if (l.name != name) l,
      OnnxEpLibrary(name: name, path: file.path),
    ];
    await _applyEpLibraries(libs);
  }

  Future<void> _removeEpLibrary(OnnxEpLibrary lib) async {
    final settings = context.read<SettingsService>();
    await _applyEpLibraries([
      for (final l in settings.onnxEpLibraries)
        if (l.name != lib.name || l.path != lib.path) l,
    ]);
  }

  static String _guessRegistrationName(String path) {
    var stem = p.basenameWithoutExtension(path);
    // Versioned Unix names keep their extension in the stem (libfoo.so.1).
    stem = stem.replaceFirst(RegExp(r'\.so(\.\d+)*$'), '');
    if (stem.startsWith('lib')) stem = stem.substring(3);
    const prefix = 'onnxruntime_providers_';
    if (stem.startsWith(prefix)) stem = stem.substring(prefix.length);
    const known = {
      'qnn': 'QNN',
      'openvino': 'OpenVINO',
      'vitisai': 'VitisAI',
      'nv_tensorrt_rtx': 'NvTensorRTRTX',
      'tensorrt': 'Tensorrt',
      'cuda': 'CUDA',
      'migraphx': 'MIGraphX',
      'webgpu': 'WebGpu',
      'dml': 'Dml',
    };
    final short = known[stem.toLowerCase()] ??
        (stem.isEmpty ? '' : stem[0].toUpperCase() + stem.substring(1));
    return short.isEmpty ? '' : '${short}ExecutionProvider';
  }

  Future<String?> _askRegistrationName(String path) async {
    final t = CP.of(context);
    final controller =
        TextEditingController(text: _guessRegistrationName(path));
    return showDialog<String>(
      context: context,
      builder: (context) => CDialog(
        title: 'Registration name',
        actions: [
          CButton(
              label: 'Cancel',
              kind: CButtonKind.quiet,
              onPressed: () => Navigator.pop(context)),
          CButton(
              label: 'Register',
              kind: CButtonKind.primary,
              onPressed: () => Navigator.pop(context, controller.text.trim())),
        ],
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(p.basename(path), style: t.monoSmallDim),
            const SizedBox(height: 10),
            TextField(
              controller: controller,
              autofocus: true,
              style: t.mono,
              cursorWidth: 1.5,
              cursorRadius: Radius.zero,
              decoration: InputDecoration(
                isDense: true,
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 11),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide(color: t.line),
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide(color: t.line),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide(color: t.text, width: 1.5),
                ),
              ),
              onSubmitted: (v) => Navigator.pop(context, v.trim()),
            ),
            const SizedBox(height: 10),
            Text(
                'The name the provider expects to be registered under; '
                'Qualcomm\'s QNN plugin requires QNNExecutionProvider.',
                style: t.micro),
          ],
        ),
      ),
    );
  }

  /// Switch the Windows ML catalog; enabling it re-enumerates, which installs
  /// whatever certified providers fit the machine.
  Future<void> _applyWinml({required bool enabled, required String path}) async {
    final settings = context.read<SettingsService>();
    final service = context.read<BenchmarkService>();
    await settings.setOnnxWinml(enabled: enabled, path: path);
    if (!mounted) return;
    await service.setOnnxWinml(enabled: enabled, path: path);
    if (!mounted) return;
    _refreshOnnx();
  }

  Future<void> _pickWinmlDir() async {
    final dir = await getDirectoryPath();
    if (dir == null || !mounted) return;
    final settings = context.read<SettingsService>();
    await _applyWinml(enabled: settings.onnxWinml, path: dir);
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final settings = context.watch<SettingsService>();
    final running = context.select<BenchmarkService, bool>((s) => s.isRunning);

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            const CHeader(title: 'Settings'),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 40),
                children: [
                  const CSection(label: 'Appearance'),
                  const SizedBox(height: 10),
                  CPanel(
                    child: CRow(
                      rule: false,
                      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
                      child: Row(
                        children: [
                          Icon(Icons.dark_mode_outlined,
                              size: 15, color: t.dim),
                          const SizedBox(width: 10),
                          Expanded(child: Text('Theme', style: t.mono)),
                          _ThemeToggle(
                            value: settings.themeMode,
                            onChanged: settings.setThemeMode,
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 22),
                  const CSection(label: 'ONNX Runtime'),
                  const SizedBox(height: 10),
                  _RuntimePanel(
                    view: _RuntimeView.onnx(_onnx),
                    savedPath: settings.onnxLibraryPath,
                    locked: running,
                    onPick: () => _pickLibrary(_Runtime.onnx),
                    onReset: () => _resetLibrary(_Runtime.onnx),
                  ),
                  if (_desktop) ...[
                    const SizedBox(height: 22),
                    const CSection(label: 'Plugin execution providers'),
                    const SizedBox(height: 10),
                    _EpLibrariesPanel(
                      libraries: settings.onnxEpLibraries,
                      status: _onnx?.epLibraries ?? const [],
                      locked: running,
                      onAdd: _addEpLibrary,
                      onRemove: _removeEpLibrary,
                    ),
                  ],
                  if (Platform.isWindows) ...[
                    const SizedBox(height: 22),
                    const CSection(label: 'Windows ML'),
                    const SizedBox(height: 10),
                    _WinmlPanel(
                      enabled: settings.onnxWinml,
                      path: settings.onnxWinmlPath,
                      status: _onnx,
                      locked: running,
                      onToggle: (on) =>
                          _applyWinml(enabled: on, path: settings.onnxWinmlPath),
                      onPickDir: _pickWinmlDir,
                      onClearDir: () =>
                          _applyWinml(enabled: settings.onnxWinml, path: ''),
                    ),
                  ],
                  const SizedBox(height: 22),
                  const CSection(label: 'LiteRT'),
                  const SizedBox(height: 10),
                  _RuntimePanel(
                    view: _RuntimeView.litert(_litert),
                    savedPath: settings.litertLibraryPath,
                    locked: running,
                    onPick: () => _pickLibrary(_Runtime.litert),
                    onReset: () => _resetLibrary(_Runtime.litert),
                  ),
                  const SizedBox(height: 22),
                  const CSection(label: 'Diagnostics'),
                  const SizedBox(height: 10),
                  _DiagnosticsPanel(
                    verbose: settings.verbose,
                    onChanged: settings.setVerbose,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Which loadable runtime a picker row is about.
enum _Runtime {
  onnx('onnxruntime'),
  litert('litert');

  const _Runtime(this.dirName);

  /// Where an imported copy lives inside the Android sandbox.
  final String dirName;
}

/// What the panel shows for one runtime, whichever backend reported it.
class _RuntimeView {
  const _RuntimeView({
    required this.known,
    required this.available,
    required this.fixed,
    required this.title,
    required this.path,
    required this.error,
    required this.hint,
    required this.fixedHint,
  });

  /// A status has been read at all (false while "Checking…").
  final bool known;
  final bool available;

  /// Built into the app or otherwise not choosable.
  final bool fixed;
  final String title;
  final String path;
  final String error;
  final String hint;
  final String fixedHint;

  factory _RuntimeView.onnx(OnnxStatus? s) => _RuntimeView(
        known: s != null,
        available: s?.available ?? false,
        // iOS links ONNX Runtime into the app — Apple's pod is a static
        // framework and iOS will not dlopen another one — so there is
        // nothing to choose.
        fixed: s?.linkedIn ?? Platform.isIOS,
        title: s == null
            ? 'Checking…'
            : s.available
                ? 'ONNX Runtime ${s.version}'
                : 'No runtime loaded',
        path: s?.path ?? '',
        error: s?.error ?? '',
        hint: _onnxRuntimeHint,
        fixedHint: 'Linked into the app; there is nothing else to choose.',
      );

  /// The engine's own hint text, worded to what actually follows it on this
  /// platform: a plugin-library section and (Windows) Windows ML on
  /// desktop, neither of which the phone build shows — there the app adds
  /// a vendor NPU's plugin on its own when the device has one.
  static String get _onnxRuntimeHint {
    if (Platform.isAndroid) {
      return 'The engine ONNX runs on. A vendor NPU plugin (Qualcomm\'s QNN) '
          'is added automatically when this device has one.';
    }
    final more = Platform.isWindows
        ? 'a plugin library below, or Windows ML further down'
        : 'a plugin library below';
    return 'The engine ONNX runs on — its build decides which providers '
        'exist by default (always CPU, plus GPU/NPU if it shipped one). '
        'Add a vendor NPU it left out with $more.';
  }

  factory _RuntimeView.litert(LitertStatus? s) => _RuntimeView(
        known: s != null,
        available: s?.available ?? false,
        // iOS embeds Google's LiteRT dylibs in the app bundle, and loads no
        // library from anywhere else -- so there is nothing to choose,
        // though unlike ONNX Runtime it is still loaded, not linked.
        fixed: Platform.isIOS,
        title: s == null
            ? 'Checking…'
            : s.available
                ? 'LiteRT (${s.version})'
                : 'No runtime loaded',
        path: s?.path ?? '',
        error: s?.error ?? '',
        hint: 'Bundled on mobile; if none is found, choose a pip '
            'ai-edge-litert libLiteRt.',
        fixedHint: 'Bundled with the app; iOS loads nothing else.',
      );
}

class _RuntimePanel extends StatelessWidget {
  const _RuntimePanel({
    required this.view,
    required this.savedPath,
    required this.locked,
    required this.onPick,
    required this.onReset,
  });

  final _RuntimeView view;
  final String savedPath;

  /// A run is in flight; the loader hands out a pointer to the runtime it
  /// loaded, so swapping libraries underneath one is not on offer.
  final bool locked;

  final VoidCallback onPick;
  final VoidCallback onReset;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final s = view;
    final tint = ClpeakTheme.categoryColor(BenchCategory.ai,
        brightness: Theme.of(context).brightness);
    final fixed = s.fixed;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              CRow(
                accent: s.available ? tint : null,
                child: Row(
                  children: [
                    Expanded(child: Text(s.title, style: t.mono)),
                    if (s.known)
                      CTag(
                        text: s.available ? 'loaded' : 'absent',
                        color: s.available ? tint : t.dim,
                      ),
                  ],
                ),
              ),
              CRow(
                rule: s.error.isNotEmpty,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.folder_outlined, size: 15, color: t.dim),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        fixed
                            ? 'Built into the app'
                            : s.path.isNotEmpty
                                ? p.basename(s.path)
                                : savedPath.isNotEmpty
                                    ? p.basename(savedPath)
                                    : 'Found by name on the system paths',
                        style: t.monoSmallDim,
                      ),
                    ),
                  ],
                ),
              ),
              if (s.error.isNotEmpty)
                CRow(
                  rule: false,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.error_outline, size: 15, color: t.danger),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(s.error,
                            style: t.monoSmallDim.copyWith(color: t.danger)),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Text(
          fixed ? s.fixedHint : s.hint,
          style: t.micro.copyWith(color: t.dim),
        ),
        if (!fixed) ...[
          const SizedBox(height: 12),
          Row(
            children: [
              CButton(
                label: 'Choose library…',
                icon: Icons.folder_open,
                onPressed: locked ? null : onPick,
              ),
              const SizedBox(width: 8),
              CButton(
                label: 'Use default',
                onPressed: locked || savedPath.isEmpty ? null : onReset,
              ),
            ],
          ),
          if (locked) ...[
            const SizedBox(height: 8),
            Text('Not while a run is in flight.',
                style: t.micro.copyWith(color: t.dim)),
          ],
        ],
      ],
    );
  }
}

/// The plugin execution-provider libraries registered on the ONNX Runtime
/// (1.22+), each with how the last enumeration found it.
class _EpLibrariesPanel extends StatelessWidget {
  const _EpLibrariesPanel({
    required this.libraries,
    required this.status,
    required this.locked,
    required this.onAdd,
    required this.onRemove,
  });

  final List<OnnxEpLibrary> libraries;
  final List<OnnxEpLibraryStatus> status;
  final bool locked;
  final VoidCallback onAdd;
  final ValueChanged<OnnxEpLibrary> onRemove;

  OnnxEpLibraryStatus? _statusOf(OnnxEpLibrary lib) {
    for (final s in status) {
      if (s.name == lib.name && s.path == lib.path) return s;
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final tint = ClpeakTheme.categoryColor(BenchCategory.ai,
        brightness: Theme.of(context).brightness);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              if (libraries.isEmpty)
                CRow(
                  rule: false,
                  child: Text('None registered', style: t.monoSmallDim),
                ),
              for (var i = 0; i < libraries.length; i++)
                Builder(builder: (context) {
                  final lib = libraries[i];
                  final st = _statusOf(lib);
                  final registered = st?.registered ?? false;
                  return CRow(
                    rule: i < libraries.length - 1,
                    accent: registered ? tint : null,
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.extension_outlined, size: 15, color: t.dim),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(lib.name, style: t.mono),
                              const SizedBox(height: 3),
                              Text(p.basename(lib.path), style: t.monoSmallDim),
                              if (st != null && !registered) ...[
                                const SizedBox(height: 3),
                                Text(st.error,
                                    style: t.monoSmallDim
                                        .copyWith(color: t.danger)),
                              ],
                            ],
                          ),
                        ),
                        const SizedBox(width: 8),
                        CTag(
                          text: st == null
                              ? 'pending'
                              : registered
                                  ? 'registered'
                                  : 'failed',
                          color: st == null
                              ? t.dim
                              : registered
                                  ? tint
                                  : t.danger,
                        ),
                        const SizedBox(width: 4),
                        CIconButton(
                          icon: Icons.close,
                          tooltip: 'Remove',
                          onPressed: locked ? null : () => onRemove(lib),
                        ),
                      ],
                    ),
                  );
                }),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'Add a provider the runtime above does not already include — most '
          'vendor NPUs ship this way now, separately from ONNX Runtime '
          'itself: Qualcomm\'s QNN plugin (onnxruntime_providers_qnn) reaches '
          'the Hexagon NPU this way on a stock ONNX Runtime 1.24 or newer.',
          style: t.micro.copyWith(color: t.dim),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            CButton(
              label: 'Add library…',
              icon: Icons.add,
              onPressed: locked ? null : onAdd,
            ),
          ],
        ),
      ],
    );
  }
}

/// Windows ML's execution-provider catalog: the switch, where its DLL is,
/// and what the last enumeration made of it.
class _WinmlPanel extends StatelessWidget {
  const _WinmlPanel({
    required this.enabled,
    required this.path,
    required this.status,
    required this.locked,
    required this.onToggle,
    required this.onPickDir,
    required this.onClearDir,
  });

  final bool enabled;
  final String path;
  final OnnxStatus? status;
  final bool locked;
  final ValueChanged<bool> onToggle;
  final VoidCallback onPickDir;
  final VoidCallback onClearDir;

  /// What older native builds reported while nothing had resolved the
  /// catalog yet; current ones leave the error empty instead. Both mean
  /// pending, never a failure.
  static const _pendingError = 'not resolved yet; enumerate or run first';

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final s = status;
    final resolved = s != null && s.winmlEnabled && s.winmlPath.isNotEmpty;
    // Pending until the next enumeration or run resolves the catalog -- the
    // native status leaves both empty for that (older builds sent the
    // sentence below as the error, which is still treated as pending here).
    // Like the plugin list's empty-until-enumerated, it is not a failure.
    final isPending = enabled &&
        (s == null ||
            (s.winmlEnabled &&
                s.winmlPath.isEmpty &&
                (s.winmlError.isEmpty || s.winmlError == _pendingError)));
    final error = s != null && s.winmlEnabled && !isPending ? s.winmlError : '';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              CRow(
                padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
                child: Row(
                  children: [
                    Icon(Icons.storefront_outlined, size: 15, color: t.dim),
                    const SizedBox(width: 10),
                    Expanded(
                        child: Text('Store execution providers',
                            style: t.mono)),
                    CSwitch(
                        value: enabled,
                        onChanged: locked ? (_) {} : onToggle),
                  ],
                ),
              ),
              CRow(
                rule: enabled && (error.isNotEmpty || isPending),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.folder_outlined, size: 15, color: t.dim),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        resolved
                            ? s.winmlPath
                            : path.isNotEmpty
                                ? path
                                : 'Microsoft.Windows.AI.MachineLearning.dll '
                                    'beside the runtime or the app',
                        style: t.monoSmallDim,
                      ),
                    ),
                  ],
                ),
              ),
              if (enabled && isPending)
                CRow(
                  rule: false,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.hourglass_empty, size: 15, color: t.dim),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text('Waiting for enumeration…',
                            style: t.monoSmallDim),
                      ),
                    ],
                  ),
                ),
              if (enabled && error.isNotEmpty)
                CRow(
                  rule: false,
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Icon(Icons.error_outline, size: 15, color: t.danger),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(error,
                            style: t.monoSmallDim.copyWith(color: t.danger)),
                      ),
                    ],
                  ),
                ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'The automatic version of the plugin providers above: turning this '
          'on installs the vendor providers (Qualcomm QNN, Intel OpenVINO, '
          'AMD Vitis AI, NVIDIA TensorRT for RTX) that fit this machine from '
          'the Microsoft Store on first use — a download — instead of you '
          'naming the library yourself. The catalog DLL comes with the '
          'Microsoft.Windows.AI.MachineLearning package, not with clpeak.',
          style: t.micro.copyWith(color: t.dim),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            CButton(
              label: 'Choose folder…',
              icon: Icons.folder_open,
              onPressed: locked ? null : onPickDir,
            ),
            const SizedBox(width: 8),
            CButton(
              label: 'Use default',
              onPressed: locked || path.isEmpty ? null : onClearDir,
            ),
          ],
        ),
      ],
    );
  }
}

/// The verbose switch, and the way to the issue tracker it exists for.
class _DiagnosticsPanel extends StatelessWidget {
  const _DiagnosticsPanel({required this.verbose, required this.onChanged});

  final bool verbose;
  final ValueChanged<bool> onChanged;

  static final _issuesUrl =
      Uri.parse('https://github.com/krrishnarraj/clpeak/issues/new');

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              CRow(
                padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
                child: Row(
                  children: [
                    Icon(Icons.bug_report_outlined, size: 15, color: t.dim),
                    const SizedBox(width: 10),
                    Expanded(
                        child: Text('Verbose diagnostics', style: t.mono)),
                    CSwitch(value: verbose, onChanged: onChanged),
                  ],
                ),
              ),
              CRow(
                rule: false,
                onTap: () =>
                    launchUrl(_issuesUrl, mode: LaunchMode.externalApplication),
                child: Row(
                  children: [
                    Icon(Icons.open_in_new, size: 15, color: t.dim),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Report an issue', style: t.mono),
                          const SizedBox(height: 3),
                          Text('github.com/krrishnarraj/clpeak/issues',
                              style: t.monoSmallDim),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Text(
          'Off by default. When on, saved runs include debug output — '
          'enable it before re-running a suspicious result for an issue report.',
          style: t.micro.copyWith(color: t.dim),
        ),
      ],
    );
  }
}

/// Square segmented control — hairline frame, solid block on the active cell.
class _ThemeToggle extends StatelessWidget {
  const _ThemeToggle({required this.value, required this.onChanged});

  final ThemeMode value;
  final ValueChanged<ThemeMode> onChanged;

  static const _modes = [
    (ThemeMode.system, 'Auto'),
    (ThemeMode.light, 'Light'),
    (ThemeMode.dark, 'Dark'),
  ];

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: t.line),
        borderRadius: BorderRadius.circular(CP.rControl),
      ),
      clipBehavior: Clip.antiAlias,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          for (var i = 0; i < _modes.length; i++)
            CTap(
              onTap: () => onChanged(_modes[i].$1),
              builder: (context, hovered, pressed) {
                final on = value == _modes[i].$1;
                return Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                  decoration: BoxDecoration(
                    color: on
                        ? t.inverse
                        : (hovered || pressed
                            ? t.hover
                            : Colors.transparent),
                    border: Border(
                      left: i == 0
                          ? BorderSide.none
                          : BorderSide(color: t.line),
                    ),
                  ),
                  child: Text(
                    _modes[i].$2.toUpperCase(),
                    style: t.micro
                        .copyWith(color: on ? t.onInverse : t.dim),
                  ),
                );
              },
            ),
        ],
      ),
    );
  }
}
