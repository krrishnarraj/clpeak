import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../ffi/xpupeak_bindings.dart';
import '../../services/benchmark_service.dart';
import '../../services/settings_service.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/kit.dart';

/// Appearance, which ONNX Runtime and LiteRT libraries the two backends
/// measure, and whether runs record verbose diagnostics.
///
/// The runtime picker is the reason this screen exists.  Unlike every other
/// backend, ONNX has no single driver on a machine: NPU vendors ship their own
/// builds, and which one is loaded decides which execution providers appear at
/// all — a stock build offers CPU and nothing else.  So the library is a
/// setting.  Where the backends run in this process (mobile) it is set once
/// per launch: the first runtime that loads fixes it (with Windows ML, which
/// can choose the runtime and adds providers to it), so a change made after
/// that is saved for the next launch, and the panel shows both — the runtime
/// active now, and the one the next launch loads.  On desktop every
/// enumeration and run is a new engine process, so a change applies to the
/// next one ([BenchmarkService.runtimeFixedOnceLoaded]).  LiteRT's library
/// works the same way.  Plugin libraries stay live.
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

  /// Apply a library choice: persist it and hand it to the native loader,
  /// which loads it now if no runtime has loaded yet (and the service
  /// re-enumerates) or keeps it for the next launch.  Either way the status
  /// read afterwards says which.
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
            Text(path, style: t.caption),
            const SizedBox(height: 12),
            // The kit's search field, as a plain text field: the grey
            // track, an ink edge while it has focus.
            TextField(
              controller: controller,
              autofocus: true,
              style: t.body.copyWith(fontSize: 14.5),
              cursorColor: t.ink,
              cursorWidth: 1.5,
              decoration: InputDecoration(
                isDense: true,
                filled: true,
                fillColor: t.surface2,
                contentPadding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide.none,
                ),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide.none,
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(CP.rControl),
                  borderSide: BorderSide(color: t.ink, width: 1.5),
                ),
              ),
              onSubmitted: (v) => Navigator.pop(context, v.trim()),
            ),
            const SizedBox(height: 12),
            Text(
                'The name the provider expects to be registered under; '
                'Qualcomm\'s QNN plugin requires QNNExecutionProvider.',
                style: t.prose),
          ],
        ),
      ),
    );
  }

  /// Switch the Windows ML catalog, part of the runtime setup: before a
  /// runtime has loaded this re-enumerates, which installs whatever certified
  /// providers fit the machine; after, it is saved for the next launch.
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
    final settings = context.watch<SettingsService>();
    final running = context.select<BenchmarkService, bool>((s) => s.isRunning);
    final perLaunch = context.read<BenchmarkService>().runtimeFixedOnceLoaded;

    return Scaffold(
      body: SafeArea(
        child: _Page(
          title: 'Settings',
          sections: [
            _Section(
              label: 'Appearance',
              child: _AppearanceCard(
                value: settings.themeMode,
                onChanged: settings.setThemeMode,
              ),
            ),
            _Section(
              label: 'ONNX Runtime',
              child: _RuntimeCard(
                view: _RuntimeView.onnx(_onnx,
                    winml: Platform.isWindows, perLaunch: perLaunch),
                savedPath: settings.onnxLibraryPath,
                locked: running,
                onPick: () => _pickLibrary(_Runtime.onnx),
                onReset: () => _resetLibrary(_Runtime.onnx),
              ),
            ),
            // Windows ML beside the library it belongs with: together they
            // are the runtime setup.
            if (Platform.isWindows)
              _Section(
                label: 'Windows ML',
                child: _WinmlCard(
                  enabled: settings.onnxWinml,
                  path: settings.onnxWinmlPath,
                  locked: running,
                  perLaunch: perLaunch,
                  onToggle: (on) =>
                      _applyWinml(enabled: on, path: settings.onnxWinmlPath),
                  onPickDir: _pickWinmlDir,
                  onClearDir: () =>
                      _applyWinml(enabled: settings.onnxWinml, path: ''),
                ),
              ),
            if (_desktop)
              _Section(
                label: 'Plugin execution providers',
                child: _EpLibrariesCard(
                  libraries: settings.onnxEpLibraries,
                  status: _onnx?.epLibraries ?? const [],
                  locked: running,
                  onAdd: _addEpLibrary,
                  onRemove: _removeEpLibrary,
                ),
              ),
            _Section(
              label: 'LiteRT',
              child: _RuntimeCard(
                view: _RuntimeView.litert(_litert, perLaunch: perLaunch),
                savedPath: settings.litertLibraryPath,
                locked: running,
                onPick: () => _pickLibrary(_Runtime.litert),
                onReset: () => _resetLibrary(_Runtime.litert),
              ),
            ),
            _Section(
              label: 'Diagnostics',
              note: 'Off by default. When on, saved runs include debug '
                  'output — enable it before re-running a suspicious result '
                  'for an issue report.',
              child: _DiagnosticsCard(
                verbose: settings.verbose,
                onChanged: settings.setVerbose,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// The page: its title, then its sections, scrolling together -- on
/// desktop in a column no wider than a settings form.
class _Page extends StatelessWidget {
  const _Page({required this.title, required this.sections});

  final String title;
  final List<Widget> sections;

  @override
  Widget build(BuildContext context) {
    final wide = cIsWide(context);
    final side = wide ? 34.0 : 16.0;
    return ListView(
      padding: EdgeInsets.only(bottom: wide ? 44 : 28),
      children: [
        CHeader(title: title),
        for (var i = 0; i < sections.length; i++)
          Padding(
            padding: EdgeInsets.fromLTRB(
                side, i == 0 ? (wide ? 14 : 12) : (wide ? 26 : 20), side, 0),
            child: Align(
              alignment: Alignment.topLeft,
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 780),
                child: sections[i],
              ),
            ),
          ),
      ],
    );
  }
}

/// A section: its head in capitals, its card, and a paragraph under it.
class _Section extends StatelessWidget {
  const _Section({required this.label, required this.child, this.note});

  final String label;
  final Widget child;
  final String? note;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CSection(label: label),
        SizedBox(height: wide ? 10 : 8),
        child,
        if (note != null) ...[
          SizedBox(height: wide ? 10 : 8),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6),
            child: Text(note!,
                style: t.prose.copyWith(fontSize: wide ? 13 : 12.5)),
          ),
        ],
      ],
    );
  }
}

/// The type of a settings card: a row's name, the line under it, and a
/// paragraph of explanation.
class _Type {
  _Type(BuildContext context)
      : t = CP.of(context),
        wide = cIsWide(context);

  final CP t;
  final bool wide;

  TextStyle get name => t.title.copyWith(fontSize: wide ? 15.5 : 15);
  TextStyle get line => t.caption.copyWith(fontSize: wide ? 13 : 12.5);
  TextStyle get prose => t.prose.copyWith(fontSize: wide ? 13.5 : 13);

  EdgeInsets get padding => wide
      ? const EdgeInsets.fromLTRB(20, 18, 20, 18)
      : const EdgeInsets.fromLTRB(18, 16, 18, 16);
}

/// Theme: System, Light or Dark, and what System means right now.
class _AppearanceCard extends StatelessWidget {
  const _AppearanceCard({required this.value, required this.onChanged});

  final ThemeMode value;
  final ValueChanged<ThemeMode> onChanged;

  @override
  Widget build(BuildContext context) {
    final type = _Type(context);
    final system = MediaQuery.platformBrightnessOf(context) == Brightness.dark
        ? 'Dark'
        : 'Light';
    final label = Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Theme', style: type.name),
        const SizedBox(height: 1),
        Text(
          value == ThemeMode.system
              ? 'Following the system: $system right now'
              : 'System is the default',
          style: type.line,
        ),
      ],
    );
    final control = CSegmented<ThemeMode>(
      segments: const [
        CSegment(ThemeMode.system, 'System'),
        CSegment(ThemeMode.light, 'Light'),
        CSegment(ThemeMode.dark, 'Dark'),
      ],
      selected: value,
      onChanged: onChanged,
      expand: true,
      semanticLabel: 'Theme',
    );
    return CPanel(
      padding: type.wide
          ? const EdgeInsets.fromLTRB(20, 16, 18, 16)
          : const EdgeInsets.fromLTRB(18, 14, 14, 14),
      child: LayoutBuilder(builder: (context, c) {
        if (c.maxWidth < 520) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [label, const SizedBox(height: 12), control],
          );
        }
        return Row(
          children: [
            Expanded(child: label),
            const SizedBox(width: 18),
            SizedBox(width: 300, child: control),
          ],
        );
      }),
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

/// One line of a runtime setup under its path: Windows ML's state, say.
class _SetupLine {
  const _SetupLine(this.text, {this.danger = false});

  final String text;
  final bool danger;
}

/// A runtime setup saved since the runtime loaded, which the next launch
/// loads: its library path (empty = found by name) and further lines.
class _NextLaunch {
  const _NextLaunch({required this.path, this.lines = const []});

  final String path;
  final List<_SetupLine> lines;
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
    this.lines = const [],
    this.next,
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

  /// The rest of the active setup, under its path (ONNX: Windows ML).
  final List<_SetupLine> lines;

  /// The setup the next launch loads, when it differs from the active one.
  final _NextLaunch? next;

  /// What older native builds reported while nothing had resolved the
  /// Windows ML catalog yet; current ones leave the error empty instead.
  /// Both mean pending, never a failure.
  static const _winmlPendingError = 'not resolved yet; enumerate or run first';

  /// Windows ML as the active runtime was set up with it.
  static _SetupLine _activeWinml(OnnxStatus s) {
    if (!s.winmlEnabled) return const _SetupLine('Windows ML off');
    if (s.winmlPath.isNotEmpty) {
      return _SetupLine('Windows ML on · ${s.winmlPath}');
    }
    if (s.winmlError.isNotEmpty && s.winmlError != _winmlPendingError) {
      return _SetupLine('Windows ML on · ${s.winmlError}', danger: true);
    }
    return const _SetupLine('Windows ML on · waiting for enumeration…');
  }

  /// [winml]: this platform has the Windows ML catalog, so its state is part
  /// of the setup shown.
  ///
  /// [perLaunch]: a runtime choice waits for the next launch once one has
  /// loaded (in-process); otherwise it applies to the next enumeration.
  factory _RuntimeView.onnx(OnnxStatus? s,
      {required bool winml, required bool perLaunch}) {
    final pending = s?.pendingRuntime;
    return _RuntimeView(
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
      hint: _onnxRuntimeHint(perLaunch),
      fixedHint: 'Linked into the app; there is nothing else to choose.',
      lines: [if (winml && s != null && s.available) _activeWinml(s)],
      next: pending == null
          ? null
          : _NextLaunch(path: pending.path, lines: [
              if (winml || pending.winml)
                _SetupLine(pending.winml
                    ? 'Windows ML on · '
                        '${pending.winmlPath.isNotEmpty ? pending.winmlPath : 'catalog beside the runtime or the app'}'
                    : 'Windows ML off'),
            ]),
    );
  }

  /// The engine's own hint text, worded to what actually follows it on this
  /// platform: Windows ML right below and a plugin-library section on
  /// desktop, neither of which the phone build shows — there the app adds a
  /// vendor NPU's plugin on its own when the device has one.
  static String _onnxRuntimeHint(bool perLaunch) {
    final once = perLaunch
        ? 'Set once per launch: after a runtime has loaded, a change takes '
            'effect the next time XpuPeak starts.'
        : 'A change applies from the next device scan and run.';
    if (Platform.isAndroid) {
      return 'The engine ONNX runs on. A vendor NPU plugin (Qualcomm\'s QNN) '
          'is added automatically when this device has one. $once';
    }
    final more = Platform.isWindows
        ? 'Windows ML below, or a plugin library'
        : 'a plugin library below';
    return 'The engine ONNX runs on — its build decides which providers '
        'exist by default (always CPU, plus GPU/NPU if it shipped one). '
        'Add a vendor NPU it left out with $more. $once';
  }

  factory _RuntimeView.litert(LitertStatus? s, {required bool perLaunch}) =>
      _RuntimeView(
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
            'ai-edge-litert libLiteRt.'
            '${perLaunch ? ' Set once per launch, like ONNX Runtime.' : ''}',
        fixedHint: 'Bundled with the app; iOS loads nothing else.',
        next: s?.pendingPath == null ? null : _NextLaunch(path: s!.pendingPath!),
      );
}

/// A status dot: filled in [color] for a runtime that loaded or a provider
/// that registered, an empty ring for one that is absent or not yet known.
class _Dot extends StatelessWidget {
  const _Dot({this.color});

  /// Null draws the ring.
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      width: 10,
      height: 10,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color,
        border: color == null ? Border.all(color: t.muted, width: 1.5) : null,
      ),
    );
  }
}

/// A row's name with its status dot before it and its lines under it, the
/// dot centred on the name's first line.
class _StatusRow extends StatelessWidget {
  const _StatusRow({
    required this.dot,
    required this.name,
    this.lines = const [],
    this.muted = false,
  });

  final Color? dot;
  final String name;

  /// Under the name: paths, setup lines, errors (in danger).
  final List<_SetupLine> lines;

  /// The whole row inactive (the next launch's setup).
  final bool muted;

  @override
  Widget build(BuildContext context) {
    final type = _Type(context);
    final t = type.t;
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: EdgeInsets.only(top: type.wide ? 6 : 5.5),
          child: _Dot(color: dot),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(name,
                  style: muted ? type.name.copyWith(color: t.muted) : type.name),
              for (final line in lines) ...[
                const SizedBox(height: 2),
                // Wraps: a path is read to its end, which is the file.
                Text(line.text,
                    style: line.danger
                        ? type.line.copyWith(color: t.danger)
                        : type.line),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

/// A row of a card's actions, and the note that a run holds them.
class _Actions extends StatelessWidget {
  const _Actions({required this.children, required this.locked});

  final List<Widget> children;
  final bool locked;

  @override
  Widget build(BuildContext context) {
    final type = _Type(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // A Wrap, so a phone-width screen puts the second button under the
        // first rather than past the edge.
        Wrap(
          spacing: 8,
          runSpacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: children,
        ),
        if (locked) ...[
          const SizedBox(height: 6),
          Text('Not while a run is in flight.', style: type.line),
        ],
      ],
    );
  }
}

class _RuntimeCard extends StatelessWidget {
  const _RuntimeCard({
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

  static const _byName = 'Found by name on the system paths';

  @override
  Widget build(BuildContext context) {
    final type = _Type(context);
    final t = type.t;
    final s = view;
    final next = s.next;

    return CPanel(
      padding: type.padding,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _StatusRow(
            dot: s.available ? t.ok : null,
            name: s.title,
            lines: [
              _SetupLine(s.fixed
                  ? 'Built into the app'
                  : s.path.isNotEmpty
                      ? s.path
                      : savedPath.isNotEmpty
                          ? savedPath
                          : _byName),
              ...s.lines,
              if (s.error.isNotEmpty) _SetupLine(s.error, danger: true),
            ],
          ),
          // The saved setup the next launch loads, inactive until then.
          if (next != null) ...[
            const SizedBox(height: 14),
            Container(
              padding: const EdgeInsets.fromLTRB(16, 13, 16, 14),
              decoration: BoxDecoration(
                color: t.well,
                borderRadius: BorderRadius.circular(CP.rWell),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _StatusRow(
                    dot: null,
                    name: 'Next launch',
                    muted: true,
                    lines: [
                      _SetupLine(next.path.isNotEmpty ? next.path : _byName),
                      ...next.lines,
                    ],
                  ),
                  const SizedBox(height: 8),
                  Padding(
                    padding: const EdgeInsets.only(left: 22),
                    child: Text('Takes effect the next time XpuPeak starts.',
                        style: t.smallMuted),
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(height: 12),
          Text(s.fixed ? s.fixedHint : s.hint, style: type.prose),
          if (!s.fixed) ...[
            const SizedBox(height: 12),
            _Actions(
              locked: locked,
              children: [
                CButton(
                  label: 'Choose library…',
                  onPressed: locked ? null : onPick,
                ),
                CButton(
                  label: 'Use default',
                  kind: CButtonKind.quiet,
                  onPressed: locked || savedPath.isEmpty ? null : onReset,
                ),
              ],
            ),
          ],
        ],
      ),
    );
  }
}

/// The plugin execution-provider libraries registered on the ONNX Runtime
/// (1.22+), each with how the last enumeration found it.
class _EpLibrariesCard extends StatelessWidget {
  const _EpLibrariesCard({
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
    final type = _Type(context);
    final t = type.t;
    return CPanel(
      padding: type.padding,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (libraries.isEmpty)
            Text('None registered',
                style: t.title.copyWith(
                    fontSize: type.wide ? 15 : 14.5,
                    fontWeight: FontWeight.w600,
                    color: t.muted)),
          for (var i = 0; i < libraries.length; i++)
            Builder(builder: (context) {
              final lib = libraries[i];
              final st = _statusOf(lib);
              final registered = st?.registered ?? false;
              final replaces = st?.replaces ?? const <String>[];
              return Container(
                padding: EdgeInsets.only(
                    top: i == 0 ? 0 : 10,
                    bottom: i == libraries.length - 1 ? 0 : 10),
                decoration: BoxDecoration(
                  border: i == libraries.length - 1
                      ? null
                      : Border(bottom: BorderSide(color: t.line)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: _StatusRow(
                        dot: st == null
                            ? null
                            : registered
                                ? t.ok
                                : t.danger,
                        name: lib.name,
                        lines: [
                          _SetupLine(lib.path),
                          for (final r in replaces)
                            _SetupLine('In place of Windows ML\'s $r'),
                          if (st != null && !registered)
                            _SetupLine(st.error, danger: true),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Padding(
                      padding: const EdgeInsets.only(top: 1),
                      child: CTag(
                        text: st == null
                            ? 'Pending'
                            : registered
                                ? 'Registered'
                                : 'Failed',
                        upper: false,
                        color: st == null
                            ? null
                            : registered
                                ? t.ok
                                : t.danger,
                      ),
                    ),
                    // Pulled up to the name's line: the button's 44 px
                    // target is taller than the row's first line.
                    Transform.translate(
                      offset: const Offset(6, -10),
                      child: CIconButton(
                        icon: Icons.close,
                        tooltip: 'Remove',
                        onPressed: locked ? null : () => onRemove(lib),
                      ),
                    ),
                  ],
                ),
              );
            }),
          const SizedBox(height: 12),
          Text(
            'Add a provider the runtime above does not already include — '
            'most vendor NPUs ship this way now, separately from ONNX Runtime '
            'itself: Qualcomm\'s QNN plugin (onnxruntime_providers_qnn) '
            'reaches the Hexagon NPU this way on a stock ONNX Runtime 1.24 or '
            'newer.',
            style: type.prose,
          ),
          const SizedBox(height: 12),
          _Actions(
            locked: locked,
            children: [
              CButton(
                label: 'Add library…',
                onPressed: locked ? null : onAdd,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// Windows ML's execution-provider catalog, as saved: the switch and where
/// its DLL is.  What the active runtime was set up with -- and what the
/// catalog made of it -- is the runtime card's to say, beside the library
/// it belongs with; a change here after a runtime has loaded shows up there
/// as the next launch's setup.
class _WinmlCard extends StatelessWidget {
  const _WinmlCard({
    required this.enabled,
    required this.path,
    required this.locked,
    required this.perLaunch,
    required this.onToggle,
    required this.onPickDir,
    required this.onClearDir,
  });

  final bool enabled;
  final String path;
  final bool locked;
  final bool perLaunch; // see _RuntimeView.onnx
  final ValueChanged<bool> onToggle;
  final VoidCallback onPickDir;
  final VoidCallback onClearDir;

  @override
  Widget build(BuildContext context) {
    final type = _Type(context);
    return CPanel(
      padding: type.padding,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Store execution providers', style: type.name),
                    const SizedBox(height: 2),
                    Text(
                      path.isNotEmpty
                          ? path
                          : 'Microsoft.Windows.AI.MachineLearning.dll '
                              'beside the runtime or the app',
                      style: type.line,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              CSwitch(
                value: enabled,
                onChanged: locked ? null : onToggle,
                semanticLabel: 'Store execution providers',
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'The automatic version of the plugin providers below: turning '
            'this on installs the vendor providers (Qualcomm QNN, Intel '
            'OpenVINO, AMD Vitis AI, NVIDIA TensorRT for RTX) that fit this '
            'machine from the Microsoft Store on first use — a download — '
            'instead of you naming the library yourself. The catalog DLL '
            'comes with the Microsoft.Windows.AI.MachineLearning package, not '
            'with XpuPeak.'
            '${perLaunch ? ' Part of the runtime setup above, so it too is set '
                'once per launch.' : ''}',
            style: type.prose,
          ),
          const SizedBox(height: 12),
          _Actions(
            locked: locked,
            children: [
              CButton(
                label: 'Choose folder…',
                onPressed: locked ? null : onPickDir,
              ),
              CButton(
                label: 'Use default',
                kind: CButtonKind.quiet,
                onPressed: locked || path.isEmpty ? null : onClearDir,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// The verbose switch, and the way to the issue tracker it exists for.
class _DiagnosticsCard extends StatelessWidget {
  const _DiagnosticsCard({required this.verbose, required this.onChanged});

  final bool verbose;
  final ValueChanged<bool> onChanged;

  static final _issuesUrl =
      Uri.parse('https://github.com/krrishnarraj/xpupeak/issues/new');

  @override
  Widget build(BuildContext context) {
    final type = _Type(context);
    final t = type.t;
    return CPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          CRow(
            padding: type.wide
                ? const EdgeInsets.fromLTRB(20, 8, 12, 8)
                : const EdgeInsets.fromLTRB(18, 6, 8, 6),
            child: Row(
              children: [
                Expanded(child: Text('Verbose diagnostics', style: type.name)),
                CSwitch(
                  value: verbose,
                  onChanged: onChanged,
                  semanticLabel: 'Verbose diagnostics',
                ),
              ],
            ),
          ),
          CRow(
            rule: false,
            padding: type.wide
                ? const EdgeInsets.fromLTRB(20, 14, 20, 14)
                : const EdgeInsets.fromLTRB(18, 12, 18, 12),
            onTap: () =>
                launchUrl(_issuesUrl, mode: LaunchMode.externalApplication),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Report an issue', style: type.name),
                      const SizedBox(height: 1),
                      Text('github.com/krrishnarraj/xpupeak/issues',
                          style: type.line),
                    ],
                  ),
                ),
                const SizedBox(width: 10),
                CGlyph(CGlyphs.external, size: 18, color: t.ink, stroke: 2),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
