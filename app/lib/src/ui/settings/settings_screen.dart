import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:provider/provider.dart';

import '../../ffi/clpeak_bindings.dart';
import '../../model/result_model.dart';
import '../../services/benchmark_service.dart';
import '../../services/settings_service.dart';
import '../../theme/clpeak_theme.dart';
import '../common/kit.dart';

/// Appearance, and which ONNX Runtime the ONNX backend measures.
///
/// The runtime picker is the reason this screen exists.  Unlike every other
/// backend, ONNX has no single driver on a machine: NPU vendors ship their own
/// builds, and which one is loaded decides which execution providers appear at
/// all — a stock build offers CPU and nothing else.  So the library is a
/// setting, and changing it re-enumerates immediately rather than asking for a
/// restart.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  OnnxStatus? _onnx;

  @override
  void initState() {
    super.initState();
    _refreshOnnx();
  }

  void _refreshOnnx() {
    setState(() => _onnx = context.read<BenchmarkService>().onnxStatus());
  }

  /// Apply a library choice: persist it, hand it to the native loader, then
  /// re-enumerate so the provider list on the run screen matches what was
  /// just chosen.
  Future<void> _applyLibrary(String path) async {
    final settings = context.read<SettingsService>();
    final service = context.read<BenchmarkService>();
    await settings.setOnnxLibraryPath(path);
    if (!mounted) return;
    service.setOnnxLibrary(path);
    _refreshOnnx();
  }

  Future<void> _pickLibrary() async {
    // No extension filter: a runtime is .so / .dylib / .dll depending on the
    // platform, and plenty of real ones are versioned
    // (libonnxruntime.so.1.27.0) where an extension filter matches nothing.
    final file = await openFile(
      acceptedTypeGroups: const [XTypeGroup(label: 'Shared library')],
    );
    if (file == null) return;
    await _applyLibrary(await _durablePath(file.path));
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
  Future<String> _durablePath(String picked) async {
    if (!Platform.isAndroid) return picked;
    final dir = Directory(
        p.join((await getApplicationSupportDirectory()).path, 'onnxruntime'));
    if (dir.existsSync()) dir.deleteSync(recursive: true);
    dir.createSync(recursive: true);
    final stamp = DateTime.now().millisecondsSinceEpoch;
    final dest = p.join(dir.path, '${stamp}_${p.basename(picked)}');
    await File(picked).copy(dest);
    return dest;
  }

  /// Back to the bundled/system runtime, and drop the imported copy with it —
  /// leaving 27 MB stranded in the sandbox would be its own small bug.
  Future<void> _resetLibrary() async {
    await _applyLibrary('');
    if (!Platform.isAndroid) return;
    final dir = Directory(
        p.join((await getApplicationSupportDirectory()).path, 'onnxruntime'));
    if (dir.existsSync()) dir.deleteSync(recursive: true);
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
                  _OnnxPanel(
                    status: _onnx,
                    savedPath: settings.onnxLibraryPath,
                    locked: running,
                    onPick: _pickLibrary,
                    onReset: _resetLibrary,
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

class _OnnxPanel extends StatelessWidget {
  const _OnnxPanel({
    required this.status,
    required this.savedPath,
    required this.locked,
    required this.onPick,
    required this.onReset,
  });

  final OnnxStatus? status;
  final String savedPath;

  /// A run is in flight; the loader hands out a pointer to the runtime it
  /// loaded, so swapping libraries underneath one is not on offer.
  final bool locked;

  final VoidCallback onPick;
  final VoidCallback onReset;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final s = status;
    final tint = ClpeakTheme.categoryColor(BenchCategory.ai,
        brightness: Theme.of(context).brightness);

    // iOS links ONNX Runtime into the app — Apple's pod is a static framework
    // and iOS will not dlopen another one — so there is nothing to choose.
    final fixed = s?.linkedIn ?? Platform.isIOS;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        CPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              CRow(
                accent: s?.available == true ? tint : null,
                child: Row(
                  children: [
                    Expanded(
                      child: Text(
                        s == null
                            ? 'Checking…'
                            : s.available
                                ? 'ONNX Runtime ${s.version}'
                                : 'No runtime loaded',
                        style: t.mono,
                      ),
                    ),
                    if (s != null)
                      CTag(
                        text: s.available ? 'loaded' : 'absent',
                        color: s.available ? tint : t.dim,
                      ),
                  ],
                ),
              ),
              CRow(
                rule: s?.error.isNotEmpty ?? false,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.folder_outlined, size: 15, color: t.dim),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        fixed
                            ? 'Built into the app'
                            : (s?.path.isNotEmpty ?? false)
                                ? s!.path
                                : savedPath.isNotEmpty
                                    ? savedPath
                                    : 'Found by name on the system paths',
                        style: t.monoSmallDim,
                      ),
                    ),
                  ],
                ),
              ),
              if (s != null && s.error.isNotEmpty)
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
          fixed
              ? 'ONNX Runtime is linked into this build, so there is no other '
                  'library to point at.'
              : 'Which runtime is loaded decides which execution providers '
                  'exist: a stock build offers CPU only, while a vendor build '
                  'brings its NPU. Changing it re-enumerates straight away.',
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
