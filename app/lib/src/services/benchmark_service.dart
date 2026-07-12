import 'dart:async';
import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../ffi/clpeak_bindings.dart';
import '../ffi/clpeak_events.dart';
import '../ffi/clpeak_runner.dart';
import '../model/catalog.dart';
import '../model/run_config.dart';
import '../model/run_document.dart';
import '../model/run_summary.dart';
import 'run_history_store.dart';

enum BenchmarkState { idle, running, cancelling, finished }

/// Central app state: device catalog, run configuration, the live run, and
/// its finalization into history.  One run at a time.
class BenchmarkService extends ChangeNotifier {
  BenchmarkService(this._bindings, this._history) {
    _catalog = BackendCatalog.fromJson(_bindings.backendCatalog());
    _config = RunConfig.allDevices(_catalog);
    version = _bindings.version();
  }

  final ClpeakBindings _bindings;
  final RunHistoryStore _history;

  late final BackendCatalog _catalog;
  late RunConfig _config;
  late final String version;

  BackendCatalog get catalog => _catalog;
  RunConfig get config => _config;

  // ── Live run state ───────────────────────────────────────────────────────

  BenchmarkState _state = BenchmarkState.idle;
  BenchmarkState get state => _state;
  bool get isRunning =>
      _state == BenchmarkState.running || _state == BenchmarkState.cancelling;

  RunDocument _document = RunDocument();
  RunDocument get document => _document;

  RunSummary? _lastSummary;
  RunSummary? get lastSummary => _lastSummary;

  String currentBackend = '';
  String currentTest = '';
  int completedTests = 0;
  int exitCode = 0;
  bool cancelled = false;
  final List<String> notes = [];

  DateTime? _startedAt;
  DateTime? get startedAt => _startedAt;

  ClpeakRun? _run;
  String? _runId;

  /// Elapsed time of the in-flight (or just-finished) run.
  Duration get elapsed => _startedAt == null
      ? Duration.zero
      : DateTime.now().difference(_startedAt!);

  void updateConfig(void Function(RunConfig) mutate) {
    mutate(_config);
    notifyListeners();
  }

  void applyPreset(RunPreset preset) {
    _config = RunConfig.preset(preset, _catalog);
    notifyListeners();
  }

  Future<void> start({RunPreset? preset}) async {
    if (isRunning) return;
    if (preset != null) _config = RunConfig.preset(preset, _catalog);
    if (!_config.hasSelection || _config.categories.isEmpty) return;

    _document = RunDocument();
    notes.clear();
    currentBackend = '';
    currentTest = '';
    completedTests = 0;
    exitCode = 0;
    cancelled = false;
    _startedAt = DateTime.now();
    _runId = _makeRunId(_startedAt!);
    _state = BenchmarkState.running;
    notifyListeners();

    final xmlPath = await _history.xmlPathFor(_runId!);
    final args = [..._config.toArgs(_catalog), '--xml-file', xmlPath];

    final run = ClpeakRunner(_bindings).start(args);
    _run = run;
    run.events.listen(_onEvent, onDone: () async {
      exitCode = await run.result.catchError((_) => 1);
      cancelled = exitCode == clpeakRunCancelled;
      await _finalize();
    });
  }

  void cancel() {
    if (_state != BenchmarkState.running) return;
    _state = BenchmarkState.cancelling;
    _run?.cancel();
    notifyListeners();
  }

  /// App-exit hook: cancel an in-flight run and wait for the native side to
  /// finish the current test and save partial results before quitting.
  Future<AppExitResponse> onExitRequested() async {
    if (!isRunning) return AppExitResponse.exit;
    cancel();
    await _run?.result.catchError((_) => 1);
    return AppExitResponse.exit;
  }

  /// Back to the dashboard after viewing a finished run.
  void reset() {
    if (isRunning) return;
    _state = BenchmarkState.idle;
    notifyListeners();
  }

  void _onEvent(ClpeakEvent event) {
    switch (event) {
      case BackendBeginEvent(:final backend):
        currentBackend = backend;
        currentTest = '';
      case DeviceEvent():
        _document
            .runFor(event.backend, event.platform, event.device, event.driver)
            .props = event.props;
      case TestBeginEvent(:final display):
        currentTest = display;
      case MetricEvent(:final entry):
        _document.addEntry(entry);
      case TestSkippedEvent():
        // Record one unsupported row so the results view can show it.
        _document.addEntry(event.toEntry());
      case TestEndEvent():
        completedTests++;
        currentTest = '';
      case NoteEvent(:final message):
        final trimmed = message.trim();
        if (trimmed.isNotEmpty) notes.add(trimmed);
      case DoneEvent():
        break; // handled via onDone/result
      case DeviceEndEvent():
      case BackendEndEvent():
        break;
    }
    notifyListeners();
  }

  Future<void> _finalize() async {
    final startedAt = _startedAt ?? DateTime.now();
    if (!_document.isEmpty) {
      final summary = RunSummary.fromDocument(
        id: _runId!,
        fileName: '$_runId.xml',
        doc: _document,
        startedAt: startedAt,
        durationMs: DateTime.now().difference(startedAt).inMilliseconds,
        cancelled: cancelled,
      );
      _lastSummary = summary;
      await _history.add(summary);
    }
    _run = null;
    _state = BenchmarkState.finished;
    notifyListeners();
  }

  static String _makeRunId(DateTime t) {
    String two(int v) => v.toString().padLeft(2, '0');
    return '${t.year}${two(t.month)}${two(t.day)}_'
        '${two(t.hour)}${two(t.minute)}${two(t.second)}';
  }
}
