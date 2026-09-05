import 'dart:async';
import 'dart:isolate';
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
import 'screen_wake.dart';

enum BenchmarkState { idle, running, cancelling, finished }

/// Central app state: device catalog, run configuration, the live run, and
/// its finalization into history.  One run at a time.
///
/// The device catalog loads asynchronously: the constructor does no native
/// enumeration (that can take seconds while an NPU toolchain compiles its
/// viability probes), so the first frame never waits for it.  Call [init]
/// once at startup; [catalogReady] turns true when the first load lands and
/// [ready] completes for anyone that must wait (run start, AUTORUN).  Runs
/// never see a partial catalog — [start] refuses while loading.
class BenchmarkService extends ChangeNotifier {
  BenchmarkService(this._bindings, this._history) {
    _catalog = BackendCatalog(const []);
    _config = RunConfig.allDevices(_catalog);
    version = _bindings.version();
  }

  final ClpeakBindings _bindings;
  final RunHistoryStore _history;

  late BackendCatalog _catalog;
  late RunConfig _config;
  late final String version;

  BackendCatalog get catalog => _catalog;
  RunConfig get config => _config;

  bool _loadingCatalog = false;
  bool get isLoadingCatalog => _loadingCatalog;

  /// True once the first catalog load has landed (or failed — see
  /// [catalogError], in which case history viewing still works).
  bool _catalogReady = false;
  bool get catalogReady => _catalogReady;

  String? _catalogError;
  String? get catalogError => _catalogError;

  final Completer<void> _readyCompleter = Completer<void>();
  Future<void> get ready => _readyCompleter.future;

  void _markReady() {
    if (!_readyCompleter.isCompleted) _readyCompleter.complete();
  }

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

  // ── Live-update throttle ─────────────────────────────────────────────────
  //
  // Every rebuild this notifier triggers ends in a presented frame, which is
  // GPU work on the device currently being benchmarked.  Native events arrive
  // in bursts (one per metric, several per test), so during a run they are
  // coalesced onto a fixed low-rate tick instead of notifying per event.
  // Everything outside a run notifies immediately.
  static const _liveTick = Duration(milliseconds: 250);
  Timer? _liveTimer;
  bool _liveDirty = false;

  void _notifyLive() {
    if (_liveTimer == null) {
      notifyListeners(); // not throttled outside a run
      return;
    }
    _liveDirty = true;
  }

  void _startLiveTicker() {
    _liveTimer?.cancel();
    _liveTimer = Timer.periodic(_liveTick, (_) {
      if (!_liveDirty) return;
      _liveDirty = false;
      notifyListeners();
    });
  }

  void _stopLiveTicker() {
    _liveTimer?.cancel();
    _liveTimer = null;
    _liveDirty = false;
  }

  @override
  void dispose() {
    _stopLiveTicker();
    ScreenWake.release();
    super.dispose();
  }

  /// Elapsed time of the in-flight (or just-finished) run.
  Duration get elapsed => _startedAt == null
      ? Duration.zero
      : DateTime.now().difference(_startedAt!);

  void updateConfig(void Function(RunConfig) mutate) {
    mutate(_config);
    notifyListeners();
  }

  /// Which ONNX Runtime the backend has loaded, or why none is.
  OnnxStatus onnxStatus() => _bindings.onnxStatus();

  /// Point the ONNX backend at a library and re-enumerate, so the device
  /// list reflects the providers the new runtime brings.  Empty path = back
  /// to searching the conventional names.
  Future<void> setOnnxLibrary(String path) async {
    if (isRunning) return;
    // A startup load may still be in flight; re-enumerating under it would
    // be dropped by the single-flight guard, leaving a stale catalog.
    // Awaiting a null future is a no-op, so this is free when idle.
    await _catalogFlight;
    if (isRunning) return;
    _bindings.setOnnxLibrary(path);
    await reloadCatalog();
  }

  /// Re-enumerate after something changed what the native side can see —
  /// today only the ONNX Runtime the settings screen chose.
  ///
  /// Selections survive where they still mean something: a device the user
  /// had turned off stays off, one that has gone away is dropped, and a
  /// backend that has just appeared comes in fully selected, which is what
  /// picking a runtime was asking for.
  Future<void> reloadCatalog() => _refreshCatalog();

  /// First load at startup.  Concurrent callers join the in-flight load
  /// instead of starting another enumeration.
  Future<void> init() => _refreshCatalog();

  Future<void>? _catalogFlight;

  Future<void> _refreshCatalog() {
    if (isRunning) return Future.value();
    return _catalogFlight ??= _loadCatalog().whenComplete(() {
      _catalogFlight = null;
    });
  }

  /// Single-flight catalog load shared by [init] and [reloadCatalog]: the
  /// blocking native enumeration runs on a worker isolate while the UI
  /// stays interactive.  Same process, so the native viability memo the
  /// probe fills is shared with later runs for free.
  Future<void> _loadCatalog() async {
    _loadingCatalog = true;
    notifyListeners();
    try {
      // Static closure: captures only sendable values, so it can cross to
      // the worker, which reopens the library there (same pattern as runs).
      final json = await Isolate.run(_fetchCatalogJson);
      _catalog = BackendCatalog.fromJson(json);
      _catalogError = null;

      final fresh = RunConfig.allDevices(_catalog,
          maxTimeMs: _config.maxTimeMs, maxTimeCpuMs: _config.maxTimeCpuMs);
      for (final backend in _catalog.usable) {
        final previous = _config.selectedDevices[backend.name];
        if (previous == null) continue; // newly present: keep it all selected
        final present = fresh.selectedDevices[backend.name] ?? const {};
        fresh.selectedDevices[backend.name] =
            previous.where(present.contains).toSet();
      }
      fresh.categories
        ..clear()
        ..addAll(_config.categories);
      _config = fresh;
    } catch (e) {
      // Native library unloadable or catalog unparsable: history viewing
      // must keep working, so record the failure and carry an empty catalog.
      _catalogError = e.toString();
      _catalog = BackendCatalog(const []);
      _config = RunConfig.allDevices(_catalog,
          maxTimeMs: _config.maxTimeMs, maxTimeCpuMs: _config.maxTimeCpuMs);
    } finally {
      _loadingCatalog = false;
      _catalogReady = true;
      _markReady();
      notifyListeners();
    }
  }

  // Static so the Isolate.run closure captures only sendable values.
  static Map<String, dynamic> _fetchCatalogJson() =>
      ClpeakBindings.open().backendCatalog();

  void applyPreset(RunPreset preset) {
    _config = RunConfig.preset(preset, _catalog);
    notifyListeners();
  }

  Future<void> start({RunPreset? preset}) async {
    if (isRunning) return;
    // Never run against a partial catalog: device indices are positions in
    // the enumerated list, so a run started mid-load would address the
    // wrong devices.  The UI gates Run the same way; this is the backstop.
    if (!catalogReady) return;
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
    _startLiveTicker();
    // Held until _finalize(), which the run's event stream always reaches --
    // it closes on the native `done` event and on a failed launch alike.
    ScreenWake.acquire();

    final resultPath = await _history.filePathFor(_runId!);
    final args = [..._config.toArgs(_catalog), '-o', resultPath];

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
            .runFor(event.backend, event.platform, event.device, event.driver,
                event.deviceIndex)
            .props = event.props;
      case TestBeginEvent(:final header):
        // The test's row is created here, from the header the native side
        // resolved: shape, direction and unit are known before the first
        // reading arrives, so nothing has to be back-filled off the rows.
        _document
            .runFor(event.backend, event.platform, event.device, event.driver,
                event.deviceIndex)
            .openTest(header);
        currentTest = header.title;
      case MetricEvent(:final testKey, :final metric):
        _document
            .runFor(event.backend, event.platform, event.device, event.driver,
                event.deviceIndex)
            .findTest(testKey)
            ?.metrics
            .add(metric);
      case TestSkippedEvent(:final header):
        // One row per named reading, as the file records them -- a whole-test
        // skip used to collapse to a single nameless placeholder.
        _document
            .runFor(event.backend, event.platform, event.device, event.driver,
                event.deviceIndex)
            .openTest(header)
            .metrics
            .addAll(event.toMetrics());
      case TestEndEvent():
        completedTests++;
        currentTest = '';
      case NoteEvent(:final message):
        final trimmed = message.trim();
        if (trimmed.isNotEmpty) {
          notes.add(trimmed);
          _document.notes.add(RunNote(
              backend: event.backend,
              device: event.device,
              message: trimmed));
        }
      case DoneEvent():
        break; // handled via onDone/result
      case DeviceEndEvent():
      case BackendEndEvent():
        break;
    }
    _notifyLive();
  }

  Future<void> _finalize() async {
    _stopLiveTicker(); // back to immediate notifications
    ScreenWake.release();
    final startedAt = _startedAt ?? DateTime.now();
    if (!_document.isEmpty) {
      final summary = RunSummary.fromDocument(
        id: _runId!,
        fileName: RunHistoryStore.fileNameFor(_runId!),
        doc: _document,
        startedAt: startedAt,
        durationMs: DateTime.now().difference(startedAt).inMilliseconds,
        cancelled: cancelled,
      );
      // Index first: listeners (History) re-read the store as soon as
      // lastSummary changes, so the row must already be on disk.
      await _history.add(summary);
      _lastSummary = summary;
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
