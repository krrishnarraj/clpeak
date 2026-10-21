import 'dart:async';
import 'dart:io';
import 'dart:ui';

import 'package:flutter/foundation.dart';

import '../ffi/clpeak_bindings.dart';
import '../ffi/clpeak_engine.dart';
import '../ffi/clpeak_events.dart';
import '../ffi/clpeak_runner.dart';
import '../model/catalog.dart';
import '../model/run_config.dart';
import '../model/run_document.dart';
import '../model/run_summary.dart';
import 'process_exit.dart';
import 'resume_offer.dart';
import 'run_history_store.dart';
import 'run_notice.dart';
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
///
/// Enumeration and runs go through [ClpeakEngine]: on desktop a process of
/// their own, in this one when none is given (tests, and mobile).
class BenchmarkService extends ChangeNotifier {
  BenchmarkService(this._bindings, this._history, {ClpeakEngine? engine})
      : _engine = engine ?? InProcessEngine(_bindings) {
    _catalog = BackendCatalog(const []);
    _config = RunConfig.allDevices(_catalog);
    version = _bindings.version();
  }

  final ClpeakBindings _bindings;
  final ClpeakEngine _engine;
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

  /// Runs ended since launch, however they ended: History re-reads its
  /// files at each, since one that died, or a continuation under the id of
  /// the run it continues, publishes no new [lastSummary].
  int _runsEnded = 0;
  int get runsEnded => _runsEnded;

  String currentBackend = '';
  String currentTest = '';

  /// Whether the in-flight (or just-finished) run records verbose
  /// diagnostics (`--verbose`).  The live view hides its DIAGNOSTICS panel
  /// when this is false, so a non-verbose run shows no diagnostics UI.
  bool runVerbose = false;

  /// The run is paused until the device cools: the latest of the native
  /// side's "thermal status ..." lines is the pause's warning, not the info
  /// line that ends it (src/common/thermal.cpp).
  bool coolingDown = false;

  /// The run is paused (Pause in the live view): it holds at its next
  /// boundary between readings until it is resumed (include/common/
  /// screen_hold.h), wherever the app is meanwhile.
  bool paused = false;

  /// The paused run has reached that boundary and waits there; until then
  /// the reading in flight finishes.
  bool held = false;
  int completedTests = 0;
  int exitCode = 0;
  bool cancelled = false;

  /// Why the last run ended early without being cancelled -- the engine
  /// process crashed, say -- or null; see [EngineRun.failure].
  String? runFailure;

  /// Whether that run's diagnostic sidecar survived it, so History lists it
  /// under "Runs that did not finish".
  bool runFailureLogKept = false;

  DateTime? _startedAt;
  DateTime? get startedAt => _startedAt;

  /// The in-flight (or just-finished) run continues one the app died in:
  /// its meta as the checkpoint held it, or null for a run of its own.
  RunMeta? _resumedFrom;
  bool get wasResumed => _resumedFrom != null;

  /// Devices the run's command line selects, for the paused notification's
  /// "device 3 of 9"; 0 when unknown.
  int _plannedDevices = 0;

  /// The run's time as the native side saved it ([DoneEvent.durationS]).
  double? _doneDurationS;

  /// The app is hidden (Android and iOS): set from [onLifecycleState].
  bool _offScreen = false;

  /// The newest run the app died in, when it can be continued and the user
  /// has not dismissed it: what the dashboard offers.  Refreshed after each
  /// catalog load and each run.
  ResumeOffer? _resumeOffer;
  ResumeOffer? get resumeOffer => _resumeOffer;

  EngineRun? _run;
  String? _runId;
  String? _resultPath;

  /// Id of the run in flight (its files in history are named by it), or
  /// null.  History uses it to tell a live run-log sidecar from a crashed
  /// run's.
  String? get inFlightRunId => isRunning ? _runId : null;

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
  OnnxStatus onnxStatus() => _engine.onnxStatus();

  /// Which LiteRT the backend has loaded, or why none is.
  LitertStatus litertStatus() => _engine.litertStatus();

  /// Whether a runtime choice waits for the next launch once a runtime has
  /// loaded ([ClpeakEngine.runtimeFixedOnceLoaded]); on desktop it applies
  /// to the next enumeration and run instead.
  bool get runtimeFixedOnceLoaded => _engine.runtimeFixedOnceLoaded;

  /// Point the LiteRT backend at a library; see [setOnnxLibrary] for when
  /// that re-enumerates and why the startup load is awaited first.
  Future<void> setLitertLibrary(String path) async {
    if (isRunning) return;
    await _catalogFlight;
    if (isRunning) return;
    final fixed = _runtimeFixed(_engine.litertStatus().available);
    _engine.setLitertLibrary(path);
    if (!fixed) await reloadCatalog();
  }

  /// Point the ONNX backend at a library.  In-process the runtime setup is
  /// fixed for the launch by the first runtime that loads
  /// (src/onnx/onnx_runtime.h): while none has, the choice applies now and
  /// the catalog is re-enumerated so the device list shows its providers;
  /// once one has, the native side keeps the choice for the next launch
  /// ([OnnxStatus.pendingRuntime]) and there is nothing to re-enumerate.  An
  /// engine process loads the setup fresh, so there it always applies now.
  /// Empty path = back to searching the conventional names.
  Future<void> setOnnxLibrary(String path) async {
    if (isRunning) return;
    // A startup load may still be in flight; re-enumerating under it would
    // be dropped by the single-flight guard, leaving a stale catalog.
    // Awaiting a null future is a no-op, so this is free when idle.
    await _catalogFlight;
    if (isRunning) return;
    final fixed = _runtimeFixed(_engine.onnxStatus().available);
    _engine.setOnnxLibrary(path);
    if (!fixed) await reloadCatalog();
  }

  /// Replace the ONNX backend's plugin execution-provider libraries and
  /// re-enumerate: unlike the runtime setup they stay live, the enumeration
  /// registering what was added and unregistering what was removed.  Same
  /// single-flight contract as [setOnnxLibrary].
  Future<void> setOnnxEpLibraries(List<OnnxEpLibrary> libs) async {
    if (isRunning) return;
    await _catalogFlight;
    if (isRunning) return;
    _engine.setOnnxEpLibraries(libs);
    await reloadCatalog();
  }

  /// Switch the Windows ML execution-provider catalog: part of the runtime
  /// setup, so the same contract as [setOnnxLibrary].  Enumeration is what
  /// installs and registers the catalog's providers, so the first one after
  /// enabling can take as long as the download does.
  Future<void> setOnnxWinml({required bool enabled, required String path}) async {
    if (isRunning) return;
    await _catalogFlight;
    if (isRunning) return;
    final fixed = _runtimeFixed(_engine.onnxStatus().available);
    _engine.setOnnxWinml(enabled: enabled, path: path);
    if (!fixed) await reloadCatalog();
  }

  /// Whether a runtime choice made now waits for the next launch: in-process,
  /// once the runtime it replaces has loaded.
  bool _runtimeFixed(bool loaded) => _engine.runtimeFixedOnceLoaded && loaded;

  /// Re-enumerate after something changed what the native side can see —
  /// a runtime the settings screen chose before any had loaded, or the
  /// plugin libraries.
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
  /// enumeration runs off the UI thread -- in the engine process, or on a
  /// worker isolate -- while the UI stays interactive.
  Future<void> _loadCatalog() async {
    _loadingCatalog = true;
    notifyListeners();
    try {
      final json = await _engine.backendCatalog();
      _catalog = BackendCatalog.fromJson(json);
      _catalogError = null;

      final fresh = RunConfig.allDevices(_catalog,
          maxTimeMs: _config.maxTimeMs,
          maxTimeMlGpuMs: _config.maxTimeMlGpuMs,
          maxTimeCpuMs: _config.maxTimeCpuMs);
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
          maxTimeMs: _config.maxTimeMs,
          maxTimeMlGpuMs: _config.maxTimeMlGpuMs,
          maxTimeCpuMs: _config.maxTimeCpuMs);
    } finally {
      _loadingCatalog = false;
      _catalogReady = true;
      _markReady();
      notifyListeners();
    }
    // Whether a run the app died in can be continued depends on the
    // devices it finds now.
    unawaited(refreshResumeOffer());
  }

  void applyPreset(RunPreset preset) {
    _config = RunConfig.preset(preset, _catalog);
    notifyListeners();
  }

  /// Launch a run.  [verbose] passes `--verbose`, so the document written
  /// for history carries the debug-level diagnostics as well (see
  /// SettingsService.verbose) — a per-launch argument rather than run
  /// configuration, because it is an app setting read at the moment of
  /// launch, not part of what the run measures.
  Future<void> start({RunPreset? preset, bool verbose = false}) async {
    if (isRunning) return;
    // Never run against a partial catalog: device indices are positions in
    // the enumerated list, so a run started mid-load would address the
    // wrong devices.  The UI gates Run the same way; this is the backstop.
    if (!catalogReady) return;
    if (preset != null) _config = RunConfig.preset(preset, _catalog);
    if (!_config.hasSelection || _config.categories.isEmpty) return;

    // The first run is when to ask to post the paused notification: the
    // one moment the request makes sense, with the app on screen.
    await RunNotice.askOnce();
    if (isRunning) return;

    final id = _makeRunId(DateTime.now());
    // What a continuation must find unchanged to be the same run, noted
    // before anything can end the process.
    await _history.writeResumeState(id,
        catalog: ResumeOffer.fingerprint(_catalog),
        os: Platform.operatingSystemVersion);
    _resumedFrom = null;
    await _launch(
      id: id,
      document: RunDocument(),
      verbose: verbose,
      plannedDevices: _config.selectedDevices.values
          .fold(0, (n, devices) => n + devices.length),
      args: (resultPath) => [
        ..._config.toArgs(_catalog),
        if (verbose) '--verbose',
        '-o',
        resultPath,
      ],
    );
  }

  /// Continue a run the app died in (`--resume`): what its checkpoint holds
  /// shows at once, and the native side runs only the rest, with the run's
  /// own options, into the same document.  Verbosity is the run's own too.
  Future<void> resume(ResumeOffer offer) async {
    if (isRunning || !catalogReady || !offer.canContinue) return;
    final saved = await _history.loadCheckpoint(offer.id);
    if (saved == null || isRunning) return;
    final meta = saved.meta;
    _resumedFrom = meta;
    // A live run has no meta until it is over, a continued one included.
    saved.meta = null;
    await _launch(
      id: offer.id,
      document: saved,
      verbose: meta?.verbose ?? false,
      plannedDevices:
          ResumeOffer.plannedDevices(meta?.argv ?? const [], _catalog),
      args: (resultPath) => [
        '--resume',
        resultPath,
        if (offer.skipCrashed) '--skip-crashed',
      ],
    );
  }

  Future<void> _launch({
    required String id,
    required RunDocument document,
    required bool verbose,
    required int plannedDevices,
    required List<String> Function(String resultPath) args,
  }) async {
    _document = document;
    currentBackend = '';
    currentTest = '';
    runVerbose = verbose;
    coolingDown = false;
    paused = false;
    held = false;
    completedTests = document.testCount;
    exitCode = 0;
    cancelled = false;
    runFailure = null;
    runFailureLogKept = false;
    _plannedDevices = plannedDevices;
    _doneDurationS = null;
    // The results screen exports this run's file, and none until it has one.
    _lastSummary = null;
    _resumeOffer = null;
    _startedAt = DateTime.now();
    _runId = id;
    _state = BenchmarkState.running;
    notifyListeners();
    _startLiveTicker();
    // Held until _finalize(), which the run's event stream always reaches --
    // it closes on the native `done` event and on a failed launch alike.
    ScreenWake.acquire();
    // So that Android's record of this process's death names the run, if
    // the run is what it dies in (ProcessExit).
    ProcessExit.markRun(id);

    final resultPath = await _history.filePathFor(id);
    _resultPath = resultPath;

    final run = _engine.start(args(resultPath));
    _run = run;
    run.events.listen(_onEvent, onDone: () async {
      exitCode = await run.result.catchError((_) => 1);
      cancelled = exitCode == clpeakRunCancelled;
      runFailure = run.failure;
      if (runFailure == null && exitCode == clpeakRunBadArgs) {
        // Refused before anything ran -- a continuation the native side
        // will not resume says why on the log.
        final why = _document.log.lastWhere(
            (e) => e.level == LogLevel.error,
            orElse: () => const LogEntry(
                level: LogLevel.error, message: 'the run could not start'));
        runFailure = why.message.replaceFirst(RegExp(r'^clpeak: '), '');
      }
      if (runFailure != null) {
        // The sidecar the native side derives from -o (.json -> .log).
        final log = '${resultPath.substring(0, resultPath.length - 5)}.log';
        runFailureLogKept = await File(log).exists();
      }
      await _finalize();
    });
  }

  void cancel() {
    if (_state != BenchmarkState.running) return;
    _state = BenchmarkState.cancelling;
    paused = false;
    held = false;
    _run?.cancel();
    RunNotice.clear();
    notifyListeners();
  }

  /// Pause the run, or let a paused one go on.  The reading in flight
  /// finishes first: one cut in two would measure the pause.
  void setPaused(bool pause) {
    if (_state != BenchmarkState.running || paused == pause) return;
    paused = pause;
    held = false;
    _run?.setPaused(pause);
    notifyListeners();
  }

  /// Whether a run the app died in can be continued, as things are now;
  /// History asks it for every such run, the dashboard -- with [now] and
  /// [newerRunStarted], its narrower terms -- for the newest.
  ResumeOffer decideResume(CrashLog log,
          {DateTime? now, DateTime? newerRunStarted}) =>
      ResumeOffer.decide(
        log,
        appVersion: version,
        osVersion: Platform.operatingSystemVersion,
        catalog: _catalogFingerprint,
        now: now,
        newerRunStarted: newerRunStarted,
      );

  /// Whether a cancelled run can be continued, as things are now.
  ResumeOffer decideCancelled(RunSummary run,
          {required RunCheckpointInfo? checkpoint,
          required RunResumeState? state}) =>
      ResumeOffer.decideCancelled(
        run,
        checkpoint: checkpoint,
        state: state,
        appVersion: version,
        osVersion: Platform.operatingSystemVersion,
        catalog: _catalogFingerprint,
      );

  String? get _catalogFingerprint => catalogReady && _catalogError == null
      ? ResumeOffer.fingerprint(_catalog)
      : null;

  /// Find the newest run the app died in and whether to offer it.
  Future<void> refreshResumeOffer() async {
    if (isRunning) return;
    ResumeOffer? offer;
    try {
      final logs = await _history.listCrashLogs(inFlightId: inFlightRunId);
      final newest = logs.isEmpty ? null : logs.first;
      if (newest != null && !(newest.resumeState?.dismissed ?? false)) {
        final decided = decideResume(newest,
            now: DateTime.now(),
            newerRunStarted:
                ResumeOffer.newerRunThan(newest, await _history.list()));
        if (decided.canContinue) offer = decided;
      }
    } catch (_) {
      // An offer, never a reason for the dashboard to fail.
    }
    if (isRunning) return;
    _resumeOffer = offer;
    notifyListeners();
  }

  /// The dashboard's offer turned down: History keeps the run, with its
  /// Continue while the run still allows one.
  Future<void> dismissResumeOffer() async {
    final offer = _resumeOffer;
    if (offer == null) return;
    _resumeOffer = null;
    notifyListeners();
    await _history.dismissResume(offer.id);
  }

  /// App-exit hook: cancel an in-flight run and wait for the native side to
  /// finish the current test and save partial results before quitting.
  Future<AppExitResponse> onExitRequested() async {
    if (!isRunning) return AppExitResponse.exit;
    cancel();
    await _run?.result.catchError((_) => 1);
    return AppExitResponse.exit;
  }

  /// App-lifecycle hook: the engine is told whether the app is on screen,
  /// and a run in this process waits at its next boundary between readings
  /// while it is not.  On a phone an app off screen runs on background cores
  /// at background clocks until it is frozen: a Galaxy Z Fold8 froze clpeak
  /// about 45 and 70 s after it left, once inside a test, and its CPU
  /// backend, which started as the app came back, ran on four of eight
  /// cores.  The run's log records each change.  Desktop runs live in the
  /// engine process, which nothing pauses.
  void onLifecycleState(AppLifecycleState state) {
    if (!Platform.isAndroid && !Platform.isIOS) return;
    if (state == AppLifecycleState.hidden) {
      _engine.setOnScreen(false);
      _offScreen = true;
      // Posted now, while the process still runs: the phone freezes it
      // soon after (RunNotice).
      if (_state == BenchmarkState.running) {
        RunNotice.show(title: 'Run paused', body: _pausedText());
      }
    } else if (state == AppLifecycleState.resumed) {
      _engine.setOnScreen(true);
      _offScreen = false;
      RunNotice.clear();
    }
  }

  /// "CPU · device 3 of 9 · 41 tests done. Tap to continue."  No test total:
  /// a run decides which tests a device runs as it goes.  A run the person
  /// paused stays paused once the app is back.
  String _pausedText() {
    final devices = _document.runs.length;
    final parts = [
      if (currentBackend.isNotEmpty) currentBackend,
      if (devices > 0 && _plannedDevices >= devices)
        'device $devices of $_plannedDevices',
      '$completedTests ${completedTests == 1 ? 'test' : 'tests'} done',
    ];
    final tap = paused ? 'Tap to go back to it.' : 'Tap to continue.';
    return '${parts.join(' · ')}. $tap';
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
      case LogEntryEvent(:final entry):
        // The same entry the native side records on the file's `log`, so
        // the live view and the reopened run show one stream.
        _document.log.add(entry);
        if (entry.source.isEmpty &&
            entry.message.startsWith('thermal status ')) {
          coolingDown = entry.level == LogLevel.warning;
        }
        // The paused run's own line once it waits (src/common/
        // screen_hold.cpp).
        if (paused &&
            entry.source.isEmpty &&
            entry.message == 'held until the run is resumed') {
          held = true;
        }
      case DoneEvent(:final durationS):
        _doneDurationS = durationS; // the rest via onDone/result
      case DeviceEndEvent():
      case BackendEndEvent():
        break;
    }
    _notifyLive();
  }

  Future<void> _finalize() async {
    _stopLiveTicker(); // back to immediate notifications
    ScreenWake.release();
    paused = false;
    held = false;
    ProcessExit.markRun(null);
    final startedAt = _startedAt ?? DateTime.now();
    // Indexed only once written: a run whose engine died has no document,
    // just the sidecar History lists on its own.  A cancelled run continued
    // has its first part's result on disk already, so the file has to be
    // this launch's (with slack for a filesystem's coarse timestamps).
    final result = _resultPath == null ? null : File(_resultPath!);
    final saved = runFailure == null &&
        result != null &&
        await result.exists() &&
        !(await result.lastModified())
            .isBefore(startedAt.subtract(const Duration(seconds: 2)));
    // A run whose last test was in flight as the app left ends off screen:
    // where its paused notice was, say so.
    if (_offScreen && saved) {
      RunNotice.show(
          title: 'Run finished',
          body: '$completedTests tests. Tap to see the results.');
    } else {
      RunNotice.clear();
    }
    if (!_document.isEmpty && saved) {
      // A continued run is the run it continues: listed from when that
      // started, for the time all its parts ran.
      final prior = _resumedFrom;
      final firstStart = prior == null
          ? startedAt
          : (DateTime.tryParse(prior.generatedAt)?.toLocal() ?? startedAt);
      final priorMs =
          prior == null ? 0 : (prior.durationSeconds * 1000).round();
      final done = _doneDurationS;
      final summary = RunSummary.fromDocument(
        id: _runId!,
        fileName: RunHistoryStore.fileNameFor(_runId!),
        doc: _document,
        startedAt: firstStart,
        // What the document says; the checkpoint's own time stops at its
        // last write, short of where the part before ended.
        durationMs: done != null
            ? (done * 1000).round()
            : priorMs + DateTime.now().difference(startedAt).inMilliseconds,
        cancelled: cancelled,
      );
      // Index first: History re-reads the store as soon as the run is
      // over, so the row must already be on disk.
      await _history.add(summary);
      _lastSummary = summary;
    }
    // A cancelled run keeps its note with its checkpoint, to be continued.
    if (saved && !cancelled && _runId != null) {
      await _history.clearResumeState(_runId!);
    }
    _run = null;
    _state = BenchmarkState.finished;
    _runsEnded++;
    notifyListeners();
    unawaited(refreshResumeOffer());
  }

  static String _makeRunId(DateTime t) {
    String two(int v) => v.toString().padLeft(2, '0');
    return '${t.year}${two(t.month)}${two(t.day)}_'
        '${two(t.hour)}${two(t.minute)}${two(t.second)}';
  }
}
