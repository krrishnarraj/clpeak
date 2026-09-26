import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import '../model/run_document.dart';
import 'clpeak_bindings.dart';
import 'clpeak_events.dart';
import 'clpeak_library.dart';
import 'clpeak_runner.dart';

/// [EngineRun.result] when the engine process ended without finishing the
/// run: it crashed, or never started.  App-side only -- the CLPEAK_RUN_*
/// codes clpeak_launch returns are in src/ffi/clpeak_ffi.h.
const int clpeakRunEngineFailed = -4;

/// Where the device catalog is enumerated and runs execute.
///
/// On desktop, in `clpeak-engine`: a process started for each catalog and
/// each run (clpeak_engine_main, src/ffi/clpeak_ffi.h).  This process holds
/// the GUI toolkit and a GL driver, and a vendor runtime loaded beside them
/// can break on what they bring: HIP did on a Radeon 890M, whose GL driver
/// had loaded a second LLVM (src/ffi/engine.cpp).  The engine process also
/// loads the runtime setup afresh every time, and a native crash there ends
/// the run, not the app.  On Android and iOS -- and on a desktop build
/// without the engine host -- everything runs in this process over FFI.
abstract interface class ClpeakEngine {
  /// The engine process on desktop when its host and the library are both
  /// found, this process otherwise.
  static ClpeakEngine forPlatform(ClpeakBindings bindings) {
    final desktop = Platform.isMacOS || Platform.isLinux || Platform.isWindows;
    if (desktop) {
      final host = clpeakEngineHostPath();
      final library = clpeakLibraryPath();
      if (host != null && library != null) {
        return ChildProcessEngine(host: host, library: library);
      }
    }
    return InProcessEngine(bindings);
  }

  /// Whether the runtime setup (the ONNX Runtime library and Windows ML, the
  /// LiteRT library) is fixed for this app launch once a runtime has loaded
  /// -- true in-process (src/onnx/onnx_runtime.h), where a later choice waits
  /// for the next launch.  An engine process loads it fresh every time.
  bool get runtimeFixedOnceLoaded;

  /// Enumerate every backend: the inventoryToJson() document.  The statuses
  /// below describe the runtimes this enumeration loaded.
  Future<Map<String, dynamic>> backendCatalog();

  OnnxStatus onnxStatus();
  LitertStatus litertStatus();

  // The runtime setup, as ClpeakBindings documents each; between runs only.
  void setOnnxLibrary(String path);
  void setOnnxEpLibraries(List<OnnxEpLibrary> libs);
  void setOnnxWinml({required bool enabled, required String path});
  void setLitertLibrary(String path);
  void setLitertNpuStageDir(String dir);

  /// Start a run with CLI-grammar arguments, without a program name.
  EngineRun start(List<String> args);
}

/// Everything in this process, over the clpeak_ffi C ABI.
class InProcessEngine implements ClpeakEngine {
  InProcessEngine(this._bindings);

  final ClpeakBindings _bindings;

  @override
  bool get runtimeFixedOnceLoaded => true;

  /// The blocking enumeration runs on a worker isolate while the UI stays
  /// interactive.  Same process, so the native viability memo the probes
  /// fill is shared with later runs for free.
  @override
  Future<Map<String, dynamic>> backendCatalog() => Isolate.run(_fetchCatalog);

  // Static so the Isolate.run closure captures only sendable values; the
  // worker reopens the library there (the same pattern as runs).
  static Map<String, dynamic> _fetchCatalog() =>
      ClpeakBindings.open().backendCatalog();

  @override
  OnnxStatus onnxStatus() => _bindings.onnxStatus();

  @override
  LitertStatus litertStatus() => _bindings.litertStatus();

  @override
  void setOnnxLibrary(String path) => _bindings.setOnnxLibrary(path);

  @override
  void setOnnxEpLibraries(List<OnnxEpLibrary> libs) =>
      _bindings.setOnnxEpLibraries(libs);

  @override
  void setOnnxWinml({required bool enabled, required String path}) =>
      _bindings.setOnnxWinml(enabled: enabled, path: path);

  @override
  void setLitertLibrary(String path) => _bindings.setLitertLibrary(path);

  @override
  void setLitertNpuStageDir(String dir) => _bindings.setLitertNpuStageDir(dir);

  @override
  EngineRun start(List<String> args) => ClpeakRunner(_bindings).start(args);
}

/// Everything in `clpeak-engine` processes.  The runtime setup is kept here
/// and handed to each process as it starts, through the same clpeak_set_*
/// calls the in-process engine makes -- not as run flags, which the saved
/// document would record.
class ChildProcessEngine implements ClpeakEngine {
  ChildProcessEngine({required this.host, required this.library});

  /// The engine host executable, and the clpeak_ffi library it loads.
  final String host;
  final String library;

  String _onnxLibrary = '';
  List<OnnxEpLibrary> _onnxEpLibraries = const [];
  bool _onnxWinml = false;
  String _onnxWinmlPath = '';
  String _litertLibrary = '';

  // What the last catalog's process loaded; nothing until the first lands.
  OnnxStatus _onnx = const OnnxStatus.unavailable('');
  LitertStatus _litert = const LitertStatus.unavailable('');

  @override
  bool get runtimeFixedOnceLoaded => false;

  List<String> _arguments(String mode, [List<String> run = const []]) => [
        library,
        mode,
        if (_onnxLibrary.isNotEmpty) ...['--set-onnx-library', _onnxLibrary],
        for (final l in _onnxEpLibraries)
          ...['--set-onnx-ep', '${l.implicit ? '!' : ''}${l.name}=${l.path}'],
        if (_onnxWinml) ...['--set-onnx-winml', _onnxWinmlPath],
        if (_litertLibrary.isNotEmpty) ...['--set-litert-library', _litertLibrary],
        if (run.isNotEmpty) ...['--', ...run],
      ];

  @override
  Future<Map<String, dynamic>> backendCatalog() async {
    final process = await Process.start(host, _arguments('catalog'));
    unawaited(process.stdin.close().catchError((Object _) {}));
    final err = _EngineStderr(process.stderr);
    final out = await process.stdout
        .transform(const Utf8Decoder(allowMalformed: true))
        .join();
    final code = await process.exitCode;
    for (final line in const LineSplitter().convert(out).reversed) {
      Object? doc;
      try {
        doc = jsonDecode(line);
      } on FormatException {
        continue;
      }
      if (doc is Map<String, dynamic> && doc['t'] == 'catalog') {
        _onnx = OnnxStatus.fromJson(
            doc['onnx'] as Map<String, dynamic>? ?? const {});
        _litert = LitertStatus.fromJson(
            doc['litert'] as Map<String, dynamic>? ?? const {});
        return doc['catalog'] as Map<String, dynamic>? ??
            const {'backends': []};
      }
    }
    throw EngineFailure(
        'the device scan ${describeEngineExit(code)}', err.lastLine);
  }

  @override
  OnnxStatus onnxStatus() => _onnx;

  @override
  LitertStatus litertStatus() => _litert;

  @override
  void setOnnxLibrary(String path) => _onnxLibrary = path;

  @override
  void setOnnxEpLibraries(List<OnnxEpLibrary> libs) =>
      _onnxEpLibraries = List.unmodifiable(libs);

  @override
  void setOnnxWinml({required bool enabled, required String path}) {
    _onnxWinml = enabled;
    _onnxWinmlPath = path;
  }

  @override
  void setLitertLibrary(String path) => _litertLibrary = path;

  /// Android only; an engine process never runs there.
  @override
  void setLitertNpuStageDir(String dir) {}

  @override
  EngineRun start(List<String> args) =>
      ChildRun(host, _arguments('launch', args));
}

/// A run in an engine process: its events arrive one JSON document per line
/// of stdout, a cancel is a line on its stdin.  The same drain barrier as in
/// process: [events] closes once `done` has been read and the process is
/// gone.  A process that exits without `done` crashed, and the run ends with
/// [failure] and a synthesized `done`.
class ChildRun implements EngineRun {
  ChildRun(String host, List<String> arguments) {
    unawaited(_run(host, arguments).catchError((Object e) {
      // Nothing of the above should throw; if it does, the run still ends.
      if (!_rc.isCompleted) {
        if (!_events.isClosed) {
          _fail('clpeak-engine: $e', 'The benchmark engine stopped unexpectedly');
        }
        _rc.complete(clpeakRunEngineFailed);
      }
      if (!_events.isClosed) unawaited(_events.close());
    }));
  }

  final _events = StreamController<ClpeakEvent>();
  final _rc = Completer<int>();
  Process? _process;
  bool _cancelRequested = false;
  DoneEvent? _done;
  String? _failure;

  @override
  Stream<ClpeakEvent> get events => _events.stream;

  @override
  Future<int> get result => _rc.future;

  @override
  String? get failure => _failure;

  @override
  void cancel() {
    _cancelRequested = true;
    _send('cancel'); // before the process is up, it is sent once it is
  }

  void _send(String line) {
    final process = _process;
    if (process == null) return;
    try {
      process.stdin.writeln(line);
    } catch (_) {
      // The engine has already gone; its exit says how.
    }
  }

  Future<void> _run(String host, List<String> arguments) async {
    final Process process;
    try {
      process = await Process.start(host, arguments);
    } catch (e) {
      _fail('clpeak-engine did not start: $e',
          'The benchmark engine did not start');
      _rc.complete(clpeakRunEngineFailed);
      await _events.close();
      return;
    }
    _process = process;
    // Writes to an engine that has exited fail; its exit code says why.
    unawaited(process.stdin.done.catchError((Object _) {}));
    if (_cancelRequested) _send('cancel');
    final err = _EngineStderr(process.stderr);
    // Its end is taken now: stdout usually closes before the exit code is
    // in, and a listener added after that would wait for nothing.
    final drained = Completer<void>();
    final lines = process.stdout
        .transform(const Utf8Decoder(allowMalformed: true))
        .transform(const LineSplitter())
        .listen(_onLine, onDone: drained.complete, onError: (Object _) {});
    final code = await process.exitCode;
    // The events pipe closes with the process, which keeps it from any
    // process it starts itself; the grace only guards against one that
    // did not.
    await drained.future
        .timeout(const Duration(seconds: 5), onTimeout: () {});
    await lines.cancel();
    // Closing stdin once the run is over; before, it would be a cancel.
    try {
      await process.stdin.close();
    } catch (_) {}
    if (_done == null) {
      final how = describeEngineExit(code);
      final detail = err.lastLine;
      _fail('clpeak-engine $how${detail.isEmpty ? '' : ': $detail'}',
          'The benchmark engine $how before the run finished');
    }
    _rc.complete(_done?.status ?? clpeakRunEngineFailed);
    await _events.close();
  }

  void _onLine(String line) {
    // Nothing after `done` belongs to the run: a late relayed log line.
    if (line.isEmpty || _done != null) return;
    ClpeakEvent event;
    try {
      event = ClpeakEvent.fromJson(jsonDecode(line) as Map<String, dynamic>);
    } catch (_) {
      event = LogEntryEvent(LogEntry(
          level: LogLevel.warning, message: 'undecodable event: $line'));
    }
    _events.add(event);
    if (event is DoneEvent) _done = event;
  }

  /// [log] goes on the run's diagnostic stream, [failure] is the sentence
  /// the results screen shows.
  void _fail(String log, String failure) {
    _failure = failure;
    _events.add(LogEntryEvent(LogEntry(level: LogLevel.error, message: log)));
    _events.add(
        const DoneEvent(status: clpeakRunEngineFailed, cancelled: false));
  }
}

/// A catalog the engine process could not produce.
class EngineFailure implements Exception {
  EngineFailure(this.what, [this.detail = '']);

  final String what; // "the device scan crashed (signal 11, SIGSEGV)"
  final String detail; // the engine's last line of stderr, if any

  @override
  String toString() {
    final head = '${what[0].toUpperCase()}${what.substring(1)}';
    return detail.isEmpty ? head : '$head: $detail';
  }
}

/// How an engine process that did not finish ended, as "crashed (signal 11,
/// SIGSEGV)" or "exited with code 69".
String describeEngineExit(int code) {
  if (Platform.isWindows) {
    // An NTSTATUS error (0xC…) is an unhandled exception, Windows' crash.
    final status = code & 0xFFFFFFFF;
    if (status >= 0xC0000000) {
      return 'crashed (exception 0x${status.toRadixString(16)})';
    }
  } else if (code < 0) {
    final signal = -code;
    final name = {
      4: 'SIGILL',
      6: 'SIGABRT',
      8: 'SIGFPE',
      9: 'SIGKILL',
      11: 'SIGSEGV',
      15: 'SIGTERM',
      if (Platform.isMacOS) 10: 'SIGBUS' else 7: 'SIGBUS',
    }[signal];
    return 'crashed (signal $signal${name == null ? '' : ', $name'})';
  }
  return 'exited with code $code';
}

/// The engine's stderr, forwarded to this process's: where it went when the
/// engine ran here, so a runtime's own logging (AMD_LOG_LEVEL, say) still
/// reaches the terminal the app was started from.  The tail is kept for the
/// message when the engine dies.
class _EngineStderr {
  _EngineStderr(Stream<List<int>> stream) {
    stream.listen((chunk) {
      if (_forward) {
        try {
          stderr.add(chunk);
        } catch (_) {
          _forward = false; // no stderr here (a Windows GUI launch)
        }
      }
      _tail.addAll(chunk);
      if (_tail.length > _keep) _tail.removeRange(0, _tail.length - _keep);
    }, onError: (Object _) {});
  }

  static const _keep = 4096;
  final _tail = <int>[];
  bool _forward = true;

  /// The last non-empty line written so far, capped for a one-line message.
  String get lastLine {
    final lines = utf8
        .decode(_tail, allowMalformed: true)
        .split('\n')
        .map((l) => l.trim())
        .where((l) => l.isNotEmpty);
    if (lines.isEmpty) return '';
    final last = lines.last;
    return last.length > 300 ? '${last.substring(0, 300)}…' : last;
  }
}
