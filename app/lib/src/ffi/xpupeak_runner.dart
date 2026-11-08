import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'xpupeak_bindings.dart';
import '../model/run_document.dart';
import 'xpupeak_events.dart';

/// One in-flight benchmark run, wherever it executes: in this process
/// ([XpupeakRun]) or in the engine process (`ChildRun`, xpupeak_engine.dart).
abstract interface class EngineRun {
  /// Decoded run events, ending with [DoneEvent].
  Stream<XpupeakEvent> get events;

  /// The xpupeak_launch return code (XPUPEAK_RUN_* / OR'd backend status).
  Future<int> get result;

  /// Why the run ended without finishing, when it did not end by itself: the
  /// engine process crashed or could not start.  Null otherwise.
  String? get failure;

  /// Request cooperative cancellation; the current test finishes first.
  void cancel();

  /// Pause the run at its next boundary between readings, or let it go on
  /// (xpupeak_set_paused).
  void setPaused(bool paused);
}

/// A single in-flight benchmark run in this process.
///
/// Threading contract: the blocking `xpupeak_launch` runs in a short-lived
/// worker isolate; native events fire on that isolate's thread and are
/// marshalled back here through a `NativeCallable.listener`, so [events]
/// delivers on the main isolate in emission order.  The final native `done`
/// event is the drain barrier — the listener is closed only after both the
/// done event has been consumed and the launch call has returned.
class XpupeakRun implements EngineRun {
  XpupeakRun._(this._bindings, List<String> args) {
    _callable = NativeCallable<XpupeakEventCallbackNative>.listener(_onNative);
    _spawn(_callable.nativeFunction.address, args).then((rc) {
      _launchReturned = true;
      _rc.complete(rc);
      _maybeFinish();
    }, onError: (Object e, StackTrace st) {
      _launchReturned = true;
      _rc.completeError(e, st);
      _doneSeen = true; // no more events will come
      _maybeFinish();
    });
  }

  final XpupeakBindings _bindings;
  final _events = StreamController<XpupeakEvent>();
  final _rc = Completer<int>();
  late final NativeCallable<XpupeakEventCallbackNative> _callable;
  bool _doneSeen = false;
  bool _launchReturned = false;
  bool _finished = false;

  @override
  Stream<XpupeakEvent> get events => _events.stream;

  @override
  Future<int> get result => _rc.future;

  /// Always null: a native crash here ends the app along with the run.
  @override
  String? get failure => null;

  @override
  void cancel() => _bindings.requestCancel();

  @override
  void setPaused(bool paused) => _bindings.setPaused(paused);

  void _onNative(Pointer<Void> userData, Pointer<Utf8> eventJson) {
    final json = _bindings.takeString(eventJson);
    if (json == null || _finished) return;
    XpupeakEvent event;
    try {
      event = XpupeakEvent.fromJson(jsonDecode(json) as Map<String, dynamic>);
    } catch (_) {
      event = LogEntryEvent(LogEntry(
          level: LogLevel.warning, message: 'undecodable event: $json'));
    }
    _events.add(event);
    if (event is DoneEvent) {
      _doneSeen = true;
      _maybeFinish();
    }
  }

  void _maybeFinish() {
    if (_finished || !_doneSeen || !_launchReturned) return;
    _finished = true;
    _events.close();
    _callable.close();
  }

  // Static so the Isolate.run closure captures only sendable values (an
  // instance-method closure would drag `this` along and fail to send).
  static Future<int> _spawn(int callbackAddress, List<String> args) {
    final argsCopy = List<String>.of(args);
    return Isolate.run(() => _launchBlocking(callbackAddress, argsCopy));
  }
}

/// Launches benchmark runs over the xpupeak_ffi C ABI.
class XpupeakRunner {
  XpupeakRunner(this._bindings);

  final XpupeakBindings _bindings;

  /// Start a run with CLI-grammar arguments (without the leading program
  /// name — it is prepended here).
  XpupeakRun start(List<String> args) => XpupeakRun._(_bindings, args);
}

/// Runs in the worker isolate: reopens the library, marshals argv, and makes
/// the blocking launch call against the listener's native function pointer.
int _launchBlocking(int callbackAddress, List<String> args) {
  final bindings = XpupeakBindings.open();
  final argv = ['xpupeak', ...args];
  final argvPtrs = calloc<Pointer<Utf8>>(argv.length);
  final allocated = <Pointer<Utf8>>[];
  try {
    for (var i = 0; i < argv.length; i++) {
      final s = argv[i].toNativeUtf8();
      allocated.add(s);
      argvPtrs[i] = s;
    }
    final callback =
        Pointer<NativeFunction<XpupeakEventCallbackNative>>.fromAddress(
            callbackAddress);
    return bindings.launch(argv.length, argvPtrs, callback, nullptr);
  } finally {
    for (final p in allocated) {
      calloc.free(p);
    }
    calloc.free(argvPtrs);
  }
}
