import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'clpeak_bindings.dart';
import 'clpeak_events.dart';

/// A single in-flight benchmark run.
///
/// Threading contract: the blocking `clpeak_launch` runs in a short-lived
/// worker isolate; native events fire on that isolate's thread and are
/// marshalled back here through a `NativeCallable.listener`, so [events]
/// delivers on the main isolate in emission order.  The final native `done`
/// event is the drain barrier — the listener is closed only after both the
/// done event has been consumed and the launch call has returned.
class ClpeakRun {
  ClpeakRun._(this._bindings, List<String> args) {
    _callable = NativeCallable<ClpeakEventCallbackNative>.listener(_onNative);
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

  final ClpeakBindings _bindings;
  final _events = StreamController<ClpeakEvent>();
  final _rc = Completer<int>();
  late final NativeCallable<ClpeakEventCallbackNative> _callable;
  bool _doneSeen = false;
  bool _launchReturned = false;
  bool _finished = false;

  /// Decoded run events, ending with [DoneEvent].
  Stream<ClpeakEvent> get events => _events.stream;

  /// The clpeak_launch return code (CLPEAK_RUN_* / OR'd backend status).
  Future<int> get result => _rc.future;

  /// Request cooperative cancellation; the current test finishes first.
  void cancel() => _bindings.requestCancel();

  void _onNative(Pointer<Void> userData, Pointer<Utf8> eventJson) {
    final json = _bindings.takeString(eventJson);
    if (json == null || _finished) return;
    ClpeakEvent event;
    try {
      event = ClpeakEvent.fromJson(jsonDecode(json) as Map<String, dynamic>);
    } catch (_) {
      event = NoteEvent(json);
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

/// Launches benchmark runs over the clpeak_ffi C ABI.
class ClpeakRunner {
  ClpeakRunner(this._bindings);

  final ClpeakBindings _bindings;

  /// Start a run with CLI-grammar arguments (without the leading program
  /// name — it is prepended here).
  ClpeakRun start(List<String> args) => ClpeakRun._(_bindings, args);
}

/// Runs in the worker isolate: reopens the library, marshals argv, and makes
/// the blocking launch call against the listener's native function pointer.
int _launchBlocking(int callbackAddress, List<String> args) {
  final bindings = ClpeakBindings.open();
  final argv = ['clpeak', ...args];
  final argvPtrs = calloc<Pointer<Utf8>>(argv.length);
  final allocated = <Pointer<Utf8>>[];
  try {
    for (var i = 0; i < argv.length; i++) {
      final s = argv[i].toNativeUtf8();
      allocated.add(s);
      argvPtrs[i] = s;
    }
    final callback =
        Pointer<NativeFunction<ClpeakEventCallbackNative>>.fromAddress(
            callbackAddress);
    return bindings.launch(argv.length, argvPtrs, callback, nullptr);
  } finally {
    for (final p in allocated) {
      calloc.free(p);
    }
    calloc.free(argvPtrs);
  }
}
