import 'dart:convert';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// How Android says the app's process ended, for a run it ended.
///
/// On a phone a run executes in the app's own process, so a native crash in
/// a driver or the low-memory killer takes both down, and the run-log
/// sidecar stops at the last line before -- which cannot say which of the
/// two it was.  Android keeps a record of each of the app's process deaths
/// (ApplicationExitInfo): why, how much memory the process held, and for a
/// native crash the tombstone.  While a run is in flight its id is stamped
/// on the process, so the record names the run it ended, and
/// RunHistoryStore appends it to that run's sidecar on the next launch.
///
/// Android only, and best effort: the record is a diagnosis, never a reason
/// for a run or a listing to fail.  The platform half is
/// `android/app/src/main/kotlin/kr/clpeak/MainActivity.kt`.
class ProcessExit {
  const ProcessExit._();

  static const _channel = MethodChannel('kr.clpeak/process');

  /// Stamp [runId] on this process, or clear the stamp with null.
  static Future<void> markRun(String? runId) async {
    if (!Platform.isAndroid) return;
    try {
      await _channel.invokeMethod<void>('markRun', runId);
    } catch (e) {
      debugPrint('clpeak: process state summary: $e');
    }
  }

  /// The app's recent process deaths that had a run in flight.
  static Future<List<ProcessExitRecord>> records() async {
    if (!Platform.isAndroid) return const [];
    try {
      final raw = await _channel.invokeListMethod<Map>('exits') ?? const [];
      return [for (final m in raw) ProcessExitRecord.fromMap(m)];
    } catch (e) {
      debugPrint('clpeak: process exit records: $e');
      return const [];
    }
  }
}

/// One ApplicationExitInfo, as the platform channel hands it over.
class ProcessExitRecord {
  const ProcessExitRecord({
    required this.runId,
    required this.at,
    required this.reason,
    this.status = 0,
    this.description = '',
    this.importance = 0,
    this.pssKb = 0,
    this.rssKb = 0,
    this.trace,
  });

  factory ProcessExitRecord.fromMap(Map m) => ProcessExitRecord(
        runId: m['run'] as String? ?? '',
        at: DateTime.fromMillisecondsSinceEpoch(
            (m['timestamp_ms'] as num?)?.toInt() ?? 0),
        reason: m['reason'] as String? ?? 'UNKNOWN',
        status: (m['status'] as num?)?.toInt() ?? 0,
        description: m['description'] as String? ?? '',
        importance: (m['importance'] as num?)?.toInt() ?? 0,
        pssKb: (m['pss_kb'] as num?)?.toInt() ?? 0,
        rssKb: (m['rss_kb'] as num?)?.toInt() ?? 0,
        trace: m['trace'] as Uint8List?,
      );

  final String runId;
  final DateTime at;

  /// ApplicationExitInfo.REASON_* without the prefix: LOW_MEMORY,
  /// SIGNALED, CRASH_NATIVE, ANR, ...
  final String reason;

  /// The signal for SIGNALED and CRASH_NATIVE, the exit code for EXIT_SELF.
  final int status;
  final String description;

  /// ActivityManager.RunningAppProcessInfo.IMPORTANCE_*: whether the app
  /// was on screen when it went.
  final int importance;
  final int pssKb;
  final int rssKb;

  /// CRASH_NATIVE: the tombstone (protobuf).  ANR: the thread dump (text).
  final Uint8List? trace;

  /// The sidecar entry's text: one line saying how, then the trace.
  String get message {
    final facts = <String>[
      if (reason == 'SIGNALED' || reason == 'CRASH_NATIVE')
        'signal $status'
      else if (reason == 'EXIT_SELF')
        'exit code $status',
      _importanceName(importance),
      if (pssKb > 0) 'pss ${_mb(pssKb)}',
      if (rssKb > 0) 'rss ${_mb(rssKb)}',
    ];
    final head = StringBuffer('process ended: $reason')
      ..write(facts.isEmpty ? '' : ' (${facts.join(', ')})')
      ..write(description.isEmpty ? '' : ': $description');
    final t = trace;
    if (t == null || t.isEmpty) return head.toString();
    final body = reason == 'CRASH_NATIVE'
        ? tombstoneText(t)
        : _head(utf8.decode(t, allowMalformed: true), 32 << 10);
    return '$head\n$body';
  }

  static String _mb(int kb) => '${(kb / 1024).toStringAsFixed(0)} MB';

  static String _importanceName(int i) => switch (i) {
        100 => 'foreground',
        125 => 'foreground service',
        200 => 'visible',
        230 => 'perceptible',
        300 => 'service',
        325 => 'top, screen off',
        350 => 'heavy weight',
        400 => 'cached',
        1000 => 'gone',
        _ => 'importance $i',
      };

  // An ANR dump runs to hundreds of KB; the threads it is read for come
  // first.
  static String _head(String s, int max) =>
      s.length <= max
          ? s
          : '${s.substring(0, max)}\n... (${s.length - max} more characters)';
}

/// A native crash's tombstone, as debuggerd prints the parts a report needs:
/// the signal, the abort message and causes, the crashing thread's
/// backtrace, and the last lines the process logged.  The input is the
/// protobuf of AOSP `system/core/debuggerd/proto/tombstone.proto`, which is
/// what ApplicationExitInfo hands back for REASON_CRASH_NATIVE.
String tombstoneText(Uint8List bytes, {int logLines = 50}) {
  try {
    return _Tombstone.parse(bytes).text(logLines);
  } catch (e) {
    return 'tombstone of ${bytes.length} bytes could not be read: $e';
  }
}

class _Tombstone {
  int tid = 0;
  _Signal? signal;
  String abortMessage = '';
  final causes = <String>[];
  final threads = <int, _Thread>{};
  final logs = <String>[];

  static _Tombstone parse(Uint8List b) {
    final t = _Tombstone();
    final r = _Pb(b, 0, b.length);
    while (r.more) {
      final (field, wire) = r.tag();
      switch (field) {
        case 6 when wire == 0:
          t.tid = r.varint();
        case 10 when wire == 2:
          t.signal = _Signal.parse(r.sub());
        case 14 when wire == 2:
          t.abortMessage = r.string();
        case 15 when wire == 2:
          final c = r.sub();
          while (c.more) {
            final (f, w) = c.tag();
            if (f == 1 && w == 2) {
              t.causes.add(c.string());
            } else {
              c.skip(w);
            }
          }
        case 16 when wire == 2:
          // map<uint32, Thread>: an entry message of key = 1, value = 2.
          final e = r.sub();
          var key = 0;
          _Thread? thread;
          while (e.more) {
            final (f, w) = e.tag();
            if (f == 1 && w == 0) {
              key = e.varint();
            } else if (f == 2 && w == 2) {
              thread = _Thread.parse(e.sub());
            } else {
              e.skip(w);
            }
          }
          if (thread != null) t.threads[key] = thread;
        case 18 when wire == 2:
          t.logs.addAll(_logBuffer(r.sub()));
        default:
          r.skip(wire);
      }
    }
    return t;
  }

  String text(int logLines) {
    final out = StringBuffer();
    final s = signal;
    if (s != null) {
      out.write(
          'signal ${s.number} (${s.name}), code ${s.code} (${s.codeName})');
      if (s.hasFaultAddress) {
        out.write(', fault addr 0x${_hex(s.faultAddress, 16)}');
      }
      out.writeln();
    }
    if (abortMessage.isNotEmpty) out.writeln("Abort message: '$abortMessage'");
    for (final c in causes) {
      out.writeln('Cause: $c');
    }
    final thread = threads[tid];
    if (thread != null) {
      out.writeln('backtrace (thread $tid "${thread.name}"):');
      for (final note in thread.notes) {
        out.writeln('  NOTE: $note');
      }
      for (var i = 0; i < thread.frames.length; i++) {
        out.writeln('  #${i.toString().padLeft(2, '0')} ${thread.frames[i]}');
      }
    }
    if (logs.isNotEmpty) {
      final from = logs.length > logLines ? logs.length - logLines : 0;
      out.writeln('log (last ${logs.length - from} lines):');
      for (final l in logs.skip(from)) {
        out.writeln('  $l');
      }
    }
    return out.toString().trimRight();
  }

  static List<String> _logBuffer(_Pb r) {
    final lines = <String>[];
    while (r.more) {
      final (f, w) = r.tag();
      if (f != 2 || w != 2) {
        r.skip(w);
        continue;
      }
      final m = r.sub();
      var timestamp = '', tag = '', message = '';
      var pid = 0, tid = 0, priority = 0;
      while (m.more) {
        final (mf, mw) = m.tag();
        switch (mf) {
          case 1 when mw == 2:
            timestamp = m.string();
          case 2 when mw == 0:
            pid = m.varint();
          case 3 when mw == 0:
            tid = m.varint();
          case 4 when mw == 0:
            priority = m.varint();
          case 5 when mw == 2:
            tag = m.string();
          case 6 when mw == 2:
            message = m.string();
          default:
            m.skip(mw);
        }
      }
      const levels = 'VVVDIWEF';   // android_LogPriority 2..7
      final level = priority >= 2 && priority <= 7 ? levels[priority] : '?';
      lines.add('$timestamp $pid $tid $level $tag: ${message.trimRight()}');
    }
    return lines;
  }
}

class _Signal {
  int number = 0;
  String name = '';
  int code = 0;
  String codeName = '';
  bool hasFaultAddress = false;
  int faultAddress = 0;

  static _Signal parse(_Pb r) {
    final s = _Signal();
    while (r.more) {
      final (f, w) = r.tag();
      switch (f) {
        case 1 when w == 0:
          s.number = r.varint();
        case 2 when w == 2:
          s.name = r.string();
        case 3 when w == 0:
          s.code = r.varint();
        case 4 when w == 2:
          s.codeName = r.string();
        case 8 when w == 0:
          s.hasFaultAddress = r.varint() != 0;
        case 9 when w == 0:
          s.faultAddress = r.varint();
        default:
          r.skip(w);
      }
    }
    return s;
  }
}

class _Thread {
  String name = '';
  final notes = <String>[];
  final frames = <String>[];

  static _Thread parse(_Pb r) {
    final t = _Thread();
    while (r.more) {
      final (f, w) = r.tag();
      switch (f) {
        case 2 when w == 2:
          t.name = r.string();
        case 4 when w == 2:
          t.frames.add(_frame(r.sub()));
        case 7 when w == 2:
          t.notes.add(r.string());
        default:
          r.skip(w);
      }
    }
    return t;
  }

  // As debuggerd prints one: "pc 000000000004e8cc  /lib64/libc.so (abort+164)
  // (BuildId: ...)".
  static String _frame(_Pb r) {
    var relPc = 0, offset = 0;
    var function = '', file = '', buildId = '';
    while (r.more) {
      final (f, w) = r.tag();
      switch (f) {
        case 1 when w == 0:
          relPc = r.varint();
        case 4 when w == 2:
          function = r.string();
        case 5 when w == 0:
          offset = r.varint();
        case 6 when w == 2:
          file = r.string();
        case 8 when w == 2:
          buildId = r.string();
        default:
          r.skip(w);
      }
    }
    final s = StringBuffer('pc ${_hex(relPc, 16)}  $file');
    if (function.isNotEmpty) s.write(' ($function+$offset)');
    if (buildId.isNotEmpty) s.write(' (BuildId: $buildId)');
    return s.toString();
  }
}

// A uint64 held in a Dart int, which is signed: a tagged or kernel address
// would print negative.
String _hex(int v, int width) =>
    BigInt.from(v).toUnsigned(64).toRadixString(16).padLeft(width, '0');

/// The protobuf wire format, as far as reading a tombstone needs it.
class _Pb {
  _Pb(this.b, this.pos, this.end);

  final Uint8List b;
  int pos;
  final int end;

  bool get more => pos < end;

  int varint() {
    var result = 0, shift = 0;
    while (true) {
      if (pos >= end) throw const FormatException('truncated varint');
      final x = b[pos++];
      if (shift < 64) result |= (x & 0x7f) << shift;
      if (x < 0x80) return result;
      shift += 7;
    }
  }

  (int, int) tag() {
    final t = varint();
    return (t >> 3, t & 7);
  }

  int _length() {
    final n = varint();
    if (n < 0 || n > end - pos) throw const FormatException('truncated field');
    return n;
  }

  _Pb sub() {
    final n = _length();
    final r = _Pb(b, pos, pos + n);
    pos += n;
    return r;
  }

  String string() {
    final n = _length();
    final s = utf8.decode(Uint8List.sublistView(b, pos, pos + n),
        allowMalformed: true);
    pos += n;
    return s;
  }

  void skip(int wire) {
    switch (wire) {
      case 0:
        varint();
      case 1:
        pos += 8;
      case 2:
        final n = _length(); // moves pos past the length first
        pos += n;
      case 5:
        pos += 4;
      default:
        throw FormatException('wire type $wire');
    }
    if (pos > end) throw const FormatException('truncated field');
  }
}
