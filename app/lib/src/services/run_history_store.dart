import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../model/run_document.dart';
import '../model/run_summary.dart';
import 'process_exit.dart';

/// Persists every run under `<base>/runs/`:
///   `<id>.clpeak.json`  the run document, written by the NATIVE side
///                       (clpeak_launch -o) — also the export artifact
///   `<id>.clpeak.log`   the run's diagnostic stream, streamed line by line
///                       by the native side WHILE the run is in flight and
///                       removed once the document is written.  One that
///                       outlives its run is the record of a run the
///                       process died in — see [listCrashLogs]
///   `<id>.clpeak.partial.json`
///                       the run's checkpoint, rewritten by the native side
///                       as each benchmark finishes and removed with the
///                       log: what such a run finished, which `--resume`
///                       continues (include/common/run_checkpoint.h)
///   `<id>.clpeak.resume.json`
///                       the app's own note on a run in flight: the device
///                       catalog it started on, and whether its Continue
///                       offer was dismissed ([RunResumeState])
///   index.json          {"runs":[RunSummary...]} for a fast history list
///
/// Orphan documents (present on disk but missing from the index, e.g. after
/// an app kill mid-finalize) are re-adopted on load.
///
/// Files are read here with dart:convert rather than through the native
/// library: the saved document is already the shape the UI renders, so a
/// round trip through FFI would buy nothing — and history stays readable even
/// when the native library cannot be loaded at all.
class RunHistoryStore {
  RunHistoryStore({
    Directory? directoryOverride,
    Future<List<ProcessExitRecord>> Function()? processExits,
  })  : _override = directoryOverride,
        _processExits = processExits ?? ProcessExit.records;

  final Directory? _override;
  final Future<List<ProcessExitRecord>> Function() _processExits;
  Directory? _dir;

  /// Suffix rather than plain `.json`: it names the format at a glance in a
  /// directory a user is expected to browse, and keeps the index sidecar
  /// (index.json) out of the orphan scan.
  static const fileSuffix = '.clpeak.json';

  /// The sidecar the native side derives from `-o`: `.json` -> `.log`
  /// (RunLog::sidecarPathFor in include/common/run_log.h).
  static const logSuffix = '.clpeak.log';

  /// The checkpoint the native side derives from `-o`: `.json` ->
  /// `.partial.json` (RunCheckpoint::pathFor in
  /// include/common/run_checkpoint.h).
  static const partialSuffix = '.clpeak.partial.json';

  static const resumeSuffix = '.clpeak.resume.json';

  static String fileNameFor(String id) => '$id$fileSuffix';
  static String logFileNameFor(String id) => '$id$logSuffix';
  static String partialFileNameFor(String id) => '$id$partialSuffix';
  static String resumeFileNameFor(String id) => '$id$resumeSuffix';

  Future<Directory> runsDirectory() async {
    if (_dir != null) return _dir!;
    final dir =
        _override ?? Directory(p.join((await baseDirectory()).path, 'runs'));
    await dir.create(recursive: true);
    _dir = dir;
    return dir;
  }

  /// Desktop: `$HOME/.clpeak` (`%USERPROFILE%\.clpeak` on Windows). The
  /// documents directory is deliberately not used there — on macOS the first
  /// touch of `~/Documents` raises a TCC consent dialog, and a benchmark tool
  /// has no business asking for the user's documents.
  ///
  /// Mobile keeps the per-app documents directory: it is inside the app
  /// sandbox (no permission involved), and is the location iOS exposes to the
  /// Files app / iTunes file sharing.
  static Future<Directory> baseDirectory() async {
    if (Platform.isAndroid || Platform.isIOS) {
      final docs = await getApplicationDocumentsDirectory();
      return Directory(p.join(docs.path, 'clpeak'));
    }
    final env = Platform.environment;
    final home = env['HOME'] ?? env['USERPROFILE'];
    if (home != null && home.isNotEmpty) {
      return Directory(p.join(home, '.clpeak'));
    }
    // No home directory (odd service/CI environments) — fall back to the
    // platform's per-app support dir, which never needs consent either.
    return getApplicationSupportDirectory();
  }

  File _indexFile(Directory dir) => File(p.join(dir.path, 'index.json'));

  /// Absolute path a new run's document should be written to.
  Future<String> filePathFor(String id) async =>
      p.join((await runsDirectory()).path, fileNameFor(id));

  Future<List<RunSummary>> _readIndex(Directory dir) async {
    final f = _indexFile(dir);
    if (!await f.exists()) return [];
    try {
      final doc = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      return [
        for (final r in (doc['runs'] as List? ?? const []))
          RunSummary.fromJson(r as Map<String, dynamic>)
      ];
    } catch (_) {
      return [];
    }
  }

  Future<void> _writeIndex(Directory dir, List<RunSummary> runs) async {
    final doc = {'runs': [for (final r in runs) r.toJson()]};
    await _indexFile(dir)
        .writeAsString(const JsonEncoder.withIndent(' ').convert(doc));
  }

  /// History rows, newest first, adopting any orphan documents.
  ///
  /// Only v3 documents are listed: the index may still hold pre-v3 rows
  /// (v2 tracked `<id>.xml`), which this build can no longer open, so such
  /// rows are hidden — and pruned from the index — rather than listed only
  /// to fail on tap. The same goes for rows whose file went missing or no
  /// longer parses as the current version.
  Future<List<RunSummary>> list() async {
    final dir = await runsDirectory();
    var runs = await _readIndex(dir);

    if (runs.isNotEmpty) {
      final kept = <RunSummary>[];
      var pruned = false;
      for (final r in runs) {
        if (!r.fileName.endsWith(fileSuffix)) {
          pruned = true;
          continue;
        }
        final f = File(p.join(dir.path, p.basename(r.fileName)));
        if (!await f.exists() || !await _isCurrentVersion(f)) {
          pruned = true;
          continue;
        }
        kept.add(r);
      }
      if (pruned) {
        await _writeIndex(dir, kept);
      }
      runs = kept;
    }
    final known = {for (final r in runs) r.fileName};

    var adopted = false;
    await for (final f in dir.list()) {
      if (f is! File || !f.path.endsWith(fileSuffix)) continue;
      final name = p.basename(f.path);
      if (known.contains(name)) continue;
      final doc = await _readDocument(f);
      if (doc == null) continue;
      final stat = await f.stat();
      DateTime startedAt;
      if (doc.meta?.generatedAt.isNotEmpty ?? false) {
        // Local, as the index's own are: the file says UTC.
        startedAt = DateTime.tryParse(doc.meta!.generatedAt)?.toLocal() ??
            stat.modified;
      } else {
        startedAt = stat.modified;
      }
      final durationMs = ((doc.meta?.durationSeconds ?? 0) * 1000).toInt();
      runs.add(RunSummary.fromDocument(
        id: name.substring(0, name.length - fileSuffix.length),
        fileName: name,
        doc: doc,
        startedAt: startedAt,
        durationMs: durationMs,
        cancelled: doc.meta?.cancelled ?? false,
      ));
      adopted = true;
    }
    runs.sort((a, b) => b.startedAt.compareTo(a.startedAt));
    if (adopted) await _writeIndex(dir, runs);
    return runs;
  }

  Future<void> add(RunSummary summary) async {
    final dir = await runsDirectory();
    final runs = await _readIndex(dir)
      ..removeWhere((r) => r.id == summary.id)
      ..add(summary);
    runs.sort((a, b) => b.startedAt.compareTo(a.startedAt));
    await _writeIndex(dir, runs);
  }

  /// Set (or clear, with an empty string) a run's user-given name.
  Future<void> rename(RunSummary summary, String name) async {
    final dir = await runsDirectory();
    final runs = await _readIndex(dir);
    final i = runs.indexWhere((r) => r.id == summary.id);
    if (i < 0) return;
    runs[i] = runs[i].withName(name.trim());
    await _writeIndex(dir, runs);
  }

  Future<void> delete(RunSummary summary) async {
    final dir = await runsDirectory();
    final file = File(p.join(dir.path, summary.fileName));
    if (await file.exists()) await file.delete();
    // A cancelled run's checkpoint and note go with it.
    await _deleteQuietly([
      File(p.join(dir.path, partialFileNameFor(summary.id))),
      File(p.join(dir.path, resumeFileNameFor(summary.id))),
    ]);
    final runs = await _readIndex(dir)
      ..removeWhere((r) => r.id == summary.id);
    await _writeIndex(dir, runs);
  }

  // ── Runs that never finished ─────────────────────────────────────────────

  /// Sidecars left behind by runs whose document was never written — the
  /// process was killed or crashed natively mid-run — newest first.  A
  /// sidecar whose document does exist is stale (the run finished, the
  /// removal did not) and is cleaned up here instead of listed -- unless the
  /// run's checkpoint is there too: a cancelled run keeps its checkpoint, and
  /// one continued that died in turn left its sidecar beside the result its
  /// first part saved.
  ///
  /// [inFlightId] is the run currently executing, whose sidecar is live and
  /// not a crash.
  Future<List<CrashLog>> listCrashLogs({String? inFlightId}) async {
    final dir = await runsDirectory();
    await _appendProcessExits(dir);
    final out = <CrashLog>[];
    await for (final f in dir.list()) {
      if (f is! File || !f.path.endsWith(logSuffix)) continue;
      final name = p.basename(f.path);
      final id = name.substring(0, name.length - logSuffix.length);
      if (id == inFlightId) continue;
      if (await _finished(dir, id)) {
        await _deleteQuietly([
          f,
          File(p.join(dir.path, resumeFileNameFor(id))),
        ]);
        continue;
      }
      final log = await CrashLog.read(
        f,
        id: id,
        checkpoint: File(p.join(dir.path, partialFileNameFor(id))),
        resumeState: File(p.join(dir.path, resumeFileNameFor(id))),
      );
      if (log != null) out.add(log);
    }
    out.sort((a, b) => b.startedAt.compareTo(a.startedAt));
    return out;
  }

  /// The run's result is saved and nothing of it is left to continue.
  static Future<bool> _finished(Directory dir, String id) async =>
      await File(p.join(dir.path, fileNameFor(id))).exists() &&
      !await File(p.join(dir.path, partialFileNameFor(id))).exists();

  static Future<void> _deleteQuietly(List<File> files) async {
    for (final f in files) {
      try {
        if (await f.exists()) await f.delete();
      } catch (_) {}
    }
  }

  /// Android's record of how the process ended (ProcessExit), appended
  /// once to the sidecar of the run it ended -- after everything the run
  /// recorded, which is where it happened -- so the file a user exports
  /// says whether a driver crashed or the phone ran out of memory.
  Future<void> _appendProcessExits(Directory dir) async {
    final List<ProcessExitRecord> exits;
    try {
      exits = await _processExits();
    } catch (_) {
      return;
    }
    for (final exit in exits) {
      if (exit.runId.isEmpty || p.basename(exit.runId) != exit.runId) continue;
      final log = File(p.join(dir.path, logFileNameFor(exit.runId)));
      try {
        if (!await log.exists() || await _finished(dir, exit.runId)) {
          continue;
        }
        final raf = await log.open(mode: FileMode.append);
        try {
          final size = await raf.length();
          final head = await _readAt(raf, 0, 64 << 10);
          final tail = await _readAt(
              raf, size > _exitTail ? size - _exitTail : 0, _exitTail);
          DateTime? started;
          final nl = head.indexOf('\n');
          if (nl > 0) {
            try {
              final header =
                  jsonDecode(head.substring(0, nl)) as Map<String, dynamic>;
              started =
                  DateTime.tryParse(header['generated_at'] as String? ?? '');
            } catch (_) {}
          }
          final entry = jsonEncode({
            'elapsed_s': started == null
                ? 0
                : exit.at.difference(started).inMilliseconds / 1000.0,
            'level': 'error',
            'source': 'android',
            'message': exit.message,
          });
          // Records stay with Android across launches, and one run's log
          // can hold more than one: a run that was resumed and died again.
          // The same record makes the same line, which is already there
          // somewhere when it was appended before.
          if (await _hasLine(log, entry)) continue;
          // A crash can cut the last line short; the entry starts its own.
          final prefix = tail.isEmpty || tail.endsWith('\n') ? '' : '\n';
          await raf.setPosition(size);
          await raf.writeString('$prefix$entry\n');
          await raf.flush();
        } finally {
          await raf.close();
        }
      } catch (_) {
        // A diagnosis, never a reason for the listing to fail.
      }
    }
  }

  // Enough of the file's end to see whether its last line is whole.
  static const _exitTail = 4 << 10;

  static Future<String> _readAt(RandomAccessFile raf, int at, int max) async {
    await raf.setPosition(at);
    return utf8.decode(await raf.read(max), allowMalformed: true);
  }

  // Whether a sidecar holds `line` whole; streamed, since a verbose run's
  // log can be tens of MB.
  static Future<bool> _hasLine(File f, String line) async {
    await for (final l in f
        .openRead()
        .transform(const Utf8Decoder(allowMalformed: true))
        .transform(const LineSplitter())) {
      if (l == line) return true;
    }
    return false;
  }

  Future<File> crashLogFile(CrashLog log) async {
    final dir = await runsDirectory();
    return File(p.join(dir.path, log.fileName));
  }

  /// Everything of the run: a cancelled run continued that died in turn
  /// has its first part's result too.
  Future<void> deleteCrashLog(CrashLog log) async {
    final dir = await runsDirectory();
    await _deleteQuietly([
      File(p.join(dir.path, log.fileName)),
      File(p.join(dir.path, partialFileNameFor(log.id))),
      File(p.join(dir.path, resumeFileNameFor(log.id))),
      File(p.join(dir.path, fileNameFor(log.id))),
    ]);
    final runs = await _readIndex(dir);
    if (runs.any((r) => r.id == log.id)) {
      await _writeIndex(dir, runs..removeWhere((r) => r.id == log.id));
    }
  }

  /// What a run that did not finish had finished, for viewing and for the
  /// live view of its continuation: its checkpoint as a document, or null.
  Future<RunDocument?> loadCheckpoint(String id) async {
    final dir = await runsDirectory();
    try {
      final f = File(p.join(dir.path, partialFileNameFor(id)));
      final doc = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      if (doc['schema'] != 'clpeak/run-partial' ||
          (doc['format_version'] as num?)?.toInt() != formatVersion) {
        return null;
      }
      return RunDocument.fromJson(doc);
    } catch (_) {
      return null;
    }
  }

  /// Note the device catalog and OS release a run starts on, beside its
  /// files ([RunResumeState]).
  Future<void> writeResumeState(String id,
      {required String catalog, required String os}) async {
    final dir = await runsDirectory();
    try {
      await File(p.join(dir.path, resumeFileNameFor(id))).writeAsString(
          jsonEncode(RunResumeState(catalog: catalog, os: os).toJson()));
    } catch (_) {}
  }

  /// What a cancelled run kept to be continued: its checkpoint, or null.
  Future<RunCheckpointInfo?> checkpointInfo(String id) async {
    final dir = await runsDirectory();
    return RunCheckpointInfo.read(
        File(p.join(dir.path, partialFileNameFor(id))));
  }

  /// The note the app left beside a run when it started it, or null.
  Future<RunResumeState?> resumeState(String id) async {
    final dir = await runsDirectory();
    return RunResumeState.read(File(p.join(dir.path, resumeFileNameFor(id))));
  }

  /// The user turned down continuing this run: History keeps it, the
  /// dashboard stops offering it.
  Future<void> dismissResume(String id) async {
    final dir = await runsDirectory();
    final file = File(p.join(dir.path, resumeFileNameFor(id)));
    final state = await RunResumeState.read(file) ?? const RunResumeState();
    try {
      await file.writeAsString(
          jsonEncode(RunResumeState(
                  catalog: state.catalog, os: state.os, dismissed: true)
              .toJson()));
    } catch (_) {}
  }

  /// A run's note once its document is written and it has nothing left to
  /// continue.
  Future<void> clearResumeState(String id) async {
    final dir = await runsDirectory();
    await _deleteQuietly([File(p.join(dir.path, resumeFileNameFor(id)))]);
  }

  /// Load a saved run for viewing.
  Future<RunDocument?> load(RunSummary summary) async {
    final dir = await runsDirectory();
    return _readDocument(File(p.join(dir.path, summary.fileName)));
  }

  /// Parse one saved document, or null when it is unreadable, not JSON, or
  /// written by a clpeak whose format this build does not know.
  Future<RunDocument?> _readDocument(File f) async {
    try {
      final doc = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      if (!_matchesCurrentVersion(doc)) return null;
      return RunDocument.fromJson(doc);
    } catch (_) {
      return null;
    }
  }

  /// Lightweight version probe for history rows: answers whether a file is a
  /// document this build renders, without building the full model.
  Future<bool> _isCurrentVersion(File f) async {
    try {
      final doc = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      return _matchesCurrentVersion(doc);
    } catch (_) {
      return false;
    }
  }

  bool _matchesCurrentVersion(Map<String, dynamic> doc) =>
      (doc['format_version'] as num?)?.toInt() == formatVersion;

  Future<File> documentFile(RunSummary summary) async {
    final dir = await runsDirectory();
    return File(p.join(dir.path, summary.fileName));
  }

  Future<bool> fileExists(String fileName) async {
    final dir = await runsDirectory();
    return File(p.join(dir.path, p.basename(fileName))).exists();
  }

  Future<String> nextAvailableFileName(String desiredName) async {
    final dir = await runsDirectory();
    var base = p.basename(desiredName);
    if (!base.endsWith(fileSuffix)) {
      final dot = base.lastIndexOf('.');
      base = (dot > 0 ? base.substring(0, dot) : base) + fileSuffix;
    }
    var candidate = base;
    var i = 1;
    while (await File(p.join(dir.path, candidate)).exists()) {
      final without = base.substring(0, base.length - fileSuffix.length);
      candidate = '${without}_$i$fileSuffix';
      i++;
    }
    return candidate;
  }

  /// Import a run document from outside the runs directory.
  ///
  /// Validates the JSON, checks `format_version`, copies the file into the
  /// store and updates the index.  When [overwrite] is false and a file with
  /// the same name already exists a [FileSystemException] is thrown so the
  /// caller can prompt the user (overwrite vs. rename).
  Future<RunSummary> importExternalFile(
    File sourceFile, {
    String? targetFileName,
    bool overwrite = false,
  }) async {
    final raw = await sourceFile.readAsString();
    return importContent(raw,
        fileName: targetFileName ?? p.basename(sourceFile.path),
        overwrite: overwrite);
  }

  /// Import from an in-memory JSON string (e.g. an `XFile` picked via
  /// `file_selector` where the path may not be a normal file).
  Future<RunSummary> importContent(
    String raw, {
    required String fileName,
    bool overwrite = false,
  }) async {
    late Map<String, dynamic> json;
    try {
      json = jsonDecode(raw) as Map<String, dynamic>;
    } catch (e) {
      throw FormatException('Not valid JSON: $e');
    }
    if ((json['format_version'] as num?)?.toInt() != formatVersion) {
      throw FormatException(
          'Unsupported format version ${json['format_version']} – expected $formatVersion');
    }
    if (json['schema'] == 'clpeak/run-partial') {
      throw const FormatException(
          'This is the checkpoint of a run that did not finish, not a result');
    }
    final doc = RunDocument.fromJson(json);
    final dir = await runsDirectory();
    var targetName = p.basename(fileName);
    if (!targetName.endsWith(fileSuffix)) {
      final dot = targetName.lastIndexOf('.');
      targetName =
          (dot > 0 ? targetName.substring(0, dot) : targetName) + fileSuffix;
    }
    final targetFile = File(p.join(dir.path, targetName));
    if (await targetFile.exists() && !overwrite) {
      throw FileSystemException('File already exists', targetFile.path);
    }
    await targetFile.writeAsString(raw);
    final id = targetName.substring(0, targetName.length - fileSuffix.length);
    DateTime startedAt;
    final generated = doc.meta?.generatedAt ?? '';
    if (generated.isNotEmpty) {
      startedAt =
          DateTime.tryParse(generated) ?? (await targetFile.stat()).modified;
    } else {
      startedAt = (await targetFile.stat()).modified;
    }
    final durationMs = ((doc.meta?.durationSeconds ?? 0) * 1000).toInt();
    final summary = RunSummary.fromDocument(
      id: id,
      fileName: targetName,
      doc: doc,
      startedAt: startedAt,
      durationMs: durationMs,
      cancelled: doc.meta?.cancelled ?? false,
    );
    await add(summary);
    return summary;
  }
}

/// A run the process died in.  Its document was never written; the run-log
/// sidecar the native side streamed as it went is the record of it, and the
/// thing to attach to a bug report.  Read from the sidecar's header line
/// (the run's identity) and the entries after it, with the run's checkpoint
/// beside it when there is one -- what it finished, which Continue resumes.
class CrashLog {
  const CrashLog({
    required this.id,
    required this.fileName,
    required this.startedAt,
    required this.clpeakVersion,
    required this.verbose,
    required this.entries,
    required this.backends,
    required this.lastEntry,
    this.processExit,
    this.hostOs = '',
    this.hostOsVersion = '',
    this.lastTestTitle = '',
    this.lastTestBackend = '',
    this.diedOffScreen = false,
    this.checkpoint,
    this.resumeState,
  });

  final String id;
  final String fileName;
  final DateTime startedAt;
  final String clpeakVersion;
  final bool verbose;

  /// Lines after the header.
  final int entries;

  /// Backends the log saw, in order of first appearance.
  final List<String> backends;

  /// The last line recorded before the process stopped — the one that
  /// usually says where.
  final LogEntry? lastEntry;

  /// How Android says the process ended (`source: "android"`, appended on
  /// a later launch by RunHistoryStore) — what the run's own lines cannot
  /// say.  Not counted in [entries], and never the [lastEntry].  For a run
  /// that was resumed, only the latest part's.
  final LogEntry? processExit;

  /// The OS the run started on, from the sidecar's header.
  final String hostOs;
  final String hostOsVersion;

  /// The test the run was in when it stopped, as the log titled it when it
  /// opened ("Memory latency (pointer-chase)"), and its backend.
  final String lastTestTitle;
  final String lastTestBackend;

  /// The app had left the screen, and not come back, when the run stopped:
  /// the phone, or the user, ended it there.  False when it stopped on
  /// screen -- a crash -- and where no app said (the desktop, the CLI).
  final bool diedOffScreen;

  /// What it finished, when it left a checkpoint.
  final RunCheckpointInfo? checkpoint;

  /// What the app noted when it started the run.
  final RunResumeState? resumeState;

  /// Parse a sidecar, or null when it is not one (no header line, or a
  /// header from a format this build does not read).  Streamed: a verbose
  /// run's sidecar can be tens of MB.
  static Future<CrashLog?> read(File file,
      {required String id, File? checkpoint, File? resumeState}) async {
    Map<String, dynamic>? header;
    final backends = <String>[];
    LogEntry? last;
    LogEntry? exit;
    var count = 0;
    var testTitle = '';
    var testBackend = '';
    var offScreen = false;
    try {
      await for (final line in file
          .openRead()
          .transform(const Utf8Decoder(allowMalformed: true))
          .transform(const LineSplitter())) {
        if (header == null) {
          try {
            header = jsonDecode(line) as Map<String, dynamic>;
          } catch (_) {
            return null;
          }
          if (header['schema'] != 'clpeak/run-log' ||
              (header['format_version'] as num?)?.toInt() != formatVersion) {
            return null;
          }
          continue;
        }
        if (line.trim().isEmpty) continue;
        try {
          final entry =
              LogEntry.fromJson(jsonDecode(line) as Map<String, dynamic>);
          if (entry.source == 'android') {
            exit = entry;
            continue;
          }
          count++;
          last = entry;
          if (entry.backend.isNotEmpty && !backends.contains(entry.backend)) {
            backends.add(entry.backend);
          }
          if (entry.source == 'app') {
            // Its other lines -- a pause -- say nothing of where it was.
            if (entry.message.startsWith('the app left the screen')) {
              offScreen = true;
            } else if (entry.message.startsWith('the app is back on screen')) {
              offScreen = false;
            }
          } else if (entry.source.isEmpty) {
            // A part resumed after the one before died starts on screen,
            // and what ended the part before is not this part's.
            if (entry.message.startsWith('resumed the run')) {
              offScreen = false;
              exit = null;
            }
            // The line a test opens with: "    Title [variant]".
            final m = entry.message;
            if (entry.test.isNotEmpty &&
                entry.level == LogLevel.info &&
                m.length > 4 &&
                m.startsWith('    ') &&
                m[4] != ' ') {
              testTitle = m.trim();
              testBackend = entry.backend;
            }
          }
        } catch (_) {
          // A line cut short by the crash itself.
        }
      }
    } catch (_) {
      return null;
    }
    final h = header;
    if (h == null) return null;
    final stat = await file.stat();
    final host = h['host'] as Map<String, dynamic>? ?? const {};
    return CrashLog(
      id: id,
      fileName: p.basename(file.path),
      // Local, as History's saved runs are: the header says UTC.
      startedAt: DateTime.tryParse(h['generated_at'] as String? ?? '')
              ?.toLocal() ??
          stat.modified,
      clpeakVersion: h['clpeak_version'] as String? ?? '',
      verbose: (h['invocation'] as Map<String, dynamic>?)?['verbose']
              as bool? ??
          false,
      entries: count,
      backends: backends,
      lastEntry: last,
      processExit: exit,
      hostOs: host['os'] as String? ?? '',
      hostOsVersion: host['os_version'] as String? ?? '',
      lastTestTitle: testTitle,
      lastTestBackend: testBackend,
      diedOffScreen: offScreen,
      checkpoint:
          checkpoint == null ? null : await RunCheckpointInfo.read(checkpoint),
      resumeState:
          resumeState == null ? null : await RunResumeState.read(resumeState),
    );
  }
}

/// The checkpoint of a run that did not finish (include/common/
/// run_checkpoint.h), as far as History needs it: how much it holds, and
/// what it was run by.
class RunCheckpointInfo {
  const RunCheckpointInfo({
    required this.writtenAt,
    required this.tests,
    required this.clpeakVersion,
    required this.hostOs,
    required this.hostOsVersion,
    required this.argv,
  });

  /// When the run last finished something -- about when it stopped.
  final DateTime writtenAt;

  /// Tests it holds, over every device.
  final int tests;
  final String clpeakVersion;
  final String hostOs;
  final String hostOsVersion;

  /// The command line the run started with, which a continuation keeps.
  final List<String> argv;

  static Future<RunCheckpointInfo?> read(File f) async {
    try {
      if (!await f.exists()) return null;
      final doc = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      if (doc['schema'] != 'clpeak/run-partial' ||
          (doc['format_version'] as num?)?.toInt() != formatVersion) {
        return null;
      }
      final meta = RunMeta.fromJson(doc);
      var tests = 0;
      for (final d in (doc['devices'] as List? ?? const [])) {
        tests += ((d as Map<String, dynamic>)['tests'] as List? ?? const [])
            .length;
      }
      return RunCheckpointInfo(
        writtenAt: (await f.stat()).modified,
        tests: tests,
        clpeakVersion: meta.clpeakVersion,
        hostOs: meta.host['os']?.toString() ?? '',
        hostOsVersion: meta.host['os_version']?.toString() ?? '',
        argv: meta.argv,
      );
    } catch (_) {
      return null;
    }
  }
}

/// What the app notes beside a run when it starts one: the device catalog
/// it started on -- devices and runtime versions -- and the OS release, which
/// a continuation must find unchanged to be the same run, and whether the
/// user dismissed its Continue offer.
class RunResumeState {
  const RunResumeState(
      {this.catalog = '', this.os = '', this.dismissed = false});

  /// ResumeOffer.fingerprint of the catalog.
  final String catalog;

  /// Platform.operatingSystemVersion.
  final String os;
  final bool dismissed;

  Map<String, dynamic> toJson() => {
        'catalog': catalog,
        'os': os,
        if (dismissed) 'dismissed': true,
      };

  static Future<RunResumeState?> read(File f) async {
    try {
      if (!await f.exists()) return null;
      final m = jsonDecode(await f.readAsString()) as Map<String, dynamic>;
      return RunResumeState(
        catalog: m['catalog'] as String? ?? '',
        os: m['os'] as String? ?? '',
        dismissed: m['dismissed'] as bool? ?? false,
      );
    } catch (_) {
      return null;
    }
  }
}

/// Dump-format version this build reads — must match RESULT_FORMAT_VERSION in
/// include/common/run_document.h.  A file from another version is skipped
/// rather than half-parsed.
const int formatVersion = 3;
