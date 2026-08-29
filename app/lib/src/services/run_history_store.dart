import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../model/run_document.dart';
import '../model/run_summary.dart';

/// Persists every run under `<base>/runs/`:
///   `<id>.clpeak.json`  the run document, written by the NATIVE side
///                       (clpeak_launch -o) — also the export artifact
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
  RunHistoryStore({Directory? directoryOverride}) : _override = directoryOverride;

  final Directory? _override;
  Directory? _dir;

  /// Suffix rather than plain `.json`: it names the format at a glance in a
  /// directory a user is expected to browse, and keeps the index sidecar
  /// (index.json) out of the orphan scan.
  static const fileSuffix = '.clpeak.json';

  static String fileNameFor(String id) => '$id$fileSuffix';

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
  Future<List<RunSummary>> list() async {
    final dir = await runsDirectory();
    final runs = await _readIndex(dir);
    final known = {for (final r in runs) r.fileName};

    var adopted = false;
    await for (final f in dir.list()) {
      if (f is! File || !f.path.endsWith(fileSuffix)) continue;
      final name = p.basename(f.path);
      if (known.contains(name)) continue;
      final doc = await _readDocument(f);
      if (doc == null) continue;
      final stat = await f.stat();
      runs.add(RunSummary.fromDocument(
        id: name.substring(0, name.length - fileSuffix.length),
        fileName: name,
        doc: doc,
        startedAt: stat.modified,
        durationMs: 0,
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
    final runs = await _readIndex(dir)
      ..removeWhere((r) => r.id == summary.id);
    await _writeIndex(dir, runs);
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
      if ((doc['format_version'] as num?)?.toInt() != formatVersion) return null;
      return RunDocument.fromJson(doc);
    } catch (_) {
      return null;
    }
  }

  Future<File> documentFile(RunSummary summary) async {
    final dir = await runsDirectory();
    return File(p.join(dir.path, summary.fileName));
  }
}

/// Dump-format version this build reads — must match RESULT_FORMAT_VERSION in
/// include/common/run_document.h.  A file from another version is skipped
/// rather than half-parsed.
const int formatVersion = 3;
