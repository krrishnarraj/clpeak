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
      DateTime startedAt;
      if (doc.meta?.generatedAt.isNotEmpty ?? false) {
        startedAt = DateTime.tryParse(doc.meta!.generatedAt) ?? stat.modified;
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

/// Dump-format version this build reads — must match RESULT_FORMAT_VERSION in
/// include/common/run_document.h.  A file from another version is skipped
/// rather than half-parsed.
const int formatVersion = 3;
