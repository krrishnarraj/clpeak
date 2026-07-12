import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

import '../ffi/clpeak_bindings.dart';
import '../model/run_document.dart';
import '../model/run_summary.dart';

/// Persists every run under `<documents>/clpeak/runs/`:
///   `<id>.xml`   canonical schema-v2 XML written by the NATIVE side
///                (clpeak_launch --xml-file) — also the export artifact
///   index.json   {"runs":[RunSummary...]} for a fast history list
///
/// Orphan XMLs (present on disk but missing from the index, e.g. after an
/// app kill mid-finalize) are re-adopted on load via the native loader.
class RunHistoryStore {
  RunHistoryStore(this._bindings, {Directory? directoryOverride})
      : _override = directoryOverride;

  final ClpeakBindings _bindings;
  final Directory? _override;
  Directory? _dir;

  Future<Directory> runsDirectory() async {
    if (_dir != null) return _dir!;
    final base = _override ?? await getApplicationDocumentsDirectory();
    final dir = _override ?? Directory(p.join(base.path, 'clpeak', 'runs'));
    await dir.create(recursive: true);
    _dir = dir;
    return dir;
  }

  File _indexFile(Directory dir) => File(p.join(dir.path, 'index.json'));

  /// Absolute path a new run's XML should be written to.
  Future<String> xmlPathFor(String id) async =>
      p.join((await runsDirectory()).path, '$id.xml');

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

  /// History rows, newest first, adopting any orphan XML files.
  Future<List<RunSummary>> list() async {
    final dir = await runsDirectory();
    final runs = await _readIndex(dir);
    final known = {for (final r in runs) r.fileName};

    var adopted = false;
    await for (final f in dir.list()) {
      if (f is! File || !f.path.endsWith('.xml')) continue;
      final name = p.basename(f.path);
      if (known.contains(name)) continue;
      final doc = _bindings.loadResultFile(f.path);
      if (doc == null) continue;
      final stat = await f.stat();
      runs.add(RunSummary.fromDocument(
        id: p.basenameWithoutExtension(name),
        fileName: name,
        doc: RunDocument.fromEntriesJson(doc),
        startedAt: stat.modified,
        durationMs: 0,
        cancelled: false,
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
    final xml = File(p.join(dir.path, summary.fileName));
    if (await xml.exists()) await xml.delete();
    final runs = await _readIndex(dir)
      ..removeWhere((r) => r.id == summary.id);
    await _writeIndex(dir, runs);
  }

  /// Load a saved run for viewing (XML → native loader → document).
  Future<RunDocument?> load(RunSummary summary) async {
    final dir = await runsDirectory();
    final doc = _bindings.loadResultFile(p.join(dir.path, summary.fileName));
    if (doc == null) return null;
    return RunDocument.fromEntriesJson(doc);
  }

  Future<File> xmlFile(RunSummary summary) async {
    final dir = await runsDirectory();
    return File(p.join(dir.path, summary.fileName));
  }
}
