/// Bringing saved runs into a compare (lib/src/model/compare.dart).  Every
/// run a compare reads is one History holds: one this app ran, or a file
/// from another machine that Import copied in (`importContent` /
/// `importExternalFile`, which refuse what is not a format 4 result), so a
/// run from anywhere is read the one way History reads its own.
library;

import '../model/compare.dart';
import '../model/headlines.dart';
import '../model/run_summary.dart';
import 'run_history_store.dart';

extension CompareLoading on RunHistoryStore {
  /// [summary]'s run as one side of a compare; null where its file can no
  /// longer be read as this build's format.
  Future<CompareRun?> loadForCompare(RunSummary summary) async {
    final doc = await load(summary);
    return doc == null ? null : CompareRun(doc, summary: summary);
  }

  /// [a] against [b], paired as `pairProcessors` pairs them -- one run
  /// twice, where both are one row; null where either cannot be read.
  Future<RunCompare?> compareRuns(RunSummary a, RunSummary b,
      {required HeadlineCatalog catalog}) async {
    final runA = await loadForCompare(a);
    if (runA == null) return null;
    final runB = a.id == b.id ? runA : await loadForCompare(b);
    if (runB == null) return null;
    return RunCompare.of(runA, runB, catalog: catalog);
  }
}
