import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:share_plus/share_plus.dart';

/// Exports a saved run document — or the run-log sidecar of a run that never
/// finished: native share sheet on mobile, save-file dialog on desktop.
class ExportService {
  Future<void> exportRun(File file, {required String suggestedName}) =>
      _export(file,
          suggestedName: suggestedName,
          mimeType: 'application/json',
          label: 'clpeak results',
          extensions: const ['json']);

  /// The NDJSON run log left behind when the app stopped mid-run.
  Future<void> exportRunLog(File file, {required String suggestedName}) =>
      _export(file,
          suggestedName: suggestedName,
          mimeType: 'text/plain',
          label: 'clpeak run log',
          extensions: const ['log']);

  Future<void> _export(
    File file, {
    required String suggestedName,
    required String mimeType,
    required String label,
    required List<String> extensions,
  }) async {
    if (Platform.isAndroid || Platform.isIOS) {
      await SharePlus.instance.share(ShareParams(
        files: [XFile(file.path, mimeType: mimeType)],
        subject: suggestedName,
      ));
      return;
    }
    final location = await getSaveLocation(
      suggestedName: suggestedName,
      acceptedTypeGroups: [XTypeGroup(label: label, extensions: extensions)],
    );
    if (location == null) return; // user cancelled
    await file.copy(location.path);
  }
}
