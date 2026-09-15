import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:share_plus/share_plus.dart';

/// Exports a saved run document: native share sheet on mobile, save-file
/// dialog on desktop.
class ExportService {
  Future<void> exportRun(File file, {required String suggestedName}) async {
    if (Platform.isAndroid || Platform.isIOS) {
      await SharePlus.instance.share(ShareParams(
        files: [XFile(file.path, mimeType: 'application/json')],
        subject: suggestedName,
      ));
      return;
    }
    final location = await getSaveLocation(
      suggestedName: suggestedName,
      acceptedTypeGroups: const [
        XTypeGroup(label: 'clpeak results', extensions: ['json'])
      ],
    );
    if (location == null) return; // user cancelled
    await file.copy(location.path);
  }
}
