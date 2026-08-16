import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:share_plus/share_plus.dart';

/// Exports a saved run's XML: native share sheet on mobile, save-file dialog
/// on desktop.
class ExportService {
  Future<void> exportXml(File xml, {required String suggestedName}) async {
    if (Platform.isAndroid || Platform.isIOS) {
      await SharePlus.instance.share(ShareParams(
        files: [XFile(xml.path, mimeType: 'application/xml')],
        subject: suggestedName,
      ));
      return;
    }
    final location = await getSaveLocation(
      suggestedName: suggestedName,
      acceptedTypeGroups: const [
        XTypeGroup(label: 'clpeak results', extensions: ['xml'])
      ],
    );
    if (location == null) return; // user cancelled
    await xml.copy(location.path);
  }
}
