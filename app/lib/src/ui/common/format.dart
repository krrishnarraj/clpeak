/// Small shared formatting helpers, and the words every screen names a
/// backend and a headline by.  The model keeps each word as the file
/// spells it; these change only what people read.
library;

import '../../model/headlines.dart';

/// A backend as people read it: `Core ML`, `ONNX Runtime` (the CLI's
/// `--help` names), and every other as the file spells it.  A key, a
/// route or a `--devices` token keeps the file's word.
String backendName(String backend) => switch (backend) {
      'CoreML' => 'Core ML',
      'ONNX' => 'ONNX Runtime',
      _ => backend,
    };

/// A headline as a card or a compare's tile names it: the catalog's own
/// measures by their short names ([short]: a phone's column), any other by
/// its catalog title.
String headlineLabel(Headline h, {bool short = false}) => switch (h.id) {
      'peak-float' => short ? 'Float' : 'Floating point',
      'peak-integer' => 'Integer',
      'peak-memory-bandwidth' => short ? 'Memory' : 'Memory bandwidth',
      _ => h.title,
    };

String formatBytes(int bytes) {
  const gb = 1024 * 1024 * 1024;
  const mb = 1024 * 1024;
  if (bytes >= gb) {
    final v = bytes / gb;
    return v == v.roundToDouble()
        ? '${v.round()} GB'
        : '${v.toStringAsFixed(1)} GB';
  }
  if (bytes >= mb) return '${(bytes / mb).round()} MB';
  return '${(bytes / 1024).round()} KB';
}

String formatDate(DateTime t) {
  String two(int v) => v.toString().padLeft(2, '0');
  return '${t.year}-${two(t.month)}-${two(t.day)} ${two(t.hour)}:${two(t.minute)}';
}

/// A day as a person writes it: `20 Oct 2026`.
String formatDay(DateTime t) {
  const months = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  return '${t.day} ${months[t.month - 1]} ${t.year}';
}

String formatDuration(Duration d) {
  if (d.inHours > 0) {
    return '${d.inHours}h ${d.inMinutes % 60}m';
  }
  if (d.inMinutes > 0) {
    return '${d.inMinutes}m ${d.inSeconds % 60}s';
  }
  return '${d.inSeconds}s';
}
