/// Editing a saved document in place (docs/format-v4.md, Compatibility:
/// "Rewriting keeps the unknown").
///
/// The app writes only `annotations`, and it writes them into the file's own
/// text: the member is replaced, inserted or removed, and every other byte
/// stays as the native writer laid it down -- every key this build does not
/// know included, so an older app never strips a newer file, and the log
/// keeps its one entry per line.  The edit is checked by parsing the result:
/// only `annotations` may differ from the original.  Where the text is not
/// laid out so a splice can be made, the document is re-encoded from its
/// parsed tree instead, which keeps every key all the same.
library;

import 'dart:convert';

/// [raw] with its `annotations` set to [annotations]: blank values dropped,
/// keys sorted as the native writer sorts them, and the member left out
/// altogether when none remain.  Throws [FormatException] when [raw] is not
/// a JSON object.
String withAnnotations(String raw, Map<String, String> annotations) {
  final root = jsonDecode(raw);
  if (root is! Map<String, dynamic>) {
    throw const FormatException('not a run document');
  }
  final wanted = <String, String>{
    for (final key in (annotations.keys.toList()..sort()))
      if (key.trim().isNotEmpty && annotations[key]!.trim().isNotEmpty)
        key.trim(): annotations[key]!.trim(),
  };
  final expected = _withMember(root, wanted);

  final spliced = _splice(raw, wanted);
  if (spliced != null) {
    try {
      if (_jsonEquals(jsonDecode(spliced), expected)) return spliced;
    } on FormatException {
      // Fall through to the re-encode.
    }
  }
  return '${const JsonEncoder.withIndent('  ').convert(expected)}\n';
}

/// The members the native writer puts `annotations` after, nearest first.
const _anchors = ['conditions', 'system', 'invocation', 'build'];

/// [root] with `annotations` set (or removed), in the writer's place.
Map<String, dynamic> _withMember(
    Map<String, dynamic> root, Map<String, String> wanted) {
  final out = <String, dynamic>{};
  final anchor = _anchors.firstWhere(root.containsKey, orElse: () => '');
  var placed = root.containsKey('annotations');
  for (final e in root.entries) {
    if (e.key == 'annotations') {
      if (wanted.isNotEmpty) out['annotations'] = wanted;
      continue;
    }
    out[e.key] = e.value;
    if (!placed && e.key == anchor && wanted.isNotEmpty) {
      out['annotations'] = wanted;
      placed = true;
    }
  }
  if (!placed && wanted.isNotEmpty) out['annotations'] = wanted;
  return out;
}

/// One top-level member of the document's text: where its key starts and
/// its value starts and ends.
class _Member {
  _Member(this.key, this.start, this.valueStart, this.valueEnd);

  final String key;
  final int start;
  final int valueStart;
  final int valueEnd;
}

/// The new text, or null when the document's text is not one a splice can
/// be made in.
String? _splice(String raw, Map<String, String> wanted) {
  final members = _topLevelMembers(raw);
  if (members == null || members.isEmpty) return null;
  final existing = members.where((m) => m.key == 'annotations').toList();
  if (existing.length > 1) return null;

  final newline = raw.contains('\r\n') ? '\r\n' : '\n';
  final indent = _indentBefore(raw, members.first.start);
  String rendered() {
    final inner = '$indent$indent';
    final lines = [
      for (final e in wanted.entries)
        '$inner${jsonEncode(e.key)}: ${jsonEncode(e.value)}',
    ];
    return '{$newline${lines.join(',$newline')}$newline$indent}';
  }

  if (existing.isNotEmpty) {
    final m = existing.single;
    if (wanted.isNotEmpty) {
      return raw.replaceRange(m.valueStart, m.valueEnd, rendered());
    }
    // Removed: the member and the comma that joined it to its neighbour.
    final i = members.indexOf(m);
    if (i > 0) {
      return raw.replaceRange(members[i - 1].valueEnd, m.valueEnd, '');
    }
    if (members.length > 1) {
      return raw.replaceRange(m.start, members[1].start, '');
    }
    return null;
  }

  if (wanted.isEmpty) return raw;
  _Member? after;
  for (final key in _anchors) {
    for (final m in members) {
      if (m.key == key) after ??= m;
    }
    if (after != null) break;
  }
  after ??= members.last;
  return raw.replaceRange(after.valueEnd, after.valueEnd,
      ',$newline$indent"annotations": ${rendered()}');
}

/// The whitespace a line holds before [at]: the document's indent.
String _indentBefore(String raw, int at) {
  var i = at;
  while (i > 0 && (raw[i - 1] == ' ' || raw[i - 1] == '\t')) {
    i--;
  }
  return raw.substring(i, at);
}

/// The top-level object's members, in text order; null when [s] is not an
/// object.  Only called on text that already parsed as JSON.
List<_Member>? _topLevelMembers(String s) {
  var i = _skipSpace(s, 0);
  if (i >= s.length || s[i] != '{') return null;
  i++;
  final out = <_Member>[];
  while (true) {
    i = _skipSpace(s, i);
    if (i >= s.length) return null;
    if (s[i] == '}') return out;
    if (s[i] == ',') {
      i++;
      continue;
    }
    if (s[i] != '"') return null;
    final start = i;
    final keyEnd = _skipString(s, i);
    final key = jsonDecode(s.substring(start, keyEnd)) as String;
    i = _skipSpace(s, keyEnd);
    if (i >= s.length || s[i] != ':') return null;
    final valueStart = _skipSpace(s, i + 1);
    final valueEnd = _skipValue(s, valueStart);
    if (valueEnd < 0) return null;
    out.add(_Member(key, start, valueStart, valueEnd));
    i = valueEnd;
  }
}

int _skipSpace(String s, int i) {
  while (i < s.length && ' \t\r\n'.contains(s[i])) {
    i++;
  }
  return i;
}

/// Past the string that opens at [i].
int _skipString(String s, int i) {
  i++;
  while (i < s.length) {
    final c = s[i];
    if (c == '\\') {
      i += 2;
    } else if (c == '"') {
      return i + 1;
    } else {
      i++;
    }
  }
  return s.length;
}

/// Past the value that starts at [i], or -1.
int _skipValue(String s, int i) {
  if (i >= s.length) return -1;
  final c = s[i];
  if (c == '"') return _skipString(s, i);
  if (c == '{' || c == '[') {
    var depth = 0;
    while (i < s.length) {
      final d = s[i];
      if (d == '"') {
        i = _skipString(s, i);
        continue;
      }
      if (d == '{' || d == '[') depth++;
      if (d == '}' || d == ']') {
        depth--;
        if (depth == 0) return i + 1;
      }
      i++;
    }
    return -1;
  }
  // A number, true, false or null.
  while (i < s.length && !',}] \t\r\n'.contains(s[i])) {
    i++;
  }
  return i;
}

bool _jsonEquals(Object? a, Object? b) {
  if (a is Map && b is Map) {
    if (a.length != b.length) return false;
    for (final key in a.keys) {
      if (!b.containsKey(key) || !_jsonEquals(a[key], b[key])) return false;
    }
    return true;
  }
  if (a is List && b is List) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (!_jsonEquals(a[i], b[i])) return false;
    }
    return true;
  }
  return a == b;
}
