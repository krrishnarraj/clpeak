// The Details editor's write (lib/src/model/document_edit.dart): a run's
// annotations go into its file's own text, and every key this build does not
// know -- a newer XpuPeak's -- survives, as does every byte around them.
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:xpupeak/src/model/document_edit.dart';
import 'package:xpupeak/src/model/run_document.dart';

String _laptop() =>
    File('test/fixtures/soc_laptop.xpupeak.json').readAsStringSync();

/// The laptop as a newer build might write it: keys, a dim and an `ext`
/// this build does not know, at the top, on a device and on a reading.
String _fromANewerBuild() => _laptop()
    .replaceFirst('"format_minor": 0', '"format_minor": 3')
    .replaceFirst('  "packages": [',
        '  "power_trace": {"samples": [1.5, 2.25], "unit": "W"},\n  "packages": [')
    .replaceFirst('"link": "registry",',
        '"link": "registry",\n      "future_device_key": "kept",')
    .replaceFirst('"dims": {"dtype": "fp16"},\n              "value": 4.1e+12',
        '"dims": {"dtype": "fp16", "lanes": 32},\n              "value": 4.1e+12,\n'
            '              "ext": {"cycles": 81920}');

Map<String, dynamic> _parse(String s) => jsonDecode(s) as Map<String, dynamic>;

void main() {
  test('editing the details keeps every key the app does not know', () {
    final before = _fromANewerBuild();
    final after = withAnnotations(before, {
      'system.name': 'Work laptop',
      'cooling': 'stock',
      'notes': 'lid closed',
    });

    final old = _parse(before);
    final edited = _parse(after);
    expect(edited['annotations'],
        {'cooling': 'stock', 'notes': 'lid closed', 'system.name': 'Work laptop'});
    // Only the annotations differ.
    expect(edited..remove('annotations'), old..remove('annotations'));
    expect(after, contains('"power_trace": {"samples": [1.5, 2.25], "unit": "W"}'));
    expect(after, contains('"future_device_key": "kept"'));
    expect(after, contains('"ext": {"cycles": 81920}'));
    expect(after, contains('"dims": {"dtype": "fp16", "lanes": 32}'));

    // In place: the text around the member is the file's own, byte for byte,
    // the log still one entry per line.
    const was = '  "annotations": {\n    "cooling": "stock"\n  },\n';
    const now = '  "annotations": {\n    "cooling": "stock",\n'
        '    "notes": "lid closed",\n    "system.name": "Work laptop"\n  },\n';
    expect(before, contains(was));
    expect(after, before.replaceFirst(was, now));

    // And the app reads its name back.
    expect(RunDocument.fromJson(_parse(after)).meta!.systemName, 'Work laptop');
  });

  test('details go where the CLI writes them, and blank ones go away', () {
    final none = withAnnotations(_laptop(), {'cooling': '  '});
    expect(_parse(none).containsKey('annotations'), isFalse);
    expect(none, isNot(contains('annotations')));

    final added = withAnnotations(none, {'memory': 'LPDDR5-6400'});
    final keys = _parse(added).keys.toList();
    expect(keys.indexOf('annotations'), keys.indexOf('conditions') + 1);
    expect(added,
        contains('  },\n  "annotations": {\n    "memory": "LPDDR5-6400"\n  },\n'
            '  "packages": ['));
  });

  test('a document that cannot be spliced is rewritten with every key', () {
    // Two annotations members: no single place to edit.
    const odd = '{"annotations": {"a": "1"}, "keep": [true, {"k": 2}], '
        '"annotations": {"b": "2"}}';
    final out = _parse(withAnnotations(odd, {'cooling': 'water'}));
    expect(out, {
      'annotations': {'cooling': 'water'},
      'keep': [
        true,
        {'k': 2}
      ],
    });
  });

  test('a file that is not a document is refused', () {
    expect(() => withAnnotations('[1, 2]', {'a': 'b'}),
        throwsA(isA<FormatException>()));
  });
}
