import 'dart:convert';
import 'dart:io';

import 'package:xpupeak/src/services/run_history_store.dart';
import 'package:flutter_test/flutter_test.dart';

String v4Doc() => jsonEncode({
      'schema': 'xpupeak/run',
      'format_version': 4,
      'devices': [],
    });

String v3Doc() => jsonEncode({
      'format_version': 3,
      'devices': [],
    });

Map<String, dynamic> indexRow(String id, String fileName) => {
      'id': id,
      'startedAt': DateTime(2026, 8, 1).toIso8601String(),
      'durationMs': 100,
      'devices': ['M1'],
      'backends': ['CPU'],
      'cancelled': false,
      'fileName': fileName,
    };

Future<Directory> makeStore() async {
  final tmp = Directory.systemTemp.createTempSync('xpupeak_hist_v4');
  return tmp;
}

void main() {
  test('list() shows only v4 documents, pruning the rest', () async {
    final dir = await makeStore();
    addTearDown(() => dir.deleteSync(recursive: true));

    // On-disk files: one v4, one v3-in-v4-clothing, one legacy xml.
    await File('${dir.path}/good.xpupeak.json').writeAsString(v4Doc());
    await File('${dir.path}/oldver.xpupeak.json').writeAsString(v3Doc());
    await File('${dir.path}/legacy.xml').writeAsString('<run/>');
    // Orphan v4 file (in index nowhere) must be adopted; orphan junk ignored.
    await File('${dir.path}/orphan.xpupeak.json').writeAsString(v4Doc());
    await File('${dir.path}/orphan_junk.xpupeak.json')
        .writeAsString(v3Doc());

    await File('${dir.path}/index.json').writeAsString(jsonEncode({
      'runs': [
        indexRow('good', 'good.xpupeak.json'),
        indexRow('oldver', 'oldver.xpupeak.json'),
        indexRow('legacy', 'legacy.xml'),
        indexRow('gone', 'gone.xpupeak.json'),
      ],
    }));

    final store = RunHistoryStore(directoryOverride: dir);
    final runs = await store.list();
    final names = {for (final r in runs) r.fileName};

    expect(names, contains('good.xpupeak.json'));
    expect(names, contains('orphan.xpupeak.json'));
    expect(names, isNot(contains('oldver.xpupeak.json')));
    expect(names, isNot(contains('legacy.xml')));
    expect(names, isNot(contains('gone.xpupeak.json')));

    // The index was pruned so the hidden rows stay hidden.
    final index = jsonDecode(
        await File('${dir.path}/index.json').readAsString()) as Map<String, dynamic>;
    final indexed =
        [(index['runs'] as List).map((r) => (r as Map)['fileName'])].expand((e) => e).toSet();
    expect(indexed, isNot(contains('legacy.xml')));
    expect(indexed, isNot(contains('oldver.xpupeak.json')));
    expect(indexed, isNot(contains('gone.xpupeak.json')));
  });
}
