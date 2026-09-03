import 'dart:convert';
import 'dart:io';

import 'package:clpeak/src/services/run_history_store.dart';
import 'package:flutter_test/flutter_test.dart';

String v3Doc() => jsonEncode({
      'format_version': 3,
      'devices': [],
    });

String v2Doc() => jsonEncode({
      'format_version': 2,
      'entries': [],
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
  final tmp = Directory.systemTemp.createTempSync('clpeak_hist_v3');
  return tmp;
}

void main() {
  test('list() shows only v3 documents, pruning the rest', () async {
    final dir = await makeStore();
    addTearDown(() => dir.deleteSync(recursive: true));

    // On-disk files: one v3, one v2-in-v3-clothing, one legacy xml.
    await File('${dir.path}/good.clpeak.json').writeAsString(v3Doc());
    await File('${dir.path}/oldver.clpeak.json').writeAsString(v2Doc());
    await File('${dir.path}/legacy.xml').writeAsString('<run/>');
    // Orphan v3 file (in index nowhere) must be adopted; orphan junk ignored.
    await File('${dir.path}/orphan.clpeak.json').writeAsString(v3Doc());
    await File('${dir.path}/orphan_junk.clpeak.json')
        .writeAsString(v2Doc());

    await File('${dir.path}/index.json').writeAsString(jsonEncode({
      'runs': [
        indexRow('good', 'good.clpeak.json'),
        indexRow('oldver', 'oldver.clpeak.json'),
        indexRow('legacy', 'legacy.xml'),
        indexRow('gone', 'gone.clpeak.json'),
      ],
    }));

    final store = RunHistoryStore(directoryOverride: dir);
    final runs = await store.list();
    final names = {for (final r in runs) r.fileName};

    expect(names, contains('good.clpeak.json'));
    expect(names, contains('orphan.clpeak.json'));
    expect(names, isNot(contains('oldver.clpeak.json')));
    expect(names, isNot(contains('legacy.xml')));
    expect(names, isNot(contains('gone.clpeak.json')));

    // The index was pruned so the hidden rows stay hidden.
    final index = jsonDecode(
        await File('${dir.path}/index.json').readAsString()) as Map<String, dynamic>;
    final indexed =
        [(index['runs'] as List).map((r) => (r as Map)['fileName'])].expand((e) => e).toSet();
    expect(indexed, isNot(contains('legacy.xml')));
    expect(indexed, isNot(contains('oldver.clpeak.json')));
    expect(indexed, isNot(contains('gone.clpeak.json')));
  });
}
