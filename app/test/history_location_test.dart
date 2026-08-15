// Where run history lives. The desktop answer must stay out of ~/Documents:
// touching it on macOS raises a TCC consent dialog on first run.
import 'dart:io';

import 'package:clpeak/src/services/run_history_store.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

void main() {
  test('desktop history lives in \$HOME/.clpeak, not the documents dir',
      () async {
    final home =
        Platform.environment['HOME'] ?? Platform.environment['USERPROFILE'];
    expect(home, isNotNull, reason: 'test host has no home directory');

    // Resolves without path_provider — no plugin channel in a pure-Dart test.
    final base = await RunHistoryStore.baseDirectory();
    expect(base.path, p.join(home!, '.clpeak'));
  },
      skip: Platform.isAndroid || Platform.isIOS
          ? 'mobile keeps the sandboxed app documents dir'
          : null);
}
