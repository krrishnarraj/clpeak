import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:provider/provider.dart';

import '../../model/document_edit.dart';
import '../../model/run_document.dart';
import '../../model/run_summary.dart';
import '../../services/run_history_store.dart';
import '../../theme/xpupeak_theme.dart';
import '../common/kit.dart';

/// The annotations docs/format-v4.md recommends, as the editor lists them:
/// the key, its label, and what goes in it.
const _recommended = <(String, String, String)>[
  ('system.name', 'Name',
      "the machine's name, where the OS gives none or for a custom build"),
  ('cooling', 'Cooling', "stock, a cooler's model, passive, …"),
  ('power_limit', 'Power limit', 'a power or TDP limit set by hand'),
  ('overclock', 'Overclock', 'an overclock or undervolt'),
  ('memory', 'Memory', 'the memory configuration: channels, speed, timings'),
  ('gpu_board', 'GPU board', "the maker of a discrete GPU's board"),
  ('notes', 'Notes', 'anything else'),
];

/// Edit what only a person knows about a saved run -- its `annotations` --
/// and write them into its file, in place (document_edit.dart), so they
/// travel with it when it is exported.  A cancelled run's checkpoint gets
/// them too, since continuing the run starts from the checkpoint.  The
/// history index is refreshed from the file, so its list shows the name.
Future<void> editRunDetails(BuildContext context, RunSummary summary,
    {required VoidCallback onSaved}) async {
  final history = context.read<RunHistoryStore>();
  final messenger = ScaffoldMessenger.of(context);
  final file = await history.documentFile(summary);
  final String raw;
  final RunMeta meta;
  try {
    raw = await file.readAsString();
    meta = RunDocument.fromJson(jsonDecode(raw) as Map<String, dynamic>).meta!;
  } catch (_) {
    messenger.showSnackBar(
        const SnackBar(content: Text('Could not read this result file.')));
    return;
  }
  if (!context.mounted) return;
  final edited = await showDialog<Map<String, String>>(
    context: context,
    builder: (context) => _DetailsDialog(meta: meta),
  );
  if (edited == null) return;
  try {
    final text = withAnnotations(raw, edited);
    await _replace(file, text);
    final checkpoint = File(
        p.join(file.parent.path, RunHistoryStore.partialFileNameFor(summary.id)));
    if (await checkpoint.exists()) {
      await _replace(checkpoint,
          withAnnotations(await checkpoint.readAsString(), edited));
    }
    final saved = RunDocument.fromJson(jsonDecode(text) as Map<String, dynamic>);
    await history.add(RunSummary.fromDocument(
      id: summary.id,
      fileName: summary.fileName,
      doc: saved,
      startedAt: summary.startedAt,
      durationMs: summary.durationMs,
      cancelled: summary.cancelled,
    ).withName(summary.name));
    onSaved();
  } catch (e) {
    messenger.showSnackBar(
        SnackBar(content: Text('Could not save the details: $e')));
  }
}

/// Write [text] over [file] through a rename, so a failure part-way leaves
/// the run as it was rather than half a file.
Future<void> _replace(File file, String text) async {
  final tmp = File('${file.path}.tmp');
  await tmp.writeAsString(text, flush: true);
  await tmp.rename(file.path);
}

class _DetailsDialog extends StatefulWidget {
  const _DetailsDialog({required this.meta});

  final RunMeta meta;

  @override
  State<_DetailsDialog> createState() => _DetailsDialogState();
}

class _DetailsDialogState extends State<_DetailsDialog> {
  late final Map<String, TextEditingController> _fields;

  /// The recommended keys, then any other the file holds -- a newer build's
  /// or a site's -- which are kept, and editable, as written.
  late final List<(String, String, String)> _rows;

  @override
  void initState() {
    super.initState();
    final current = widget.meta.annotations;
    final known = {for (final r in _recommended) r.$1};
    _rows = [
      ..._recommended,
      for (final key in current.keys.toList()..sort())
        if (!known.contains(key)) (key, key, ''),
    ];
    _fields = {
      for (final (key, _, _) in _rows)
        key: TextEditingController(text: current[key] ?? ''),
    };
  }

  @override
  void dispose() {
    for (final c in _fields.values) {
      c.dispose();
    }
    super.dispose();
  }

  Map<String, String> _values() => {
        for (final e in _fields.entries) e.key: e.value.text,
      };

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final s = widget.meta.systemInfo;
    // What a blank name falls back to.
    final measured = [s.name, s.model, s.board, s.soc]
        .firstWhere((v) => v.isNotEmpty, orElse: () => '');
    InputBorder border(Color c, [double w = 1]) => OutlineInputBorder(
          borderRadius: BorderRadius.circular(CP.rControl),
          borderSide: BorderSide(color: c, width: w),
        );
    return CDialog(
      title: 'Run details',
      actions: [
        CButton(
            label: 'Cancel',
            kind: CButtonKind.quiet,
            onPressed: () => Navigator.pop(context)),
        CButton(
            label: 'Save',
            kind: CButtonKind.primary,
            onPressed: () => Navigator.pop(context, _values())),
      ],
      child: ConstrainedBox(
        constraints:
            BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.6),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'What only you know about this run. Saved into its file, '
                'so it travels with it.',
                style: t.monoSmallDim,
              ),
              for (final (key, label, hint) in _rows) ...[
                const SizedBox(height: 12),
                Text(label.toUpperCase(), style: t.micro),
                const SizedBox(height: 5),
                TextField(
                  key: ValueKey('details-$key'),
                  controller: _fields[key],
                  style: t.mono,
                  cursorWidth: 1.5,
                  cursorRadius: Radius.zero,
                  minLines: key == 'notes' ? 2 : 1,
                  maxLines: key == 'notes' ? 5 : 1,
                  decoration: InputDecoration(
                    isDense: true,
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 11),
                    hintText: key == 'system.name' && measured.isNotEmpty
                        ? measured
                        : hint,
                    hintStyle: t.mono.copyWith(color: t.faint),
                    border: border(t.line),
                    enabledBorder: border(t.line),
                    focusedBorder: border(t.text, 1.5),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
