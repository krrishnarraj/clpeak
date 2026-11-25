import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../../theme/xpupeak_theme.dart';
import '../common/kit.dart';

/// The licences of what XpuPeak is built with, a package to a row: the
/// Flutter packages, and the bundled Sora typeface `main()` registers.  In
/// the app's frame rather than Material's LicensePage, and opened inside the
/// content, beside the sidebar.
class LicensesScreen extends StatefulWidget {
  const LicensesScreen({super.key, required this.version});

  final String version;

  static Route<void> route(String version) => MaterialPageRoute<void>(
        builder: (_) => LicensesScreen(version: version),
      );

  @override
  State<LicensesScreen> createState() => _LicensesScreenState();
}

/// One package and the licence entries that name it.
class _Package {
  _Package(this.name, this.entries);

  final String name;
  final List<LicenseEntry> entries;
}

class _LicensesScreenState extends State<LicensesScreen> {
  late final Future<List<_Package>> _packages = _collect();

  /// Every registered entry, by the packages it names, in name order.  An
  /// entry the registry cannot read ends the list there: what was read is
  /// still worth showing.
  static Future<List<_Package>> _collect() async {
    final byName = <String, List<LicenseEntry>>{};
    try {
      await for (final entry in LicenseRegistry.licenses) {
        for (final name in entry.packages) {
          byName.putIfAbsent(name, () => []).add(entry);
        }
      }
    } on Object {
      // Keep what was read.
    }
    return [for (final e in byName.entries) _Package(e.key, e.value)]
      ..sort((a, b) => a.name.toLowerCase().compareTo(b.name.toLowerCase()));
  }

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final side = wide ? 34.0 : 16.0;
    return Scaffold(
      body: SafeArea(
        child: FutureBuilder<List<_Package>>(
          future: _packages,
          builder: (context, snapshot) {
            final packages = snapshot.data;
            final header = CHeader(
              title: 'Licenses',
              subtitle: packages == null
                  ? 'XpuPeak v${widget.version}'
                  : 'XpuPeak v${widget.version} · ${packages.length} '
                      '${packages.length == 1 ? 'package' : 'packages'}',
              onBack: () => Navigator.of(context).pop(),
            );
            final Widget body;
            if (packages == null) {
              // Static words, no spinner: reading the list takes a moment.
              body = Padding(
                padding: const EdgeInsets.symmetric(horizontal: 6),
                child: Text('Reading the licenses…', style: t.prose),
              );
            } else if (packages.isEmpty) {
              body = const CEmptyBox(text: 'No licenses are registered.');
            } else {
              body = CPanel(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    for (var i = 0; i < packages.length; i++)
                      _PackageRow(
                        package: packages[i],
                        rule: i < packages.length - 1,
                      ),
                  ],
                ),
              );
            }
            return ListView(
              padding: EdgeInsets.only(bottom: wide ? 44 : 28),
              children: [
                header,
                Padding(
                  padding: EdgeInsets.fromLTRB(side, wide ? 14 : 12, side, 0),
                  child: Align(
                    alignment: Alignment.topLeft,
                    child: ConstrainedBox(
                      constraints: const BoxConstraints(maxWidth: 780),
                      child: body,
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _PackageRow extends StatelessWidget {
  const _PackageRow({required this.package, required this.rule});

  final _Package package;
  final bool rule;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final n = package.entries.length;
    return CRow(
      rule: rule,
      padding: wide
          ? const EdgeInsets.fromLTRB(20, 12, 16, 12)
          : const EdgeInsets.fromLTRB(18, 11, 12, 11),
      onTap: () => Navigator.of(context).push(MaterialPageRoute<void>(
        builder: (_) => _PackageLicenses(package: package),
      )),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(package.name,
                    style: t.title.copyWith(fontSize: wide ? 15.5 : 15)),
                Text('$n ${n == 1 ? 'license' : 'licenses'}',
                    style: t.caption.copyWith(fontSize: wide ? 13 : 12.5)),
              ],
            ),
          ),
          const SizedBox(width: 10),
          CGlyph(CGlyphs.forward, size: 18, color: t.ink, stroke: 2.1),
        ],
      ),
    );
  }
}

/// One package's licence texts, as paragraphs of prose with a hairline
/// between entries.  Built lazily: some packages' texts run to thousands of
/// lines.
class _PackageLicenses extends StatelessWidget {
  const _PackageLicenses({required this.package});

  final _Package package;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    final side = wide ? 34.0 : 16.0;
    final n = package.entries.length;
    // The rows of the list after its header: each entry's paragraphs, a
    // rule (null) between entries.
    final rows = <LicenseParagraph?>[
      for (var i = 0; i < n; i++) ...[
        if (i > 0) null,
        ...package.entries[i].paragraphs,
      ],
    ];
    final style = t.body.copyWith(fontSize: wide ? 13.5 : 13, height: 1.55);
    return Scaffold(
      body: SafeArea(
        child: ListView.builder(
          padding: EdgeInsets.only(bottom: wide ? 44 : 28),
          itemCount: rows.length + 1,
          itemBuilder: (context, i) {
            if (i == 0) {
              return CHeader(
                title: package.name,
                subtitle: '$n ${n == 1 ? 'license' : 'licenses'}',
                onBack: () => Navigator.of(context).pop(),
              );
            }
            final p = rows[i - 1];
            final Widget row;
            if (p == null) {
              row = Padding(
                padding: const EdgeInsets.symmetric(vertical: 18),
                child: Container(height: 1, color: t.line),
              );
            } else {
              final centred = p.indent == LicenseParagraph.centeredIndent;
              row = Padding(
                padding: EdgeInsets.only(
                    top: i == 1 ? (wide ? 14 : 12) : 10,
                    left: centred ? 0 : 6 + 16.0 * p.indent,
                    right: 6),
                child: Text(p.text,
                    style: style,
                    textAlign: centred ? TextAlign.center : TextAlign.start),
              );
            }
            return Padding(
              padding: EdgeInsets.symmetric(horizontal: side),
              child: Align(
                alignment: Alignment.topLeft,
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 780),
                  // The column's width, so a centred line is centred in it.
                  child: SizedBox(width: double.infinity, child: row),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}
