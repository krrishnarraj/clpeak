import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../services/benchmark_service.dart';
import '../../theme/xpupeak_theme.dart';
import '../app.dart';
import '../common/kit.dart';
import 'licenses_screen.dart';

/// What XpuPeak is, its version, its source and the licences of what it
/// is built with -- the bundled Sora among them (main.dart registers it).
class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  static final _repoUrl = Uri.parse('https://github.com/krrishnarraj/xpupeak');

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final wide = cIsWide(context);
    // The version alone: watching the service would rebuild this tab, kept
    // alive behind the others, on every tick of a run.
    final version = context.select<BenchmarkService, String>((s) => s.version);
    final side = wide ? 34.0 : 16.0;
    final name = t.title.copyWith(fontSize: wide ? 15.5 : 15);
    final line = t.caption.copyWith(fontSize: wide ? 13 : 12.5);
    final rowPadding = wide
        ? const EdgeInsets.fromLTRB(20, 14, 20, 14)
        : const EdgeInsets.fromLTRB(18, 12, 18, 12);

    Widget link({
      required String title,
      required String detail,
      required String glyph,
      required VoidCallback onTap,
      bool rule = true,
    }) =>
        CRow(
          rule: rule,
          padding: rowPadding,
          onTap: onTap,
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: name),
                    const SizedBox(height: 1),
                    Text(detail, style: line),
                  ],
                ),
              ),
              const SizedBox(width: 10),
              CGlyph(glyph, size: 18, color: t.ink, stroke: 2.1),
            ],
          ),
        );

    final sections = <Widget>[
      CPanel(
        padding: wide
            ? const EdgeInsets.fromLTRB(22, 22, 22, 22)
            : const EdgeInsets.fromLTRB(18, 18, 18, 18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CAppMark(size: wide ? 52 : 46),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('XpuPeak',
                          style: t.wordmark.copyWith(fontSize: wide ? 24 : 22)),
                      const SizedBox(height: 2),
                      Text('v$version',
                          style: line.copyWith(
                              fontFeatures: const [
                                FontFeature.tabularFigures()
                              ])),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),
            Text(
              'CPU, GPU and NPU peaks: the compute throughput, memory bandwidth and latency each one actually reaches. '
              "Small, tight kernels run alongside each vendor's own tuned libraries, through every API that exposes a device, so the results line up side by side.",
              style: t.body.copyWith(
                  fontSize: wide ? 15 : 14, height: 1.55, color: t.muted),
            ),
          ],
        ),
      ),
      Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const CSection(label: 'Project'),
          SizedBox(height: wide ? 10 : 8),
          CPanel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                link(
                  title: 'Source code',
                  detail: 'github.com/krrishnarraj/xpupeak',
                  glyph: CGlyphs.external,
                  onTap: () => launchUrl(_repoUrl,
                      mode: LaunchMode.externalApplication),
                ),
                link(
                  title: 'Open-source licenses',
                  detail: 'The packages XpuPeak is built with, and the Sora '
                      'typeface',
                  glyph: CGlyphs.forward,
                  rule: false,
                  onTap: () =>
                      Navigator.of(context).push(LicensesScreen.route(version)),
                ),
              ],
            ),
          ),
        ],
      ),
    ];

    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: EdgeInsets.only(bottom: wide ? 44 : 28),
          children: [
            const CHeader(title: 'About'),
            for (var i = 0; i < sections.length; i++)
              Padding(
                padding: EdgeInsets.fromLTRB(side,
                    i == 0 ? (wide ? 14 : 12) : (wide ? 26 : 20), side, 0),
                child: Align(
                  alignment: Alignment.topLeft,
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 780),
                    child: sections[i],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
