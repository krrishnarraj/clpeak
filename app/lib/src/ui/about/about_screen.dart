import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../services/benchmark_service.dart';
import '../../services/settings_service.dart';
import '../../theme/clpeak_theme.dart';
import '../app.dart';
import '../common/kit.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  static final _repoUrl = Uri.parse('https://github.com/krrishnarraj/clpeak');

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final service = context.watch<BenchmarkService>();
    final settings = context.watch<SettingsService>();

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            const CHeader(title: 'About', subtitle: 'clpeak'),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 40),
                children: [
                  CPanel(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            const CAppMark(size: 34),
                            const SizedBox(width: 12),
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('clpeak', style: t.wordmark),
                                const SizedBox(height: 3),
                                Text('v${service.version}',
                                    style: t.micro
                                        .copyWith(letterSpacing: 0.6)),
                              ],
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'A synthetic micro-benchmark for measuring the peak achievable compute performance of CPUs and GPUs. '
                          'It exercises tight vector, MAD, and MMA kernels, together with vendor-optimized GEMM libraries, to expose peak hardware throughput.',
                          style: t.body,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 22),
                  const CSection(label: 'Settings'),
                  const SizedBox(height: 10),
                  CPanel(
                    child: CRow(
                      rule: false,
                      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
                      child: Row(
                        children: [
                          Icon(Icons.dark_mode_outlined,
                              size: 15, color: t.dim),
                          const SizedBox(width: 10),
                          Expanded(child: Text('Theme', style: t.mono)),
                          _ThemeToggle(
                            value: settings.themeMode,
                            onChanged: settings.setThemeMode,
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 22),
                  const CSection(label: 'Project'),
                  const SizedBox(height: 10),
                  CPanel(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        CRow(
                          onTap: () => launchUrl(_repoUrl,
                              mode: LaunchMode.externalApplication),
                          child: Row(
                            children: [
                              Icon(Icons.code, size: 15, color: t.dim),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment:
                                      CrossAxisAlignment.start,
                                  children: [
                                    Text('Source code', style: t.mono),
                                    const SizedBox(height: 3),
                                    Text('github.com/krrishnarraj/clpeak',
                                        style: t.monoSmallDim),
                                  ],
                                ),
                              ),
                              Icon(Icons.open_in_new,
                                  size: 13, color: t.faint),
                            ],
                          ),
                        ),
                        CRow(
                          rule: false,
                          onTap: () => showLicensePage(
                            context: context,
                            applicationName: 'clpeak',
                            applicationVersion: service.version,
                          ),
                          child: Row(
                            children: [
                              Icon(Icons.description_outlined,
                                  size: 15, color: t.dim),
                              const SizedBox(width: 10),
                              Expanded(
                                  child: Text('Open-source licenses',
                                      style: t.mono)),
                              Icon(Icons.chevron_right,
                                  size: 15, color: t.faint),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Square segmented control — hairline frame, solid block on the active cell.
class _ThemeToggle extends StatelessWidget {
  const _ThemeToggle({required this.value, required this.onChanged});

  final ThemeMode value;
  final ValueChanged<ThemeMode> onChanged;

  static const _modes = [
    (ThemeMode.system, 'Auto'),
    (ThemeMode.light, 'Light'),
    (ThemeMode.dark, 'Dark'),
  ];

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: t.line),
        borderRadius: BorderRadius.circular(CP.rControl),
      ),
      clipBehavior: Clip.antiAlias,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          for (var i = 0; i < _modes.length; i++)
            CTap(
              onTap: () => onChanged(_modes[i].$1),
              builder: (context, hovered, pressed) {
                final on = value == _modes[i].$1;
                return Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                  decoration: BoxDecoration(
                    color: on
                        ? t.inverse
                        : (hovered || pressed
                            ? t.hover
                            : Colors.transparent),
                    border: Border(
                      left: i == 0
                          ? BorderSide.none
                          : BorderSide(color: t.line),
                    ),
                  ),
                  child: Text(
                    _modes[i].$2.toUpperCase(),
                    style: t.micro
                        .copyWith(color: on ? t.onInverse : t.dim),
                  ),
                );
              },
            ),
        ],
      ),
    );
  }
}
