import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../services/benchmark_service.dart';
import '../../services/settings_service.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  static final _repoUrl = Uri.parse('https://github.com/krrishnarraj/clpeak');

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    final settings = context.watch<SettingsService>();
    final scheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(title: const Text('About')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 44,
                        height: 44,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          color: scheme.primary.withValues(alpha: 0.15),
                        ),
                        child: Icon(Icons.bolt, color: scheme.primary),
                      ),
                      const SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('clpeak',
                              style: Theme.of(context)
                                  .textTheme
                                  .titleLarge
                                  ?.copyWith(fontWeight: FontWeight.w700)),
                          Text('v${service.version}',
                              style: Theme.of(context)
                                  .textTheme
                                  .bodySmall
                                  ?.copyWith(color: scheme.outline)),
                        ],
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Text(
                    'A synthetic benchmark that measures peak compute '
                    '(fp16/fp32/fp64/integer), memory bandwidth, transfer '
                    'bandwidth, and kernel latency across OpenCL, Vulkan, '
                    'CUDA, ROCm, Metal, oneAPI and native CPU backends.',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Column(
              children: [
                ListTile(
                  leading: const Icon(Icons.dark_mode_outlined),
                  title: const Text('Theme'),
                  trailing: SegmentedButton<ThemeMode>(
                    showSelectedIcon: false,
                    segments: const [
                      ButtonSegment(
                          value: ThemeMode.dark, label: Text('Dark')),
                      ButtonSegment(
                          value: ThemeMode.light, label: Text('Light')),
                      ButtonSegment(
                          value: ThemeMode.system, label: Text('Auto')),
                    ],
                    selected: {settings.themeMode},
                    onSelectionChanged: (s) =>
                        settings.setThemeMode(s.first),
                  ),
                ),
                const Divider(),
                ListTile(
                  leading: const Icon(Icons.code),
                  title: const Text('Source code'),
                  subtitle: const Text('github.com/krrishnarraj/clpeak'),
                  trailing: const Icon(Icons.open_in_new, size: 18),
                  onTap: () =>
                      launchUrl(_repoUrl, mode: LaunchMode.externalApplication),
                ),
                const Divider(),
                ListTile(
                  leading: const Icon(Icons.description_outlined),
                  title: const Text('Open-source licenses'),
                  onTap: () => showLicensePage(
                    context: context,
                    applicationName: 'clpeak',
                    applicationVersion: service.version,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
