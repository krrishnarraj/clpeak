import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/benchmark_service.dart';
import '../services/settings_service.dart';
import '../theme/clpeak_theme.dart';
import 'about/about_screen.dart';
import 'dashboard/dashboard_screen.dart';
import 'history/history_screen.dart';
import 'live_run/live_run_screen.dart';
import 'results/results_screen.dart';

class ClpeakApp extends StatelessWidget {
  const ClpeakApp({super.key});

  @override
  Widget build(BuildContext context) {
    final settings = context.watch<SettingsService>();
    return MaterialApp(
      title: 'clpeak',
      debugShowCheckedModeBanner: false,
      theme: ClpeakTheme.light(),
      darkTheme: ClpeakTheme.dark(),
      themeMode: settings.themeMode,
      home: const ClpeakShell(),
    );
  }
}

/// Adaptive navigation shell: NavigationRail on wide layouts, NavigationBar
/// on phones.  The first tab hosts the whole run lifecycle
/// (dashboard → live run → results) so a run survives tab switches.
class ClpeakShell extends StatefulWidget {
  const ClpeakShell({super.key});

  @override
  State<ClpeakShell> createState() => _ClpeakShellState();
}

class _ClpeakShellState extends State<ClpeakShell> {
  int _tab = 0;

  static const _destinations = [
    (icon: Icons.speed_outlined, selected: Icons.speed, label: 'Benchmark'),
    (icon: Icons.history_outlined, selected: Icons.history, label: 'History'),
    (icon: Icons.info_outline, selected: Icons.info, label: 'About'),
  ];

  @override
  Widget build(BuildContext context) {
    final wide = MediaQuery.sizeOf(context).width >= 900;

    final content = IndexedStack(
      index: _tab,
      children: const [
        BenchmarkTab(),
        HistoryScreen(),
        AboutScreen(),
      ],
    );

    if (wide) {
      return Scaffold(
        body: Row(
          children: [
            NavigationRail(
              selectedIndex: _tab,
              onDestinationSelected: (i) => setState(() => _tab = i),
              labelType: NavigationRailLabelType.all,
              leading: const Padding(
                padding: EdgeInsets.only(top: 8, bottom: 12),
                child: _AppMark(),
              ),
              destinations: [
                for (final d in _destinations)
                  NavigationRailDestination(
                    icon: Icon(d.icon),
                    selectedIcon: Icon(d.selected),
                    label: Text(d.label),
                  ),
              ],
            ),
            const VerticalDivider(width: 1),
            Expanded(child: content),
          ],
        ),
      );
    }

    return Scaffold(
      body: content,
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (i) => setState(() => _tab = i),
        destinations: [
          for (final d in _destinations)
            NavigationDestination(
              icon: Icon(d.icon),
              selectedIcon: Icon(d.selected),
              label: d.label,
            ),
        ],
      ),
    );
  }
}

class _AppMark extends StatelessWidget {
  const _AppMark();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        color: scheme.primary.withValues(alpha: 0.15),
      ),
      child: Icon(Icons.bolt, color: scheme.primary),
    );
  }
}

/// Hosts the run lifecycle inside the first tab.
class BenchmarkTab extends StatelessWidget {
  const BenchmarkTab({super.key});

  @override
  Widget build(BuildContext context) {
    final service = context.watch<BenchmarkService>();
    return switch (service.state) {
      BenchmarkState.idle => const DashboardScreen(),
      BenchmarkState.running ||
      BenchmarkState.cancelling =>
        const LiveRunScreen(),
      BenchmarkState.finished => LiveResultsScreen(service: service),
    };
  }
}
