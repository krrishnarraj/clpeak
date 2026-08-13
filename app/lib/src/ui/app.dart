import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/benchmark_service.dart';
import '../services/settings_service.dart';
import '../theme/clpeak_theme.dart';
import 'about/about_screen.dart';
import 'common/kit.dart';
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

/// Adaptive navigation shell: a console sidebar on wide layouts, a hairline
/// tab strip on phones.  The first tab hosts the whole run lifecycle
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
    final t = CP.of(context);
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
            _Sidebar(
              selected: _tab,
              destinations: _destinations,
              onSelected: (i) => setState(() => _tab = i),
            ),
            Container(width: 1, color: t.line),
            Expanded(child: content),
          ],
        ),
      );
    }

    return Scaffold(
      body: content,
      bottomNavigationBar: _TabStrip(
        selected: _tab,
        destinations: _destinations,
        onSelected: (i) => setState(() => _tab = i),
      ),
    );
  }
}

typedef _Destination = ({IconData icon, IconData selected, String label});

/// Wide-layout navigation: a fixed console column.  Selection is a solid bar
/// on the left edge plus full-strength text — no pill, no tint.
class _Sidebar extends StatelessWidget {
  const _Sidebar({
    required this.selected,
    required this.destinations,
    required this.onSelected,
  });

  final int selected;
  final List<_Destination> destinations;
  final ValueChanged<int> onSelected;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final version = context.select<BenchmarkService, String>((s) => s.version);

    return Container(
      width: 196,
      color: t.bg,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 20, 16, 16),
            child: Row(
              children: [
                const CAppMark(size: 22),
                const SizedBox(width: 9),
                Text('clpeak',
                    style: t.title.copyWith(fontSize: 15, letterSpacing: -0.6)),
              ],
            ),
          ),
          Container(height: 1, color: t.line),
          const SizedBox(height: 8),
          for (var i = 0; i < destinations.length; i++)
            _NavItem(
              destination: destinations[i],
              selected: i == selected,
              onTap: () => onSelected(i),
            ),
          const Spacer(),
          Container(height: 1, color: t.line),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
            child: Text('v$version',
                style: t.micro.copyWith(letterSpacing: 0.6),
                maxLines: 1,
                overflow: TextOverflow.ellipsis),
          ),
        ],
      ),
    );
  }
}

class _NavItem extends StatelessWidget {
  const _NavItem({
    required this.destination,
    required this.selected,
    required this.onTap,
  });

  final _Destination destination;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onTap,
      builder: (context, hovered, pressed) => Container(
        height: 38,
        color: selected
            ? t.hover
            : (hovered || pressed ? t.hover.withValues(alpha: 0.6)
                                  : Colors.transparent),
        child: Row(
          children: [
            Container(
              width: 2,
              height: 38,
              color: selected ? t.text : Colors.transparent,
            ),
            const SizedBox(width: 14),
            Icon(selected ? destination.selected : destination.icon,
                size: 15, color: selected ? t.text : t.dim),
            const SizedBox(width: 10),
            Text(
              destination.label,
              style: t.micro.copyWith(
                fontSize: 11,
                letterSpacing: 0.9,
                color: selected ? t.text : t.dim,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Phone-layout navigation: a hairline strip, selection marked by a rule above
/// the item rather than a Material pill.
class _TabStrip extends StatelessWidget {
  const _TabStrip({
    required this.selected,
    required this.destinations,
    required this.onSelected,
  });

  final int selected;
  final List<_Destination> destinations;
  final ValueChanged<int> onSelected;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      decoration: BoxDecoration(
        color: t.panel,
        border: Border(top: BorderSide(color: t.line)),
      ),
      child: SafeArea(
        top: false,
        child: SizedBox(
          height: 56,
          child: Row(
            children: [
              for (var i = 0; i < destinations.length; i++)
                Expanded(
                  child: CTap(
                    onTap: () => onSelected(i),
                    builder: (context, hovered, pressed) {
                      final on = i == selected;
                      return Container(
                        color: pressed ? t.hover : Colors.transparent,
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              width: 18,
                              height: 2,
                              color: on ? t.text : Colors.transparent,
                            ),
                            const SizedBox(height: 8),
                            Icon(
                                on
                                    ? destinations[i].selected
                                    : destinations[i].icon,
                                size: 17,
                                color: on ? t.text : t.dim),
                            const SizedBox(height: 5),
                            Text(
                              destinations[i].label,
                              style: t.micro.copyWith(
                                fontSize: 9.5,
                                letterSpacing: 0.8,
                                color: on ? t.text : t.dim,
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// The wordmark glyph: a solid block with the bolt knocked out of it.
class CAppMark extends StatelessWidget {
  const CAppMark({super.key, this.size = 22});

  final double size;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      width: size,
      height: size,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: t.inverse,
        borderRadius: BorderRadius.circular(CP.rControl),
      ),
      child: Icon(Icons.bolt, size: size * 0.68, color: t.onInverse),
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
