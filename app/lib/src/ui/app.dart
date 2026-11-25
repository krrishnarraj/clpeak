import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:provider/provider.dart';

import '../model/run_config.dart';
import '../model/run_summary.dart';
import '../services/benchmark_service.dart';
import '../services/run_history_store.dart';
import '../services/settings_service.dart';
import '../theme/xpupeak_theme.dart';
import 'about/about_screen.dart';
import 'common/format.dart';
import 'common/kit.dart';
import 'common/run_route.dart';
import 'dashboard/dashboard_screen.dart';
import 'history/history_screen.dart';
import 'live_run/live_run_screen.dart';
import 'results/results_screen.dart';
import 'run_config/run_config_screen.dart';
import 'settings/settings_screen.dart';

class XpuPeakApp extends StatelessWidget {
  const XpuPeakApp({super.key});

  @override
  Widget build(BuildContext context) {
    // The theme alone: the settings' other changes (the runs opened last)
    // are the shell's to show.
    final mode = context.select<SettingsService, ThemeMode>((s) => s.themeMode);
    return MaterialApp(
      title: 'XpuPeak',
      debugShowCheckedModeBanner: false,
      theme: XpuPeakTheme.light(),
      darkTheme: XpuPeakTheme.dark(),
      themeMode: mode,
      home: const XpuPeakShell(),
    );
  }
}

/// The app's frame.  Desktop (from [cWideWidth]): a sidebar -- the
/// wordmark, the four destinations with History's count of saved runs, the
/// runs opened last and "All history" -- beside the content.  Phone: the
/// content over a bottom bar of the same four destinations.
///
/// The content is a navigator of its own.  Its first route holds the four
/// destinations in an IndexedStack, so each keeps its state, and the first
/// hosts the whole run lifecycle (dashboard → live run → results) so a run
/// survives switching.  What a screen pushes -- a saved run's results, a
/// compare, a custom run -- opens over them inside the content: beside the
/// sidebar on desktop; over the bottom bar on a phone, where back returns.
/// Choosing a destination closes what was pushed.
class XpuPeakShell extends StatefulWidget {
  const XpuPeakShell({super.key});

  @override
  State<XpuPeakShell> createState() => _XpuPeakShellState();
}

class _XpuPeakShellState extends State<XpuPeakShell> {
  static const _destinations = [
    (glyph: CGlyphs.bench, label: 'Benchmark'),
    (glyph: CGlyphs.history, label: 'History'),
    (glyph: CGlyphs.settings, label: 'Settings'),
    (glyph: CGlyphs.about, label: 'About'),
  ];

  static const _history = 1;

  static const _screens = <Widget>[
    BenchmarkTab(),
    HistoryScreen(),
    SettingsScreen(),
    AboutScreen(),
  ];

  final _tab = ValueNotifier<int>(0);

  /// Keeps the content -- its navigator and every route on it -- across a
  /// switch between the desktop and phone layouts.
  final _contentKey = GlobalKey();
  final _navigatorKey = GlobalKey<NavigatorState>();
  late final _pages =
      _PageStack(onChanged: _pagesChanged, onRunOpened: _noteOpened);

  /// A run was in flight at the last build: a run that starts while another
  /// tab is showing -- Continue in History -- is watched where runs are.
  bool _wasRunning = false;

  late int _seenRunsEnded;
  late final RunHistoryStore _store;

  /// The saved runs, as the index lists them; null until read.
  List<RunSummary>? _runs;
  Map<String, RunOrigin> _origins = const {};
  int _readGeneration = 0;

  @override
  void initState() {
    super.initState();
    _store = context.read<RunHistoryStore>();
    _store.changes.addListener(_readRuns);
    _seenRunsEnded = context.read<BenchmarkService>().runsEnded;
    _readRuns();
  }

  @override
  void dispose() {
    _store.changes.removeListener(_readRuns);
    _tab.dispose();
    super.dispose();
  }

  Future<void> _readRuns() async {
    final generation = ++_readGeneration;
    try {
      final runs = await _store.indexed();
      final origins = await _store.origins();
      if (!mounted || generation != _readGeneration) return;
      setState(() {
        _runs = runs;
        _origins = origins;
      });
    } on Object {
      // The sidebar keeps what it showed: History is where a run that
      // cannot be read says so.
    }
  }

  void _pagesChanged() {
    if (mounted) setState(() {});
  }

  void _noteOpened(String id) {
    if (mounted) context.read<SettingsService>().noteRunOpened(id);
  }

  /// Show destination [i], closing whatever was pushed over the tabs --
  /// at once, with no transition.
  void _show(int i) {
    final navigator = _navigatorKey.currentState;
    if (navigator != null) {
      for (final route in _pages.routes.reversed.toList()) {
        if (route.isActive) navigator.removeRoute(route);
      }
    }
    if (_tab.value != i) setState(() => _tab.value = i);
  }

  /// Open a recent run's results, over History.
  Future<void> _openRun(RunSummary summary, {required String? liveResult}) async {
    if (summary.id == liveResult) {
      // The run that just finished is the Benchmark tab's.
      _show(0);
      return;
    }
    if (summary.id == _pages.topRunId) return;
    final messenger = ScaffoldMessenger.of(context);
    final doc = await _store.load(summary);
    if (!mounted) return;
    if (doc == null) {
      messenger.showSnackBar(
          const SnackBar(content: Text('Could not read this result file.')));
      return;
    }
    _show(_history);
    unawaited(_navigatorKey.currentState?.push(MaterialPageRoute<void>(
      settings: savedRunRoute(summary.id),
      builder: (_) => SavedResultsScreen(document: doc, summary: summary),
    )));
  }

  /// The sidebar's Run benchmark: the dashboard's own Run, a full run of
  /// every detected device -- watched in the Benchmark tab, which the shell
  /// shows as the run starts.
  void _runBenchmark() {
    context.read<BenchmarkService>().start(
        preset: RunPreset.full,
        verbose: context.read<SettingsService>().verbose);
  }

  /// The sidebar's Customize: the custom run, over the Benchmark tab as the
  /// dashboard opens it -- unless it is open already.
  void _customize() {
    final routes = _pages.routes;
    if (routes.isNotEmpty &&
        routes.last.settings.name == RunConfigScreen.routeName) {
      return;
    }
    _show(0);
    unawaited(_navigatorKey.currentState?.push(RunConfigScreen.route()));
  }

  Route<dynamic>? _route(RouteSettings settings) =>
      settings.name != Navigator.defaultRouteName
          ? null
          : PageRouteBuilder<void>(
              settings: settings,
              pageBuilder: (context, _, _) => ValueListenableBuilder<int>(
                valueListenable: _tab,
                builder: (context, tab, _) =>
                    IndexedStack(index: tab, children: _screens),
              ),
              transitionDuration: Duration.zero,
              reverseTransitionDuration: Duration.zero,
            );

  @override
  Widget build(BuildContext context) {
    final wide = cIsWide(context);
    final running = context.select<BenchmarkService, bool>((s) => s.isRunning);
    // The sidebar's two pills, as the dashboard offers them: once devices
    // have been found, and not while a run is in flight (the Benchmark tab
    // shows the run then, not the dashboard).  A bool, so the shell is not
    // rebuilt on every tick of a run.
    final canRun = context.select<BenchmarkService, bool>((s) =>
        s.catalogReady &&
        !s.isRunning &&
        s.catalog.backends.any((b) => b.available && b.hasDevices));
    if (running && !_wasRunning) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _show(0);
      });
    }
    _wasRunning = running;

    // A run that ended with its results saved has been opened: they are on
    // screen in the Benchmark tab.
    final ended = context.select<BenchmarkService, int>((s) => s.runsEnded);
    if (ended != _seenRunsEnded) {
      _seenRunsEnded = ended;
      final saved = context.read<BenchmarkService>().lastSummary;
      if (saved != null) {
        WidgetsBinding.instance.addPostFrameCallback((_) => _noteOpened(saved.id));
      }
    }
    final liveResult = context.select<BenchmarkService, String?>((s) =>
        s.state == BenchmarkState.finished ? s.lastSummary?.id : null);
    final recentIds =
        context.select<SettingsService, List<String>>((s) => s.recentRunIds);

    final content = KeyedSubtree(
      key: _contentKey,
      child: NavigatorPopHandler<Object?>(
        onPopWithResult: (_) => _navigatorKey.currentState?.maybePop(),
        child: Navigator(
          key: _navigatorKey,
          observers: [_pages],
          onGenerateRoute: _route,
        ),
      ),
    );

    if (!wide) {
      return Scaffold(
        body: content,
        // A page over the tabs has its own way back.
        bottomNavigationBar: _pages.hasPages
            ? null
            : _BottomBar(
                selected: _tab.value,
                destinations: _destinations,
                onSelected: _show,
              ),
      );
    }

    final runs = _runs;
    final byId = {for (final r in runs ?? const <RunSummary>[]) r.id: r};
    // The open run: the one a page over the tabs shows, or the one the
    // Benchmark tab has just finished.
    final open = _pages.hasPages
        ? _pages.topRunId
        : (_tab.value == 0 ? liveResult : null);
    return Scaffold(
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _Sidebar(
            selected: _tab.value,
            destinations: _destinations,
            onSelected: _show,
            historyCount: runs?.length,
            recent: [
              for (final id in recentIds)
                if (byId[id] case final run?)
                  (
                    run: run,
                    origin: _origins[run.id],
                    open: run.id == open,
                  ),
            ],
            onOpenRun: (run) => _openRun(run, liveResult: liveResult),
            onRun: canRun ? _runBenchmark : null,
            onCustomize: canRun ? _customize : null,
          ),
          Expanded(child: content),
        ],
      ),
    );
  }
}

/// The routes pushed over the shell's tabs, as its navigator reports them:
/// whether a page covers the tabs, which saved run the top page shows, and
/// each saved run opened ([savedRunRoute]).
class _PageStack extends NavigatorObserver {
  _PageStack({required this.onChanged, required this.onRunOpened});

  final VoidCallback onChanged;
  final ValueChanged<String> onRunOpened;

  /// Every route over the tabs, bottom to top: pages, and the menus and
  /// sheets over them.
  final routes = <Route<dynamic>>[];

  Iterable<PageRoute<dynamic>> get _pages => routes.whereType<PageRoute<dynamic>>();

  bool get hasPages => _pages.isNotEmpty;

  String? get topRunId => hasPages ? savedRunOf(_pages.last) : null;

  static bool _isTabs(Route<dynamic> route) =>
      route.settings.name == Navigator.defaultRouteName;

  @override
  void didPush(Route<dynamic> route, Route<dynamic>? previousRoute) {
    if (_isTabs(route)) return;
    routes.add(route);
    final run = savedRunOf(route);
    _soon(() {
      if (run != null) onRunOpened(run);
      onChanged();
    });
  }

  @override
  void didPop(Route<dynamic> route, Route<dynamic>? previousRoute) =>
      _drop(route);

  @override
  void didRemove(Route<dynamic> route, Route<dynamic>? previousRoute) =>
      _drop(route);

  @override
  void didReplace({Route<dynamic>? newRoute, Route<dynamic>? oldRoute}) {
    final i = oldRoute == null ? -1 : routes.indexOf(oldRoute);
    if (newRoute == null || _isTabs(newRoute)) {
      if (i >= 0) routes.removeAt(i);
    } else if (i >= 0) {
      routes[i] = newRoute;
    } else {
      routes.add(newRoute);
    }
    _soon(onChanged);
  }

  void _drop(Route<dynamic> route) {
    if (routes.remove(route)) _soon(onChanged);
  }

  /// Now, unless a frame is being built: a navigator reports some changes
  /// mid-build, where the shell cannot rebuild.
  static void _soon(VoidCallback f) {
    if (SchedulerBinding.instance.schedulerPhase == SchedulerPhase.idle) {
      f();
    } else {
      scheduleMicrotask(f);
    }
  }
}

typedef _Destination = ({String glyph, String label});

typedef _Recent = ({RunSummary run, RunOrigin? origin, bool open});

/// Desktop navigation: the wordmark, the app's two run actions, the
/// destinations, and the runs opened last -- the open one marked -- with a
/// way to all of them.
class _Sidebar extends StatelessWidget {
  const _Sidebar({
    required this.selected,
    required this.destinations,
    required this.onSelected,
    required this.historyCount,
    required this.recent,
    required this.onOpenRun,
    required this.onRun,
    required this.onCustomize,
  });

  final int selected;
  final List<_Destination> destinations;
  final ValueChanged<int> onSelected;

  /// Saved runs, once read.
  final int? historyCount;
  final List<_Recent> recent;
  final ValueChanged<RunSummary> onOpenRun;

  /// Run benchmark and Customize; null while they are not on offer.
  final VoidCallback? onRun;
  final VoidCallback? onCustomize;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      width: 264,
      decoration: BoxDecoration(
        color: t.side,
        border: Border(right: BorderSide(color: t.line)),
      ),
      child: SafeArea(
        right: false,
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(18, 24, 18, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(6, 0, 6, 0),
                child: Row(
                  children: [
                    const CAppMark(),
                    const SizedBox(width: 11),
                    Text('XpuPeak', style: t.wordmark),
                  ],
                ),
              ),
              const SizedBox(height: 26),
              CButton(
                label: 'Run benchmark',
                glyph: CGlyphs.play,
                kind: CButtonKind.primary,
                stretch: true,
                onPressed: onRun,
              ),
              const SizedBox(height: 10),
              // The width of the column, at the secondary pill's height.
              SizedBox(
                height: 46,
                child: CButton(label: 'Customize', onPressed: onCustomize),
              ),
              const SizedBox(height: 26),
              for (var i = 0; i < destinations.length; i++) ...[
                if (i > 0) const SizedBox(height: 2),
                _NavItem(
                  destination: destinations[i],
                  selected: i == selected,
                  count: i == _XpuPeakShellState._history &&
                          (historyCount ?? 0) > 0
                      ? '$historyCount'
                      : null,
                  onTap: () => onSelected(i),
                ),
              ],
              if (recent.isNotEmpty) ...[
                Padding(
                  padding: const EdgeInsets.fromLTRB(12, 30, 12, 4),
                  child: Text('RECENTLY OPENED',
                      style: t.capsHead
                          .copyWith(fontSize: 11.5, fontWeight: FontWeight.w700)),
                ),
                for (final r in recent) ...[
                  const SizedBox(height: 10),
                  _RecentRun(recent: r, onTap: () => onOpenRun(r.run)),
                ],
                const SizedBox(height: 10),
                _AllHistory(onTap: () => onSelected(_XpuPeakShellState._history)),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

class _NavItem extends StatelessWidget {
  const _NavItem({
    required this.destination,
    required this.selected,
    required this.count,
    required this.onTap,
  });

  final _Destination destination;
  final bool selected;
  final String? count;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final fg = selected ? t.ink : t.muted;
    return CTap(
      onTap: onTap,
      selected: selected,
      builder: (context, hovered, pressed) => Container(
        height: 44,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        decoration: BoxDecoration(
          color: selected
              ? t.surface2
              : (hovered || pressed
                  ? t.surface2.withValues(alpha: 0.6)
                  : Colors.transparent),
          borderRadius: BorderRadius.circular(CP.rControl),
        ),
        child: Row(
          children: [
            CGlyph(destination.glyph, size: 20, color: fg),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                destination.label,
                style: t.label.copyWith(
                  fontWeight: selected ? FontWeight.w700 : FontWeight.w600,
                  color: fg,
                ),
              ),
            ),
            if (count != null)
              CCount(count!, fill: selected ? t.ground : t.surface2),
          ],
        ),
      ),
    );
  }
}

/// One of the runs opened last: its name, its day and where it came from.
class _RecentRun extends StatelessWidget {
  const _RecentRun({required this.recent, required this.onTap});

  final _Recent recent;
  final VoidCallback onTap;

  /// This machine, as its owner calls it.
  static String get _here => Platform.isMacOS
      ? 'this Mac'
      : Platform.isWindows
          ? 'this PC'
          : Platform.isLinux
              ? 'this computer'
              : 'this device';

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    final run = recent.run;
    final name = run.name.isNotEmpty
        ? run.name
        : (run.systemName.isNotEmpty ? run.systemName : run.displayTitle);
    final origin = switch (recent.origin) {
      RunOrigin.measured => ' · $_here',
      RunOrigin.imported => ' · imported',
      null => '',
    };
    return CTap(
      onTap: onTap,
      selected: recent.open,
      focusRadius: BorderRadius.circular(CP.rRow),
      builder: (context, hovered, pressed) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
        decoration: BoxDecoration(
          color: recent.open
              ? t.surface2
              : (hovered || pressed
                  ? t.surface2.withValues(alpha: 0.6)
                  : Colors.transparent),
          borderRadius: BorderRadius.circular(CP.rRow),
        ),
        child: Row(
          children: [
            Container(
              width: 7,
              height: 7,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: recent.open ? t.ink : Colors.transparent,
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: t.body.copyWith(fontWeight: FontWeight.w700)),
                  Text('${formatDay(run.startedAt)}$origin',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: t.smallMuted),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _AllHistory extends StatelessWidget {
  const _AllHistory({required this.onTap});

  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return CTap(
      onTap: onTap,
      builder: (context, hovered, pressed) => Container(
        height: 44,
        padding: const EdgeInsets.fromLTRB(29, 0, 12, 0),
        decoration: BoxDecoration(
          color: hovered || pressed
              ? t.surface2.withValues(alpha: 0.6)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(CP.rControl),
        ),
        child: Row(
          children: [
            Text('All history',
                style: t.label.copyWith(fontSize: 13.5, color: t.ink)),
            const SizedBox(width: 6),
            CGlyph(CGlyphs.forward, size: 16, color: t.ink, stroke: 2.2),
          ],
        ),
      ),
    );
  }
}

/// Phone navigation: the four destinations under a hairline, the selected
/// one in ink.
class _BottomBar extends StatelessWidget {
  const _BottomBar({
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
        color: t.ground,
        border: Border(top: BorderSide(color: t.line)),
      ),
      child: SafeArea(
        top: false,
        child: SizedBox(
          height: 66,
          child: Row(
            children: [
              for (var i = 0; i < destinations.length; i++)
                Expanded(
                  child: CTap(
                    onTap: () => onSelected(i),
                    selected: i == selected,
                    insetFocus: true,
                    builder: (context, hovered, pressed) {
                      final on = i == selected;
                      final fg = on ? t.ink : t.muted;
                      return Container(
                        color: pressed ? t.surface2 : Colors.transparent,
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            CGlyph(destinations[i].glyph, size: 22, color: fg),
                            const SizedBox(height: 4),
                            Text(
                              destinations[i].label,
                              style: t.small.copyWith(
                                fontWeight:
                                    on ? FontWeight.w700 : FontWeight.w600,
                                color: fg,
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

/// The app mark: an ink tile with the peak line knocked out of it.
class CAppMark extends StatelessWidget {
  const CAppMark({super.key, this.size = 38});

  final double size;

  @override
  Widget build(BuildContext context) {
    final t = CP.of(context);
    return Container(
      width: size,
      height: size,
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: t.ink,
        borderRadius: BorderRadius.circular(size * 13 / 38),
      ),
      child: CGlyph(CGlyphs.peak,
          size: size * 22 / 38, color: t.inkOn, stroke: 2.3),
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
