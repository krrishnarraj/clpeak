import '../model/catalog.dart';
import '../model/run_document.dart';
import '../model/run_summary.dart';
import 'run_history_store.dart';

/// Whether a run that stopped before it finished can be continued -- `clpeak
/// --resume`, which runs only what its checkpoint does not hold and saves
/// every part as one result (include/common/run_resume.h) -- and how the app
/// says what happened to it.  A run stops so when the app dies in it
/// ([CrashLog]), and when it is cancelled, which keeps its checkpoint beside
/// the result it saved.
///
/// A run is continued only as the same run: by the clpeak that started it,
/// on the same OS release, devices and runtimes.  The native side refuses
/// another build or OS release itself; the devices and runtimes are what the
/// app noted beside the run when it started it ([RunResumeState]).  History
/// offers every run that can be; the dashboard only the newest the app died
/// in, within a day, and while no newer run has started since.
class ResumeOffer {
  const ResumeOffer({
    required this.id,
    required this.startedAt,
    this.checkpoint,
    this.log,
    this.headline = '',
    this.canContinue = false,
    this.pending = false,
    this.whyNot = '',
    this.skipCrashed = false,
  });

  /// The run's id: its files in history are named by it.
  final String id;
  final DateTime startedAt;

  /// What it finished, when it left a checkpoint.
  final RunCheckpointInfo? checkpoint;

  /// The run's log, for a run the app died in; null for one that was
  /// cancelled.
  final CrashLog? log;

  /// What happened, in a sentence: "Android closed clpeak during CPU ·
  /// Memory latency (pointer-chase)."  Empty for a cancelled run.
  final String headline;

  final bool canContinue;

  /// The devices are still being detected, so it cannot be told yet.
  final bool pending;

  /// Why it cannot be continued, in a phrase: "clpeak was updated since".
  final String whyNot;

  /// The run died on screen -- a crash, not the phone or the user ending
  /// it -- so a continuation does not run the test it died in again
  /// (`--skip-crashed`): it would most likely crash the same way.
  final bool skipCrashed;

  /// Tests its checkpoint holds.
  int get testsSaved => checkpoint?.tests ?? 0;

  /// How long the dashboard offers a run: the parts of one run are best
  /// measured within a day of each other, not across days of changing
  /// conditions.
  static const window = Duration(hours: 24);

  /// A run the app died in.  [now] and [newerRunStarted] are the
  /// dashboard's: given, a run that stopped more than a [window] ago, or
  /// before another started, is not offered there.
  static ResumeOffer decide(
    CrashLog log, {
    required String appVersion,
    required String osVersion,
    required String? catalog,
    DateTime? now,
    DateTime? newerRunStarted,
  }) {
    final reason = exitReason(log);
    final skip = !log.diedOffScreen &&
        reason != 'USER_REQUESTED' &&
        reason != 'USER_STOPPED';
    final offer = _decide(
      ResumeOffer(
          id: log.id,
          startedAt: log.startedAt,
          checkpoint: log.checkpoint,
          log: log,
          headline: '${who(log, reason)}${where(log)}.',
          skipCrashed: skip),
      log.resumeState,
      appVersion: appVersion,
      osVersion: osVersion,
      catalog: catalog,
    );
    final checkpoint = log.checkpoint;
    if (!offer.canContinue || checkpoint == null) return offer;
    if (now != null && now.difference(checkpoint.writtenAt) > window) {
      return offer._no('it stopped more than a day ago');
    }
    if (newerRunStarted != null && newerRunStarted.isAfter(log.startedAt)) {
      return offer._no('a newer run has started since');
    }
    return offer;
  }

  /// A run that was cancelled: its result holds what it measured, and the
  /// checkpoint it kept, what it finished.
  static ResumeOffer decideCancelled(
    RunSummary run, {
    required RunCheckpointInfo? checkpoint,
    required RunResumeState? state,
    required String appVersion,
    required String osVersion,
    required String? catalog,
  }) =>
      _decide(
        ResumeOffer(
            id: run.id, startedAt: run.startedAt, checkpoint: checkpoint),
        state,
        appVersion: appVersion,
        osVersion: osVersion,
        catalog: catalog,
      );

  // Whether `run` -- not yet continuable -- is the same run here and now.
  static ResumeOffer _decide(
    ResumeOffer run,
    RunResumeState? state, {
    required String appVersion,
    required String osVersion,
    required String? catalog,
  }) {
    final checkpoint = run.checkpoint;
    if (checkpoint == null) {
      return run._no('it saved nothing to continue from');
    }
    if (checkpoint.clpeakVersion != appVersion) {
      return run._no('clpeak was updated since it started');
    }
    if (state != null && state.os.isNotEmpty && state.os != osVersion) {
      return run._no('the system was updated since it started');
    }
    if (catalog == null) return run._with(pending: true);
    if (state != null && state.catalog.isNotEmpty && state.catalog != catalog) {
      return run._no('the devices or runtimes clpeak finds have changed since');
    }
    return run._with(canContinue: true);
  }

  ResumeOffer _no(String why) => _with(whyNot: why);

  ResumeOffer _with({
    bool canContinue = false,
    bool pending = false,
    String whyNot = '',
  }) =>
      ResumeOffer(
        id: id,
        startedAt: startedAt,
        checkpoint: checkpoint,
        log: log,
        headline: headline,
        canContinue: canContinue,
        pending: pending,
        whyNot: whyNot,
        skipCrashed: skipCrashed,
      );

  /// Android's reason for the process's end (ApplicationExitInfo.REASON_*
  /// without the prefix), from the line the app adds to the run's log, or
  /// null where there is none -- iOS, the desktop, an older record.
  static String? exitReason(CrashLog log) {
    final m = RegExp(r'^process ended: ([A-Z_]+)')
        .firstMatch(log.processExit?.message ?? '');
    return m?.group(1);
  }

  /// Who ended the run, as the person reading the offer would put it.
  static String who(CrashLog log, String? reason) => switch (reason) {
        'USER_REQUESTED' || 'USER_STOPPED' => 'You closed clpeak',
        'CRASH_NATIVE' || 'CRASH' => 'clpeak crashed',
        'ANR' => 'clpeak stopped responding',
        'LOW_MEMORY' when !log.diedOffScreen =>
          'The phone ran out of memory',
        null => log.diedOffScreen ? 'clpeak was closed' : 'clpeak stopped',
        _ => log.diedOffScreen ? 'Android closed clpeak' : 'clpeak stopped',
      };

  /// " during CPU · Memory latency (pointer-chase)", when the log says.
  static String where(CrashLog log) {
    if (log.lastTestTitle.isEmpty) return '';
    final backend =
        log.lastTestBackend.isEmpty ? '' : '${log.lastTestBackend} · ';
    return ' during $backend${log.lastTestTitle}';
  }

  /// A device catalog as a run notes it, one line per usable backend: its
  /// name, its runtime facts (the ONNX Runtime version, the OS release) and
  /// its devices by index.  A continuation runs on the same devices and
  /// runtimes when this matches.
  static String fingerprint(BackendCatalog catalog) => [
        for (final b in catalog.usable)
          [
            b.name,
            b.info,
            for (final p in b.platforms)
              for (final d in p.devices) '${d.index}:${d.name}',
          ].join('|'),
      ].join('\n');

  /// How many devices a run's command line selects: its `--devices` list,
  /// or every device of the catalog when it names none.
  static int plannedDevices(List<String> argv, BackendCatalog catalog) {
    final at = argv.indexOf('--devices');
    if (at >= 0 && at + 1 < argv.length) {
      return argv[at + 1].split(',').where((s) => s.isNotEmpty).length;
    }
    return catalog.usable.fold(0, (n, b) => n + b.deviceCount);
  }

  /// The run's continuation, once it is one: the latest part's start and
  /// what ended the part before, for the results header.
  static ResumedPart? latestPart(RunDocument doc) {
    final parts = doc.meta?.resumed ?? const [];
    return parts.isEmpty ? null : parts.last;
  }
}
