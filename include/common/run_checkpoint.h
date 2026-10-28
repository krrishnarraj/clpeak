#ifndef CLPEAK_RUN_CHECKPOINT_H
#define CLPEAK_RUN_CHECKPOINT_H

#include <chrono>
#include <set>
#include <string>

#include <common/benchmark_enums.h>
#include <common/run_document.h>

// ── The run's checkpoint ───────────────────────────────────────────────────
//
// A phone ends an app it has frozen off screen once it wants the memory, a
// user swipes it away, a driver crashes the process -- and a run can take
// most of an hour.  So with -o the finished part of the run is kept on disk
// as it goes: `<output>.partial.json` (`.json` swapped for `.partial.json`),
// a v3 document of every Benchmark finished so far with schema
// "clpeak/run-partial", no log (the sidecar keeps that, run_log.h), and the
// run's progress: the backends finished, and per device the whole device or
// the Benchmarks finished on it.
//
// It is rewritten whole, through a temporary file and a rename so that a
// death mid-write leaves the one before, at each boundary that adds to it: a
// Benchmark finishing (Peak::admit's guard), a device or a backend closing
// (the logger's scopes), and the run about to wait at a boundary between
// readings -- a hold or a thermal pause, during which a phone may end the
// app.  Nothing is being timed at those moments.  It is
// removed with the sidecar once the document is saved, so one left behind is
// a run that did not finish, and `--resume <output>` continues it
// (run_resume.h) -- except that a cancelled run keeps it beside the document
// it saved, which can then be continued the same way.
//
// One per run, owned by the host beside its RunLog and current() while it
// lives.  Called on the run's thread only.
class RunCheckpoint
{
public:
    // `doc` is the host's document: the backends already folded in.
    explicit RunCheckpoint(RunDocument &doc);
    ~RunCheckpoint();

    RunCheckpoint(const RunCheckpoint &) = delete;
    RunCheckpoint &operator=(const RunCheckpoint &) = delete;

    // The run's checkpoint, or nullptr outside a run.
    static RunCheckpoint *current();

    // `run.clpeak.json` -> `run.clpeak.partial.json`; a name without `.json`
    // gets `.partial.json` appended.
    static std::string pathFor(const std::string &outputFile);

    // Start keeping the checkpoint of `outputFile`.
    void open(const std::string &outputFile);

    // Stop; remove the file once the document it stood in for is saved.
    void close(bool remove);

    // A resumed run: what the parts before it finished, and how long they ran.
    void seed(const RunProgress &progress, double priorRunningS);

    // A resumed run: this part's `resumed` record -- when it started, and why
    // the part before ended -- which every rewrite carries as far as it goes,
    // so a part that is ended in turn is still one of the run's parts.
    // `before` is what the parts before left.
    void beginPart(const ResumeRecord &part, const RunDocument &before);

    // This part's record as of `doc`: with `from`, the first test in run
    // order that the parts before did not have, and where --skip-crashed
    // recorded its row.
    ResumeRecord partRecord(const RunDocument &doc) const;

    // Seconds spent running: the parts before this one, and this one so far.
    double runningS() const;

    const RunProgress &progress() const { return prog; }

    // ---- What finished ---------------------------------------------------
    // `live` is the document of the backend in flight -- its logger's --
    // which the host folds into its own only once the backend is done.
    void benchmarkFinished(const std::string &backend, int deviceIndex, Benchmark b,
                           const RunDocument &live);
    void deviceFinished(const std::string &backend, int deviceIndex,
                        const RunDocument &live);
    void backendFinished(const std::string &backend, const RunDocument &live);

    // The run is about to wait at a boundary between readings on this
    // device: save what it has, the tests of the Benchmark in flight among
    // it, as held (RunProgress::Device::held).  If the app is ended while it
    // waits, a resumed run still measures that Benchmark again from its
    // start -- a runner cannot begin part-way -- but keeps the readings the
    // held tests had, and takes from its own only the ones they lack
    // (RunDocument::append).
    void held(const std::string &backend, int deviceIndex, const RunDocument &live);

    // ---- --skip-crashed ----------------------------------------------------
    // The part before died in a Benchmark: the first one a resumed run is
    // admitted to is where, so it is recorded instead of run (Peak::admit).
    struct Crash {
        std::string backend;           // where the part before was last, by its log
        int         deviceIndex = -1;
        std::string testId;
        std::string variant;
        std::string title;
        std::string reason;            // the row's: why the test is not run again
    };
    void armCrashSkip(Crash crash);
    // True once, for the first Benchmark asked after armCrashSkip().
    bool takeCrashSkip();
    const Crash &crash() const { return crashInfo; }

    // Where the skipped Benchmark's row was recorded, for `resumed[].skipped`.
    void noteCrashRecorded(const RunPlace &where) { crashPlace = where; }
    const RunPlace &crashRecorded() const { return crashPlace; }

private:
    void write(const RunDocument &live);

    // Every test the document has for this device, the ones in `live`
    // included: those of finished Benchmarks once one finishes.
    void finishTests(const std::string &backend, int deviceIndex, const RunDocument &live);

    RunDocument &doc;
    std::string  path;
    RunProgress  prog;
    double       priorS = 0.0;
    std::chrono::steady_clock::time_point start;
    bool         warned = false;       // a failed write said so once

    bool         armed = false;
    Crash        crashInfo;
    RunPlace     crashPlace;

    bool                  inPart = false;
    ResumeRecord          partStart;
    std::set<std::string> priorTests;   // device key + '\n' + test key
    std::set<std::string> finished;     // the same keys: tests of finished Benchmarks
};

#endif  // CLPEAK_RUN_CHECKPOINT_H
