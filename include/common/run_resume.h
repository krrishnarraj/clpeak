#ifndef CLPEAK_RUN_RESUME_H
#define CLPEAK_RUN_RESUME_H

#include <string>
#include <vector>

#include <common/options.h>
#include <common/run_document.h>

class RunLog;
class RunCheckpoint;

// ── Continuing an interrupted run ──────────────────────────────────────────
//
// `--resume <output>` continues the run whose -o file <output> was, after the
// process it ran in ended part-way or it was cancelled; the checkpoint's own
// path names the run as well.  Its checkpoint (`<output>.partial.json`,
// run_checkpoint.h) holds what it finished and the options it started with;
// its sidecar (`<output>.log`, run_log.h) what it logged -- or, for a run that
// was cancelled, the document it saved, whose sidecar went once it was
// written.  The run goes on
// with those options and skips the backends, devices and Benchmarks the part
// before finished.  The Benchmark it was in when it died runs again from its
// start -- or with --skip-crashed, for a process that crashed rather than
// being ended, is recorded as where it died and not run, since a crash is
// likely to repeat.  The parts end in one document under the same name, with
// a `resumed` record per part after the first (docs/format-v3.md), and one
// log: the sidecar is continued, and the document's log starts with the part
// before's.
//
// Only the build that started a run continues it, on the OS release it
// started on: another build can run other tests in another order, and the
// parts would not be one run.  Both hosts (src/cli/main.cpp,
// src/ffi/clpeak_ffi.cpp) go through these three calls in this order, and
// take this part's `resumed` record from the checkpoint once it is over
// (RunCheckpoint::partRecord).

struct ResumePlan {
    std::string outputFile;           // the run's -o file, which this part writes
    RunDocument seed;                 // the checkpoint: meta and finished devices
    RunProgress progress;
    std::vector<LogEntry> priorLog;   // the sidecar's entries, in order
    bool        sidecarFound = false;
    bool        cancelled = false;    // the part before was cancelled, and saved
    double      priorRunningS = -1.0; // then: the parts' time, as the document says
    std::string after;                // why the part before ended, when its log says
    std::string startedAt;            // ISO-8601 UTC: this part's start
    double      clockOffsetS = 0.0;   // the same moment on the run's clock
};

// Read the checkpoint and the sidecar of the run `path` names: its -o file, or
// its checkpoint.  False, with `error` saying why in a line for the user, when
// there is nothing to continue or it cannot be continued here.
bool loadResumePlan(const std::string &path, ResumePlan &plan, std::string &error);

// The continued run's options: the run's own, parsed again from its
// checkpoint, writing to the same file and skipping what it finished.
// `given` is what --resume was called with, for --skip-crashed.
bool resumeOptions(const ResumePlan &plan, const CliOptions &given, CliOptions &out,
                   std::string &error);

// Seed the host's document, log and checkpoint with the parts before.  Before
// the sidecar is opened.
void seedResume(const ResumePlan &plan, const CliOptions &opts, RunDocument &doc,
                RunLog &log, RunCheckpoint &checkpoint);

#endif  // CLPEAK_RUN_RESUME_H
