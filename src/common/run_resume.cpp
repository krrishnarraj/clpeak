#include <common/run_resume.h>
#include <common/common.h>
#include <common/host_info.h>
#include <common/logger.h>
#include <common/run_checkpoint.h>
#include <common/run_log.h>
#include <version.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace {

// The line each part after the first starts with on the log; the app's
// reading of the sidecar (CrashLog.read) knows it too.
const char kResumedLine[] = "resumed the run started at ";

const std::string kPartialExt = ".partial.json";

bool endsWith(const std::string &s, const std::string &tail)
{
    return s.size() > tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// The -o file a checkpoint belongs to: `X.partial.json` is X.json's, or X's
// for a run whose -o named no .json, which its command line says
// (RunCheckpoint::pathFor).
std::string resultPathOf(const std::string &partial, const RunMeta &meta)
{
    const std::string stem = partial.substr(0, partial.size() - kPartialExt.size());
    const std::vector<std::string> &argv = meta.invocation.argv;
    for (size_t i = 0; i + 1 < argv.size(); i++)
        if ((argv[i] == "-o" || argv[i] == "--output") && !endsWith(argv[i + 1], ".json"))
            return stem;
    return stem + ".json";
}

// The part a log entry belongs to: those at or after the latest part's start
// on the run's clock are the latest part's.
double latestPartStart(const RunMeta &meta)
{
    return meta.resumed.empty() ? -1.0 : meta.resumed.back().elapsedS;
}

void readSidecar(const std::string &path, ResumePlan &plan)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return;
    plan.sidecarFound = true;
    std::string line;
    bool header = true;
    while (std::getline(f, line))
    {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (header)
        {
            header = false;
            continue;
        }
        LogEntry e;
        if (logEntryFromJson(line, e)) plan.priorLog.push_back(std::move(e));
    }
}

// A cancelled run saved its document and dropped its sidecar, so the parts
// before are read from the document instead: its log, and its duration,
// which is exact where a log's span is not.
void readSavedRun(const std::string &path, ResumePlan &plan)
{
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) return;
    RunDocument doc;
    if (!loadRunJson(path, doc)) return;
    plan.priorLog      = std::move(doc.log);
    plan.priorRunningS = doc.meta.durationS;
    plan.cancelled     = doc.meta.cancelled;
}

// What ended the latest part, as Android recorded it: the line the app adds
// to the sidecar from ApplicationExitInfo (app/lib/src/services/
// process_exit.dart), its first line -- the rest is a tombstone.
std::string endOfLatestPart(const ResumePlan &plan)
{
    const double from = latestPartStart(plan.seed.meta);
    for (auto it = plan.priorLog.rbegin(); it != plan.priorLog.rend(); ++it)
    {
        if (it->elapsedS < from) break;
        if (it->source == "android") return it->message.substr(0, it->message.find('\n'));
    }
    return "";
}

// Where the latest part was when it died: the last line its log scoped to a
// test, and that test's title from the line that opened it
// (logger::mirrorEvent's "    Title [variant]").
RunCheckpoint::Crash crashSite(const ResumePlan &plan)
{
    RunCheckpoint::Crash crash;
    const double from = latestPartStart(plan.seed.meta);
    std::string key;
    for (auto it = plan.priorLog.rbegin(); it != plan.priorLog.rend(); ++it)
    {
        if (it->elapsedS < from) break;
        if (it->test.empty() || !it->source.empty()) continue;
        crash.backend     = it->backend;
        crash.deviceIndex = it->deviceIndex;
        key               = it->test;
        break;
    }
    if (!key.empty())
    {
        const size_t at = key.find('@');
        crash.testId  = key.substr(0, at);
        crash.variant = at == std::string::npos ? "" : key.substr(at + 1);
        for (auto it = plan.priorLog.rbegin(); it != plan.priorLog.rend(); ++it)
        {
            if (it->test != key || it->level != clpeak::LogLevel::Info ||
                it->message.compare(0, 4, "    ") != 0 || it->message.size() <= 4 ||
                it->message[4] == ' ')
                continue;
            std::string title = it->message.substr(4);
            const std::string suffix = " [" + crash.variant + "]";
            if (!crash.variant.empty() && title.size() > suffix.size() &&
                title.compare(title.size() - suffix.size(), suffix.size(), suffix) == 0)
                title.resize(title.size() - suffix.size());
            crash.title = title;
            break;
        }
    }
    crash.reason = "clpeak ended here before the run was resumed";
    if (!plan.after.empty()) crash.reason += " (" + plan.after + ")";
    crash.reason += ", so this was not run again";
    return crash;
}

// Seconds the parts before ran, from their log: each from its start -- the
// run's, or its "resumed" line -- to its last line before the process ended.
// Exact where the checkpoint is not: that is rewritten only as something
// finishes, and a part's time after its last write would be lost.
double priorRunning(const ResumePlan &plan)
{
    if (plan.priorRunningS >= 0.0) return plan.priorRunningS;
    if (plan.priorLog.empty()) return plan.seed.meta.durationS;
    double total = 0.0, start = 0.0, last = 0.0;
    for (const LogEntry &e : plan.priorLog)
    {
        if (e.source.empty() && e.message.rfind(kResumedLine, 0) == 0)
        {
            total += last - start;
            start = last = e.elapsedS;
            continue;
        }
        if (e.source == "android") continue;   // the app's, after the end
        last = std::max(last, e.elapsedS);
    }
    return total + (last - start);
}

} // namespace

bool loadResumePlan(const std::string &path, ResumePlan &plan, std::string &error)
{
    plan = ResumePlan();

    const bool        byCheckpoint = endsWith(path, kPartialExt);
    const std::string partial      = byCheckpoint ? path : RunCheckpoint::pathFor(path);
    std::error_code ec;
    if (!std::filesystem::exists(partial, ec))
    {
        error = byCheckpoint ? "clpeak: nothing to resume at " + path + ": there is no such file"
                : std::filesystem::exists(path, ec)
                    ? "clpeak: " + path + " is a finished run; there is nothing to resume"
                    : "clpeak: nothing to resume at " + path + ": a run that did not "
                      "finish leaves " + partial + " beside it, and there is none";
        return false;
    }
    std::string why;
    if (!loadPartialJson(partial, plan.seed, plan.progress, why))
    {
        error = "clpeak: cannot resume " + path + ": " + why;
        return false;
    }
    plan.outputFile = byCheckpoint ? resultPathOf(path, plan.seed.meta) : path;
    const std::string &outputFile = plan.outputFile;

    const RunMeta &meta = plan.seed.meta;
    if (meta.clpeakVersion != CLPEAK_VERSION_STR)
    {
        error = "clpeak: " + outputFile + " was started by clpeak " + meta.clpeakVersion +
                ", and a run is continued only by the build that started it; this is " +
                CLPEAK_VERSION_STR;
        return false;
    }
    const HostInfo host = probeHost();
    if (meta.host.os != host.os || meta.host.osVersion != host.osVersion)
    {
        error = "clpeak: " + outputFile + " was started on " + meta.host.os + " " +
                meta.host.osVersion + ", and a run is continued only on the release it "
                "started on; this is " + host.os + " " + host.osVersion;
        return false;
    }
    if (meta.invocation.argv.empty())
    {
        error = "clpeak: cannot resume " + outputFile + ": its checkpoint records no options";
        return false;
    }

    readSidecar(RunLog::sidecarPathFor(outputFile), plan);
    if (!plan.sidecarFound) readSavedRun(outputFile, plan);
    plan.after = plan.cancelled ? "cancelled" : endOfLatestPart(plan);

    // The run's clock goes on from its start, the time between the parts
    // included -- and never behind what the part before logged, whatever
    // the wall clock did meanwhile.
    plan.startedAt = isoTimestampUtc();
    long long started = 0, now = 0;
    double    offset  = 0.0;
    if (parseIsoTimestampUtc(meta.generatedAt, started) &&
        parseIsoTimestampUtc(plan.startedAt, now))
        offset = static_cast<double>(now - started);
    for (const LogEntry &e : plan.priorLog) offset = std::max(offset, e.elapsedS);
    for (const ResumeRecord &r : meta.resumed) offset = std::max(offset, r.elapsedS);
    plan.clockOffsetS = offset;
    return true;
}

bool resumeOptions(const ResumePlan &plan, const CliOptions &given, CliOptions &out,
                   std::string &error)
{
    std::vector<std::string> args = plan.seed.meta.invocation.argv;
    std::vector<char *> argv;
    for (std::string &a : args) argv.push_back(&a[0]);
    CliOptions opts;
    std::string why;
    if (!parseCliOptionsNoExit(static_cast<int>(argv.size()), argv.data(), opts, why))
    {
        while (!why.empty() && why.back() == '\n') why.pop_back();
        error = "clpeak: cannot resume " + plan.outputFile + ": its options do not parse (" +
                why + ")";
        return false;
    }

    // Wherever the first part was told to write, this part writes the file
    // it was resumed by.
    opts.outputFile   = plan.outputFile;
    opts.enableOutput = true;
    opts.resumeFile   = given.resumeFile;
    opts.skipCrashed  = given.skipCrashed;
    if (opts.skipCrashed && plan.cancelled)
    {
        error = "clpeak: " + plan.outputFile + " was cancelled, not ended by a crash; "
                "resume it without --skip-crashed";
        return false;
    }

    for (const std::string &name : plan.progress.backendsDone)
    {
        Backend b;
        if (backendFromName(name, b)) opts.resume.backendsDone.set(static_cast<size_t>(b));
    }
    for (const RunProgress::Device &d : plan.progress.devices)
    {
        Backend b;
        if (!backendFromName(d.backend, b) || d.deviceIndex < 0) continue;
        DeviceSelector sel;
        sel.backend = b;
        sel.index   = d.deviceIndex;
        if (d.done)
        {
            opts.resume.devicesDone.push_back(sel);
            continue;
        }
        std::bitset<static_cast<size_t>(Benchmark::COUNT)> done;
        for (const std::string &flag : d.benchmarks)
        {
            Benchmark bench;
            if (benchmarkFromFlag(flag, bench)) done.set(static_cast<size_t>(bench));
        }
        if (done.any()) opts.resume.benchmarksDone.emplace_back(sel, done);
    }

    out = std::move(opts);
    return true;
}

void seedResume(const ResumePlan &plan, const CliOptions &opts, RunDocument &doc,
                RunLog &log, RunCheckpoint &checkpoint)
{
    // The run's own: when it started, how it was asked to run, its parts.
    doc.meta    = plan.seed.meta;
    doc.devices = plan.seed.devices;

    // The document's log starts with the parts before, as the sidecar holds
    // them -- less the reading lines a run without --verbose keeps off its
    // document, which the devices above hold instead.
    std::vector<LogEntry> prior;
    for (const LogEntry &e : plan.priorLog)
        if (plan.seed.meta.invocation.verbose || !logger::isReadingLine(e))
            prior.push_back(e);
    log.seedPrior(std::move(prior), plan.clockOffsetS);

    checkpoint.seed(plan.progress, priorRunning(plan));
    if (opts.skipCrashed) checkpoint.armCrashSkip(crashSite(plan));
    ResumeRecord part;
    part.at       = plan.startedAt;
    part.elapsedS = plan.clockOffsetS;
    part.after    = plan.after;
    checkpoint.beginPart(part, plan.seed);

    size_t tests = 0;
    for (const DeviceResult &d : plan.seed.devices) tests += d.tests.size();
    CLPEAK_LOG(Info, "%s%s, which had measured %zu tests when it stopped%s%s", kResumedLine,
               plan.seed.meta.generatedAt.c_str(), tests, plan.after.empty() ? "" : ": ",
               plan.after.c_str());
    // Where it stopped while held: what it measures again keeps those readings.
    for (const RunProgress::Device &d : plan.progress.devices)
        for (const std::string &t : d.held)
            CLPEAK_LOG(Info, "it stopped while held in %s (%s device %d): that test runs again, "
                       "and the readings it had are kept",
                       t.c_str(), d.backend.c_str(), d.deviceIndex);
}
