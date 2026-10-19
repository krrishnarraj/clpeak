#include <common/run_checkpoint.h>
#include <common/common.h>
#include <common/options.h>   // benchmarkFlag

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace {
RunCheckpoint *g_current = nullptr;
}

RunCheckpoint::RunCheckpoint(RunDocument &doc)
    : doc(doc), start(std::chrono::steady_clock::now())
{
    g_current = this;
}

RunCheckpoint::~RunCheckpoint()
{
    if (g_current == this) g_current = nullptr;
}

RunCheckpoint *RunCheckpoint::current()
{
    return g_current;
}

std::string RunCheckpoint::pathFor(const std::string &outputFile)
{
    const std::string ext = ".json";
    if (outputFile.size() > ext.size() &&
        outputFile.compare(outputFile.size() - ext.size(), ext.size(), ext) == 0)
        return outputFile.substr(0, outputFile.size() - ext.size()) + ".partial.json";
    return outputFile + ".partial.json";
}

void RunCheckpoint::open(const std::string &outputFile)
{
    path = pathFor(outputFile);
    // At once, before anything has finished: a run that dies in its first
    // Benchmark can still be continued -- from the start, with its own
    // options, under its own name.
    write(RunDocument());
}

void RunCheckpoint::close(bool remove)
{
    if (path.empty()) return;
    if (remove)
    {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    path.clear();
}

void RunCheckpoint::seed(const RunProgress &progress, double priorRunningS)
{
    prog   = progress;
    priorS = priorRunningS;
}

void RunCheckpoint::beginPart(const ResumeRecord &part, const RunDocument &before)
{
    inPart    = true;
    partStart = part;
    priorTests.clear();
    for (const DeviceResult &d : before.devices)
        for (const TestResult &t : d.tests) priorTests.insert(d.key() + '\n' + t.key());
}

ResumeRecord RunCheckpoint::partRecord(const RunDocument &d) const
{
    ResumeRecord r = partStart;
    r.skipped      = crashPlace;
    for (const DeviceResult &dev : d.devices)
        for (const TestResult &t : dev.tests)
        {
            if (priorTests.count(dev.key() + '\n' + t.key())) continue;
            if (!r.skipped.empty() && r.skipped.backend == dev.backend &&
                r.skipped.deviceIndex == dev.deviceIndex && r.skipped.test == t.key())
                continue;
            r.from.backend     = dev.backend;
            r.from.device      = dev.name;
            r.from.deviceIndex = dev.deviceIndex;
            r.from.test        = t.key();
            return r;
        }
    return r;
}

double RunCheckpoint::runningS() const
{
    return priorS +
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void RunCheckpoint::benchmarkFinished(const std::string &backend, int deviceIndex,
                                      Benchmark b, const RunDocument &live)
{
    RunProgress::Device &d = prog.device(backend, deviceIndex);
    const std::string flag = benchmarkFlag(b);
    if (std::find(d.benchmarks.begin(), d.benchmarks.end(), flag) == d.benchmarks.end())
        d.benchmarks.push_back(flag);
    write(live);
}

void RunCheckpoint::deviceFinished(const std::string &backend, int deviceIndex,
                                   const RunDocument &live)
{
    RunProgress::Device &d = prog.device(backend, deviceIndex);
    d.done = true;
    d.benchmarks.clear();
    write(live);
}

void RunCheckpoint::backendFinished(const std::string &backend, const RunDocument &live)
{
    if (!prog.backendDone(backend)) prog.backendsDone.push_back(backend);
    // Its devices go with it: a resumed run does not construct the backend.
    prog.devices.erase(std::remove_if(prog.devices.begin(), prog.devices.end(),
                                      [&](const RunProgress::Device &d) {
                                          return d.backend == backend;
                                      }),
                       prog.devices.end());
    write(live);
}

void RunCheckpoint::armCrashSkip(Crash crash)
{
    crashInfo = std::move(crash);
    armed     = true;
}

bool RunCheckpoint::takeCrashSkip()
{
    if (!armed) return false;
    armed = false;
    return true;
}

void RunCheckpoint::write(const RunDocument &live)
{
    if (path.empty()) return;

    // The host's document has the backends already folded in; the one in
    // flight is still its logger's.
    RunDocument snapshot;
    snapshot.meta           = doc.meta;
    snapshot.meta.durationS = runningS();
    snapshot.devices        = doc.devices;
    snapshot.append(live);
    if (inPart) snapshot.meta.resumed.push_back(partRecord(snapshot));

    // Written beside and renamed over: a death mid-write leaves the
    // checkpoint before it rather than half of this one.
    const std::string tmp = path + ".tmp";
    bool ok = false;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (f.is_open())
        {
            f << partialRunToJson(snapshot, prog);
            f.flush();
            ok = f.good();
        }
    }
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (ok && !ec) return;

    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);
    if (warned) return;
    warned = true;
    const std::string why = ec ? " (" + ec.message() + ")" : std::string();
    CLPEAK_LOG(Warning,
               "cannot write the run's checkpoint %s%s; if the run is ended part-way, it "
               "cannot be continued",
               path.c_str(), why.c_str());
}
