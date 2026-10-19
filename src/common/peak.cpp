#include <common/peak.h>
#include <common/common.h>
#include <common/options.h>
#include <common/run_checkpoint.h>

#include <exception>

void Peak::applyOptions(const CliOptions &opts)
{
    clpeak::setVerbose(opts.verbose);
    forceIters     = opts.forceIters;
    specifiedIters = opts.iters;
    warmupCount    = opts.warmupCount;
    targetTimeUs   = opts.targetTimeUs;
    enabledTests      = opts.enabledTests;
    enabledCategories = opts.enabledCategories;

    // Only the items that name this backend.  The run loop already skips a
    // backend with none when --devices was given (CliOptions::backendEnabled),
    // so an empty list here means "every device".
    selectedDevices.clear();
    for (const DeviceSelector &sel : opts.devices)
        if (sel.backend == backend())
            selectedDevices.push_back(sel.index);

    resumeDevicesDone.clear();
    resumeBenchmarksDone.clear();
    for (const DeviceSelector &sel : opts.resume.devicesDone)
        if (sel.backend == backend())
            resumeDevicesDone.push_back(sel.index);
    for (const auto &done : opts.resume.benchmarksDone)
        if (done.first.backend == backend())
            resumeBenchmarksDone[done.first.index] |= done.second;
}

bool Peak::resumedPast(Benchmark b) const
{
    if (resumeBenchmarksDone.empty() || !log) return false;
    auto it = resumeBenchmarksDone.find(log->openDeviceIndex());
    return it != resumeBenchmarksDone.end() && it->second.test(static_cast<size_t>(b));
}

Peak::Admission Peak::admit(Benchmark b)
{
    bool admitted = isAllowed(b);
    if (admitted)
        if (RunCheckpoint *checkpoint = RunCheckpoint::current())
            if (checkpoint->takeCrashSkip())
            {
                // Where the part of the run before this one died: say so in
                // its place, and finish it, so that a later resume does not
                // come back to it either.
                recordCrashed(b);
                benchmarkFinished(b);
                admitted = false;
            }
    return Admission(this, b, admitted);
}

void Peak::benchmarkFinished(Benchmark b)
{
    RunCheckpoint *checkpoint = RunCheckpoint::current();
    if (!checkpoint || !log) return;
    const int device = log->openDeviceIndex();
    if (device < 0) return;
    checkpoint->benchmarkFinished(backendInfo(backend()).name, device, b, log->doc);
}

void Peak::recordCrashed(Benchmark b)
{
    RunCheckpoint *checkpoint = RunCheckpoint::current();
    if (!checkpoint || !log || log->openDeviceIndex() < 0) return;
    const RunCheckpoint::Crash &crash = checkpoint->crash();

    // The test the part before was last in, by its log, when that is on this
    // device -- it nearly always is: the first Benchmark not finished on the
    // device it was on is the one it died in.  Otherwise the Benchmark's flag
    // names the place.
    logger::TestSpec spec;
    const std::string name = backendInfo(backend()).name;
    if (crash.backend == name && crash.deviceIndex == log->openDeviceIndex() &&
        !crash.testId.empty())
    {
        spec.tag     = crash.testId;
        spec.variant = crash.variant;
        spec.display = crash.title.empty() ? crash.testId : crash.title;
    }
    else
    {
        spec.tag     = benchmarkFlag(b);
        spec.display = benchmarkFlag(b);
    }
    spec.category = categoryOf(b);
    if (!log->recordUnmeasured(spec, "crashed", ResultStatus::Error, crash.reason))
        return;
    RunPlace where;
    where.backend     = name;
    where.device      = log->openDeviceName();
    where.deviceIndex = log->openDeviceIndex();
    where.test        = spec.variant.empty() ? spec.tag : spec.tag + "@" + spec.variant;
    checkpoint->noteCrashRecorded(where);
}

// ── Admission ──────────────────────────────────────────────────────────────

Peak::Admission::Admission(Peak *peak, Benchmark b, bool admitted)
    : peak(peak), bench(b), admitted(admitted),
      exceptionsAtEntry(std::uncaught_exceptions())
{
}

Peak::Admission::Admission(Admission &&other) noexcept
    : peak(other.peak), bench(other.bench), admitted(other.admitted),
      exceptionsAtEntry(other.exceptionsAtEntry)
{
    other.peak     = nullptr;
    other.admitted = false;
}

Peak::Admission::~Admission()
{
    if (!admitted || !peak) return;
    // Cut short: a cancel stops a runner at its next test boundary, and an
    // exception unwinding through here never let it finish.
    if (clpeak::cancelRequested() || std::uncaught_exceptions() > exceptionsAtEntry)
        return;
    peak->benchmarkFinished(bench);
}
