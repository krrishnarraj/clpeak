#include <common/logger.h>
#include <cassert>

namespace {

// Descriptions are prose, and all three dump formats (CSV, XML, JSON) are
// line-oriented -- one row, one line -- so a literal newline in one would cut
// a record in half for the loaders.  Authors write them as C++ string
// literals wrapped across source lines, which is exactly where stray newlines
// and tabs creep in, so collapse every whitespace run to one space once,
// here, instead of trusting every call site to do it.
std::string oneLine(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    bool pendingSpace = false;
    for (char c : s)
    {
        const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                         c == '\f' || c == '\v');
        if (ws)
        {
            pendingSpace = !out.empty();
            continue;
        }
        if (pendingSpace) out += ' ';
        pendingSpace = false;
        out += c;
    }
    return out;
}

} // namespace

// ── Constructor ────────────────────────────────────────────────────────────

logger::logger(std::string compareFileName)
    : compareEnabled(!compareFileName.empty())
{
    if (compareEnabled)
        baseline = buildBaselineMap(loadResultFile(compareFileName));
}

// ── Event / entry construction ─────────────────────────────────────────────

LogEvent logger::makeEvent(LogEvent::Kind kind) const
{
    LogEvent e;
    e.kind        = kind;
    e.backend     = curBackend;
    e.platform    = curPlatform;
    e.device      = curDevice;
    e.driver      = curDriver;
    e.testTag     = curTest;
    e.testDisplay = curTestDisplay;
    e.unit        = curUnit;
    e.category    = curCategory;
    e.testDescription = curTestDescription;
    return e;
}

ResultEntry logger::makeEntry(const std::string &metric, ResultStatus status,
                              float value, const std::string &reason,
                              const std::string &metricDescription) const
{
    ResultEntry e;
    e.backend  = curBackend;
    e.platform = curPlatform;
    e.device   = curDevice;
    e.driver   = curDriver;
    e.category = categoryString(curCategory);
    e.test     = curTest;
    e.metric   = metric;
    e.unit     = curUnit;
    e.status   = status;
    e.value    = value;
    e.reason   = reason;
    e.display  = curTestDisplay;
    e.description       = curTestDescription;
    e.metricDescription = oneLine(metricDescription);
    return e;
}

// ── Top-level entry ────────────────────────────────────────────────────────

logger::BackendScope logger::beginBackend(const std::string &name)
{
    return BackendScope(this, name);
}

void logger::note(const std::string &msg)
{
    LogEvent e = makeEvent(LogEvent::Kind::Note);
    e.message = msg;
    onEvent(e);
}

// ── BackendScope ───────────────────────────────────────────────────────────

logger::BackendScope::BackendScope(logger *log, const std::string &name)
    : log(log)
{
    assert(log->contextDepth == 0);
    log->curBackend   = name;
    log->curPlatform.clear();
    log->curDevice.clear();
    log->curDriver.clear();
    log->contextDepth = 1;
    log->onEvent(log->makeEvent(LogEvent::Kind::BackendBegin));
}

logger::BackendScope::~BackendScope()
{
    if (!closed)
        end();
}

logger::BackendScope::BackendScope(BackendScope &&other) noexcept
    : log(other.log), closed(other.closed)
{
    other.log    = nullptr;
    other.closed = true;
}

void logger::BackendScope::end()
{
    if (closed) return;
    closed = true;
    assert(log->contextDepth == 1);
    log->onEvent(log->makeEvent(LogEvent::Kind::BackendEnd));
    log->curBackend.clear();
    log->contextDepth = 0;
}

logger::DeviceScope logger::BackendScope::beginDevice(const DeviceSpec &spec)
{
    assert(!closed);
    assert(log->contextDepth == 1);
    return DeviceScope(log, spec);
}

// ── DeviceScope ────────────────────────────────────────────────────────────

logger::DeviceScope::DeviceScope(logger *log, const DeviceSpec &spec)
    : log(log)
{
    assert(log->contextDepth == 1);

    log->curPlatform = spec.platform.empty() ? log->curBackend : spec.platform;
    log->curDevice   = spec.name;
    log->curDriver   = spec.driver_version;
    log->contextDepth = 2;

    DeviceInfo info;
    info.backend  = log->curBackend;
    info.platform = log->curPlatform;
    info.device   = log->curDevice;
    info.driver   = log->curDriver;
    info.props    = spec.props;

    // Re-opening the same device (a backend that enumerates it twice) must not
    // duplicate the row the dump formats key off.
    bool known = false;
    for (DeviceInfo &d : log->devices)
    {
        if (d.key() != info.key()) continue;
        d.props = info.props;
        known   = true;
        break;
    }
    if (!known)
        log->devices.push_back(info);

    LogEvent e = log->makeEvent(LogEvent::Kind::DeviceBegin);
    e.props            = spec.props;
    e.platformIndex    = spec.platform_index;
    e.deviceIndex      = spec.device_index;
    e.showPlatformLine = (log->curPlatform != log->curBackend);
    log->onEvent(e);
}

logger::DeviceScope::~DeviceScope()
{
    if (!closed)
        end();
}

logger::DeviceScope::DeviceScope(DeviceScope &&other) noexcept
    : log(other.log), closed(other.closed)
{
    other.log    = nullptr;
    other.closed = true;
}

void logger::DeviceScope::end()
{
    if (closed) return;
    closed = true;
    assert(log->contextDepth == 2);
    log->onEvent(log->makeEvent(LogEvent::Kind::DeviceEnd));
    log->curDevice.clear();
    log->curDriver.clear();
    log->curPlatform.clear();
    log->contextDepth = 1;
}

void logger::closeOpenTest()
{
    if (contextDepth != 3) return;
    onEvent(makeEvent(LogEvent::Kind::TestEnd));
    curTest.clear();
    curTestDisplay.clear();
    curTestDescription.clear();
    curUnit.clear();
    curCategory  = Category::Unknown;
    contextDepth = 2;
    curTestSeq   = 0;          // any live TestScope for it is now stale
}

logger::TestScope logger::DeviceScope::beginTest(const TestSpec &spec)
{
    assert(!closed);
    assert(log->contextDepth == 2);
    // Overlapping TestScopes are a caller bug: the previous test's rows are
    // still pending in buffering channels (LoggerText renders a whole test at
    // TestEnd).  Rather than let them be dropped -- which is silent, because
    // the ResultStore keeps them and only the text output loses rows -- close
    // the open test first so its output is complete and correctly attributed,
    // and say so.  The assert above catches this in debug builds; the note and
    // the implicit close are what make release builds behave sanely.
    if (log->contextDepth == 3)
    {
        std::string prev = log->curTest;
        log->closeOpenTest();
        log->note("logger: test '" + spec.tag + "' opened while '" + prev +
                  "' was still open; closed it first (call end() on the "
                  "previous TestScope before starting a sibling test)");
    }
    return TestScope(log, spec);
}

// ── TestScope ──────────────────────────────────────────────────────────────

logger::TestScope::TestScope(logger *log, const TestSpec &spec)
    : log(log)
{
    assert(log->contextDepth == 2);

    log->curTest        = spec.tag;
    log->curTestDisplay = spec.display;
    log->curTestDescription = oneLine(spec.description);
    log->curUnit        = spec.unit;
    log->curCategory    = (spec.category != Category::Unknown)
                              ? spec.category
                              : categoryFromUnit(spec.unit);
    log->contextDepth = 3;
    seq = ++log->testSeqCounter;
    log->curTestSeq = seq;

    log->onEvent(log->makeEvent(LogEvent::Kind::TestBegin));
}

logger::TestScope::~TestScope()
{
    if (!closed)
        end();
}

logger::TestScope::TestScope(TestScope &&other) noexcept
    : log(other.log), closed(other.closed), seq(other.seq)
{
    other.log    = nullptr;
    other.closed = true;
}

void logger::TestScope::emit(std::string metric, float value, EmitOptions opts)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    LogEvent e  = log->makeEvent(LogEvent::Kind::Metric);
    e.entry     = log->makeEntry(metric, ResultStatus::Ok, value, "",
                                 opts.description);
    e.subMetric = opts.subMetric;
    log->results.push_back(e.entry);

    log->onEvent(e);
}

void logger::TestScope::emit(std::string metric, float value,
                             const char *description)
{
    EmitOptions opts;
    if (description) opts.description = description;
    emit(std::move(metric), value, std::move(opts));
}

void logger::TestScope::skip(std::string metric, ResultStatus status,
                             std::string reason, std::string description)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    LogEvent e = log->makeEvent(LogEvent::Kind::Metric);
    e.entry    = log->makeEntry(metric, status, 0.0f, reason, description);
    log->results.push_back(e.entry);

    log->onEvent(e);
}

void logger::TestScope::skipAll(std::initializer_list<std::string> metrics,
                                ResultStatus status, std::string reason)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    LogEvent e = log->makeEvent(LogEvent::Kind::TestSkippedAll);
    e.status   = status;
    e.reason   = reason;

    for (const auto &metric : metrics)
    {
        log->results.push_back(log->makeEntry(metric, status, 0.0f, reason));
        e.metricNames.push_back(metric);
    }

    log->onEvent(e);
}

void logger::TestScope::end()
{
    if (closed) return;
    closed = true;
    // Already implicitly closed by a sibling beginTest (see closeOpenTest):
    // its TestEnd has been emitted, so emitting another here would leave the
    // stream with two TestEnds for one TestBegin.
    if (log->curTestSeq != seq) return;
    assert(log->contextDepth == 3);
    log->closeOpenTest();
}
