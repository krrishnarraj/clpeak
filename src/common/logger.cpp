#include <common/logger.h>
#include <cassert>

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
    return e;
}

ResultEntry logger::makeEntry(const std::string &metric, ResultStatus status,
                              float value, const std::string &reason) const
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

logger::TestScope logger::DeviceScope::beginTest(const TestSpec &spec)
{
    assert(!closed);
    assert(log->contextDepth == 2);
    return TestScope(log, spec);
}

// ── TestScope ──────────────────────────────────────────────────────────────

logger::TestScope::TestScope(logger *log, const TestSpec &spec)
    : log(log)
{
    assert(log->contextDepth == 2);

    log->curTest        = spec.tag;
    log->curTestDisplay = spec.display;
    log->curUnit        = spec.unit;
    log->curCategory    = (spec.category != Category::Unknown)
                              ? spec.category
                              : categoryFromUnit(spec.unit);
    log->contextDepth = 3;

    log->onEvent(log->makeEvent(LogEvent::Kind::TestBegin));
}

logger::TestScope::~TestScope()
{
    if (!closed)
        end();
}

logger::TestScope::TestScope(TestScope &&other) noexcept
    : log(other.log), closed(other.closed)
{
    other.log    = nullptr;
    other.closed = true;
}

void logger::TestScope::emit(std::string metric, float value, EmitOptions opts)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    LogEvent e  = log->makeEvent(LogEvent::Kind::Metric);
    e.entry     = log->makeEntry(metric, ResultStatus::Ok, value, "");
    e.subMetric = opts.subMetric;
    log->results.push_back(e.entry);

    log->onEvent(e);
}

void logger::TestScope::skip(std::string metric, ResultStatus status,
                             std::string reason)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    LogEvent e = log->makeEvent(LogEvent::Kind::Metric);
    e.entry    = log->makeEntry(metric, status, 0.0f, reason);
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
    assert(log->contextDepth == 3);
    log->onEvent(log->makeEvent(LogEvent::Kind::TestEnd));
    log->curTest.clear();
    log->curTestDisplay.clear();
    log->curUnit.clear();
    log->curCategory = Category::Unknown;
    log->contextDepth = 2;
}
