#include <common/logger.h>
#include <cassert>
#include <sstream>

// ── Constructor ────────────────────────────────────────────────────────────

logger::logger(std::string compareFileName)
    : compareEnabled(!compareFileName.empty())
{
    if (compareEnabled)
        baseline = buildBaselineMap(loadResultFile(compareFileName));
}

// ── Top-level entry ────────────────────────────────────────────────────────

logger::BackendScope logger::beginBackend(const std::string &name)
{
    return BackendScope(this, name);
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
    log->onBackendBegin(name);
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
    log->onBackendEnd();
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

    bool showPlatformLine = (log->curPlatform != log->curBackend);

    std::string displayPlatform = log->curPlatform;
    if (spec.platform_index >= 0 && showPlatformLine)
        displayPlatform = std::to_string(spec.platform_index) + ": " + displayPlatform;

    std::string displayDevice = spec.name;
    if (spec.device_index >= 0)
        displayDevice = std::to_string(spec.device_index) + ": " + displayDevice;

    log->onDeviceBegin(displayDevice, displayPlatform,
                       spec.driver_version, spec.props,
                       showPlatformLine);
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
    log->onDeviceEnd();
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

    log->curTest     = spec.tag;
    log->curUnit     = spec.unit;
    log->curCategory = (spec.category != Category::Unknown)
                           ? spec.category
                           : categoryFromUnit(spec.unit);
    log->contextDepth = 3;

    log->onTestBegin(spec.tag, spec.display, spec.unit);
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

    // Build ResultEntry and push
    ResultEntry e;
    e.backend  = log->curBackend;
    e.platform = log->curPlatform;
    e.device   = log->curDevice;
    e.driver   = log->curDriver;
    e.category = categoryString(log->curCategory);
    e.test     = log->curTest;
    e.metric   = metric;
    e.unit     = log->curUnit;
    e.status   = ResultStatus::Ok;
    e.value    = value;
    log->results.push_back(e);

    log->onMetricEmitted(e, value, opts.subMetric);
}

void logger::TestScope::skip(std::string metric, ResultStatus status,
                             std::string reason)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    // Build ResultEntry and push
    ResultEntry e;
    e.backend  = log->curBackend;
    e.platform = log->curPlatform;
    e.device   = log->curDevice;
    e.driver   = log->curDriver;
    e.category = categoryString(log->curCategory);
    e.test     = log->curTest;
    e.metric   = metric;
    e.unit     = log->curUnit;
    e.status   = status;
    e.value    = 0.0f;
    e.reason   = reason;
    log->results.push_back(e);

    log->onMetricSkipped(e);
}

void logger::TestScope::skipAll(std::initializer_list<std::string> metrics,
                                ResultStatus status, std::string reason)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    log->onTestSkippedAll(status, reason);

    for (const auto &metric : metrics)
    {
        ResultEntry e;
        e.backend  = log->curBackend;
        e.platform = log->curPlatform;
        e.device   = log->curDevice;
        e.driver   = log->curDriver;
        e.category = categoryString(log->curCategory);
        e.test     = log->curTest;
        e.metric   = metric;
        e.unit     = log->curUnit;
        e.status   = status;
        e.value    = 0.0f;
        e.reason   = reason;
        log->results.push_back(e);
    }
}

void logger::TestScope::end()
{
    if (closed) return;
    closed = true;
    assert(log->contextDepth == 3);
    log->onTestEnd();
    log->curTest.clear();
    log->curUnit.clear();
    log->curCategory = Category::Unknown;
    log->contextDepth = 2;
}
