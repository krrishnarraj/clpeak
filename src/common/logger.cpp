#include <common/logger.h>
#include <cassert>

namespace {

// Descriptions are prose, and they end up in a JSON document and in aligned
// terminal output, neither of which wants an embedded newline.  Authors write
// them as C++ string literals wrapped across source lines, which is exactly
// where stray newlines and tabs creep in, so collapse every whitespace run to
// one space once, here, instead of trusting every call site to do it.
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

// Driver-reported names come as the driver spells them, padding included:
// Intel's OpenCL runtime returns "AMD Ryzen Threadripper PRO 3955WX 16-Cores"
// with five trailing spaces.  That padding reached the document, the device's
// identity key and the GUI, where it is at best untidy and at worst two names
// for one device if a driver ever changes how much of it there is.  Trimmed
// once here rather than in eight backends.
std::string trimmed(const std::string &s)
{
    const char *ws = " \t\n\r";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

} // namespace

// ── Constructor ────────────────────────────────────────────────────────────

logger::logger(std::string compareFileName)
    : compareEnabled(!compareFileName.empty())
{
    if (!compareEnabled) return;
    RunDocument base;
    if (loadRunJson(compareFileName, base))
        baseline = buildBaselineMap(base);
}

// ── Event construction ─────────────────────────────────────────────────────

TestResult *logger::openTest()
{
    if (curDeviceIdx == kNoIndex || curTestIdx == kNoIndex) return nullptr;
    return &doc.devices[curDeviceIdx].tests[curTestIdx];
}

const TestResult *logger::openTest() const
{
    return const_cast<logger *>(this)->openTest();
}

LogEvent logger::makeEvent(LogEvent::Kind kind) const
{
    LogEvent e;
    e.kind     = kind;
    e.backend  = curBackend;
    e.platform = curPlatform;
    e.device      = curDevice;
    e.driver      = curDriver;
    e.deviceIndex = curDeviceIndex;

    if (const TestResult *t = openTest())
    {
        e.testId          = t->id;
        e.testTitle       = t->title;
        e.testVariant     = t->variant;
        e.testAxis        = t->axis;
        e.testDescription = t->description;
        e.category        = t->category;
        e.shape           = t->shape;
        e.direction       = t->direction;
        e.unit            = t->unit;
        e.quantity        = t->quantity;
        e.scale           = t->scale;
    }
    return e;
}

MetricResult &logger::record(MetricResult m)
{
    TestResult *t = openTest();
    assert(t && "record() outside an open test");
    t->metrics.push_back(std::move(m));
    return t->metrics.back();
}

// ── Top-level entry ────────────────────────────────────────────────────────

logger::BackendScope logger::beginBackend(const std::string &name)
{
    return BackendScope(this, name);
}

void logger::note(const std::string &msg)
{
    // Kept on the document as well as dispatched: a note is usually the only
    // record of *why* something is missing ("library not found"), and without
    // it a reopened run reads as hardware that simply lacks the feature.
    doc.notes.push_back({curBackend, curDevice, oneLine(msg)});

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

    log->curPlatform    = spec.platform.empty() ? log->curBackend
                                                : trimmed(spec.platform);
    log->curDevice      = trimmed(spec.name);
    log->curDriver      = trimmed(spec.driver_version);
    log->curDeviceIndex = spec.device_index;
    log->contextDepth = 2;
    log->curTestIdx   = logger::kNoIndex;

    DeviceResult fresh;
    fresh.backend       = log->curBackend;
    fresh.platform      = log->curPlatform;
    fresh.name          = log->curDevice;
    fresh.driver        = log->curDriver;
    fresh.type          = spec.type;
    fresh.platformIndex = spec.platform_index;
    fresh.deviceIndex   = spec.device_index;
    fresh.properties    = spec.props;

    // Re-opening the same device (a backend that enumerates it twice) must
    // append to the block already recorded, not start a second one.
    if (DeviceResult *known = log->doc.findDevice(fresh.key()))
    {
        known->properties = fresh.properties;
        known->driver     = fresh.driver;
        if (known->type == DeviceType::Unknown) known->type = fresh.type;
        log->curDeviceIdx = static_cast<std::size_t>(known - &log->doc.devices[0]);
    }
    else
    {
        log->doc.devices.push_back(fresh);
        log->curDeviceIdx = log->doc.devices.size() - 1;
    }

    LogEvent e = log->makeEvent(LogEvent::Kind::DeviceBegin);
    e.props            = spec.props;
    e.type             = spec.type;
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
    log->curDeviceIndex = -1;
    log->curDeviceIdx = logger::kNoIndex;
    log->contextDepth = 1;
}

void logger::closeOpenTest()
{
    if (contextDepth != 3) return;
    onEvent(makeEvent(LogEvent::Kind::TestEnd));
    curTestIdx   = kNoIndex;
    contextDepth = 2;
    curTestSeq   = 0;          // any live TestScope for it is now stale
}

logger::TestScope logger::DeviceScope::beginTest(const TestSpec &spec)
{
    assert(!closed);
    assert(log->contextDepth == 2);
    // Overlapping TestScopes are a caller bug: the previous test's readings
    // are still pending in buffering channels (LoggerText renders a whole test
    // at TestEnd).  Rather than let them be dropped -- which is silent,
    // because the document keeps them and only the text output loses rows --
    // close the open test first so its output is complete and correctly
    // attributed, and say so.  The assert above catches this in debug builds;
    // the note and the implicit close are what make release builds behave
    // sanely.
    if (log->contextDepth == 3)
    {
        std::string prev = log->openTest() ? log->openTest()->id : std::string();
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
    assert(log->curDeviceIdx != logger::kNoIndex);

    TestResult fresh;
    fresh.id          = spec.tag;
    fresh.title       = spec.display.empty() ? spec.tag : spec.display;
    fresh.variant     = spec.variant;
    fresh.axis        = spec.axis;
    fresh.description = oneLine(spec.description);
    fresh.shape       = spec.shape;

    // Resolve the authored unit token exactly once.  Everything downstream --
    // the CLI header, the GUI's SI scaling, the direction of a compare delta
    // -- reads the resolved fields, so no consumer repeats this lookup.
    const UnitInfo u = unitInfo(spec.unit);
    fresh.unit      = u.symbol;
    fresh.quantity  = u.quantity;
    fresh.scale     = u.scale;
    fresh.direction = (spec.direction != Direction::FromUnit) ? spec.direction
                                                              : u.direction;
    fresh.category  = (spec.category != Category::Unknown)
                          ? spec.category
                          : categoryFromUnit(spec.unit);

    DeviceResult &dev  = log->doc.devices[log->curDeviceIdx];
    bool          reopened = false;
    if (TestResult *known = dev.findTest(fresh.key()))
    {
        // Reopen: the first open defined this test, so only readings are
        // added.  A reading measured in another unit carries its own on
        // EmitOptions::unit.
        reopened        = true;
        log->curTestIdx = static_cast<std::size_t>(known - &dev.tests[0]);
    }
    else
    {
        dev.tests.push_back(fresh);
        log->curTestIdx = dev.tests.size() - 1;
    }

    log->contextDepth = 3;
    seq = ++log->testSeqCounter;
    log->curTestSeq = seq;

    LogEvent e = log->makeEvent(LogEvent::Kind::TestBegin);
    e.reopened   = reopened;
    e.streaming  = spec.streaming;
    // What this opening asked for, not what the test settled on at its first.
    e.openedUnit = u.symbol;
    log->onEvent(e);
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

    MetricResult m;
    m.id          = std::move(metric);
    m.label       = opts.label;
    m.status      = ResultStatus::Ok;
    m.value       = value;
    m.description = oneLine(opts.description);
    m.direction   = opts.direction;
    if (!opts.unit.empty())
    {
        const UnitInfo u = unitInfo(opts.unit);
        m.hasUnit  = true;
        m.unit     = u.symbol;
        m.quantity = u.quantity;
        m.scale    = u.scale;
        // A unit override with no explicit direction takes that unit's, not
        // the test's: a `us` reading inside a throughput test is still
        // lower-is-better.
        if (m.direction == Direction::FromUnit) m.direction = u.direction;
    }

    LogEvent e = log->makeEvent(LogEvent::Kind::Metric);
    e.metric   = log->record(std::move(m));
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

    MetricResult m;
    m.id          = std::move(metric);
    m.status      = status;
    m.reason      = std::move(reason);
    m.description = oneLine(description);

    LogEvent e = log->makeEvent(LogEvent::Kind::Metric);
    e.metric   = log->record(std::move(m));
    log->onEvent(e);
}

void logger::TestScope::skip(std::string metric, ResultStatus status,
                             std::string reason, EmitOptions opts)
{
    assert(!closed);
    assert(log->contextDepth == 3);

    MetricResult m;
    m.id          = std::move(metric);
    m.status      = status;
    m.reason      = std::move(reason);
    m.description = oneLine(opts.description);
    m.label       = opts.label;
    m.direction   = opts.direction;
    // A reading that never ran still records the unit it would have been in:
    // that is what tells a reader an unsupported int8 row was never going to
    // be flops, and what keeps the row honest when the test it sits in
    // reports something else.
    if (!opts.unit.empty())
    {
        const UnitInfo u = unitInfo(opts.unit);
        m.hasUnit  = true;
        m.unit     = u.symbol;
        m.quantity = u.quantity;
        m.scale    = u.scale;
        if (m.direction == Direction::FromUnit) m.direction = u.direction;
    }

    LogEvent e = log->makeEvent(LogEvent::Kind::Metric);
    e.metric   = log->record(std::move(m));
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
        MetricResult m;
        m.id     = metric;
        m.status = status;
        m.reason = reason;
        log->record(std::move(m));
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
