#include <common/logger_text.h>
#include <common/measures.h>
#include <common/path_scrub.h>
#include <common/run_compare.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

// ── Constants ──────────────────────────────────────────────────────────────

static const int MIN_METRIC_PAD = 8;   // minimum column width for metric names

// --describe wraps prose to this width.  Fixed rather than read off the
// terminal: output is routinely piped, redirected and diffed, and a width that
// changes with the window would make those results differ run to run.
static const int DESCRIBE_WIDTH = 80;

// Narrowest prose column worth wrapping to; below it, deep indentation would
// leave a ragged one-word-per-line column, so let the text overrun instead.
static const int MIN_DESCRIBE_COLUMN = 32;

static std::string statusTag(ResultStatus status)
{
    switch (status)
    {
    case ResultStatus::Unsupported: return "unsupported";
    case ResultStatus::Skipped:     return "skipped";
    case ResultStatus::Error:       return "error";
    default:                        return "unknown";
    }
}

// ── Event dispatch ─────────────────────────────────────────────────────────

void LoggerText::onEvent(const LogEvent &e)
{
    switch (e.kind)
    {
    case LogEvent::Kind::BackendBegin:   renderBackendBegin(e);   break;
    case LogEvent::Kind::DeviceBegin:    renderDeviceBegin(e);    break;
    case LogEvent::Kind::TestBegin:      renderTestBegin(e);      break;
    case LogEvent::Kind::Metric:         renderMetric(e);         break;
    case LogEvent::Kind::TestSkippedAll: renderTestSkippedAll(e); break;
    case LogEvent::Kind::TestEnd:
        renderTestEnd();
        noteClosedTest(e.testKey());
        break;
    case LogEvent::Kind::DeviceEnd:      renderDeviceEnd();       break;
    case LogEvent::Kind::BackendEnd:     renderBackendEnd();      break;
    case LogEvent::Kind::Log:            renderLog(e.log);        break;
    }
}

// ── Log ────────────────────────────────────────────────────────────────────

void LoggerText::renderLog(const LogEntry &l)
{
    // Library diagnostics and XpuPeak's own debug/info lines are the --verbose
    // channel: stderr, flushed per line, so the last one before a driver-side
    // crash is on screen -- and written past any console capture in progress,
    // or the line would be captured again.
    if (!l.source.empty() || l.level == xpupeak::LogLevel::Debug ||
        l.level == xpupeak::LogLevel::Info)
    {
        if (!verbose) return;
        xpupeak::stderrWrite(l.message + "\n");
        return;
    }
    // XpuPeak's own warnings and errors print inline with the results, as
    // notes always have -- before the message, or it would appear above rows
    // that were measured before it.
    flushMetrics();
    out << l.message << "\n";
    out.flush();
}

// ── BackendBegin ───────────────────────────────────────────────────────────

void LoggerText::renderBackendBegin(const LogEvent &e)
{
    indentLevel = 0;
    out << "Backend: " << e.backend << "\n";
    out.flush();
}

// ── DeviceBegin ────────────────────────────────────────────────────────────

void LoggerText::renderDeviceBegin(const LogEvent &e)
{
    // Indent setup: platform line (if shown) pushes device one level deeper
    int deviceIndent = e.showPlatformLine ? 2 : 1;
    propIndent       = deviceIndent + 1;   // props indented under device
    indentLevel      = deviceIndent;

    if (e.showPlatformLine)
    {
        std::string pline = e.platformIndex >= 0
            ? "Platform " + std::to_string(e.platformIndex) + ": " + e.platform
            : "Platform: " + e.platform;
        writeLine(1, pline);
    }

    std::string dline = e.deviceIndex >= 0
        ? "Device " + std::to_string(e.deviceIndex) + ": " + e.device
        : "Device: " + e.device;
    writeLine(deviceIndent, dline);

    // Properties
    if (!e.driver.empty())
    {
        std::string dvLabel = "Driver version";
        while (dvLabel.size() < 17)
            dvLabel += ' ';
        writeLine(propIndent, dvLabel + ": " + e.driver);
    }

    for (const auto &prop : e.props)
    {
        std::string line = prop.key;
        // Right-align key to match "Driver version  " width (17 chars)
        while (line.size() < 17)
            line += ' ';
        line += ": " + prop.value;
        writeLine(propIndent, line);
    }

    out.flush();
}

// ── TestBegin ──────────────────────────────────────────────────────────────

void LoggerText::renderTestBegin(const LogEvent &e)
{
    // Reopening the test that just closed, with its header still standing:
    // keep accumulating into the same stanza, under the header already
    // printed for it.  Rows carry their own units, so a reopen with a different
    // unit joins the same table rather than
    // starting a look-alike stanza beside it.
    if (e.reopened && e.testKey() == lastClosedTest && stanzaHasHeader)
        return;

    flushMetrics();
    mergedPad       = 0;
    stanzaHasHeader = true;

    metricIndent = propIndent + 1;   // metrics indented one more than props
    indentLevel  = propIndent;       // test header at prop level

    // Build the header.  In verbose mode the header is printed
    // immediately so that backend XPUPEAK_VLOG lines (stderr) land
    // under it, not above it.  In default mode the header is deferred
    // until we know the test has visible output, so a fully-skipped
    // test leaves no orphan header behind.  The header always
    // precedes its --describe prose.
    std::string header = e.testTitle;
    if (!e.testVariant.empty()) header += " [" + e.testVariant + "]";
    if (verbose)
    {
        out << "\n";
        writeLine(header);
        if (describe && (!e.testDescription.empty() || !e.testAxis.empty()))
        {
            if (!e.testDescription.empty())
                writeWrapped(metricIndent * 2, e.testDescription);
            if (!e.testAxis.empty())
                writeWrapped(metricIndent * 2, "Readings vary by " + e.testAxis + ".");
            out << "\n";
        }
        mPendingHeader.clear();
        mPendingDescription.clear();
        mPendingAxis.clear();
    }
    else
    {
        mPendingHeader = header;
        if (describe)
        {
            mPendingDescription = e.testDescription;
            mPendingAxis        = e.testAxis;
        }
        else
        {
            mPendingDescription.clear();
            mPendingAxis.clear();
        }
    }

    metricLines.clear();
    out.flush();
}

// ── Metric ─────────────────────────────────────────────────────────────────

void LoggerText::renderMetric(const LogEvent &e)
{
    // In non-verbose mode, skip unsupported/skipped/error readings —
    // they clutter the table and are only useful when debugging.
    if (!verbose && e.metric.status != ResultStatus::Ok)
        return;

    MetricLine ml;
    ml.label       = e.metric.displayLabel();
    ml.value       = e.metric.value;
    ml.status      = e.metric.status;
    ml.reason      = e.metric.reason;
    ml.description = e.metric.description;
    // Taken now, while the reading's device is the open one.
    if (compareEnabled)
        ml.baselineNote = baselineNote(e);
    // The unit context this row prints in: the reading's own when it
    // overrides its test's (an int8 row in ops inside a GEMM test in flops),
    // else the test's.  Resolved here so the print path does no unit-table
    // lookups.
    if (e.metric.hasUnit)
    {
        ml.unit.symbol   = e.metric.unit;
        ml.unit.quantity = e.metric.quantity;
    }
    else
    {
        ml.unit.symbol   = e.unit;
        ml.unit.quantity = e.quantity;
    }
    ml.direction = (e.metric.direction == Direction::FromUnit) ? e.direction
                                                               : e.metric.direction;

    metricLines.push_back(std::move(ml));
    flushMetrics();
}

// ── TestSkippedAll ─────────────────────────────────────────────────────────

void LoggerText::renderTestSkippedAll(const LogEvent &e)
{
    if (!verbose)
        return;

    // Flush the deferred header (and its --describe prose) first so the
    // skip line appears under its own test, not orphaned under the
    // previous one.
    if (!mPendingHeader.empty())
    {
        out << "\n";
        writeLine(mPendingHeader);
        if (!mPendingDescription.empty())
            writeWrapped(metricIndent * 2, mPendingDescription);
        if (!mPendingAxis.empty())
            writeWrapped(metricIndent * 2, "Readings vary by " + mPendingAxis + ".");
        if (!mPendingDescription.empty() || !mPendingAxis.empty())
            out << "\n";
        mPendingHeader.clear();
        mPendingDescription.clear();
        mPendingAxis.clear();
    }
    else
    {
        mPendingDescription.clear();
        mPendingAxis.clear();
    }

    // One line, not one per metric: a whole-test skip is a single fact, and
    // the reason is identical on every reading it stands in for.  The document
    // still records each named reading, so the file is complete.
    writeLine(metricIndent, "[" + statusTag(e.status) + "] " + e.reason);
    out.flush();
}

// ── TestEnd ────────────────────────────────────────────────────────────────

void LoggerText::renderTestEnd()
{
    flushMetrics();
    indentLevel = propIndent;
}

void LoggerText::noteClosedTest(const std::string &key)
{
    lastClosedTest = key;
}

// ── DeviceEnd ──────────────────────────────────────────────────────────────

void LoggerText::renderDeviceEnd()
{
    // Ensure any remaining metrics are flushed (shouldn't happen if well-formed)
    if (!metricLines.empty())
        flushMetrics();

    out << "\n";
    indentLevel = 0;
    out.flush();
}

// ── BackendEnd ─────────────────────────────────────────────────────────────

void LoggerText::renderBackendEnd()
{
    // Rows are held until the stanza ends (see renderTestEnd), so the last
    // test of the last device flushes here.
    flushMetrics();
    indentLevel = 0;
    out.flush();
}

// ── flushMetrics ───────────────────────────────────────────────────────────

void LoggerText::flushMetrics()
{
    if (metricLines.empty())
    {
        // The stanza produced nothing visible, so its deferred header goes —
        // and with it the header a reopen would otherwise continue under (a
        // reopened test re-heads itself instead: see renderTestBegin).
        if (!mPendingHeader.empty())
            stanzaHasHeader = false;
        mPendingHeader.clear();
        mPendingDescription.clear();
        mPendingAxis.clear();
        return;
    }

    // Print the deferred test header if we have one.  The --describe prose
    // was deferred with it so a fully-skipped test does not leave an empty
    // header behind.
    if (!mPendingHeader.empty())
    {
        out << "\n";
        writeLine(mPendingHeader);
        if (!mPendingDescription.empty())
            writeWrapped(metricIndent * 2, mPendingDescription);
        if (!mPendingAxis.empty())
            writeWrapped(metricIndent * 2, "Readings vary by " + mPendingAxis + ".");
        if (!mPendingDescription.empty() || !mPendingAxis.empty())
            out << "\n";
        mPendingHeader.clear();
        mPendingDescription.clear();
        mPendingAxis.clear();
    }
    else
    {
        mPendingDescription.clear();
        mPendingAxis.clear();
    }

    // Compute the maximum metric name width in this test.  `mergedPad` is the
    // widest label already printed in this stanza, so a test written in
    // several blocks stays one aligned table.
    int maxWidth = MIN_METRIC_PAD > mergedPad ? MIN_METRIC_PAD : mergedPad;
    for (const auto &ml : metricLines)
    {
        int w = static_cast<int>(ml.label.size());
        if (w > maxWidth)
            maxWidth = w;
    }
    mergedPad = maxWidth;

    for (const auto &ml : metricLines)
    {
        // Build padded metric name
        std::string padded = ml.label;
        while (static_cast<int>(padded.size()) < maxWidth)
            padded += ' ';

        if (ml.status == ResultStatus::Ok)
        {
            // Value and unit are chosen together: the mantissa is scaled to
            // the reading's magnitude and the unit beside it is the one that
            // matches (4476 GFLOPS prints as "4.48 TFLOPS") — the same pair
            // the GUI's value column shows.  The unit lives on the row, not
            // the test header, because a heterogeneous test's rows need not
            // share one (a GEMM test holds TFLOPS and TOPS alike).
            const ScaledValue sv = formatScaledValue(ml.value, ml.unit);

            // Print metric line without trailing newline (baseline delta may follow)
            out << indentStr(metricIndent) << padded << " : " << sv.text;
            if (!sv.unit.empty())
                out << " " << sv.unit;

            // Baseline delta on the same line (if enabled)
            out << ml.baselineNote;

            out << "\n";
        }
        else
        {
            out << indentStr(metricIndent) << padded << " : ["
                << statusTag(ml.status) << "] " << ml.reason << "\n";
        }

        // The reading's own note on the line below it, left-aligned with the
        // value column: one straight edge for every note in the test, and
        // clearly subordinate to the row it belongs to.
        if (describe && !ml.description.empty())
            writeWrapped(metricIndent * 2 + maxWidth + 3, ml.description);
    }

    metricLines.clear();
    out.flush();
}

// ── Helpers ────────────────────────────────────────────────────────────────

std::string LoggerText::indentStr(int level) const
{
    if (level <= 0)
        return "";
    return std::string(static_cast<size_t>(level) * 2, ' ');
}

void LoggerText::writeLine(int level, const std::string &text)
{
    out << indentStr(level) << text << "\n";
}

void LoggerText::writeLine(const std::string &text)
{
    writeLine(indentLevel, text);
}

void LoggerText::writeWrapped(int column, const std::string &text)
{
    if (column < 0)
        column = 0;
    const std::string pad(static_cast<size_t>(column), ' ');

    int width = DESCRIBE_WIDTH - column;
    if (width < MIN_DESCRIBE_COLUMN)
        width = MIN_DESCRIBE_COLUMN;

    // Descriptions arrive whitespace-collapsed (logger.cpp), so splitting on
    // single spaces is enough to recover the words.
    std::string line;
    std::istringstream words(text);
    std::string word;
    while (words >> word)
    {
        if (!line.empty() &&
            static_cast<int>(line.size() + 1 + word.size()) > width)
        {
            out << pad << line << "\n";
            line.clear();
        }
        if (!line.empty())
            line += ' ';
        line += word;           // a word longer than the column overruns it
    }
    if (!line.empty())
        out << pad << line << "\n";
}


// ── Text reports of a whole run ─────────────────────────────────────────────
//
// The app's Summary and compare, as text: the numbers, sides and words are
// the measure engine's and the compare model's (include/common/measures.h,
// run_compare.h), and what is laid around them here is the CLI's own.

namespace {

std::string trimmedText(const std::string &s)
{
    const char *ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

std::string joinedWith(const std::vector<std::string> &parts, const char *sep)
{
    std::string out;
    for (size_t i = 0; i < parts.size(); i++)
    {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

std::string padTo(const std::string &s, size_t width)
{
    return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

std::string scaled(double value, const UnitInfo &unit)
{
    const ScaledValue sv = formatScaledValue(value, unit);
    return sv.unit.empty() ? sv.text : sv.text + " " + sv.unit;
}

// The system as a person would name it, as the app's RunMeta.systemName: the
// `system.name` annotation, else the OS's marketing name, else the model id,
// the board or the SoC.
std::string systemName(const RunMeta &m)
{
    const auto it = m.annotations.find("system.name");
    if (it != m.annotations.end() && !trimmedText(it->second).empty())
        return trimmedText(it->second);
    for (const std::string *s : {&m.system.name, &m.system.model, &m.system.board, &m.system.soc})
        if (!trimmedText(*s).empty()) return trimmedText(*s);
    return "";
}

const PackageInfo *packageNamed(const RunDocument &run, const std::string &id)
{
    for (const PackageInfo &p : run.packages)
        if (trimmedText(p.id) == trimmedText(id)) return &p;
    return nullptr;
}

std::string gigahertz(unsigned mhz)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GHz", mhz / 1000.0);
    return buf;
}

std::string memoryText(std::uint64_t bytes)
{
    char buf[32];
    const double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    if (gb >= 1.0)
        std::snprintf(buf, sizeof(buf), "%.1f GB", gb);
    else
        std::snprintf(buf, sizeof(buf), "%.0f MB", gb * 1024.0);
    return buf;
}

// One processor's facts in a line, as the app's processorFacts() lists them.
std::string processorFacts(const xpupeak::ProcessorMeasures &g, const RunDocument &run)
{
    const ProcessorInfo *p = g.processor;
    const DeviceResult *dev = g.devices.empty() ? nullptr : g.devices.front();
    const HardwareFacts none;
    const HardwareFacts &f = p ? p->facts : (dev ? dev->facts : none);
    std::vector<std::string> parts;
    if (g.kind() == "cpu")
    {
        const CpuFacts &c = f.cpu;
        if (c.cores) parts.push_back(std::to_string(c.cores) + " cores");
        if (c.threads && c.threads != c.cores) parts.push_back(std::to_string(c.threads) + " threads");
        std::vector<std::string> kinds;
        unsigned maxMhz = 0;
        for (const CpuCluster &k : c.clusters)
        {
            maxMhz = std::max(maxMhz, k.maxMhz);
            if (!k.cores || k.kind.empty()) continue;
            const std::string tag = k.kind == "prime" ? "X" : k.kind == "performance" ? "P"
                                  : k.kind == "efficiency" ? "E" : k.kind == "low_power" ? "LP"
                                                                                    : " " + k.kind;
            kinds.push_back(std::to_string(k.cores) + tag);
        }
        if (kinds.size() > 1) parts.push_back(joinedWith(kinds, " + "));
        if (maxMhz) parts.push_back(gigahertz(maxMhz));
        if (c.sockets > 1) parts.push_back(std::to_string(c.sockets) + " sockets");
        if (!c.isa.empty()) parts.push_back(c.isa);
        for (const CpuEngine &e : c.engines)
            if (!e.name.empty()) parts.push_back(e.name);
    }
    else
    {
        if (f.computeUnits)
        {
            const std::string unit = f.computeUnit.empty() ? "unit" : f.computeUnit;
            parts.push_back(std::to_string(f.computeUnits) + " " + unit + (f.computeUnits == 1 ? "" : "s"));
        }
        if (!f.arch.empty()) parts.push_back(f.arch);
        if (f.clockMhz) parts.push_back(gigahertz(f.clockMhz));
        if (f.integrated == 1)
            parts.push_back("integrated");
        else if (f.memoryBytes)
            parts.push_back(memoryText(f.memoryBytes));
    }
    if (!p && parts.empty() && dev)
    {
        if (!dev->platform.empty() && dev->platform != dev->backend) parts.push_back(dev->platform);
        if (!dev->driver.empty()) parts.push_back(dev->driver);
    }
    if (p)
        if (const PackageInfo *pkg = packageNamed(run, p->package))
            if (!trimmedText(pkg->name).empty() && trimmedText(pkg->name) != trimmedText(p->name))
                parts.push_back("in " + trimmedText(pkg->name));
    return joinedWith(parts, ", ");
}

// A device by its backend, and by its backend's name for it where that
// backend reached the processor more than once, as the app labels it.
std::string deviceLabel(const DeviceResult &d, const std::vector<const DeviceResult *> &all)
{
    const auto same = std::count_if(all.begin(), all.end(), [&](const DeviceResult *r) {
        return r->backend == d.backend;
    });
    return same > 1 ? d.backend + " " + d.name + " #" + std::to_string(d.deviceIndex) : d.backend;
}

// A headline's reading in a person's words: its data type and qualifiers,
// or its test's title where it names neither (the app's _words()).
std::string readingWords(const xpupeak::MeasuredReading &r)
{
    const std::vector<std::string> words = xpupeak::describeReading(r);
    return words.empty() ? r.test->title : joinedWith(words, ", ");
}

std::string factorText(double factor)
{
    char buf[32];
    // Near 1, a third digit: 1.002x is not 1.00x.
    if (factor < 1.1)
        std::snprintf(buf, sizeof(buf), "%.3fx", factor);
    else if (factor < 10.0)
        std::snprintf(buf, sizeof(buf), "%.2fx", factor);
    else if (factor < 100.0)
        std::snprintf(buf, sizeof(buf), "%.1fx", factor);
    else
        std::snprintf(buf, sizeof(buf), "%.0fx", factor);
    return buf;
}

// A verdict from this run's side: "1.19x better", "2.14x worse", "the same",
// "method changed"; "better" alone where a value is not above zero.
std::string verdictText(const xpupeak::Verdict &v)
{
    if (v.methodChanged) return "method changed";
    if (v.isTie()) return "the same";
    const std::string word = v.winner == xpupeak::CompareSide::A ? "better" : "worse";
    return v.hasFactor ? factorText(v.factor) + " " + word : word;
}

std::string valueWithBackend(const xpupeak::CompareValue &v)
{
    return scaled(v.value(), v.units()) + " " + v.backend();
}

std::string groupTitle(const xpupeak::ProcessorMeasures &g)
{
    const std::string id = g.linked() ? g.id : (g.devices.empty() ? g.id : g.devices.front()->backend);
    const std::string name = g.name().empty() ? "(unnamed)" : g.name();
    return id + "  " + name;
}

std::string categoryWord(Category c)
{
    return c == Category::Unknown ? "other" : categoryString(c);
}

// One measure row: the family, then what its reading measured.
std::string keyLabel(const xpupeak::MeasureKey &key, const std::vector<std::string> &words)
{
    return words.empty() ? key.family : key.family + "  " + joinedWith(words, ", ");
}

void printKeyRows(const std::vector<xpupeak::KeyCompare> &rows, std::ostream &out)
{
    size_t labelWidth = 0, aWidth = 0, bWidth = 0;
    std::vector<std::string> labels, as, bs;
    for (const xpupeak::KeyCompare &k : rows)
    {
        labels.push_back(keyLabel(k.key, k.label()));
        as.push_back(valueWithBackend(k.a.best));
        bs.push_back(valueWithBackend(k.b.best));
        labelWidth = std::max(labelWidth, labels.back().size());
        aWidth     = std::max(aWidth, as.back().size());
        bWidth     = std::max(bWidth, bs.back().size());
    }
    labelWidth = std::min<size_t>(labelWidth, 56);
    Category last = Category::Unknown;
    for (size_t i = 0; i < rows.size(); i++)
    {
        if (i == 0 || rows[i].category != last)
            out << "    " << categoryWord(rows[i].category) << "\n";
        last = rows[i].category;
        out << "      " << padTo(labels[i], labelWidth) << "  " << padTo(as[i], aWidth) << "  vs  "
            << padTo(bs[i], bWidth) << "  " << verdictText(rows[i].verdict) << "\n";
    }
}

void printOneSided(const char *heading, const std::vector<xpupeak::MeasureRow> &rows,
                   std::ostream &out)
{
    out << "   " << heading << ":\n";
    Category last = Category::Unknown;
    for (size_t i = 0; i < rows.size(); i++)
    {
        if (i == 0 || rows[i].category != last)
            out << "    " << categoryWord(rows[i].category) << "\n";
        last = rows[i].category;
        out << "      " << keyLabel(rows[i].key, rows[i].label()) << "  "
            << valueWithBackend(rows[i].best) << "\n";
    }
}

} // namespace

void printRunSummary(const RunDocument &run, const xpupeak::HeadlineCatalog &catalog,
                     std::ostream &out)
{
    const xpupeak::RunMeasures measures = xpupeak::RunMeasures::of(run);
    out << "\nSummary: each processor's headlines, the best reading of each over every\n"
           "backend that reached it\n";
    if (run.processors.empty())
        out << "This file does not link its devices to processors, so each backend's view of\n"
               "one is listed on its own.\n";

    // Columns as wide as their widest entry over the whole summary, so every
    // processor's numbers line up.
    size_t titleWidth = 0, valueWidth = 1, wordsWidth = 0;
    for (const xpupeak::Headline &h : catalog.headlines)
        titleWidth = std::max(titleWidth, h.title.size());
    for (const xpupeak::ProcessorMeasures &g : measures.groups)
        for (const xpupeak::Headline &h : catalog.headlines)
        {
            xpupeak::HeadlineValue v;
            if (!g.headline(h, v)) continue;
            valueWidth = std::max(valueWidth, scaled(v.best.value(), v.best.units()).size());
            wordsWidth = std::max(wordsWidth, readingWords(v.best).size());
        }
    wordsWidth = std::min<size_t>(wordsWidth, 48);

    for (const xpupeak::ProcessorMeasures &g : measures.groups)
    {
        out << "\n  " << groupTitle(g);
        const std::string kind = g.kind();
        out << "  [" << (kind.empty() ? "device" : kind) << "]";
        const std::string facts = processorFacts(g, run);
        if (!facts.empty()) out << "  " << facts;
        out << "\n";

        bool any = false;
        for (const xpupeak::Headline &h : catalog.headlines)
        {
            if (!h.appliesTo(kind)) continue;
            any = true;
            out << "    " << padTo(h.title, titleWidth) << "  ";
            xpupeak::HeadlineValue v;
            if (!g.headline(h, v))
            {
                out << padTo("-", valueWidth) << "  not measured on this device\n";
                continue;
            }
            out << padTo(scaled(v.best.value(), v.best.units()), valueWidth) << "  "
                << padTo(readingWords(v.best), wordsWidth) << "  " << v.best.backend();
            if (v.byBackend.size() > 1)
                out << ", best of " << v.byBackend.size() << " backends";
            out << "\n";
        }
        if (!any)
            out << "    No headline applies to this kind of device.\n";
        if (!g.devices.empty())
        {
            std::vector<std::string> through;
            for (const DeviceResult *d : g.devices) through.push_back(deviceLabel(*d, g.devices));
            out << "    measured through " << joinedWith(through, ", ") << "\n";
        }
    }
    out << "\n";
}

void printRunCompare(const RunDocument &run, const xpupeak::CompareBaseline &baseline,
                     const xpupeak::HeadlineCatalog &catalog, bool verbose, std::ostream &out)
{
    const xpupeak::CompareRun a(run);
    const xpupeak::CompareRun b(baseline.document);
    const xpupeak::RunCompare c = xpupeak::RunCompare::of(a, b, catalog);

    const RunMeta &bm = baseline.document.meta;
    // The path as a shared file would hold it: output gets pasted into
    // issues, so no home directory's name (include/common/path_scrub.h).
    out << "\nCompared with " << xpupeak::scrubPaths(baseline.file) << "\n";
    out << "  this run: " << (systemName(run.meta).empty() ? "(unnamed system)" : systemName(run.meta))
        << "\n";
    out << "  baseline: " << (systemName(bm).empty() ? "(unnamed system)" : systemName(bm));
    std::vector<std::string> when;
    if (bm.generatedAt.size() >= 10) when.push_back(bm.generatedAt.substr(0, 10));
    if (!bm.generator.version.empty()) when.push_back(bm.generator.name + " " + bm.generator.version);
    if (!when.empty()) out << " (" << joinedWith(when, ", ") << ")";
    out << "\n";

    // Which processors it paired, and how: the first thing a compare of two
    // machines has to say.
    out << "\n Processors, paired by kind, then by model:\n";
    size_t width = 0;
    for (const xpupeak::PairCompare &p : c.pairs) width = std::max(width, groupTitle(*p.a).size());
    for (const xpupeak::ProcessorMeasures *g : c.unpairedA) width = std::max(width, groupTitle(*g).size());
    for (const xpupeak::PairCompare &p : c.pairs)
        out << "  " << padTo(groupTitle(*p.a), width) << "  with  " << groupTitle(*p.b) << "  ("
            << (p.pairedBy == xpupeak::PairedBy::Model ? "one model"
                                                       : "two models, paired by kind")
            << ")\n";
    for (const xpupeak::ProcessorMeasures *g : c.unpairedA)
        out << "  " << padTo(groupTitle(*g), width) << "  unpaired: the baseline has no other "
            << (g->kind().empty() ? "device" : g->kind()) << "\n";
    for (const xpupeak::ProcessorMeasures *g : c.unpairedB)
        out << "  " << padTo("", width) << "  unpaired in the baseline: " << groupTitle(*g) << "\n";

    for (const xpupeak::PairCompare &p : c.pairs)
    {
        out << "\n " << groupTitle(*p.a) << "  against the baseline's  " << groupTitle(*p.b) << "\n";
        // A processor one side measured nothing on -- the host CPU of a run
        // that did not reach it -- has nothing to set beside the other's.
        if (p.a->byKey.empty() || p.b->byKey.empty())
        {
            out << "   Nothing measured on it " << (p.a->byKey.empty() ? "in this run" : "in the baseline")
                << (p.a->byKey.empty() && p.b->byKey.empty() ? " or the baseline" : "") << ".\n";
            continue;
        }

        // Headlines: each side's value, which may be readings of two keys.
        size_t titleWidth = 0, aWidth = 0;
        std::vector<std::string> as;
        for (const xpupeak::HeadlineCompare &h : p.headlines)
        {
            titleWidth = std::max(titleWidth, h.headline->title.size());
            as.push_back(h.hasA ? valueWithBackend(h.a) + " (" + joinedWith(h.a.label(), ", ") + ")"
                                : "-");
            aWidth = std::max(aWidth, as.back().size());
        }
        for (size_t i = 0; i < p.headlines.size(); i++)
        {
            const xpupeak::HeadlineCompare &h = p.headlines[i];
            out << "   " << padTo(h.headline->title, titleWidth) << "  " << padTo(as[i], aWidth) << "  vs  "
                << (h.hasB ? valueWithBackend(h.b) + " (" + joinedWith(h.b.label(), ", ") + ")" : "-");
            if (h.hasVerdict) out << "  " << verdictText(h.verdict);
            out << "\n";
        }

        if (p.byReadings())
        {
            // One model: the devices both runs saw on it compared reading by
            // reading, beside the rows above.
            size_t n = 0, better = 0, worse = 0, same = 0, changed = 0, onlyA = 0, onlyB = 0;
            for (const xpupeak::DeviceCompare &d : p.devices)
            {
                for (const xpupeak::ReadingCompare &r : d.readings)
                {
                    n++;
                    const double base = r.b.value();
                    if (r.verdict.methodChanged)
                        changed++;
                    else if (r.verdict.isTie() ||
                             (r.verdict.hasRatio && base != 0.0 &&
                              std::fabs(r.a.value() - base) / std::fabs(base) < 0.0005))
                        same++;
                    else if (r.verdict.winner == xpupeak::CompareSide::A)
                        better++;
                    else
                        worse++;
                }
                onlyA += d.onlyA.size();
                onlyB += d.onlyB.size();
            }
            out << "   One model: " << n << " readings on " << p.devices.size()
                << (p.devices.size() == 1 ? " device" : " devices")
                << " compared reading by reading, beside the rows above:\n   " << better
                << " better, " << worse << " worse, " << same << " within 0.05%";
            if (changed) out << ", " << changed << " method changed";
            out << ".\n";
            if (onlyA || onlyB)
                out << "   Readings only in this run: " << onlyA << "; only in the baseline: " << onlyB
                    << ".\n";
            for (const DeviceResult *d : p.devicesOnlyA)
                out << "   Only this run saw " << d->backend << " " << d->name << ".\n";
            for (const DeviceResult *d : p.devicesOnlyB)
                out << "   Only the baseline saw " << d->backend << " " << d->name << ".\n";
            if (verbose && !p.both.empty())
            {
                out << "   Measured on both, by measure (" << p.both.size() << "):\n";
                printKeyRows(p.both, out);
            }
        }
        else
        {
            // Two models, or no device both saw: by measure key, the best of
            // each side's backends.
            if (p.sameModel())
                out << "   One model, but no device both runs saw on it: by measure.\n";
            else
            {
                // Rows above that met a baseline device of the same name
                // compared two parts that share it (an M1 Pro of 8+2 cores
                // and one of 6+2): these are the comparable figures.
                bool named = false;
                for (const DeviceResult *d : p.a->devices)
                    for (const DeviceResult *e : p.b->devices)
                        if (d->backend == e->backend && d->name == e->name &&
                            a.occurrenceOf(*d) == b.occurrenceOf(*e))
                            named = true;
                if (named)
                    out << "   Two models of one name: the notes beside the rows above set two parts\n"
                           "   side by side; by measure:\n";
            }
            out << "   Measured on both (" << p.both.size() << (p.both.size() == 1 ? " measure" : " measures")
                << "):\n";
            if (!p.both.empty()) printKeyRows(p.both, out);
        }
        if (verbose)
        {
            if (!p.onlyA.empty()) printOneSided("Only in this run", p.onlyA, out);
            if (!p.onlyB.empty()) printOneSided("Only in the baseline", p.onlyB, out);
        }
        else if (!p.onlyA.empty() || !p.onlyB.empty())
        {
            out << "   Measures only in this run: " << p.onlyA.size() << "; only in the baseline: "
                << p.onlyB.size() << " (--verbose lists them).\n";
        }
    }
    out << "\n";
}
