#include <common/logger_text.h>
#include <algorithm>
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
    case LogEvent::Kind::TestEnd:        renderTestEnd();         break;
    case LogEvent::Kind::DeviceEnd:      renderDeviceEnd();       break;
    case LogEvent::Kind::BackendEnd:     renderBackendEnd();      break;
    case LogEvent::Kind::Note:
        out << e.message;
        out.flush();
        break;
    }
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
    // Defence in depth: rows are buffered until TestEnd, so anything still
    // pending here belongs to a test that was not closed.  The scope layer
    // now closes it for us (logger::closeOpenTest), but flush rather than
    // clear so no measured row can ever be discarded silently.
    flushMetrics();

    out << "\n";

    metricIndent = propIndent + 1;   // metrics indented one more than props
    indentLevel  = propIndent;       // test header at prop level

    // Build header: display name + unit in caps, e.g. "Global memory bandwidth (GBPS)"
    std::string header = e.testDisplay;
    if (!e.unit.empty())
    {
        std::string u = e.unit;
        for (auto &c : u)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        header += " (" + u + ")";
    }

    writeLine(header);

    // What the test measures, under its own header and one step in from it,
    // then a blank line so the readings below still read as a block.
    if (describe && !e.testDescription.empty())
    {
        writeWrapped(metricIndent * 2, e.testDescription);
        out << "\n";
    }

    metricLines.clear();
    out.flush();
}

// ── Metric ─────────────────────────────────────────────────────────────────

void LoggerText::renderMetric(const LogEvent &e)
{
    metricLines.push_back({e.entry.metric, e.entry.value, e.entry.status,
                           e.entry.reason, e.subMetric, e.entry.key(),
                           e.entry.metricDescription});
}

// ── TestSkippedAll ─────────────────────────────────────────────────────────

void LoggerText::renderTestSkippedAll(const LogEvent &e)
{
    writeLine(metricIndent, "[" + statusTag(e.status) + "] " + e.reason);
    out.flush();
}

// ── TestEnd ────────────────────────────────────────────────────────────────

void LoggerText::renderTestEnd()
{
    flushMetrics();
    indentLevel = propIndent;
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
    indentLevel = 0;
    out.flush();
}

// ── flushMetrics ───────────────────────────────────────────────────────────

void LoggerText::flushMetrics()
{
    if (metricLines.empty())
        return;

    // Compute the maximum metric name width in this test
    int maxWidth = MIN_METRIC_PAD;
    for (const auto &ml : metricLines)
    {
        int w = static_cast<int>(ml.metric.size());
        if (w > maxWidth)
            maxWidth = w;
    }

    for (const auto &ml : metricLines)
    {
        int lineIndent = ml.subMetric ? metricIndent + 1 : metricIndent;
        int padTarget  = ml.subMetric ? maxWidth - 2 : maxWidth;
        if (padTarget < MIN_METRIC_PAD)
            padTarget = MIN_METRIC_PAD;

        // Build padded metric name
        std::string padded = ml.metric;
        while (static_cast<int>(padded.size()) < padTarget)
            padded += ' ';

        if (ml.status == ResultStatus::Ok)
        {
            // Format value
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << ml.value;

            // Print metric line without trailing newline (baseline delta may follow)
            out << indentStr(lineIndent) << padded << " : " << ss.str();

            // Baseline delta on the same line (if enabled)
            if (compareEnabled)
                printBaselineDelta(ml.baselineKey, ml.value);

            out << "\n";
        }
        else
        {
            out << indentStr(lineIndent) << padded << " : ["
                << statusTag(ml.status) << "] " << ml.reason << "\n";
        }

        // The reading's own note on the line below it, left-aligned with the
        // value column: one straight edge for every note in the test, and
        // clearly subordinate to the row it belongs to.
        if (describe && !ml.description.empty())
            writeWrapped(lineIndent * 2 + padTarget + 3, ml.description);
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

void LoggerText::printBaselineDelta(const std::string &key, float value)
{
    auto it = baseline.find(key);
    if (it == baseline.end())
        return;

    float base  = it->second;
    float delta = (base != 0.0f) ? 100.0f * (value - base) / base : 0.0f;

    char  sign     = (delta >= 0.0f) ? '+' : '-';
    float absDelta = (delta < 0.0f)  ? -delta : delta;

    out << "  (was " << std::fixed << std::setprecision(2) << base
        << ", " << sign << std::setprecision(1) << absDelta << "%)";
}
