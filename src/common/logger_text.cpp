#include <common/logger_text.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

// ── Constants ──────────────────────────────────────────────────────────────

static const int MIN_METRIC_PAD = 8;   // minimum column width for metric names

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
    metricLines.clear();
    out.flush();
}

// ── Metric ─────────────────────────────────────────────────────────────────

void LoggerText::renderMetric(const LogEvent &e)
{
    metricLines.push_back({e.entry.metric, e.entry.value, e.entry.status,
                           e.entry.reason, e.subMetric, e.entry.key()});
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
