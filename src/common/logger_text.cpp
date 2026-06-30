#include <common/logger_text.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

// ── Constants ──────────────────────────────────────────────────────────────

static const int MIN_METRIC_PAD = 8;   // minimum column width for metric names

// ── note ───────────────────────────────────────────────────────────────────

void LoggerText::note(const std::string &msg)
{
    out << msg;
    out.flush();
}

// ── onBackendBegin ─────────────────────────────────────────────────────────

void LoggerText::onBackendBegin(const std::string &name)
{
    indentLevel = 0;
    out << "Backend: " << name << "\n";
    out.flush();
}

// ── onDeviceBegin ──────────────────────────────────────────────────────────

void LoggerText::onDeviceBegin(const std::string &name,
                               const std::string &platform,
                               const std::string &driverVersion,
                               const std::vector<Prop> &props,
                               bool _showPlatformLine,
                               int platformIndex,
                               int deviceIndex)
{
    showPlatformLine = _showPlatformLine;

    // Indent setup: platform line (if shown) pushes device one level deeper
    int deviceIndent = showPlatformLine ? 2 : 1;
    propIndent       = deviceIndent + 1;   // props indented under device
    indentLevel      = deviceIndent;

    if (showPlatformLine)
    {
        std::string pline = platformIndex >= 0
            ? "Platform " + std::to_string(platformIndex) + ": " + platform
            : "Platform: " + platform;
        writeLine(1, pline);
    }

    std::string dline = deviceIndex >= 0
        ? "Device " + std::to_string(deviceIndex) + ": " + name
        : "Device: " + name;
    writeLine(deviceIndent, dline);

    // Properties
    if (!driverVersion.empty())
    {
        std::string dvLabel = "Driver version";
        while (dvLabel.size() < 17)
            dvLabel += ' ';
        writeLine(propIndent, dvLabel + ": " + driverVersion);
    }

    for (const auto &prop : props)
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

// ── onTestBegin ────────────────────────────────────────────────────────────

void LoggerText::onTestBegin(const std::string & /*tag*/,
                             const std::string &display,
                             const std::string &unit)
{
    out << "\n";

    metricIndent = propIndent + 1;   // metrics indented one more than props
    indentLevel  = propIndent;       // test header at prop level

    // Build header: display name + unit in caps, e.g. "Global memory bandwidth (GBPS)"
    std::string header = display;
    if (!unit.empty())
    {
        std::string u = unit;
        for (auto &c : u)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        header += " (" + u + ")";
    }

    writeLine(header);
    metricLines.clear();
    out.flush();
}

// ── onMetricEmitted ────────────────────────────────────────────────────────

void LoggerText::onMetricEmitted(const ResultEntry &e, float value, bool subMetric)
{
    metricLines.push_back({MetricLine::Ok, e.metric, value,
                           ResultStatus::Ok, "", subMetric});
}

// ── onMetricSkipped ────────────────────────────────────────────────────────

void LoggerText::onMetricSkipped(const ResultEntry &e)
{
    metricLines.push_back({MetricLine::Skipped, e.metric, 0.0f,
                           e.status, e.reason, false});
}

// ── onTestSkippedAll ───────────────────────────────────────────────────────

void LoggerText::onTestSkippedAll(ResultStatus status, const std::string &reason)
{
    std::string tag;
    switch (status)
    {
    case ResultStatus::Unsupported: tag = "unsupported"; break;
    case ResultStatus::Skipped:     tag = "skipped";     break;
    case ResultStatus::Error:       tag = "error";       break;
    default:                        tag = "unknown";     break;
    }

    writeLine(metricIndent, "[" + tag + "] " + reason);
    out.flush();
}

// ── onTestEnd ──────────────────────────────────────────────────────────────

void LoggerText::onTestEnd()
{
    flushMetrics();
    indentLevel = propIndent;
}

// ── onDeviceEnd ────────────────────────────────────────────────────────────

void LoggerText::onDeviceEnd()
{
    // Ensure any remaining metrics are flushed (shouldn't happen if well-formed)
    if (!metricLines.empty())
        flushMetrics();

    out << "\n";
    indentLevel = 0;
    out.flush();
}

// ── onBackendEnd ───────────────────────────────────────────────────────────

void LoggerText::onBackendEnd()
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

        if (ml.kind == MetricLine::Ok)
        {
            // Format value
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << ml.value;

            // Print metric line without trailing newline (baseline delta may follow)
            out << indentStr(lineIndent) << padded << " : " << ss.str();

            // Baseline delta on the same line (if enabled)
            if (compareEnabled)
                printBaselineDelta(ml.metric, ml.value);

            out << "\n";
        }
        else
        {
            std::string tag;
            switch (ml.status)
            {
            case ResultStatus::Unsupported: tag = "unsupported"; break;
            case ResultStatus::Skipped:     tag = "skipped";     break;
            case ResultStatus::Error:       tag = "error";       break;
            default:                        tag = "unknown";     break;
            }

            out << indentStr(lineIndent) << padded << " : ["
                << tag << "] " << ml.reason << "\n";
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

void LoggerText::printBaselineDelta(const std::string &metric, float value)
{
    // Build key from current context
    std::string key = curBackend + "/" + curPlatform + "/" +
                      curDevice + "/" + categoryString(curCategory) +
                      "/" + curTest + "/" + metric;

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
