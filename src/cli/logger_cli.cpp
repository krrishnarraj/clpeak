#include <cli/logger_cli.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

// ── Constants ──────────────────────────────────────────────────────────────

static const int MIN_METRIC_PAD = 8;   // minimum column width for metric names

// ── note ───────────────────────────────────────────────────────────────────

void LoggerCli::note(const std::string &msg)
{
    std::cout << msg;
    std::cout.flush();
}

// ── onBackendBegin ─────────────────────────────────────────────────────────

void LoggerCli::onBackendBegin(const std::string &name)
{
    indentLevel = 0;
    std::cout << "Backend: " << name << "\n";
    std::cout.flush();
}

// ── onDeviceBegin ──────────────────────────────────────────────────────────

void LoggerCli::onDeviceBegin(const std::string &name,
                              const std::string &platform,
                              const std::string &driverVersion,
                              const std::vector<Prop> &props,
                              bool _showPlatformLine)
{
    showPlatformLine = _showPlatformLine;

    // Indent setup: platform line (if shown) pushes device one level deeper
    int deviceIndent = showPlatformLine ? 2 : 1;
    propIndent       = deviceIndent + 1;   // props indented under device
    indentLevel      = deviceIndent;

    if (showPlatformLine)
    {
        writeLine(1, "Platform: " + platform);
    }

    writeLine(deviceIndent, "Device: " + name);

    // Properties
    if (!driverVersion.empty())
        writeLine(propIndent, "Driver version  : " + driverVersion);

    for (const auto &prop : props)
    {
        std::string line = prop.key;
        // Right-align key to match "Driver version  " width (17 chars)
        while (line.size() < 17)
            line += ' ';
        line += ": " + prop.value;
        writeLine(propIndent, line);
    }

    std::cout.flush();
}

// ── onTestBegin ────────────────────────────────────────────────────────────

void LoggerCli::onTestBegin(const std::string & /*tag*/,
                            const std::string &display,
                            const std::string & /*unit*/)
{
    std::cout << "\n";

    metricIndent = propIndent + 1;   // metrics indented one more than props
    indentLevel  = propIndent;       // test header at prop level

    writeLine(display);
    metricLines.clear();
    std::cout.flush();
}

// ── onMetricEmitted ────────────────────────────────────────────────────────

void LoggerCli::onMetricEmitted(const ResultEntry &e, float value, bool subMetric)
{
    metricLines.push_back({MetricLine::Ok, e.metric, value,
                           ResultStatus::Ok, "", subMetric});
}

// ── onMetricSkipped ────────────────────────────────────────────────────────

void LoggerCli::onMetricSkipped(const ResultEntry &e)
{
    metricLines.push_back({MetricLine::Skipped, e.metric, 0.0f,
                           e.status, e.reason, false});
}

// ── onTestSkippedAll ───────────────────────────────────────────────────────

void LoggerCli::onTestSkippedAll(ResultStatus status, const std::string &reason)
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
    std::cout.flush();
}

// ── onTestEnd ──────────────────────────────────────────────────────────────

void LoggerCli::onTestEnd()
{
    flushMetrics();
    indentLevel = propIndent;
}

// ── onDeviceEnd ────────────────────────────────────────────────────────────

void LoggerCli::onDeviceEnd()
{
    // Ensure any remaining metrics are flushed (shouldn't happen if well-formed)
    if (!metricLines.empty())
        flushMetrics();

    std::cout << "\n";
    indentLevel = 0;
    std::cout.flush();
}

// ── onBackendEnd ───────────────────────────────────────────────────────────

void LoggerCli::onBackendEnd()
{
    indentLevel = 0;
    std::cout.flush();
}

// ── flushMetrics ───────────────────────────────────────────────────────────

void LoggerCli::flushMetrics()
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
            std::cout << indentStr(lineIndent) << padded << " : " << ss.str();

            // Baseline delta on the same line (if enabled)
            if (compareEnabled)
                printBaselineDelta(ml.metric, ml.value);

            std::cout << "\n";
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

            std::cout << indentStr(lineIndent) << padded << " : ["
                      << tag << "] " << ml.reason << "\n";
        }
    }

    metricLines.clear();
    std::cout.flush();
}

// ── Helpers ────────────────────────────────────────────────────────────────

std::string LoggerCli::indentStr(int level) const
{
    if (level <= 0)
        return "";
    return std::string(static_cast<size_t>(level) * 2, ' ');
}

void LoggerCli::writeLine(int level, const std::string &text)
{
    std::cout << indentStr(level) << text << "\n";
}

void LoggerCli::writeLine(const std::string &text)
{
    writeLine(indentLevel, text);
}

void LoggerCli::printBaselineDelta(const std::string &metric, float value)
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

    std::cout << "  (was " << std::fixed << std::setprecision(2) << base
              << ", " << sign << std::setprecision(1) << absDelta << "%)";
}
