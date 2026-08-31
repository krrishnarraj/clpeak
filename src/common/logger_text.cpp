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
    case LogEvent::Kind::TestEnd:
        renderTestEnd();
        noteClosedTest(e.testKey());
        break;
    case LogEvent::Kind::DeviceEnd:      renderDeviceEnd();       break;
    case LogEvent::Kind::BackendEnd:     renderBackendEnd();      break;
    case LogEvent::Kind::Note:
        // Before the message, or it would appear above rows that were measured
        // before it.
        flushMetrics();
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
    // The unit this opening declared, which on a reopen need not be the
    // test's: the integer phase of a GEMM test reports ops, not flops.
    const std::string openUnit = e.openedUnit.empty() ? e.unit : e.openedUnit;

    // Reopening the test that just closed, in the same unit: keep accumulating
    // into the same stanza, under the header already printed for it.  Rows
    // pending from the previous block are deliberately NOT flushed here -- the
    // whole point is that the stanza's label column is computed once, over all
    // of its readings together.
    if (e.reopened && e.testKey() == lastClosedTest && openUnit == stanzaUnit)
        return;

    flushMetrics();
    mergedPad  = 0;
    stanzaUnit = openUnit;

    metricIndent = propIndent + 1;   // metrics indented one more than props
    indentLevel  = propIndent;       // test header at prop level

    // Build the header but defer printing until flushMetrics confirms
    // there are actual metric lines to display (not all readings were
    // skipped/unsupported in the default non-verbose mode).
    std::string header = e.testTitle;
    if (!e.testVariant.empty()) header += " [" + e.testVariant + "]";
    if (!openUnit.empty())      header += " (" + openUnit + ")";
    mPendingHeader = header;

    // What the test measures, under its own header and one step in from it,
    // then a blank line so the readings below still read as a block.
    if (describe && (!e.testDescription.empty() || !e.testAxis.empty()))
    {
        if (!e.testDescription.empty())
            writeWrapped(metricIndent * 2, e.testDescription);
        // The axis is the shortest possible answer to "why are there eight of
        // these?", and the one line that tells a reader whether the readings
        // below are variants of one measurement or separate measurements.
        if (!e.testAxis.empty())
            writeWrapped(metricIndent * 2, "Readings vary by " + e.testAxis + ".");
        out << "\n";
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
    ml.baselineKey = baselineKey(e.backend, e.platform, e.deviceKey(),
                                 e.testKey(), e.metric.id);
    // Only when it actually differs from the header above these rows:
    // repeating what the header already says would just be noise.
    if (e.metric.hasUnit && e.metric.unit != stanzaUnit)
        ml.unitSuffix = e.metric.unit;
    ml.direction = (e.metric.direction == Direction::FromUnit) ? e.direction
                                                               : e.metric.direction;
    metricLines.push_back(std::move(ml));
}

// ── TestSkippedAll ─────────────────────────────────────────────────────────

void LoggerText::renderTestSkippedAll(const LogEvent &e)
{
    if (!verbose)
        return;

    // One line, not one per metric: a whole-test skip is a single fact, and
    // the reason is identical on every reading it stands in for.  The document
    // still records each named reading, so the file is complete.
    writeLine(metricIndent, "[" + statusTag(e.status) + "] " + e.reason);
    out.flush();
}

// ── TestEnd ────────────────────────────────────────────────────────────────

void LoggerText::renderTestEnd()
{
    // A test's rows print when it ends, before anything the next test does.
    // Holding them any longer lets that next test's --verbose setup lines
    // (which go to stderr, unbuffered) land between a header and its readings.
    //
    // This is only safe because a merged family opens ONE scope for all its
    // readings -- see the tensor-core and cooperative-matrix runners.  Were it
    // to close and reopen per data type, each block would flush separately and
    // the label column would widen down the page.
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
        mPendingHeader.clear();
        return;
    }

    // Print the deferred test header if we have one.
    if (!mPendingHeader.empty())
    {
        out << "\n";
        writeLine(mPendingHeader);
        mPendingHeader.clear();
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
            // Format value
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << ml.value;

            // Print metric line without trailing newline (baseline delta may follow)
            out << indentStr(metricIndent) << padded << " : " << ss.str();
            if (!ml.unitSuffix.empty())
                out << " " << ml.unitSuffix;

            // Baseline delta on the same line (if enabled)
            if (compareEnabled)
                printBaselineDelta(ml.baselineKey, ml.value, ml.direction);

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

void LoggerText::printBaselineDelta(const std::string &key, double value,
                                    Direction direction)
{
    auto it = baseline.find(key);
    if (it == baseline.end())
        return;

    const double base  = it->second;
    const double delta = (base != 0.0) ? 100.0 * (value - base) / base : 0.0;

    const char   sign     = (delta >= 0.0) ? '+' : '-';
    const double absDelta = (delta < 0.0)  ? -delta : delta;

    // Which way a reading moved is not which way is better: on a latency or
    // numeric-error row, +3% is a regression.  A bare signed percentage has
    // always read as good news, so say which it is.
    const bool improved = (direction == Direction::LowerIsBetter) ? (delta < 0.0)
                                                                  : (delta > 0.0);

    out << "  (was " << std::fixed << std::setprecision(2) << base
        << ", " << sign << std::setprecision(1) << absDelta << "%";
    // Below this the figure prints as 0.0%, and calling run-to-run noise
    // "better" or "worse" would be inventing a result.
    if (absDelta >= 0.05)
        out << (improved ? " better" : " worse");
    out << ")";
}
