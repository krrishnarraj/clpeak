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
    // Reopening the test that just closed, with its header still standing:
    // keep accumulating into the same stanza, under the header already
    // printed for it.  Rows carry their own units, so a reopen in a different
    // one (a GEMM test's integer phase) joins the same table rather than
    // starting a look-alike stanza beside it.
    if (e.reopened && e.testKey() == lastClosedTest && stanzaHasHeader)
        return;

    flushMetrics();
    mergedPad       = 0;
    stanzaHasHeader = true;

    metricIndent = propIndent + 1;   // metrics indented one more than props
    indentLevel  = propIndent;       // test header at prop level

    // Build the header.  In verbose mode the header is printed
    // immediately so that backend CLPEAK_VLOG lines (stderr) land
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
    ml.baselineKey = baselineKey(e.backend, e.platform, e.deviceKey(),
                                 e.testKey(), e.metric.id);
    // The unit context this row prints in: the reading's own when it
    // overrides its test's (an int8 row in ops inside a GEMM test in flops),
    // else the test's.  Resolved here so the print path does no unit-table
    // lookups.
    if (e.metric.hasUnit)
    {
        ml.unit.symbol   = e.metric.unit;
        ml.unit.quantity = e.metric.quantity;
        ml.unit.scale    = e.metric.scale;
    }
    else
    {
        ml.unit.symbol   = e.unit;
        ml.unit.quantity = e.quantity;
        ml.unit.scale    = e.scale;
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
            if (compareEnabled)
                printBaselineDelta(ml.baselineKey, ml.value, ml.direction,
                                   ml.unit);

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
                                    Direction direction, const UnitInfo &unit)
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

    // The baseline prints through the row's own scaling: "was 4476" beside a
    // row that says 4.48 TFLOPS reads as a collapse that did not happen.
    const ScaledValue sv = formatScaledValue(base, unit);
    out << "  (was " << sv.text << ", " << sign << std::fixed
        << std::setprecision(1) << absDelta << "%";
    // Below this the figure prints as 0.0%, and calling run-to-run noise
    // "better" or "worse" would be inventing a result.
    if (absDelta >= 0.05)
        out << (improved ? " better" : " worse");
    out << ")";
}
