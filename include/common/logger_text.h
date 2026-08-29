#ifndef LOGGER_TEXT_HPP
#define LOGGER_TEXT_HPP

#include <common/logger.h>
#include <iostream>
#include <vector>

// ── Text logger ────────────────────────────────────────────────────────────
//
// Renders the LogEvent stream as indented, column-aligned text with optional
// baseline-comparison deltas, and — under --describe — the plain-language
// documentation authored with each test and reading, word-wrapped and aligned
// beneath what it explains.  Output is written to an injectable std::ostream
// (defaults to std::cout), so the same formatting drives the desktop CLI and
// any future text channel (file export, …).
//
// Unavailable readings stay inline here, as `[unsupported] reason` in the
// table where they were measured: a terminal is read top to bottom, and the
// gap is most informative next to the readings that surround it.  The GUI
// collects them into a section of their own instead, where a scrollable page
// makes that the better read.
//
// The results tree itself is accumulated by the base class in `doc` — see
// run_document.h — for the host to save.

class LoggerText : public logger
{
public:
    // `out` is captured by reference and must outlive this logger.
    // `describe` (--describe) adds the plain-language explanation of each test
    // and each reading; without it the output is the bare table.
    explicit LoggerText(std::ostream &out = std::cout,
                        std::string compareFileName = "",
                        bool describe = false)
        : logger(std::move(compareFileName)), out(out), describe(describe) {}

protected:
    void onEvent(const LogEvent &e) override;

    // Destination stream for all formatted output.
    std::ostream &out;

    // --describe: render the documentation alongside the readings.
    bool describe = false;

private:
    // ── Per-event renderers ──────────────────────────────────────────────

    void renderBackendBegin(const LogEvent &e);
    void renderDeviceBegin(const LogEvent &e);
    void renderTestBegin(const LogEvent &e);
    void renderMetric(const LogEvent &e);
    void renderTestSkippedAll(const LogEvent &e);
    void renderTestEnd();
    void renderDeviceEnd();
    void renderBackendEnd();

    // ── Indentation state ────────────────────────────────────────────────

    int indentLevel  = 0;    // current base indent (0, 1, 2, 3, …)
    int propIndent   = 0;    // indent for device properties
    int metricIndent = 0;    // indent for metric lines

    // ── Metric buffering (for aligned columns within a test) ─────────────

    struct MetricLine {
        std::string  label;        // what to print in the name column
        double       value;        // valid when status == Ok
        ResultStatus status;
        std::string  reason;       // valid when status != Ok
        std::string  baselineKey;  // --compare lookup
        std::string  description;  // --describe: what this reading measures

        // Printed after the value when this reading overrides its test's
        // unit, so a TOPS row inside a TFLOPS test is not read as TFLOPS.
        std::string  unitSuffix;

        // Which way is better for this reading, so a compare delta can say
        // so rather than leaving a signed percentage to be misread.
        Direction    direction;
    };

    std::vector<MetricLine> metricLines;

    void flushMetrics();

    // ── Helpers ──────────────────────────────────────────────────────────

    std::string indentStr(int level) const;
    void writeLine(int level, const std::string &text);
    void writeLine(const std::string &text);  // uses current indentLevel
    void printBaselineDelta(const std::string &key, double value,
                            Direction direction);

    /// Write `text` as prose, word-wrapped to the output width and left-aligned
    /// at `column` spaces (a raw column, not an indent level, so a reading's
    /// note can line up under its value).
    void writeWrapped(int column, const std::string &text);
};

#endif  // LOGGER_TEXT_HPP
