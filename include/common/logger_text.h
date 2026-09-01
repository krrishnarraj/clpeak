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
                        bool describe = false,
                        bool verbose = false)
        : logger(std::move(compareFileName)), out(out), describe(describe), verbose(verbose) {}

protected:
    void onEvent(const LogEvent &e) override;

    // Destination stream for all formatted output.
    std::ostream &out;

    // --describe: render the documentation alongside the readings.
    bool describe = false;

    // --verbose: print skipped/unsupported/error readings in default
    // mode they are hidden; only shown when --verbose is passed.
    bool verbose = false;

private:
    // ── Per-event renderers ──────────────────────────────────────────────

    void renderBackendBegin(const LogEvent &e);
    void renderDeviceBegin(const LogEvent &e);
    void renderTestBegin(const LogEvent &e);
    void renderMetric(const LogEvent &e);
    void renderTestSkippedAll(const LogEvent &e);
    void renderTestEnd();
    void noteClosedTest(const std::string &key);
    void renderDeviceEnd();
    void renderBackendEnd();

    // ── Indentation state ────────────────────────────────────────────────

    int indentLevel  = 0;    // current base indent (0, 1, 2, 3, …)
    int propIndent   = 0;    // indent for device properties
    int metricIndent = 0;    // indent for metric lines

    // ── Metric formatting ────────────────────────────────────────────────

    struct MetricLine {
        std::string  label;        // what to print in the name column
        double       value;        // valid when status == Ok
        ResultStatus status;
        std::string  reason;       // valid when status != Ok
        std::string  baselineKey;  // --compare lookup
        std::string  description;  // --describe: what this reading measures

        // The unit context this row prints in: its own when the reading
        // overrides its test's (an int8 row in ops inside a GEMM test in
        // flops), else the test's.  The unit is printed beside the value,
        // auto-scaled to the magnitude, so an ops row is never read as
        // flops no matter which test it sits in.
        UnitInfo     unit;

        // Which way is better for this reading, so a compare delta can say
        // so rather than leaving a signed percentage to be misread.
        Direction    direction;
    };

    std::vector<MetricLine> metricLines;

    void flushMetrics();

    // ── Continuing a reopened test ───────────────────────────────────────
    //
    // A backend may close a test and immediately reopen it to add more
    // readings — Vulkan measures each cooperative-matrix data type in its own
    // #ifdef block, all of them into one test.  Printing the header again for
    // each would turn one test into seven look-alike stanzas, so a reopen of
    // the test that just closed continues the block.
    //
    // Every row carries its own unit, so a reopen in a DIFFERENT unit (a GEMM
    // test's floating-point phase then its integer one) continues the same
    // stanza too: one aligned table, each row labelled with what it was
    // measured in.  `stanzaHasHeader` guards the one reopen that must not
    // continue — a stanza whose deferred header was dropped because none of
    // its readings were visible (non-verbose, all skipped) re-heads on
    // reopen, or its rows would sit orphaned under the previous test's.
    //
    // Each metric flushes immediately; `mergedPad` tracks the widest label
    // seen so far in the stanza so the column widens incrementally.
    std::string lastClosedTest;
    bool        stanzaHasHeader = false;
    int         mergedPad = 0;

    // Header deferred until we know there are actual metric lines to
    // print (i.e. not all readings were skipped/unsupported in the
    // default non-verbose mode).
    std::string mPendingHeader;
    std::string mPendingDescription;
    std::string mPendingAxis;

    // ── Helpers ──────────────────────────────────────────────────────────

    std::string indentStr(int level) const;
    void writeLine(int level, const std::string &text);
    void writeLine(const std::string &text);  // uses current indentLevel
    void printBaselineDelta(const std::string &key, double value,
                            Direction direction, const UnitInfo &unit);

    /// Write `text` as prose, word-wrapped to the output width and left-aligned
    /// at `column` spaces (a raw column, not an indent level, so a reading's
    /// note can line up under its value).
    void writeWrapped(int column, const std::string &text);
};

#endif  // LOGGER_TEXT_HPP
