#ifndef LOGGER_TEXT_HPP
#define LOGGER_TEXT_HPP

#include <common/logger.h>
#include <iostream>
#include <vector>

// ── Text logger ────────────────────────────────────────────────────────────
//
// Renders the LogEvent stream as indented, column-aligned text with optional
// baseline-comparison deltas.  Output is written to an injectable
// std::ostream (defaults to std::cout), so the same formatting drives the
// desktop CLI and any future text channel (file export, …).
//
// Accumulated ResultEntry rows are stored in the base-class `results` member
// for later file dump (JSON / CSV / XML) — see result_store.h.

class LoggerText : public logger
{
public:
    // `out` is captured by reference and must outlive this logger.
    explicit LoggerText(std::ostream &out = std::cout,
                        std::string compareFileName = "")
        : logger(std::move(compareFileName)), out(out) {}

protected:
    void onEvent(const LogEvent &e) override;

    // Destination stream for all formatted output.
    std::ostream &out;

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
        std::string  metric;
        float        value;        // valid when status == Ok
        ResultStatus status;
        std::string  reason;       // valid when status != Ok
        bool         subMetric;
        std::string  baselineKey;  // ResultEntry::key() for compare lookups
    };

    std::vector<MetricLine> metricLines;

    void flushMetrics();

    // ── Helpers ──────────────────────────────────────────────────────────

    std::string indentStr(int level) const;
    void writeLine(int level, const std::string &text);
    void writeLine(const std::string &text);  // uses current indentLevel
    void printBaselineDelta(const std::string &key, float value);
};

#endif // LOGGER_TEXT_HPP
