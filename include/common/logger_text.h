#ifndef LOGGER_TEXT_HPP
#define LOGGER_TEXT_HPP

#include <common/logger.h>
#include <iostream>
#include <vector>

// ── Text logger ────────────────────────────────────────────────────────────
//
// Formats benchmark output as indented, column-aligned text with optional
// baseline-comparison deltas.  Output is written to an injectable std::ostream
// (defaults to std::cout), so the same formatting drives:
//   - the desktop CLI (std::cout),
//   - the Android app (an ostream backed by a logcat streambuf),
//   - and any future text channel (iOS, file export, …).
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

    void note(const std::string &msg) override;

protected:
    void onBackendBegin(const std::string &name) override;
    void onDeviceBegin(const std::string &name,
                       const std::string &platform,
                       const std::string &driverVersion,
                       const std::vector<Prop> &props,
                       bool showPlatformLine,
                       int platformIndex,
                       int deviceIndex) override;
    void onTestBegin(const std::string &tag,
                     const std::string &display,
                     const std::string &unit) override;
    void onMetricEmitted(const ResultEntry &e,
                         float value,
                         bool subMetric) override;
    void onMetricSkipped(const ResultEntry &e) override;
    void onTestSkippedAll(ResultStatus status,
                          const std::string &reason) override;
    void onTestEnd() override;
    void onDeviceEnd() override;
    void onBackendEnd() override;

    // Destination stream for all formatted output.
    std::ostream &out;

private:
    // ── Indentation state ────────────────────────────────────────────────

    int  indentLevel      = 0;    // current base indent (0, 1, 2, 3, …)
    int  propIndent       = 0;    // indent for device properties
    int  metricIndent     = 0;    // indent for metric lines
    bool showPlatformLine = false;

    // ── Metric buffering (for aligned columns within a test) ─────────────

    struct MetricLine {
        enum Kind { Ok, Skipped };

        Kind         kind;
        std::string  metric;
        float        value;       // valid when kind == Ok
        ResultStatus status;      // valid when kind == Skipped
        std::string  reason;      // valid when kind == Skipped
        bool         subMetric;
    };

    std::vector<MetricLine> metricLines;

    void flushMetrics();

    // ── Helpers ──────────────────────────────────────────────────────────

    std::string indentStr(int level) const;
    void writeLine(int level, const std::string &text);
    void writeLine(const std::string &text);  // uses current indentLevel
    void printBaselineDelta(const std::string &metric, float value);
};

#endif // LOGGER_TEXT_HPP
