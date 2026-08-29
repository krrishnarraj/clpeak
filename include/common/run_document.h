#ifndef CLPEAK_RUN_DOCUMENT_H
#define CLPEAK_RUN_DOCUMENT_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <common/benchmark_enums.h>
#include <common/host_info.h>
#include <common/units.h>

// ── The result document ────────────────────────────────────────────────────
//
// One run, as a tree: run -> devices -> tests -> metrics.  This is the shape
// the CLI prints, the shape the GUI renders, and the shape the file on disk
// holds, so nothing has to regroup a flat table into a hierarchy and guess at
// what belonged together.
//
// The single serialization is JSON (`saveRunJson` / `loadRunJson`).  XML and
// CSV are gone: XML's only advantage was nesting, which JSON has, and CSV
// could not carry device metadata or documentation at all.
//
// Schema and vocabulary are documented in docs/format-v3.md.

// Dump-file schema version.  Loaders reject any other version outright --
// v3 restructured the document, so there is nothing an older reader could
// salvage from it and nothing this reader can salvage from a v2 file.
constexpr int RESULT_FORMAT_VERSION = 3;

// A reading may carry no measurement when it could not be taken.  `value` is
// meaningful only for `Ok`; every other status carries a human-readable
// `reason` instead.
enum class ResultStatus {
    Ok,
    Unsupported,  // hardware / driver lacks the feature
    Skipped,      // deliberately not run
    Error         // attempted, failed at runtime (compile / OOM / timeout)
};

// Whether a test's metrics are comparable to one another.
//
//   Homogeneous   -- interchangeable variants of one measurement: float /
//                    float2 / float4, int8_dp chain depths, or a test with a
//                    single reading.  The best of them is the test's answer,
//                    so a presenter may collapse the test to that one number.
//
//   Heterogeneous -- each reading is its own measurement: cuBLASLt's nine
//                    datatypes, memory_latency's L1/L2/L3/DRAM, transfer's
//                    h2d vs d2h, a CPU test's ST vs MT.  There is no single
//                    answer, and picking the largest reading invents one.
//
// This cannot be derived from anything else.  `wmma_fp16` has one metric and
// is homogeneous; `mps-gemm-fp` has three and is not; both are tflops.  The
// same tag even differs by backend -- a GPU's global_memory_bandwidth is a
// vector-width sweep, the CPU's is read/copy/triad.  So it is authored at the
// beginTest() call site, next to the description (include/common/AGENTS.md).
//
// Heterogeneous is the default: it never invents a headline number.  A test
// that has not been visited yet is therefore verbose, never wrong.
enum class TestShape {
    Homogeneous,
    Heterogeneous
};

// Canonical lower-snake names used in the dump format.
const char  *categoryString(Category c);
Category     categoryFromString(const std::string &s);
const char  *statusString(ResultStatus s);
ResultStatus statusFromString(const std::string &s);
const char  *shapeString(TestShape s);
TestShape    shapeFromString(const std::string &s);
const char  *deviceTypeString(DeviceType t);
DeviceType   deviceTypeFromString(const std::string &s);

// ---- Metric ---------------------------------------------------------------

// One reading.  Unit fields are overrides: they are set only when this
// reading is measured in something other than its test's unit, which is what
// lets a single heterogeneous test hold both TFLOPS and TOPS readings instead
// of being split into a `-fp` and a `-int` twin.
struct MetricResult {
    std::string  id;           // stable slug within the test: "fp8_e4m3"
    std::string  label;        // display form; empty means "same as id"
    ResultStatus status = ResultStatus::Ok;
    double       value  = 0.0;
    std::string  reason;       // populated only when status != Ok

    // What this one reading means, in plain language, authored at the emit()
    // or skip() call that produces it.  Travels with the row so a reopened
    // file explains itself.
    std::string  description;

    // Unit override.  `hasUnit` false means "inherit the test's".
    bool        hasUnit  = false;
    std::string unit;                             // resolved display symbol
    Quantity    quantity = Quantity::Unknown;
    double      scale    = 1.0;

    // Direction override.  FromUnit means "inherit the test's".
    Direction   direction = Direction::FromUnit;

    const std::string &displayLabel() const { return label.empty() ? id : label; }
};

// ---- Test -----------------------------------------------------------------

struct TestResult {
    std::string id;            // canonical tag, ISA-free: "single_precision_compute"
    std::string title;         // human-readable: "Single-precision compute"

    // Runtime qualifier that is not part of the test's identity: the CPU
    // backend's detected ISA ("AVX2+FMA", "NEON + SME2 (SVL=512b)"), a GPU
    // arch, a library version.  Kept out of `id` on purpose -- baking it in is
    // what made CPU tags lossy and un-comparable across machines.
    std::string variant;

    // What varies from one metric to the next: "vector width", "data type",
    // "cache level", "direction", "threads", "pixel format".  Optional; a
    // presenter uses it as the column header over a heterogeneous test's
    // readings, and it is the plainest statement of why a test is the shape
    // it is.
    std::string axis;

    // One or two sentences on what this test measures, authored at the
    // beginTest() call site.
    std::string description;

    Category  category  = Category::Unknown;
    TestShape shape     = TestShape::Heterogeneous;
    Direction direction = Direction::HigherIsBetter;  // resolved, never FromUnit

    // Resolved from the authored unit token exactly once, at beginTest().
    std::string unit;                             // display symbol: "TFLOPS"
    Quantity    quantity = Quantity::Unknown;
    double      scale    = 1.0;                   // value * scale -> SI base

    std::vector<MetricResult> metrics;

    // Identity within a device.  `variant` participates: two ISAs of the same
    // CPU test are two tests, and comparing one against the other would be
    // meaningless.
    std::string key() const
    {
        return variant.empty() ? id : (id + "@" + variant);
    }

    // Unit context for one reading, honouring its overrides.
    const std::string &unitOf(const MetricResult &m) const
    {
        return m.hasUnit ? m.unit : unit;
    }
    Direction directionOf(const MetricResult &m) const
    {
        return (m.direction == Direction::FromUnit) ? direction : m.direction;
    }
};

// ---- Device ---------------------------------------------------------------

// Free-form per-device fact (compute units, VRAM, clocks, driver internals)
// captured once when the device is opened.  Not fields of a reading: they
// describe the device, not a measurement.
struct DeviceProp {
    std::string key;
    std::string value;
};

struct DeviceResult {
    std::string backend;       // "OpenCL" | "Vulkan" | "CUDA" | "Metal" | …
    std::string platform;      // vendor platform; defaults to the backend name
    std::string name;
    std::string driver;
    DeviceType  type = DeviceType::Unknown;

    int platformIndex = -1;
    int deviceIndex   = -1;

    std::vector<DeviceProp> properties;
    std::vector<TestResult> tests;

    // `driver` is metadata, not identity: a baseline stays comparable across
    // a driver update, which is exactly the comparison people want.
    std::string key() const
    {
        return backend + "/" + platform + "/" + name;
    }

    // Find an already-recorded test by TestResult::key(), or nullptr.  This is
    // what lets a backend reopen a test to append readings measured in a later
    // category phase (cuBLASLt's int metrics joining its fp ones).
    TestResult *findTest(const std::string &testKey);
};

// ---- Run ------------------------------------------------------------------

// An ad-hoc message emitted outside any reading (a missing library, a driver
// warning).  Kept so a reopened run can explain its own gaps instead of
// looking like the hardware simply lacks the feature.
struct RunNote {
    std::string backend;
    std::string device;
    std::string message;
};

// How clpeak was asked to run.  Recorded because every number here is
// sensitive to it: a shorter --max-time measures a different thing, and a
// selective run is not a full one even though the file looks the same shape.
struct Invocation {
    std::vector<std::string> argv;
    unsigned targetTimeUs    = 0;
    unsigned targetTimeUsCpu = 0;
    unsigned warmup          = 0;
    unsigned iters           = 0;   // 0 = calibrated per test, not forced
    std::vector<std::string> categories;  // enabled category names
    std::vector<std::string> tests;       // empty = every test enabled
};

struct RunMeta {
    std::string clpeakVersion;
    std::string generatedAt;        // ISO-8601 UTC, "2026-08-29T14:03:11Z"
    double      durationS = 0.0;

    // A cancelled run is a partial one.  Without this flag a file written
    // after a cancel is indistinguishable from a complete run, and every
    // absent test reads as unsupported hardware.
    bool        cancelled = false;

    HostInfo   host;
    Invocation invocation;
};

struct RunDocument {
    RunMeta                   meta;
    std::vector<DeviceResult> devices;
    std::vector<RunNote>      notes;

    bool empty() const { return devices.empty(); }

    // Find a device by DeviceResult::key(), or nullptr.
    DeviceResult *findDevice(const std::string &deviceKey);

    // Append another document's devices and notes.  Used to fold the
    // per-backend loggers into one file; `meta` is the caller's.
    void append(const RunDocument &other);
};

// ---- Baselines ------------------------------------------------------------

// `--compare` lookup.  Keyed on
//   backend/platform/device/test.key()/metric.id
// -- driver and category are deliberately absent, so a baseline survives a
// driver update and a test being re-filed under another category.
typedef std::map<std::string, double> BaselineMap;

std::string baselineKey(const std::string &backend, const std::string &platform,
                        const std::string &device, const std::string &testKey,
                        const std::string &metricId);

// Only Ok readings participate; there is nothing to compare a skip against.
BaselineMap buildBaselineMap(const RunDocument &doc);

// ---- Helpers --------------------------------------------------------------

// Current UTC time as ISO-8601 ("2026-08-29T14:03:11Z").  Both hosts (CLI and
// the GUI bridge) stamp a run with this, so it lives next to the field it
// fills rather than being written twice.
std::string isoTimestampUtc();

// ---- Serialization --------------------------------------------------------

// Write the document as JSON.  Returns false (after a stderr message) when
// the file cannot be opened or the stream fails while writing.
bool saveRunJson(const RunDocument &doc, const std::string &filename);

// Same document, returned as a string.
std::string runDocumentToJson(const RunDocument &doc);

// Read a document back.  Returns false (after a stderr message) for a missing
// file, malformed JSON, or a format_version this build does not write.
bool loadRunJson(const std::string &filename, RunDocument &out);

#endif  // CLPEAK_RUN_DOCUMENT_H
