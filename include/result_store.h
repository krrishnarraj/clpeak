#ifndef RESULT_STORE_H
#define RESULT_STORE_H

#include <string>
#include <vector>
#include <map>

// Dump-file schema version.  Loaders reject files written by older clpeak
// versions (v1) with a clear error.  Bumped on any breaking change to the
// fields or layout of saveJson / saveCsv / saveXml output.
constexpr int CLPEAK_FORMAT_VERSION = 2;

// Test category: every benchmark falls into exactly one of these.  Drives
// run order (fp -> int -> bandwidth -> latency on every backend) and the
// unit class (gflops/tflops, gops/tops, gbps, us).
enum class Category {
    FpCompute,
    IntCompute,
    Bandwidth,
    Latency,
    Unknown   // sentinel for back-compat paths and unknown dump-file values
};

// Result status: an emitted row may carry no measurement when the test
// could not run.  `value` is meaningful only for `Ok` rows.  Non-Ok rows
// carry a human-readable `reason` string.
enum class ResultStatus {
    Ok,
    Unsupported,  // hardware / driver lacks the feature
    Skipped,      // user deselected via CLI but emitted for completeness
    Error         // attempted, failed at runtime (compile / OOM / timeout)
};

// Canonical lower-snake names used in the dump format.  Round-trip via the
// from-string variants below.
const char *categoryString(Category c);
const char *statusString(ResultStatus s);

Category     categoryFromString(const std::string &s);
ResultStatus statusFromString(const std::string &s);

// Derive a category from a unit string (gflops/tflops -> FpCompute, etc.).
// Used by the legacy xmlOpenTag-based logger path so existing benchmark
// code keeps producing correctly-categorised rows without per-test wiring.
Category categoryFromUnit(const std::string &unit);

// A single benchmark measurement, fully qualified by its provenance.  The
// (backend, platform, device, category, test, metric) tuple is the primary
// key; `driver` is metadata only and intentionally not part of `key()` so
// baselines stay comparable across driver updates.
struct ResultEntry {
    std::string  backend;   // "OpenCL" | "Vulkan" | "CUDA" | "Metal"
    std::string  platform;  // vendor platform name
    std::string  device;
    std::string  driver;
    std::string  category;  // canonical category string (see categoryString)
    std::string  test;      // canonical test tag (e.g. "wmma")
    std::string  metric;    // variant within test (e.g. "fp16_16x16x16")
    std::string  unit;      // "gflops" | "tflops" | "gops" | "tops" | "gbps" | "us"
    ResultStatus status = ResultStatus::Ok;
    float        value  = 0.0f;
    std::string  reason;    // populated only when status != Ok

    std::string key() const
    {
        return backend + "/" + platform + "/" + device + "/" +
               category + "/" + test + "/" + metric;
    }
};

typedef std::vector<ResultEntry>      ResultStore;
typedef std::map<std::string, float>  BaselineMap;

// Build a fast lookup map from a ResultStore.  Only Ok rows participate in
// baselines; skipped/unsupported/error rows are filtered out.
BaselineMap buildBaselineMap(const ResultStore &store);

// Serialize the store to one of three formats.  Each format is
// self-describing (carries `format_version`) and produced by a single
// pass at program exit so all three formats agree on row count + ordering.
void saveJson(const ResultStore &store, const std::string &filename);
void saveCsv (const ResultStore &store, const std::string &filename);
void saveXml (const ResultStore &store, const std::string &filename);

// Parse a previously-written file.  Rejects v1 (or unversioned) files with
// a stderr message and returns an empty store.  loadResultFile dispatches
// on extension (.csv/.xml/anything else -> JSON).
ResultStore loadJson(const std::string &filename);
ResultStore loadCsv (const std::string &filename);
ResultStore loadXml (const std::string &filename);
ResultStore loadResultFile(const std::string &filename);

#endif // RESULT_STORE_H
