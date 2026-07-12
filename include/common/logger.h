#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <initializer_list>
#include <string>
#include <vector>
#include <common/result_store.h>
#include "common.h"

// ── Structured log-event stream ────────────────────────────────────────────
//
// Every observable moment of a benchmark run is one LogEvent.  The logger
// base class owns result accumulation and scope bookkeeping; derived output
// channels implement a single hook — onEvent() — and render or forward the
// stream:
//
//   LoggerText (src/common/logger_text.cpp)  → indented CLI text
//   LoggerFfi  (src/ffi/logger_ffi.cpp)      → JSON over a C callback (GUI)
//
// Backends never see LogEvent; they feed data through RAII context handles:
//
//   auto backend = log->beginBackend("OpenCL");
//   auto device  = backend.beginDevice({"M1 Pro", "Apple", "1.2.3",
//                                        {{"Compute units", "16"}}});
//   auto test    = device.beginTest({"global_bandwidth",
//                                    "Global memory bandwidth", "gbps"});
//   test.emit("float",  123.45f);
//   test.emit("float2", 456.78f);
//
// Handles auto-close on destruction.  The logger accumulates ResultEntry
// rows — backends never touch TAB / NEWLINE or call print() for structured
// data.

struct LogProp {
  std::string key;
  std::string value;
};

struct LogEvent {
  enum class Kind {
    BackendBegin,    // backend
    DeviceBegin,     // + platform/device/driver, props, indices
    TestBegin,       // + testTag/testDisplay/unit/category
    Metric,          // + entry (Ok or non-Ok), subMetric
    TestSkippedAll,  // whole test unavailable: + metricNames, status, reason
    TestEnd,
    DeviceEnd,
    BackendEnd,
    Note,            // + message (may fire at any scope depth)
  };

  Kind kind = Kind::Note;

  // Scope context — filled from the current scope state for every event
  // (empty strings when the corresponding scope is not open).
  std::string backend;
  std::string platform;
  std::string device;
  std::string driver;
  std::string testTag;
  std::string testDisplay;
  std::string unit;
  Category    category = Category::Unknown;

  // DeviceBegin
  std::vector<LogProp> props;
  int  platformIndex    = -1;
  int  deviceIndex      = -1;
  bool showPlatformLine = false;

  // Metric — the ResultEntry just recorded.  entry.status distinguishes Ok
  // (entry.value valid) from unsupported/skipped/error (entry.reason valid).
  ResultEntry entry;
  bool subMetric = false;

  // TestSkippedAll — one row per metric name was recorded in `results`
  // with the given status/reason.
  std::vector<std::string> metricNames;
  ResultStatus status = ResultStatus::Ok;
  std::string  reason;

  // Note
  std::string message;
};

class logger
{
public:
  // ── Types ──────────────────────────────────────────────────────────────

  using Prop = LogProp;

  struct DeviceSpec {
    std::string name;
    std::string platform;         // empty → auto-set to backend name
    std::string driver_version;
    std::vector<Prop> props;      // free-form properties (compute units, VRAM, …)
    int platform_index = -1;      // if >= 0, printed as "Platform N: ..."
    int device_index   = -1;      // if >= 0, printed as "Device N: ..."
  };

  struct TestSpec {
    std::string tag;              // canonical tag, e.g. "global_memory_bandwidth"
    std::string display;          // human-readable, e.g. "Global memory bandwidth"
    std::string unit;             // "gflops" | "gbps" | "us" | …
    Category    category = Category::Unknown;  // auto-derived from unit if omitted
  };

  struct EmitOptions {
    bool subMetric = false;       // extra indent for nested sub-variants
  };

  // ── RAII context handles (defined below, implemented in logger.cpp) ────

  class BackendScope;
  class DeviceScope;
  class TestScope;

  // ── Public API ─────────────────────────────────────────────────────────

  /// Begin a backend run.  Returns a handle that auto-closes on destruction.
  BackendScope beginBackend(const std::string &name);

  /// Unstructured ad-hoc message (warnings, notes, errors outside tests).
  void note(const std::string &msg);

  // ── Baseline compare ────────────────────────────────────────────────────

  bool        compareEnabled;
  BaselineMap baseline;

  // ── Accumulated metrics ─────────────────────────────────────────────────

  ResultStore results;

  explicit logger(std::string compareFileName = "");
  virtual ~logger() = default;

protected:
  // ── The single output hook ──────────────────────────────────────────────
  // Derived channels render or forward the event stream from here.

  virtual void onEvent(const LogEvent &e) = 0;

  // ── Context state ──────────────────────────────────────────────────────

  std::string curBackend;
  std::string curPlatform;
  std::string curDevice;
  std::string curDriver;
  std::string curTest;
  std::string curTestDisplay;
  std::string curUnit;
  Category    curCategory = Category::Unknown;
  int         contextDepth = 0;   // 0=none, 1=backend, 2=device, 3=test

private:
  /// New event pre-filled with the current scope context.
  LogEvent makeEvent(LogEvent::Kind kind) const;

  /// Build a ResultEntry from the current scope context.
  ResultEntry makeEntry(const std::string &metric, ResultStatus status,
                        float value, const std::string &reason) const;

  // Scope handles are friends so they can manipulate context state directly.
  friend class BackendScope;
  friend class DeviceScope;
  friend class TestScope;
};

// ── Scope handle definitions ──────────────────────────────────────────────

class logger::BackendScope
{
public:
  BackendScope(logger *log, const std::string &name);
  ~BackendScope();

  BackendScope(const BackendScope &) = delete;
  BackendScope &operator=(const BackendScope &) = delete;
  BackendScope(BackendScope &&other) noexcept;
  BackendScope &operator=(BackendScope &&) = delete;

  DeviceScope beginDevice(const DeviceSpec &spec);
  void end();

private:
  logger *log;
  bool    closed = false;
};

class logger::DeviceScope
{
public:
  DeviceScope(logger *log, const DeviceSpec &spec);
  ~DeviceScope();

  DeviceScope(const DeviceScope &) = delete;
  DeviceScope &operator=(const DeviceScope &) = delete;
  DeviceScope(DeviceScope &&other) noexcept;
  DeviceScope &operator=(DeviceScope &&) = delete;

  TestScope beginTest(const TestSpec &spec);
  void end();

private:
  logger *log;
  bool    closed = false;
};

class logger::TestScope
{
public:
  TestScope(logger *log, const TestSpec &spec);
  ~TestScope();

  TestScope(const TestScope &) = delete;
  TestScope &operator=(const TestScope &) = delete;
  TestScope(TestScope &&other) noexcept;
  TestScope &operator=(TestScope &&) = delete;

  /// Emit a successful measurement.  Records row + dispatches Metric event.
  void emit(std::string metric, float value, EmitOptions opts = {});

  /// Emit a skipped / unsupported / errored metric.
  void skip(std::string metric, ResultStatus status, std::string reason);

  /// Entire test unavailable — records one skip row per named metric and
  /// dispatches a single TestSkippedAll event.
  void skipAll(std::initializer_list<std::string> metrics,
               ResultStatus status, std::string reason);

  void end();

private:
  logger *log;
  bool    closed = false;
};

#endif  // LOGGER_HPP
