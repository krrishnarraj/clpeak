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

// Same shape the dump formats persist (result_store.h), so device metadata
// reaches the file without a conversion step.
using LogProp = DeviceProp;

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

  // TestBegin / TestSkippedAll — what the open test measures (empty until a
  // TestSpec supplies it).  Carried on the events as well as on the result
  // rows so a live consumer can show it the moment a test opens.  Per-metric
  // notes are not here: they belong to individual readings and arrive with
  // them, on `entry.metricDescription`.
  std::string testDescription;

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

    // One or two sentences on what this test measures, for readers who don't
    // already know.  Optional: an undocumented test simply shows no info
    // affordance in the GUI.  Individual readings are documented where they
    // are emitted instead — see EmitOptions::description.
    std::string description;
  };

  struct EmitOptions {
    bool subMetric = false;       // extra indent for nested sub-variants

    // What this one reading means, for readers who don't already know the
    // metric name ("DRAM x8", "float MT").  Authored where the reading is
    // emitted, so a variant's note sits with the code that measures it.
    std::string description;
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

  // Per-device metadata, in device-open order.  Kept alongside `results` so
  // the dump formats can persist it: it is the same information the GUI shows
  // live from DeviceBegin, and without it a reloaded file has no device
  // detail at all.
  DeviceInfoStore devices;

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
  std::string curTestDescription;
  std::string curUnit;
  Category    curCategory = Category::Unknown;
  int         contextDepth = 0;   // 0=none, 1=backend, 2=device, 3=test

  // Identity of the currently-open test.  A TestScope only emits its TestEnd
  // when it is still the open one, so a scope that was implicitly closed (see
  // closeOpenTest) cannot emit a second TestEnd from its destructor.
  unsigned long long testSeqCounter = 0;
  unsigned long long curTestSeq     = 0;

  /// Emit TestEnd for the open test (if any) and drop back to device scope.
  void closeOpenTest();

private:
  /// New event pre-filled with the current scope context.
  LogEvent makeEvent(LogEvent::Kind kind) const;

  /// Build a ResultEntry from the current scope context.  `metricDescription`
  /// is the note authored at the emit/skip site (empty for most readings).
  ResultEntry makeEntry(const std::string &metric, ResultStatus status,
                        float value, const std::string &reason,
                        const std::string &metricDescription = "") const;

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

  /// Same, documenting the reading in place:
  ///   test.emit("DRAM x8", ns, "Eight reads in flight at once.");
  /// Sugar for the EmitOptions form, which C++17 cannot initialize by field
  /// name.  `const char *` and not std::string on purpose: the latter makes
  /// braced calls (`emit(m, v, {true})`) ambiguous, since a braced bool also
  /// matches std::string's initializer_list constructor.  Pass `.c_str()` for
  /// a composed description, or use EmitOptions.
  void emit(std::string metric, float value, const char *description);

  /// Emit a skipped / unsupported / errored metric.  `description` documents
  /// the reading that would have been, exactly as in emit().
  void skip(std::string metric, ResultStatus status, std::string reason,
            std::string description = "");

  /// Entire test unavailable — records one skip row per named metric and
  /// dispatches a single TestSkippedAll event.
  void skipAll(std::initializer_list<std::string> metrics,
               ResultStatus status, std::string reason);

  void end();

private:
  logger             *log;
  bool                closed = false;
  unsigned long long  seq    = 0;   // matches logger::curTestSeq while open
};

#endif  // LOGGER_HPP
