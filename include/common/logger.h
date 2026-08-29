#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>
#include <common/run_document.h>
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
// Handles auto-close on destruction.  The logger builds a RunDocument
// (run_document.h) as it goes — backends never touch TAB / NEWLINE or call
// print() for structured data, and no consumer has to regroup a flat table
// back into tests.

// Same shape the document persists (run_document.h), so device metadata
// reaches the file without a conversion step.
using LogProp = DeviceProp;

struct LogEvent {
  enum class Kind {
    BackendBegin,    // backend
    DeviceBegin,     // + platform/device/driver, props, type, indices
    TestBegin,       // + the whole resolved test header (see below)
    Metric,          // + metric (Ok or non-Ok)
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

  // Open test.  Everything a live consumer needs to render the test's header
  // and scale its readings, resolved once at beginTest() so nothing
  // downstream repeats the unit-table lookup.  Empty / defaulted until a test
  // opens.  Per-reading notes are not here: they belong to individual
  // readings and arrive with them, on `metric.description`.
  std::string testId;
  std::string testTitle;
  std::string testVariant;
  std::string testAxis;
  std::string testDescription;
  Category    category  = Category::Unknown;
  TestShape   shape     = TestShape::Heterogeneous;
  Direction   direction = Direction::HigherIsBetter;
  std::string unit;                             // resolved display symbol
  Quantity    quantity  = Quantity::Unknown;
  double      scale     = 1.0;

  // True when this TestBegin reopened an already-recorded test to append more
  // readings to it, rather than starting a new one.  Channels that render a
  // test header skip it the second time round.
  bool reopened = false;

  // DeviceBegin
  std::vector<LogProp> props;
  DeviceType type              = DeviceType::Unknown;
  int        platformIndex     = -1;
  int        deviceIndex       = -1;
  bool       showPlatformLine  = false;

  // Metric — the reading just recorded.  `status` distinguishes Ok
  // (`value` valid) from unsupported/skipped/error (`reason` valid).
  MetricResult metric;

  // TestSkippedAll — one reading per metric name was recorded with the given
  // status/reason.
  std::vector<std::string> metricNames;
  ResultStatus status = ResultStatus::Ok;
  std::string  reason;

  // Note
  std::string message;

  // Identity of the open test within its device -- the same key the document
  // and the --compare baseline use.  Mirrors TestResult::key().
  std::string testKey() const
  {
    return testVariant.empty() ? testId : (testId + "@" + testVariant);
  }
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
    DeviceType type    = DeviceType::Unknown;
  };

  struct TestSpec {
    std::string tag;              // canonical tag, e.g. "global_memory_bandwidth"
    std::string display;          // human-readable, e.g. "Global memory bandwidth"
    std::string unit;             // "gflops" | "gbps" | "us" | … (see units.h)
    Category    category = Category::Unknown;  // auto-derived from unit if omitted

    // One or two sentences on what this test measures, for readers who don't
    // already know.  Optional: an undocumented test simply shows no info
    // affordance in the GUI.  Individual readings are documented where they
    // are emitted instead — see EmitOptions::description.
    std::string description;

    // Whether this test's readings are comparable to one another, and so
    // whether a presenter may collapse them to one number.  See TestShape in
    // run_document.h for what the two values mean and why neither can be
    // inferred.  Heterogeneous by default: verbose, never wrong.
    TestShape shape = TestShape::Heterogeneous;

    // Which way is better.  FromUnit (the default) takes the unit table's
    // answer — higher for throughput, lower for latency and error — which is
    // right for every test that measures what its unit suggests.
    Direction direction = Direction::FromUnit;

    // What varies from one reading to the next: "vector width", "data type",
    // "cache level", "direction", "threads", "pixel format".  Optional; the
    // GUI heads a heterogeneous test's readings with it.
    std::string axis;

    // Runtime qualifier that is not part of the test's identity — the CPU
    // backend's detected ISA, a GPU arch, a library version.  Keeping it here
    // rather than slugged onto `tag` is what keeps tags stable across
    // machines, and so keeps --compare working across them.
    std::string variant;
  };

  struct EmitOptions {
    // What this one reading means, for readers who don't already know the
    // metric name ("DRAM x8", "float MT").  Authored where the reading is
    // emitted, so a variant's note sits with the code that measures it.
    std::string description;

    // Display form when the id is not what should be shown.  Empty means the
    // id is the label, which is the usual case.
    std::string label;

    // Unit token for this one reading, when it differs from the test's — an
    // integer reading (`tops`) inside a GEMM test reported in `tflops`.  This
    // is what lets one heterogeneous test cover both, instead of the `-fp` /
    // `-int` twins that exist today only because the unit string had to
    // differ.  Empty means "the test's unit".
    std::string unit;

    // Direction override; FromUnit inherits the test's.
    Direction direction = Direction::FromUnit;
  };

  // ── RAII context handles (defined below, implemented in logger.cpp) ────

  class BackendScope;
  class DeviceScope;
  class TestScope;

  // ── Public API ─────────────────────────────────────────────────────────

  /// Begin a backend run.  Returns a handle that auto-closes on destruction.
  BackendScope beginBackend(const std::string &name);

  /// Unstructured ad-hoc message (warnings, notes, errors outside tests).
  /// Recorded on the document as well as dispatched, so a reopened run can
  /// explain its own gaps.
  void note(const std::string &msg);

  // ── Baseline compare ────────────────────────────────────────────────────

  bool        compareEnabled;
  BaselineMap baseline;

  // ── Accumulated results ─────────────────────────────────────────────────
  //
  // Devices, their tests and their readings, in emission order — the same
  // tree the file holds.  Hosts fold one of these per backend into the
  // document they save (RunDocument::append).

  RunDocument doc;

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
  int         contextDepth = 0;   // 0=none, 1=backend, 2=device, 3=test

  // Where the open scopes live in `doc`.  Indices, not pointers: appending a
  // device or a test reallocates the vector holding it.
  static constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);
  std::size_t curDeviceIdx = kNoIndex;
  std::size_t curTestIdx   = kNoIndex;

  // Identity of the currently-open test.  A TestScope only emits its TestEnd
  // when it is still the open one, so a scope that was implicitly closed (see
  // closeOpenTest) cannot emit a second TestEnd from its destructor.
  unsigned long long testSeqCounter = 0;
  unsigned long long curTestSeq     = 0;

  /// Emit TestEnd for the open test (if any) and drop back to device scope.
  void closeOpenTest();

private:
  /// New event pre-filled with the current scope context, including the open
  /// test's resolved header.
  LogEvent makeEvent(LogEvent::Kind kind) const;

  /// The open test, or nullptr when none is open.
  TestResult *openTest();
  const TestResult *openTest() const;

  /// Record a reading on the open test and return it.
  MetricResult &record(MetricResult m);

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

  /// Open a test.  A tag+variant already recorded for this device reopens
  /// that test and appends to it, which is how a backend measures the fp
  /// readings of a GEMM test in one category phase and the int ones in
  /// another without splitting it in two.  The first open defines the test's
  /// header (title, shape, unit, description); a reopen only adds readings,
  /// so a reading in another unit carries its own via EmitOptions::unit.
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

  /// Emit a successful measurement.  Records the reading + dispatches Metric.
  void emit(std::string metric, float value, EmitOptions opts = {});

  /// Same, documenting the reading in place:
  ///   test.emit("DRAM x8", ns, "Eight reads in flight at once.");
  /// Sugar for the EmitOptions form, which C++17 cannot initialize by field
  /// name.  `const char *` and not std::string on purpose: the latter makes
  /// braced calls ambiguous, since a braced string also matches std::string's
  /// initializer_list constructor.  Pass `.c_str()` for a composed
  /// description, or use EmitOptions.
  void emit(std::string metric, float value, const char *description);

  /// Emit a skipped / unsupported / errored reading.  `description` documents
  /// the reading that would have been, exactly as in emit().
  void skip(std::string metric, ResultStatus status, std::string reason,
            std::string description = "");

  /// Entire test unavailable — records one skip per named metric and
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
