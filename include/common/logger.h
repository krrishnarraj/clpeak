#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <initializer_list>
#include <string>
#include <vector>
#include <common/result_store.h>
#include "common.h"

// ── Abstract logger base class ─────────────────────────────────────────────
//
// Owns structured result accumulation (ResultEntry rows) and defines hooks
// for derived output channels (CLI → stdout, Android → JNI, iOS → Swift).
//
// Backends feed data through RAII context handles:
//
//   auto backend = log->beginBackend("OpenCL");
//   auto device  = backend.beginDevice({"M1 Pro", "Apple", "1.2.3",
//                                        {{"Compute units", "16"}}});
//   auto test    = device.beginTest({"global_bandwidth",
//                                    "Global memory bandwidth", "gbps"});
//   test.emit("float",  123.45f);
//   test.emit("float2", 456.78f);
//
// Handles auto-close on destruction.  The logger formats all output and
// accumulates ResultEntry rows — backends never touch TAB / NEWLINE or
// call print() for structured data.

class logger
{
public:
  // ── Types ──────────────────────────────────────────────────────────────

  struct Prop {
    std::string key;
    std::string value;
  };

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
  virtual void note(const std::string &msg) = 0;

  // ── Baseline compare ────────────────────────────────────────────────────

  bool        compareEnabled;
  BaselineMap baseline;

  // ── Accumulated metrics ─────────────────────────────────────────────────

  ResultStore results;

  explicit logger(std::string compareFileName = "");
  virtual ~logger() = default;

protected:
  // ── Hooks — derived classes implement for their output channel ─────────

  virtual void onBackendBegin(const std::string &name) = 0;
  virtual void onDeviceBegin(const std::string &name,
                             const std::string &platform,
                             const std::string &driverVersion,
                             const std::vector<Prop> &props,
                             bool showPlatformLine,
                             int platformIndex,
                             int deviceIndex) = 0;
  virtual void onTestBegin(const std::string &tag,
                           const std::string &display,
                           const std::string &unit) = 0;
  virtual void onMetricEmitted(const ResultEntry &e,
                               float value,
                               bool subMetric) = 0;
  virtual void onMetricSkipped(const ResultEntry &e) = 0;
  virtual void onTestSkippedAll(ResultStatus status,
                                const std::string &reason) = 0;
  virtual void onTestEnd() = 0;
  virtual void onDeviceEnd() = 0;
  virtual void onBackendEnd() = 0;

  // ── Context state ──────────────────────────────────────────────────────

  std::string curBackend;
  std::string curPlatform;
  std::string curDevice;
  std::string curDriver;
  std::string curTest;
  std::string curDisplay;
  std::string curUnit;
  Category    curCategory = Category::Unknown;
  int         contextDepth = 0;   // 0=none, 1=backend, 2=device, 3=test

private:
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

  /// Emit a successful measurement.  Prints formatted metric line + records row.
  void emit(std::string metric, float value, EmitOptions opts = {});

  /// Emit a skipped / unsupported / errored metric.
  void skip(std::string metric, ResultStatus status, std::string reason);

  /// Entire test unavailable — emits one skip row per named metric and prints
  /// a single skip message under the test header.
  void skipAll(std::initializer_list<std::string> metrics,
               ResultStatus status, std::string reason);

  void end();

private:
  logger *log;
  bool    closed = false;
};

#endif  // LOGGER_HPP
