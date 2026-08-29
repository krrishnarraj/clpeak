#ifndef CLPEAK_OPTIONS_H
#define CLPEAK_OPTIONS_H

#include <bitset>
#include <string>
#include <vector>
#include <common/benchmark_enums.h>  // Benchmark, Category
#include <common/common.h>           // DEFAULT_TARGET_TIME_US
#include <common/run_document.h>     // Invocation

// Shared CLI options populated once in entry.cpp and consumed by every
// backend.  Each backend's applyOptions() copies the relevant fields into
// its own state so the rest of its code can stay backend-flavored.
struct CliOptions {
  // Backend on/off (consumed by entry.cpp dispatcher)
  bool skipOpenCL = false;
  bool skipVulkan = false;
  bool skipCuda   = false;
  bool skipRocm   = false;
  bool skipMetal  = false;
  bool skipOneapi = false;
  bool skipCpu    = false;
  bool skipOnnx   = false;

  // OpenCL platform/device selection (OpenCL-only concept; kept here so
  // applyOptions can copy it).  Empty = run all enumerated platforms/devices.
  std::vector<unsigned long> platformIndices;
  std::vector<unsigned long> deviceIndices;

  // Per-backend device selectors.  Empty = run all enumerated devices.
  std::vector<int> vkDeviceIndices;
  std::vector<int> cudaDeviceIndices;
  std::vector<int> rocmDeviceIndices;
  std::vector<int> mtlDeviceIndices;
  std::vector<int> oneapiDeviceIndices;
  std::vector<int> onnxDeviceIndices;

  // --onnx-lib: absolute path to the onnxruntime shared library to load,
  // overriding the platform's conventional names.  Empty = search the
  // default names (see src/onnx/onnx_runtime.cpp).  Ignored on a build that
  // links ONNX Runtime statically, where there is nothing to load.
  std::string onnxLibPath;

  // Iters / warmup.  When forceIters is false, each backend's runKernel
  // calibrates iters from a one-shot timed warmup so the timed phase lands
  // at ~targetTimeUs regardless of device speed.
  bool         forceIters    = false;
  unsigned int iters         = 0;
  unsigned int warmupCount   = 2;
  unsigned int targetTimeUs  = DEFAULT_TARGET_TIME_US; // --max-time, in us
  // CPU backend uses its own (longer) budget; --max-time does not affect it.
  unsigned int targetTimeUsCpu = DEFAULT_CPU_TARGET_TIME_US; // --max-time-cpu, in us

  // Test selection.  Default: every category and every test enabled.  The
  // first positive --<test> flag flips enabledTests to allow-list mode
  // ("deny by default; enable picked"); --no-<test> always subtracts.
  // The first positive --<category> flag flips enabledCategories the same
  // way; --no-<category> always subtracts.  A test runs iff its primary
  // category is enabled AND its own bit is set (see isAllowed).
  std::bitset<static_cast<size_t>(Benchmark::COUNT)>  enabledTests;
  std::bitset<static_cast<size_t>(Category::Unknown)> enabledCategories;
  // OpenCL-only timing knob.
  bool useEventTimer = false;

  // Output / compare.  One format, one flag: `-o file` writes the v3 JSON
  // document (run_document.h).  The XML and CSV writers are gone -- XML's
  // only advantage over JSON was nesting, and CSV could carry neither device
  // metadata nor the per-test documentation.
  bool        enableOutput = false;
  std::string outputFile;
  std::string compareFile;

  // Listing mode (no benchmarks run; just print devices).
  bool listDevices = false;

  // Verbose diagnostics: print backend debug logs (kernel build logs, API /
  // launch errors, library exceptions) that are suppressed by default.
  bool verbose = false;

  // Print, alongside the readings, what each test and each reading measures
  // (the descriptions authored at the beginTest()/emit() call sites).  Off by
  // default: the plain output is a table for people who already know it.
  bool describe = false;

  CliOptions()
  {
    enabledTests.set();
    enabledCategories.set();
  }

};

// Describe how clpeak was asked to run, for the result document's `invocation`
// block.  Every number in a run is sensitive to this -- a shorter --max-time
// measures a different thing, and a selective run is not a full one even though
// the file looks the same shape -- so it is recorded rather than inferred.
// Lives here because the category and test flag-name tables do.
Invocation invocationFrom(const CliOptions &opts, int argc, char **argv);

// Parse argv into out.  On --help / --version / parse error this calls
// exit() directly (matching the previous behavior).  Returns 0 on success.
int parseCliOptions(int argc, char **argv, CliOptions &out);

// Embedding-safe variant: never calls exit().  Returns true on success;
// on failure (parse error, or --help/--version which have no meaning when
// embedded) returns false with a human-readable message in errorMsg.
// Used by clpeak_ffi so a bad argv can't kill the host GUI process.
bool parseCliOptionsNoExit(int argc, char **argv, CliOptions &out,
                           std::string &errorMsg);

#endif // CLPEAK_OPTIONS_H
