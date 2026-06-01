#ifndef CLPEAK_OPTIONS_H
#define CLPEAK_OPTIONS_H

#include <bitset>
#include <string>
#include <vector>
#include <common/benchmark_enums.h>  // Benchmark, Category
#include <common/common.h>           // DEFAULT_TARGET_TIME_US

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

  // Iters / warmup.  When forceIters is false, each backend's runKernel
  // calibrates iters from a one-shot timed warmup so the timed phase lands
  // at ~targetTimeUs regardless of device speed.
  bool         forceIters    = false;
  unsigned int iters         = 0;
  unsigned int warmupCount   = 2;
  unsigned int targetTimeUs  = DEFAULT_TARGET_TIME_US; // --max-time, in us

  // Test selection.  Default: every category and every test enabled.  The
  // first positive --<test> flag flips enabledTests to allow-list mode
  // ("deny by default; enable picked"); --no-<test> always subtracts.
  // The first positive --<category> flag flips enabledCategories the same
  // way; --no-<category> always subtracts.  A test runs iff its primary
  // category is enabled AND its own bit is set (see isAllowed).
  std::bitset<static_cast<size_t>(Benchmark::COUNT)> enabledTests;
  std::bitset<4>                                     enabledCategories;
  // OpenCL-only timing knob.
  bool useEventTimer = false;

  // Output / compare
  bool        enableXml  = false;
  std::string xmlFile;
  bool        enableJson = false;
  std::string jsonFile;
  bool        enableCsv  = false;
  std::string csvFile;
  std::string compareFile;

  // Listing mode (no benchmarks run; just print devices).
  bool listDevices = false;

  CliOptions()
  {
    enabledTests.set();
    enabledCategories.set();
  }

};

// Parse argv into out.  On --help / --version / parse error this calls
// exit() directly (matching the previous behavior).  Returns 0 on success.
int parseCliOptions(int argc, char **argv, CliOptions &out);

#endif // CLPEAK_OPTIONS_H
