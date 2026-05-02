#ifndef CLPEAK_OPTIONS_H
#define CLPEAK_OPTIONS_H

#include <bitset>
#include <string>
#include <clpeak.h> // Benchmark enum

// Shared CLI options populated once in entry.cpp and consumed by every
// backend.  Each backend's applyOptions() copies the relevant fields into
// its own state so the rest of its code can stay backend-flavored.
struct CliOptions {
  // Backend on/off (consumed by entry.cpp dispatcher)
  bool skipOpenCL = false;
  bool skipVulkan = false;
  bool skipCuda   = false;
  bool skipMetal  = false;

  // OpenCL platform/device selection (OpenCL-only concept; kept here so
  // applyOptions can copy it).  -p / --platform / --cl-platform.
  bool          forcePlatform     = false;
  unsigned long platformIndex     = 0;
  bool          forcePlatformName = false;
  std::string   platformName;
  bool          forceDevice       = false;
  unsigned long deviceIndex       = 0;
  bool          forceDeviceName   = false;
  std::string   deviceName;

  // Per-backend device selectors (-1 = run all enumerated devices).
  int vkDeviceIndex   = -1;
  int cudaDeviceIndex = -1;
  int mtlDeviceIndex  = -1;

  // Iters / warmup
  bool         forceIters    = false;
  unsigned int iters         = 0;
  unsigned int warmupCount   = 2;

  // Test selection.  Default: every category and every test enabled.  The
  // first positive --<test> flag flips enabledTests to allow-list mode
  // ("deny by default; enable picked"); --no-<test> always subtracts.
  // The first positive --<category> flag flips enabledCategories the same
  // way; --no-<category> always subtracts.  A test runs iff its primary
  // category is enabled AND its own bit is set (see isAllowed).
  std::bitset<static_cast<size_t>(Benchmark::COUNT)> enabledTests;
  std::bitset<4>                                     enabledCategories;
  bool forcedTests       = false;
  bool forcedCategories  = false;

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

  bool isTestEnabled(Benchmark b) const
  {
    return enabledTests.test(static_cast<size_t>(b));
  }
  bool isCategoryEnabled(Category c) const
  {
    if (c == Category::Unknown) return false;
    return enabledCategories.test(static_cast<size_t>(c));
  }
  bool isAllowed(Benchmark b) const
  {
    return isCategoryEnabled(categoryOf(b)) && isTestEnabled(b);
  }
};

// Parse argv into out.  On --help / --version / parse error this calls
// exit() directly (matching the previous behavior).  Returns 0 on success.
int parseCliOptions(int argc, char **argv, CliOptions &out);

#endif // CLPEAK_OPTIONS_H
