#ifndef COMMON_H
#define COMMON_H

#if defined(__APPLE__) || defined(__MACOSX) || defined(__FreeBSD__)
#include <sys/types.h>
#endif

#include <stdlib.h>
#include <chrono>
#include <string>
#include <cstdint>
#include <algorithm>
#include <common/benchmark_enums.h>

#define TAB             "  "
#define NEWLINE         "\n"

#ifdef UNUSED
#undef UNUSED
#endif
#define UNUSED(expr) do { (void)(expr); } while (0)

#if defined(__APPLE__) || defined(__MACOSX)
#define OS_NAME         "Macintosh"
#elif defined(__ANDROID__)
#define OS_NAME         "Android"
#elif defined(_WIN32)
  #if defined(_WIN64)
  #define OS_NAME     "Win64"
  #else
  #define OS_NAME     "Win32"
  #endif
#elif defined(__linux__)
  #if defined(__x86_64__)
  #define OS_NAME     "Linux x64"
  #elif defined(__i386__)
  #define OS_NAME     "Linux x86"
  #elif defined(__arm__)
  #define OS_NAME     "Linux ARM"
  #elif defined(__aarch64__)
  #define OS_NAME     "Linux ARM64"
  #else
  #define OS_NAME     "Linux unknown"
  #endif
#elif defined(__FreeBSD__)
#define OS_NAME     "FreeBSD"
#else
#define OS_NAME     "Unknown"
#endif

// Per-device benchmark tuning knobs.  All per-test iteration counts are now
// derived from `targetTimeUs` via runtime calibration (see include/calibrate.h);
// the only static iter field left is kernelLatencyIters because that test is
// structurally different (one separately-submitted dispatch per iter, so the
// watchdog only ever sees a single dispatch).
struct benchmark_config_t {
  uint64_t globalBWMaxSize;
  unsigned int computeWgsPerCU;
  unsigned int computeDPWgsPerCU;
  unsigned int targetTimeUs;          // per-test budget for the timed phase
  unsigned int kernelLatencyIters;    // separately-submitted dispatch count
  uint64_t transferBWMaxSize;

  static benchmark_config_t forDevice(DeviceType type);
};

class Timer
{
public:

  std::chrono::high_resolution_clock::time_point tick, tock;

  void start();

  // Stop and return time in micro-seconds
  float stopAndTime();
};

// Round down to next multiple of the given base with an optional maximum value
uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue = UINT64_MAX);

void populate(float *ptr, uint64_t N);
void populate(double *ptr, uint64_t N);

void trimString(std::string &str);

#endif  // COMMON_H
