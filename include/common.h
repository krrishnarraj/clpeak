#ifndef COMMON_H
#define COMMON_H

#include <CL/cl.hpp>
#if defined(__APPLE__) || defined(__MACOSX) || defined(__FreeBSD__)
#include <sys/types.h>
#endif

#include <stdlib.h>
#include <chrono>
#include <string>
#include <cstdint>

#define TAB             "  "
#define NEWLINE         "\n"
#ifndef __FreeBSD__
#define uint            unsigned int
#endif
#define ulong           unsigned long

#define MAX(X, Y)       \
  (X > Y)? X: Y

#define MIN(X, Y)       \
  (X < Y)? X: Y

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


typedef struct {
  std::string deviceName;
  std::string driverVersion;

  uint numCUs;
  uint maxWGSize;
  uint64_t maxAllocSize;
  uint64_t maxGlobalSize;
  uint maxClockFreq;

  bool halfSupported;
  bool doubleSupported;
  cl_device_type  deviceType;

  // Test specific options
  uint gloalBWIters;
  uint64_t globalBWMaxSize;
  uint computeWgsPerCU;
  uint computeDPWgsPerCU;
  uint computeIters;
  uint transferBWIters;
  uint kernelLatencyIters;
  uint64_t transferBWMaxSize;
} device_info_t;

class Timer
{
public:

  std::chrono::high_resolution_clock::time_point tick, tock;

  void start();

  // Stop and return time in micro-seconds
  float stopAndTime();
};

device_info_t getDeviceInfo(cl::Device &d);

// Return time in us for the given event
float timeInUS(cl::Event &timeEvent);

// Round down to next multiple of the given base with an optional maximum value
uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue = UINT64_MAX);

void populate(float *ptr, uint64_t N);
void populate(double *ptr, uint64_t N);

void trimString(std::string &str);

#endif  // COMMON_H

