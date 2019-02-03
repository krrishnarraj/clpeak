#ifndef COMMON_H
#define COMMON_H

#include <CL/cl.hpp>
#if defined(__APPLE__) || defined(__MACOSX) || defined(__FreeBSD__)
#include <sys/types.h>
#endif

#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

#define TAB             "  "
#define NEWLINE         "\n"
#ifndef __FreeBSD__
#define uint            unsigned int
#endif
#define ulong           unsigned long int

#define MAX(X, Y)       \
  (X > Y)? X: Y;

#define MIN(X, Y)       \
  (X < Y)? X: Y;


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
  ulong maxAllocSize;
  ulong maxGlobalSize;
  uint maxClockFreq;

  bool halfSupported;
  bool doubleSupported;
  cl_device_type  deviceType;

  // Test specific options
  int gloalBWIters;
  int computeWgsPerCU;
  int computeIters;
  int transferBWIters;
  int kernelLatencyIters;

} device_info_t;

namespace constants {
  const uint32_t thousandIterations = 1000u;
  const uint32_t hundredIterations = 100u;
  const uint32_t tenIterations = 10u;
}

using microsecondsT = double;
using occurence = uint32_t;

class Timer
{
public:
  void start();
  void stop();
  microsecondsT duration();

  // Stop and return time in micro-seconds
  float stopAndTime();

private:
  microsecondsT roundValue(long long val);

  const uint32_t nanoToMicrosecondsResolution = 1000;
  std::chrono::high_resolution_clock::time_point tick, tock;
};

struct durationTimesVec {
  std::vector<microsecondsT> enqueueTimesVector;
  std::vector<microsecondsT> flushTimesVector;

  durationTimesVec(uint32_t enqueueTimes, uint32_t flushTimes) {
    enqueueTimesVector = std::vector<microsecondsT>(enqueueTimes);
    flushTimesVector = std::vector<microsecondsT>(flushTimes);
  }

  durationTimesVec(durationTimesVec&&) = default;

  durationTimesVec& operator=(durationTimesVec&&) = default;
};

device_info_t getDeviceInfo(cl::Device &d);

// Return time in us for the given event
float timeInUS(cl::Event &timeEvent);

// Round down to next multiple of the given base with an optional maximum value
uint roundToMultipleOf(uint number, const uint base, int maxValue = -1);

void populate(float *ptr, uint N);
void populate(double *ptr, uint N);

void trimString(std::string &str);

inline microsecondsT getMininimumTime(const std::vector<microsecondsT> &sortedVector) {
  return sortedVector[0];
}

inline microsecondsT getMaximumTime(const std::vector<microsecondsT> &sortedVector) {
  return sortedVector[sortedVector.size() - 1];
}

inline microsecondsT getAverage(const std::vector<microsecondsT> &vector)
{
  microsecondsT average = 0;
  for (const auto& element : vector)
  {
    average += element;
  }

  return average / vector.size();
}

inline microsecondsT getStandardDeviation(const std::vector<microsecondsT> &vector, microsecondsT average)
{
  microsecondsT standardDeviation = 0.0f;

  for (const auto& element : vector) {
    standardDeviation += std::pow(element - average, 2);
  }

  return std::sqrt(standardDeviation / vector.size());
}

inline microsecondsT getMedian(const std::vector<microsecondsT>& sortedVector) {
  if (sortedVector.size() % 2 != 0) {
    return sortedVector[sortedVector.size() / 2];
  }
  else {
    return (sortedVector[(sortedVector.size() / 2) - 1] + sortedVector[sortedVector.size() / 2]) / 2.0;
  }
}

inline microsecondsT getEpsilon(microsecondsT average) {
  return average / 20.0f;
}

inline std::pair<microsecondsT, occurence> getMode(const std::vector<microsecondsT> &sortedVector, microsecondsT average)
{
  uint32_t counter = 0u;
  uint32_t occurence = 0u;
  microsecondsT mode = 0.0f;
  microsecondsT epsilon = getEpsilon(average);

  size_t element = 0u;
  while (element < sortedVector.size() - 1) {
    for (size_t elementIterated = 0; elementIterated < sortedVector.size() - 1; ++elementIterated) {
      if (abs(sortedVector[element] - sortedVector[elementIterated]) < epsilon) {
        ++counter;
        if (counter > occurence)
        {
          occurence = counter;
          mode = sortedVector[element];
        }
      }
      else
      {
        counter = 1u;
        if (elementIterated > element) {
          break;
        }
      }
    }
    ++element;
  }

  return std::make_pair(mode, occurence);
}

struct performanceStatistics {
  performanceStatistics(const std::vector<microsecondsT> &sortedTimes) : min(getMininimumTime(sortedTimes)), max(getMaximumTime(sortedTimes)), average(getAverage(sortedTimes)),
                                                                         standardDeviation(getStandardDeviation(sortedTimes, average)), median(getMedian(sortedTimes)),
                                                                         mode(getMode(sortedTimes, average)) {}

  performanceStatistics() = default;
  microsecondsT min;
  microsecondsT max;
  microsecondsT average;
  microsecondsT standardDeviation;
  microsecondsT median;
  std::pair<microsecondsT, occurence> mode;
};

inline void sortVector(std::vector<microsecondsT> &vector) {
  std::sort(vector.begin(), vector.end());
}

inline void sortTimes(durationTimesVec &durationTimes) {
  sortVector(durationTimes.enqueueTimesVector);
  sortVector(durationTimes.flushTimesVector);
}

inline performanceStatistics getAllStatistics(std::vector<microsecondsT> &durationTimes) {
  sortVector(durationTimes);

  return performanceStatistics(durationTimes);
}

struct performanceStatisticsPackVec {
  performanceStatisticsPackVec(const durationTimesVec &durationTimes) : enqueueStatistics(durationTimes.enqueueTimesVector),
    flushStatistics(durationTimes.flushTimesVector) {}

  performanceStatisticsPackVec() = default;

  performanceStatistics enqueueStatistics;
  performanceStatistics flushStatistics;
};

inline performanceStatisticsPackVec getAllStatistics(durationTimesVec &durationTimes) {
  sortTimes(durationTimes);

  return performanceStatisticsPackVec(durationTimes);
}

template <uint32_t batchSize, uint32_t enqueueIterations, uint32_t flushIterations, typename std::enable_if<batchSize == 1u, void>::type* = nullptr>
void checkBoundaries()
{
  static_assert(enqueueIterations % flushIterations == 0, "When batchSize is equal to 1, size of enqueue time array must be multiplication of flush time array size");
}

template <uint32_t batchSize, uint32_t enqueueIterations, uint32_t flushIterations, typename std::enable_if<batchSize != 1u, void>::type* = nullptr >
void checkBoundaries()
{
  static_assert(enqueueIterations == flushIterations, "when batchSize is bigger than 1, size of enqueue time array must be equal to flush time array size");
}

#endif  // COMMON_H
