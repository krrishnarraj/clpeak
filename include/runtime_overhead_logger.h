#pragma once

#include <clpeak.h>

inline void clPeak::printHeader(size_t queueSize, uint32_t threadSize, const cl::NDRange& globalWorkSize, const cl::NDRange& localWorkSize) {
  std::string localWorkSizeChar;
  std::string threadSizeChar;
  if (localWorkSize.dimensions() == 0) {
    localWorkSizeChar = "NULL";
  }
  else {
    localWorkSizeChar = std::to_string(localWorkSize[0]);
  }
  if (threadSize == 0) {
    threadSizeChar = "N/A";
  }
  else {
    threadSizeChar = std::to_string(threadSize);
  }
  log->print(std::string(TAB TAB TAB TAB) + std::to_string(queueSize) + std::string(" queue, ") + threadSizeChar + std::string(" threads, GlobalWorgroupSize set to ") + std::to_string(globalWorkSize[0]) + std::string(", LocalWorkgroupSize = ") + localWorkSizeChar + std::string(NEWLINE));
  log->xmlOpenTag("Configuration");
  log->xmlAppendAttribs("queues", std::to_string(queueSize));
  log->xmlAppendAttribs("threads", threadSizeChar);
  log->xmlAppendAttribs("gws", std::to_string(globalWorkSize[0]));
  log->xmlAppendAttribs("lws", localWorkSizeChar);
}

template <uint32_t enqueueIterations, uint32_t flushIterations>
inline void clPeak::printRecords(size_t enqueuesPerGivenTime, const char *lineStart)
{
  log->print(lineStart); log->print("Enqueue count   : "); log->print(enqueueIterations * enqueuesPerGivenTime); log->print(NEWLINE);
  log->print(lineStart); log->print("Flush count   : "); log->print(flushIterations); log->print(NEWLINE);
  log->print(lineStart); log->print("1 flush per "); log->print(enqueueIterations / flushIterations * enqueuesPerGivenTime); log->print(" enqueues" NEWLINE);
  log->print(lineStart); log->print("Total time for "); log->print(enqueuesPerGivenTime); log->print(" enqueues" NEWLINE);
}

template <uint32_t enqueueIterations, uint32_t flushIterations>
inline void clPeak::printRecords(size_t enqueuesPerGivenTime, const performanceStatisticsPackVec &statistics, microsecondsT warmupDuration, const char *lineStart)
{
  printRecords<enqueueIterations, flushIterations>(enqueuesPerGivenTime, lineStart);

  log->print(lineStart); log->print(TAB "Enqueue statistics" NEWLINE);
  printRecords(statistics.enqueueStatistics, warmupDuration);

  log->print(lineStart); log->print(TAB "Flush statistics" NEWLINE);
  printRecords(statistics.flushStatistics);
  log->print(NEWLINE);
}

inline void clPeak::printRecords(const performanceStatistics &statistics, microsecondsT warmupDuration, const char *lineStart)
{
  if (isVerbose)
  {
    if (warmupDuration != 0.0f)
    {
      log->print(lineStart); log->print("warmup time   : ");
      log->print(warmupDuration);     log->print(NEWLINE);
    }
    log->print(lineStart); log->print("minimum time   : ");
    log->print(statistics.min);     log->print(NEWLINE);
    log->print(lineStart); log->print("maximum time   : ");
    log->print(statistics.max);     log->print(NEWLINE);
    log->print(lineStart); log->print("average time   : ");
    log->print(statistics.average);     log->print(NEWLINE);
    log->print(lineStart); log->print("standard deviation   : ");
    log->print(statistics.standardDeviation);     log->print(NEWLINE);
    log->print(lineStart); log->print("median time   : ");
    log->print(statistics.median);     log->print(NEWLINE);
  }
  log->print(lineStart); log->print("mode   : ");
  log->print(statistics.mode, getEpsilon(statistics.average));     log->print(NEWLINE);
}

template <uint32_t enqueueIterations, uint32_t flushIterations>
inline void clPeak::logRecords(size_t enqueuesPerGivenTime, const performanceStatisticsPackVec &statistics, microsecondsT warmupDuration)
{
  log->xmlOpenTag("Case");
  log->xmlAppendAttribs("enqueue_count", enqueueIterations * enqueuesPerGivenTime);
  log->xmlAppendAttribs("flush_count", flushIterations);
  log->xmlAppendAttribs("enqueues_per_flush", enqueueIterations / flushIterations * enqueuesPerGivenTime);
  log->xmlAppendAttribs("total_time_for_enqueue_count", enqueuesPerGivenTime);

  logRecords("Enqueue", statistics.enqueueStatistics, warmupDuration);
  logRecords("Flush", statistics.flushStatistics);

  log->xmlCloseTag();     // case
}

inline void clPeak::logRecords(const char* tagName, const performanceStatistics &statistics, microsecondsT warmupDuration)
{
  log->xmlOpenTag(tagName);
  if (warmupDuration != 0.0f) {
    log->xmlRecord("warmup", warmupDuration);
  }
  log->xmlRecord("min", statistics.min);
  log->xmlRecord("max", statistics.max);
  log->xmlRecord("average", statistics.average);
  log->xmlRecord("standard_deviation", statistics.standardDeviation);
  log->xmlRecord("median", statistics.median);
  log->xmlRecord("mode", statistics.mode, getEpsilon(statistics.average));
  log->xmlCloseTag();     // tagName
}
