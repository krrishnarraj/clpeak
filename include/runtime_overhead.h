#pragma once

#include <clpeak.h>
#include <type_traits>
#include <vector>

template <uint32_t batchSize, uint32_t enqueueIterations, uint32_t flushIterations>
inline void clPeak::runKernel(cl::CommandQueue &queue, cl::Kernel &kernel, durationTimesVec &times, const cl::NDRange &globalOffsetSize,
                       const cl::NDRange &globalSize, const cl::NDRange &localSize)
{
  Timer time;
  uint32_t iteration = 0;
  static_assert(batchSize != 0u, "batchSize must be grater than 0");
  static_assert(enqueueIterations != 0u, "enqueueIterations must be grater than 0");
  static_assert(flushIterations != 0u, "flushIterations must be grater than 0");

  checkBoundaries<batchSize, enqueueIterations, flushIterations>();

  constexpr uint32_t flushAfter = enqueueIterations / flushIterations;

  for (uint32_t i = 0; i < flushIterations; i++) {
    for (uint32_t j = 0; j < flushAfter; j++) {
      time.start();
      for (uint32_t k = 0; k < batchSize; k++) {
        queue.enqueueNDRangeKernel(kernel, globalOffsetSize, globalSize, localSize);
      }
      time.stop();

      times.enqueueTimesVector[j + iteration] = time.duration();
    }

    iteration += flushAfter;
    times.flushTimesVector[i] = flushQueue(queue);
  }
}

template <uint32_t batchSize, uint32_t enqueueIterations, uint32_t flushIterations>
inline void clPeak::generateTestCase(cl::CommandQueue &queue, cl::Kernel &kernel, const cl::NDRange &globalOffsetSize, const cl::NDRange &globalSize, const cl::NDRange &localSize)
{
  durationTimesVec durationTimesStructure(enqueueIterations, flushIterations);

  ///////////////////////////////////////////////////////////////////////////

  //warmup
  auto warmupDuration = runKernel(queue, kernel, globalOffsetSize, globalSize, localSize);

  runKernel<batchSize, enqueueIterations, flushIterations>(queue, kernel, durationTimesStructure, globalOffsetSize, globalSize, localSize);

  auto statistics = getAllStatistics(durationTimesStructure);

  printRecords<enqueueIterations, flushIterations>(batchSize, statistics, warmupDuration);
  logRecords<enqueueIterations, flushIterations>(batchSize, statistics, warmupDuration);

  queue.finish();
  ///////////////////////////////////////////////////////////////////////////
}
