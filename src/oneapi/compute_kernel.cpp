#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>

// Shared helpers for the compute_*.cpp drivers.  Unlike the ROCm backend's
// runtime-source dispatch table, every compute benchmark in this backend is a
// SYCL lambda living in its own TU, so this file only defines small
// utilities reused across compute_float.cpp / compute_int.cpp.

namespace clpeak_oneapi {

uint32_t pickComputeBlocks(const oneapi_device_info_t &info,
                           uint32_t blockSize, uint32_t outElemsPerBlock,
                           uint32_t elemSize)
{
  uint64_t globalThreads = targetGlobalThreads((uint32_t)info.numCUs);
  uint64_t bytesPerBlock = (uint64_t)outElemsPerBlock * elemSize;
  uint64_t maxBlocks = bytesPerBlock ? (info.totalGlobalMem / 4 / bytesPerBlock) : 0;
  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  if (pickBlocks == 0)
    pickBlocks = 1;
  return (uint32_t)pickBlocks;
}

float computeGflops(uint64_t totalThreads, uint32_t workPerWI, float meanUs,
                    double unitDivider /*=1e9*/)
{
  if (meanUs <= 0.0f)
    return -1.0f;
  if (unitDivider <= 0.0)
    unitDivider = 1e9;
  return (float)((double)totalThreads * (double)workPerWI * 1e6 /
                 (double)meanUs / unitDivider);
}

} // namespace clpeak_oneapi

#endif // ENABLE_ONEAPI
