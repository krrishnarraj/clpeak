#ifdef ENABLE_CUDA

#include <cuda_peak.h>
#include <options.h>

void CudaPeak::applyOptions(const CliOptions &opts)
{
  forceIters     = opts.forceIters;
  specifiedIters = opts.iters;
  warmupCount    = opts.warmupCount;
  deviceIndex    = opts.cudaDeviceIndex;
  gating.copyFrom(opts);

  // File output is centralized in entry.cpp::main(); see the comment in
  // clpeak.cpp::applyOptions().
  log.reset(new logger(false, "", false, "", false, "",
                       opts.compareFile));
}

#endif // ENABLE_CUDA
