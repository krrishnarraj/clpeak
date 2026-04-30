#ifdef ENABLE_CUDA

#include <cuda_peak.h>
#include <options.h>

void CudaPeak::applyOptions(const CliOptions &opts)
{
  forceIters     = opts.forceIters;
  specifiedIters = opts.iters;
  warmupCount    = opts.warmupCount;
  listDevices    = opts.listDevices;
  deviceIndex    = opts.cudaDeviceIndex;
  enabledTests   = opts.enabledTests;

  log.reset(new logger(opts.enableXml,  opts.xmlFile,
                       opts.enableJson, opts.jsonFile,
                       opts.enableCsv,  opts.csvFile,
                       opts.compareFile));
}

#endif // ENABLE_CUDA
