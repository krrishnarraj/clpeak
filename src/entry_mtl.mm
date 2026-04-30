#ifdef ENABLE_METAL

#include <mtl_peak.h>
#include <options.h>

void MetalPeak::applyOptions(const CliOptions &opts)
{
  forceIters     = opts.forceIters;
  specifiedIters = opts.iters;
  warmupCount    = opts.warmupCount;
  listDevices    = opts.listDevices;
  deviceIndex    = opts.mtlDeviceIndex;
  enabledTests   = opts.enabledTests;

  log.reset(new logger(opts.enableXml,  opts.xmlFile,
                       opts.enableJson, opts.jsonFile,
                       opts.enableCsv,  opts.csvFile,
                       opts.compareFile));
}

#endif // ENABLE_METAL
