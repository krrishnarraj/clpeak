#ifdef ENABLE_METAL

#include <mtl_peak.h>
#include <options.h>

void MetalPeak::applyOptions(const CliOptions &opts)
{
  forceIters     = opts.forceIters;
  specifiedIters = opts.iters;
  warmupCount    = opts.warmupCount;
  deviceIndex    = opts.mtlDeviceIndex;
  enabledTests      = opts.enabledTests;
  enabledCategories = opts.enabledCategories;

  log.reset(new logger(opts.enableXml,  opts.xmlFile,
                       opts.enableJson, opts.jsonFile,
                       opts.enableCsv,  opts.csvFile,
                       opts.compareFile));
}

#endif // ENABLE_METAL
