#ifdef ENABLE_VULKAN

#include <vk_peak.h>
#include <options.h>

void vkPeak::applyOptions(const CliOptions &opts)
{
  forceIters     = opts.forceIters;
  specifiedIters = opts.iters;
  warmupCount    = opts.warmupCount;
  deviceIndex    = opts.vkDeviceIndex;
  enabledTests      = opts.enabledTests;
  enabledCategories = opts.enabledCategories;

  log.reset(new logger(opts.enableXml,  opts.xmlFile,
                       opts.enableJson, opts.jsonFile,
                       opts.enableCsv,  opts.csvFile,
                       opts.compareFile));
}

#endif // ENABLE_VULKAN
