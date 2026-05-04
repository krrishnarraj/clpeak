#ifdef ENABLE_VULKAN

#include <vk_peak.h>
#include <options.h>

void vkPeak::applyOptions(const CliOptions &opts)
{
  forceIters     = opts.forceIters;
  specifiedIters = opts.iters;
  warmupCount    = opts.warmupCount;
  deviceIndex    = opts.vkDeviceIndex;
  gating.copyFrom(opts);

  // File output is centralized in entry.cpp::main(); see the comment in
  // clpeak.cpp::applyOptions().
  log.reset(new logger(false, "", false, "", false, "",
                       opts.compareFile));
}

#endif // ENABLE_VULKAN
