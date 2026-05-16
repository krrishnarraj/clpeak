#include <common/peak.h>
#include <common/options.h>

void Peak::applyOptions(const CliOptions &opts)
{
    forceIters     = opts.forceIters;
    specifiedIters = opts.iters;
    warmupCount    = opts.warmupCount;
    targetTimeUs   = opts.targetTimeUs;
    enabledTests      = opts.enabledTests;
    enabledCategories = opts.enabledCategories;
}
