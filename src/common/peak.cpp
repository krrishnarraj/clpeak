#include <common/peak.h>
#include <common/common.h>
#include <common/options.h>

void Peak::applyOptions(const CliOptions &opts)
{
    clpeak::setVerbose(opts.verbose);
    forceIters     = opts.forceIters;
    specifiedIters = opts.iters;
    warmupCount    = opts.warmupCount;
    targetTimeUs   = opts.targetTimeUs;
    enabledTests      = opts.enabledTests;
    enabledCategories = opts.enabledCategories;
}
