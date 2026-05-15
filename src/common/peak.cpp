#include <common/peak.h>
#include <common/options.h>

Peak::Peak() = default;

void Peak::applyOptions(const CliOptions &opts)
{
    forceIters     = opts.forceIters;
    specifiedIters = opts.iters;
    warmupCount    = opts.warmupCount;
    targetTimeUs   = opts.targetTimeUs;
    gating.copyFrom(opts);
}
