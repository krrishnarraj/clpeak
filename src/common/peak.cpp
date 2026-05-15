#include <common/peak.h>
#include <common/options.h>

Peak::Peak()
    : log(new logger())
{
}

void Peak::applyOptions(const CliOptions &opts)
{
    forceIters     = opts.forceIters;
    specifiedIters = opts.iters;
    warmupCount    = opts.warmupCount;
    targetTimeUs   = opts.targetTimeUs;
    gating.copyFrom(opts);

    // Per-backend loggers handle stdout + baseline-compare deltas only.
    // File output is centralized in src/cli/main.cpp after all backends
    // have run, so a single dump file aggregates every backend's rows.
    log.reset(new logger(opts.compareFile));
}
