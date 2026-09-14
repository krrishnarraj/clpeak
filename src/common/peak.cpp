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

    // A bare index applies to every backend; backend:index only to its own.
    selectedDevices.clear();
    for (const DeviceSelector &sel : opts.devices)
        if (sel.backend == Backend::COUNT || sel.backend == backend())
            selectedDevices.push_back(sel.index);
}
