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

    // Only the items that name this backend.  The run loop already skips a
    // backend with none when --devices was given (CliOptions::backendEnabled),
    // so an empty list here means "every device".
    selectedDevices.clear();
    for (const DeviceSelector &sel : opts.devices)
        if (sel.backend == backend())
            selectedDevices.push_back(sel.index);
}
