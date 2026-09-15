#include <common/peak.h>
#include <common/common.h>
#include <common/coreml_cache.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/backend_registry.h>
#include <common/run_document.h>
#include <common/logger_text.h>
#include <common/host_info.h>
#include <version.h>
#include <chrono>
#include <iostream>

#ifdef ENABLE_ONNX
#include <onnx/onnx_peak.h>  // onnxSetLibraryOverride, for --onnx-lib
#endif

int main(int argc, char **argv)
{
    CliOptions opts;
    parseCliOptions(argc, argv, opts);
    clpeak::setVerbose(opts.verbose);
#ifdef ENABLE_ONNX
    // Before any enumeration: --onnx-lib decides which runtime gets loaded,
    // and enumerate() is what loads it.
    if (!opts.onnxLibPath.empty())
        onnxSetLibraryOverride(opts.onnxLibPath);
#endif

    const auto &backends = backendRegistry();

    // A backend asked for by name that this binary does not carry: say so,
    // or the run reads as "no devices".
    for (Backend b : opts.requestedButNotBuilt())
        std::cout << "clpeak: the " << backendInfo(b).name
                  << " backend is not in this build\n";

    // --list-devices: every enabled backend's inventory, in one format and
    // in the order a run would visit them.
    if (opts.listDevices)
    {
        std::vector<BackendInventory> invs;
        for (const auto &be : backends)
            if (opts.backendEnabled(be.id))
                invs.push_back(be.enumerate());
        printInventory(invs, std::cout);
        return 0;
    }

    RunDocument combined;
    combined.meta.clpeakVersion = CLPEAK_VERSION_STR;
    combined.meta.generatedAt   = isoTimestampUtc();
    combined.meta.host          = probeHost();
    combined.meta.invocation    = invocationFrom(opts, argc, argv);
    const auto runStart = std::chrono::steady_clock::now();

    // Run every enabled backend in order.  No devices is not an error
    // (normal in VM/CI environments).  Only real failures (driver init,
    // runtime errors) produce a non-zero status.  We OR the statuses so
    // any real error from any backend surfaces in the exit code.
    int lastError = 0;

    for (const auto &be : backends)
    {
        if (!opts.backendEnabled(be.id))
            continue;

        auto peak = be.create();
        peak->log.reset(
            new LoggerText(std::cout, opts.compareFile, opts.describe, opts.verbose));
        peak->applyOptions(opts);
        int status = peak->runAll();
        combined.append(peak->log->doc);
        // Whatever this backend had Core ML compile is finished with; the
        // backends that use Core ML clean up after themselves as they go,
        // and this is the backstop for one that does not yet (see
        // include/common/coreml_cache.h).  A no-op off Apple platforms.
        clpeak::purgeCoreMLCompileCache();

        if (status != 0)
            lastError |= status;
    }

    combined.meta.durationS = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - runStart)
                                  .count();

    // Centralized file dump.  A failed dump surfaces in the exit code like any
    // backend failure.
    if (opts.enableOutput && !saveRunJson(combined, opts.outputFile))
        lastError |= 1;

    return lastError;
}
