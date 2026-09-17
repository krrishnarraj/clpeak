#include <common/peak.h>
#include <common/common.h>
#include <common/coreml_cache.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/backend_registry.h>
#include <common/run_document.h>
#include <common/run_log.h>
#include <common/logger_text.h>
#include <common/host_info.h>
#include <version.h>
#include <iostream>
#include <vector>

#ifdef ENABLE_ONNX
#include <onnx/onnx_peak.h>  // onnxSetLibraryOverride, for --onnx-lib
#endif
#ifdef ENABLE_LITERT
#include <litert/litert_peak.h>  // litertSetLibraryOverride, for --litert-lib
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
    // Plugin providers register on the runtime's environment, which the
    // first enumeration creates: the set has to be in place before it.
    if (!opts.onnxEpLibraries.empty())
    {
        std::vector<OnnxEpLibrary> libs;
        for (const auto &e : opts.onnxEpLibraries)
            libs.push_back({e.first, e.second, true});
        onnxSetEpLibraries(std::move(libs));
    }
    if (opts.onnxWinml)
        onnxSetWinml(true, opts.onnxWinmlPath);
#endif
#ifdef ENABLE_LITERT
    if (!opts.litertLibPath.empty())
        litertSetLibraryOverride(opts.litertLibPath);
    if (!opts.litertNpuDir.empty())
        litertSetNpuDirOverride(opts.litertNpuDir);
#endif

    const auto &backends = backendRegistry();

    // --list-devices: every enabled backend's inventory, in one format and
    // in the order a run would visit them.
    if (opts.listDevices)
    {
        for (Backend b : opts.requestedButNotBuilt())
            std::cout << "clpeak: the " << backendInfo(b).name
                      << " backend is not in this build\n";
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
    for (const auto &be : backends)
        combined.meta.build.backends.push_back(backendInfo(be.id).name);
#ifdef CLPEAK_BUILD_CONFIG
    combined.meta.build.config = CLPEAK_BUILD_CONFIG;
#endif
    combined.meta.host          = probeHost();
    combined.meta.invocation    = invocationFrom(opts, argc, argv);

    // The run's diagnostic stream: every note, CLPEAK_LOG line and relayed
    // library message from here on lands on combined.log -- and, with -o,
    // on the sidecar as it happens, so a crash inside a driver still leaves
    // a record.  Between backends no logger is open to print for it, so it
    // prints the way LoggerText would.
    RunLog runLog(combined);
    runLog.setFallbackRenderer([&](const LogEntry &e) {
        if (!e.source.empty() || e.level == clpeak::LogLevel::Debug ||
            e.level == clpeak::LogLevel::Info)
        {
            if (opts.verbose) clpeak::stderrWrite(e.message + "\n");
            return;
        }
        std::cout << e.message << "\n";
    });
    if (opts.enableOutput)
        runLog.openSidecar(opts.outputFile);

    // A backend asked for by name that this binary does not carry: say so,
    // or the run reads as "no devices".
    for (Backend b : opts.requestedButNotBuilt())
        CLPEAK_LOG(Warning, "clpeak: the %s backend is not in this build",
                   backendInfo(b).name);

    // --verbose: the file also carries what --list-devices would have shown
    // -- every device each backend saw, and why a backend had none -- since
    // "my NPU is not listed" is the question a verbose dump exists to answer.
    // Costs an enumeration pass, which is why it is not on by default.
    if (opts.verbose && opts.enableOutput)
        for (const auto &be : backends)
            if (opts.backendEnabled(be.id))
                combined.inventory.push_back(be.enumerate());

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
        // --verbose with -o: the base logger mirrors a canonical transcript
        // of backend/device/test headers and metric rows onto the run's log,
        // so the file's `log` reads as the run looked live.
        const bool mirror = opts.verbose && opts.enableOutput;
        peak->log.reset(
            new LoggerText(std::cout, opts.compareFile, opts.describe, opts.verbose, mirror));
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

    combined.meta.durationS = runLog.elapsedS();
    runLog.finish();

    // Centralized file dump.  A failed dump surfaces in the exit code like any
    // backend failure -- and keeps the sidecar, which is then the only record
    // of the run.
    if (opts.enableOutput)
    {
        const bool saved = saveRunJson(combined, opts.outputFile);
        runLog.closeSidecar(/*remove=*/saved);
        if (!saved)
            lastError |= 1;
    }

    return lastError;
}
