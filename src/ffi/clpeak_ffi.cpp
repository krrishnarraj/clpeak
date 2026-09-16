#include "clpeak_ffi.h"
#include "logger_ffi.h"

#include <common/backend_registry.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/coreml_cache.h>
#include <common/options.h>
#include <common/peak.h>
#include <common/host_info.h>
#include <common/run_document.h>
#include <common/run_log.h>
#include <version.h>

#ifdef ENABLE_ONNX
#include <onnx/onnx_peak.h>  // onnxSetLibraryOverride / onnxRuntimeStatus
#endif
#ifdef ENABLE_LITERT
#include <litert/litert_peak.h>  // litertSetLibraryOverride / litertRuntimeStatus
#endif

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace
{

char *copyString(const std::string &value)
{
    char *out = static_cast<char *>(std::malloc(value.size() + 1));
    if (!out)
        return nullptr;
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

// A log entry recorded outside any backend's logger (the RunLog's fallback):
// forwarded as the same `log` event a logger would have sent.
void emitLogEntry(ClpeakEventCallback cb, void *userData, const LogEntry &entry)
{
    LogEvent e;
    e.kind        = LogEvent::Kind::Log;
    e.backend     = entry.backend;
    e.device      = entry.device;
    e.deviceIndex = entry.deviceIndex;
    e.log         = entry;
    ffiEmitJson(cb, userData, ffiEventToJson(e));
}

void emitDone(ClpeakEventCallback cb, void *userData, int status, bool cancelled)
{
    std::string json = "{\"t\":\"done\",\"status\":" + std::to_string(status) +
                       ",\"cancelled\":" + (cancelled ? "true" : "false") + "}";
    ffiEmitJson(cb, userData, json);
}

// One launch at a time — the run loop and the cancel flag are process-global.
std::atomic<bool> g_running{false};

} // namespace

const char *clpeak_version(void)
{
    return CLPEAK_VERSION_STR;
}

char *clpeak_copy_backend_catalog_json(void)
{
    std::vector<BackendInventory> inv;
    for (const auto &be : backendRegistry())
        inv.push_back(be.enumerate());
    return copyString(inventoryToJson(inv));
}

void clpeak_free_string(char *s)
{
    std::free(s);
}

char *clpeak_copy_onnx_status_json(void)
{
#ifdef ENABLE_ONNX
    const OnnxRuntimeStatus st = onnxRuntimeStatus();
    std::string json = "{\"available\":";
    json += st.available ? "true" : "false";
    json += ",\"linkedIn\":";
    json += st.linkedIn ? "true" : "false";
    json += ",\"version\":\"" + jsonEscape(st.version) + "\"";
    json += ",\"path\":\"" + jsonEscape(st.path) + "\"";
    json += ",\"error\":\"" + jsonEscape(st.error) + "\"}";
    return copyString(json);
#else
    return copyString(
        "{\"available\":false,\"linkedIn\":false,\"version\":\"\","
        "\"path\":\"\",\"error\":\"ONNX backend not built in\"}");
#endif
}

void clpeak_set_onnx_library(const char *path)
{
#ifdef ENABLE_ONNX
    onnxSetLibraryOverride(path ? path : "");
#else
    (void)path;
#endif
}

char *clpeak_copy_litert_status_json(void)
{
#ifdef ENABLE_LITERT
    const LitertRuntimeStatus st = litertRuntimeStatus();
    std::string json = "{\"available\":";
    json += st.available ? "true" : "false";
    json += ",\"version\":\"" + jsonEscape(st.version) + "\"";
    json += ",\"path\":\"" + jsonEscape(st.path) + "\"";
    json += ",\"error\":\"" + jsonEscape(st.error) + "\"}";
    return copyString(json);
#else
    return copyString(
        "{\"available\":false,\"version\":\"\",\"path\":\"\","
        "\"error\":\"LiteRT backend not built in\"}");
#endif
}

void clpeak_set_litert_library(const char *path)
{
#ifdef ENABLE_LITERT
    litertSetLibraryOverride(path ? path : "");
#else
    (void)path;
#endif
}

void clpeak_set_litert_npu_dir(const char *dir)
{
#ifdef ENABLE_LITERT
    litertSetNpuDirOverride(dir ? dir : "");
#else
    (void)dir;
#endif
}

void clpeak_request_cancel(void)
{
    clpeak::requestCancel();
}

int clpeak_launch(int argc, const char **argv,
                  ClpeakEventCallback on_event, void *user_data)
{
    bool expected = false;
    if (!g_running.compare_exchange_strong(expected, true))
        return CLPEAK_RUN_BUSY;  // no done event: the in-flight run owns the stream

    clpeak::resetCancel();

    std::vector<char *> mutableArgv;
    mutableArgv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++)
        mutableArgv.push_back(const_cast<char *>(argv[i]));

    // The run's diagnostic stream, live before the arguments are even parsed
    // so a rejected argv is on it too.  Entries a backend's logger records
    // reach the GUI through that logger; the rest through this fallback, as
    // the same `log` event.
    RunDocument combined;
    RunLog      runLog(combined);
    runLog.setFallbackRenderer([&](const LogEntry &e) {
        emitLogEntry(on_event, user_data, e);
    });

    CliOptions opts;
    std::string parseError;
    if (!parseCliOptionsNoExit(argc, mutableArgv.data(), opts, parseError))
    {
        clpeak::logMessage(clpeak::LogLevel::Error, "", parseError);
        emitDone(on_event, user_data, CLPEAK_RUN_BAD_ARGS, false);
        g_running.store(false);
        return CLPEAK_RUN_BAD_ARGS;
    }

    clpeak::setVerbose(opts.verbose);
#ifdef ENABLE_ONNX
    if (!opts.onnxLibPath.empty())
        onnxSetLibraryOverride(opts.onnxLibPath);
#endif
#ifdef ENABLE_LITERT
    if (!opts.litertLibPath.empty())
        litertSetLibraryOverride(opts.litertLibPath);
    if (!opts.litertNpuDir.empty())
        litertSetNpuDirOverride(opts.litertNpuDir);
#endif

    combined.meta.clpeakVersion = CLPEAK_VERSION_STR;
    combined.meta.generatedAt   = isoTimestampUtc();
    for (const auto &be : backendRegistry())
        combined.meta.build.backends.push_back(backendInfo(be.id).name);
    combined.meta.host          = probeHost();
    combined.meta.invocation    = invocationFrom(opts, argc, mutableArgv.data());
    // The sidecar: every entry on disk as it happens, for the native crash
    // the document never gets written after.  The app adopts one left behind
    // on its next launch (app/lib/src/services/run_history_store.dart).
    if (opts.enableOutput)
        runLog.openSidecar(opts.outputFile);
    int status = 0;

    for (Backend b : opts.requestedButNotBuilt())
        CLPEAK_LOG(Warning, "clpeak: the %s backend is not in this build",
                   backendInfo(b).name);

    // --verbose: the catalog as the run saw it, in the file.  Already
    // enumerated once for the run screen, so the memoised probes make this
    // cheap here, unlike in the CLI.
    if (opts.verbose && opts.enableOutput)
        for (const auto &be : backendRegistry())
            if (opts.backendEnabled(be.id))
                combined.inventory.push_back(be.enumerate());

    for (const auto &be : backendRegistry())
    {
        if (!opts.backendEnabled(be.id) || clpeak::cancelRequested())
            continue;

        auto peak = be.create();
        peak->log.reset(new LoggerFfi(on_event, user_data));
        peak->applyOptions(opts);
        status |= peak->runAll();
        combined.append(peak->log->doc);
        // Backstop for Core ML's compile cache, as in the CLI (see
        // include/common/coreml_cache.h); the app's own cache directory on
        // iOS and macOS is where it lands.
        clpeak::purgeCoreMLCompileCache();
    }

    bool cancelled = clpeak::cancelRequested();

    combined.meta.cancelled = cancelled;
    combined.meta.durationS = runLog.elapsedS();
    runLog.finish();

    // Centralized file dump, exactly like the CLI — also runs after a
    // cancellation so partial results get persisted.  The `cancelled` flag is
    // what tells a reader those results are partial: without it, every test
    // the run never reached looks like hardware that lacks the feature.  The
    // sidecar goes once the document is safely written, and stays if it is
    // not.
    if (opts.enableOutput)
    {
        const bool saved = saveRunJson(combined, opts.outputFile);
        runLog.closeSidecar(/*remove=*/saved);
        if (!saved)
            status |= 1;
    }

    int result = cancelled ? CLPEAK_RUN_CANCELLED : status;

    emitDone(on_event, user_data, result, cancelled);
    g_running.store(false);
    return result;
}
