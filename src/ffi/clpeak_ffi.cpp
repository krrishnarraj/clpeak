#include "clpeak_ffi.h"
#include "logger_ffi.h"

#include <common/backend_registry.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/options.h>
#include <common/peak.h>
#include <common/host_info.h>
#include <common/run_document.h>
#include <version.h>

#ifdef ENABLE_ONNX
#include <onnx/onnx_peak.h>  // onnxSetLibraryOverride / onnxRuntimeStatus
#endif

#include <atomic>
#include <chrono>
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

void emitNote(ClpeakEventCallback cb, void *userData, const std::string &msg)
{
    LogEvent e;
    e.kind    = LogEvent::Kind::Note;
    e.message = msg;
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

    CliOptions opts;
    std::string parseError;
    if (!parseCliOptionsNoExit(argc, mutableArgv.data(), opts, parseError))
    {
        emitNote(on_event, user_data, parseError);
        emitDone(on_event, user_data, CLPEAK_RUN_BAD_ARGS, false);
        g_running.store(false);
        return CLPEAK_RUN_BAD_ARGS;
    }

    clpeak::setVerbose(opts.verbose);
#ifdef ENABLE_ONNX
    if (!opts.onnxLibPath.empty())
        onnxSetLibraryOverride(opts.onnxLibPath);
#endif

    RunDocument combined;
    combined.meta.clpeakVersion = CLPEAK_VERSION_STR;
    combined.meta.generatedAt   = isoTimestampUtc();
    combined.meta.host          = probeHost();
    combined.meta.invocation    = invocationFrom(opts, argc, mutableArgv.data());
    const auto runStart = std::chrono::steady_clock::now();
    int status = 0;

    for (Backend b : opts.requestedButNotBuilt())
        emitNote(on_event, user_data,
                 std::string("clpeak: the ") + backendInfo(b).name +
                     " backend is not in this build\n");

    for (const auto &be : backendRegistry())
    {
        if (!opts.backendEnabled(be.id) || clpeak::cancelRequested())
            continue;

        auto peak = be.create();
        peak->log.reset(new LoggerFfi(on_event, user_data));
        peak->applyOptions(opts);
        status |= peak->runAll();
        combined.append(peak->log->doc);
    }

    bool cancelled = clpeak::cancelRequested();

    combined.meta.cancelled = cancelled;
    combined.meta.durationS = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - runStart)
                                  .count();

    // Centralized file dump, exactly like the CLI — also runs after a
    // cancellation so partial results get persisted.  The `cancelled` flag is
    // what tells a reader those results are partial: without it, every test
    // the run never reached looks like hardware that lacks the feature.
    if (opts.enableOutput && !saveRunJson(combined, opts.outputFile))
        status |= 1;

    int result = cancelled ? CLPEAK_RUN_CANCELLED : status;

    emitDone(on_event, user_data, result, cancelled);
    g_running.store(false);
    return result;
}
