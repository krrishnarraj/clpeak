#include "xpupeak_ffi.h"
#include "launch.h"
#include "logger_ffi.h"

#include <common/backend_registry.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/json_writer.h>
#include <common/coreml_cache.h>
#include <common/options.h>
#include <common/peak.h>
#include <common/host_info.h>
#include <common/keep_awake.h>
#include <common/screen_hold.h>
#include <common/thermal.h>
#include <common/run_checkpoint.h>
#include <common/run_document.h>
#include <common/run_log.h>
#include <common/run_resume.h>
#include <common/console_mute.h>
#include <common/hardware_link.h>
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
void emitLogEntry(XpuPeakEventCallback cb, void *userData, const LogEntry &entry)
{
    LogEvent e;
    e.kind        = LogEvent::Kind::Log;
    e.backend     = entry.backend;
    e.device      = entry.device;
    e.deviceIndex = entry.deviceIndex;
    e.log         = entry;
    ffiEmitJson(cb, userData, ffiEventToJson(e));
}

// `durationS` is the saved document's `duration_s`, negative when there is
// none: for a resumed run only the native side knows the parts' time.
void emitDone(XpuPeakEventCallback cb, void *userData, int status, bool cancelled,
              double durationS = -1.0)
{
    std::string json = "{\"t\":\"done\",\"status\":" + std::to_string(status) +
                       ",\"cancelled\":" + (cancelled ? "true" : "false");
    if (durationS >= 0.0) json += ",\"duration_s\":" + JsonWriter::fmtNum(durationS);
    ffiEmitJson(cb, userData, json + "}");
}

// One launch at a time — the run loop and the cancel flag are process-global.
std::atomic<bool> g_running{false};

} // namespace

const char *xpupeak_version(void)
{
    return XPUPEAK_VERSION_STR;
}

char *xpupeak_copy_backend_catalog_json(void)
{
    std::vector<BackendInventory> inv;
    for (const auto &be : backendRegistry())
    {
        xpupeak::ScopedConsoleMute mute(
            xpupeak::ScopedConsoleMute::Capture::Verbose, {},
            /*stderrOnly=*/true);
        inv.push_back(be.enumerate());
    }
    return copyString(inventoryToJson(inv));
}

void xpupeak_free_string(char *s)
{
    std::free(s);
}

char *xpupeak_copy_onnx_status_json(void)
{
#ifdef ENABLE_ONNX
    const OnnxRuntimeStatus st = onnxRuntimeStatus();
    std::string json = "{\"available\":";
    json += st.available ? "true" : "false";
    json += ",\"linkedIn\":";
    json += st.linkedIn ? "true" : "false";
    json += ",\"version\":\"" + jsonEscape(st.version) + "\"";
    json += ",\"path\":\"" + jsonEscape(st.path) + "\"";
    json += ",\"error\":\"" + jsonEscape(st.error) + "\"";
    json += ",\"epLibraries\":[";
    for (size_t i = 0; i < st.epLibraries.size(); i++)
    {
        const OnnxEpLibraryStatus &e = st.epLibraries[i];
        if (i) json += ",";
        json += "{\"name\":\"" + jsonEscape(e.lib.name) + "\"";
        json += ",\"path\":\"" + jsonEscape(e.lib.path) + "\"";
        json += ",\"named\":";
        json += e.lib.named ? "true" : "false";
        json += ",\"registered\":";
        json += e.registered ? "true" : "false";
        json += ",\"error\":\"" + jsonEscape(e.error) + "\"";
        json += ",\"replaces\":[";
        for (size_t k = 0; k < e.replaces.size(); k++)
            json += (k ? ",\"" : "\"") + jsonEscape(e.replaces[k]) + "\"";
        json += "]}";
    }
    json += "]";
    json += ",\"winml\":{\"enabled\":";
    json += st.winmlEnabled ? "true" : "false";
    json += ",\"path\":\"" + jsonEscape(st.winmlPath) + "\"";
    json += ",\"error\":\"" + jsonEscape(st.winmlError) + "\"}";
    if (st.pending)
    {
        json += ",\"pendingRuntime\":{\"path\":\"" + jsonEscape(st.pendingLibrary) + "\"";
        json += ",\"winml\":{\"enabled\":";
        json += st.pendingWinml ? "true" : "false";
        json += ",\"path\":\"" + jsonEscape(st.pendingWinmlPath) + "\"}}";
    }
    json += "}";
    return copyString(json);
#else
    return copyString(
        "{\"available\":false,\"linkedIn\":false,\"version\":\"\","
        "\"path\":\"\",\"error\":\"ONNX backend not built in\","
        "\"epLibraries\":[],\"winml\":{\"enabled\":false,\"path\":\"\","
        "\"error\":\"\"}}");
#endif
}

void xpupeak_set_onnx_library(const char *path)
{
#ifdef ENABLE_ONNX
    onnxSetLibraryOverride(path ? path : "");
#else
    (void)path;
#endif
}

void xpupeak_set_onnx_ep_libraries(const char *spec)
{
#ifdef ENABLE_ONNX
    std::vector<OnnxEpLibrary> libs;
    std::string text = spec ? spec : "";
    size_t pos = 0;
    while (pos <= text.size())
    {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(pos, nl - pos);
        pos = nl + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        OnnxEpLibrary lib;
        if (line[0] == '!')
        {
            lib.named = false;
            line.erase(0, 1);
        }
        const size_t eq = line.find('=');
        // A malformed line is dropped rather than registered under a wrong
        // name; the GUI validates before it gets here.
        if (eq == std::string::npos || eq == 0 || eq + 1 == line.size())
            continue;
        lib.name = line.substr(0, eq);
        lib.path = line.substr(eq + 1);
        libs.push_back(std::move(lib));
    }
    onnxSetEpLibraries(std::move(libs));
#else
    (void)spec;
#endif
}

void xpupeak_set_onnx_winml(int enabled, const char *path)
{
#ifdef ENABLE_ONNX
    onnxSetWinml(enabled != 0, path ? path : "");
#else
    (void)enabled;
    (void)path;
#endif
}

char *xpupeak_copy_litert_status_json(void)
{
#ifdef ENABLE_LITERT
    const LitertRuntimeStatus st = litertRuntimeStatus();
    std::string json = "{\"available\":";
    json += st.available ? "true" : "false";
    json += ",\"version\":\"" + jsonEscape(st.version) + "\"";
    json += ",\"path\":\"" + jsonEscape(st.path) + "\"";
    json += ",\"error\":\"" + jsonEscape(st.error) + "\"";
    if (st.pending)
        json += ",\"pendingRuntime\":{\"path\":\"" + jsonEscape(st.pendingPath) + "\"}";
    json += "}";
    return copyString(json);
#else
    return copyString(
        "{\"available\":false,\"version\":\"\",\"path\":\"\","
        "\"error\":\"LiteRT backend not built in\"}");
#endif
}

void xpupeak_set_litert_library(const char *path)
{
#ifdef ENABLE_LITERT
    litertSetLibraryOverride(path ? path : "");
#else
    (void)path;
#endif
}

void xpupeak_set_litert_npu_dir(const char *dir)
{
#ifdef ENABLE_LITERT
    litertSetNpuDirOverride(dir ? dir : "");
#else
    (void)dir;
#endif
}

void xpupeak_set_litert_npu_stage_dir(const char *dir)
{
#ifdef ENABLE_LITERT
    litertSetNpuStageDir(dir ? dir : "");
#else
    (void)dir;
#endif
}

void xpupeak_request_cancel(void)
{
    xpupeak::requestCancel();
}

void xpupeak_set_on_screen(int on_screen)
{
    xpupeak::setHostOnScreen(on_screen != 0);
}

void xpupeak_set_paused(int paused)
{
    xpupeak::setRunPaused(paused != 0);
}

int xpupeak_launch(int argc, const char **argv,
                   XpuPeakEventCallback on_event, void *user_data)
{
    return xpuPeakLaunch(argc, argv, on_event, user_data, /*resetControls=*/true);
}

int xpuPeakLaunch(int argc, const char **argv, XpuPeakEventCallback on_event,
                  void *user_data, bool resetControls)
{
    bool expected = false;
    if (!g_running.compare_exchange_strong(expected, true))
        return XPUPEAK_RUN_BUSY;  // no done event: the in-flight run owns the stream

    if (resetControls)
    {
        xpupeak::resetCancel();
        xpupeak::setRunPaused(false);
    }

    std::vector<char *> mutableArgv;
    mutableArgv.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; i++)
        mutableArgv.push_back(const_cast<char *>(argv[i]));

    // The run's diagnostic stream, live before the arguments are even parsed
    // so a rejected argv is on it too.  Entries a backend's logger records
    // reach the GUI through that logger; the rest through this fallback, as
    // the same `log` event.
    RunDocument   combined;
    RunLog        runLog(combined);
    RunCheckpoint checkpoint(combined);
    runLog.setFallbackRenderer([&](const LogEntry &e) {
        emitLogEntry(on_event, user_data, e);
    });

    CliOptions opts;
    std::string parseError;
    if (!parseCliOptionsNoExit(argc, mutableArgv.data(), opts, parseError))
    {
        xpupeak::logMessage(xpupeak::LogLevel::Error, "", parseError);
        emitDone(on_event, user_data, XPUPEAK_RUN_BAD_ARGS, false);
        g_running.store(false);
        return XPUPEAK_RUN_BAD_ARGS;
    }

    // --resume: the interrupted run's own options, from its checkpoint, less
    // what it finished (include/common/run_resume.h).
    ResumePlan resume;
    const bool resuming = !opts.resumeFile.empty();
    if (resuming)
    {
        CliOptions resumed;
        std::string error;
        if (!loadResumePlan(opts.resumeFile, resume, error) ||
            !resumeOptions(resume, opts, resumed, error))
        {
            xpupeak::logMessage(xpupeak::LogLevel::Error, "", error);
            emitDone(on_event, user_data, XPUPEAK_RUN_BAD_ARGS, false);
            g_running.store(false);
            return XPUPEAK_RUN_BAD_ARGS;
        }
        opts = std::move(resumed);
    }

    xpupeak::setVerbose(opts.verbose);
#ifdef ENABLE_ONNX
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

    if (resuming)
    {
        // The parts before: their meta, devices, log and progress.
        seedResume(resume, opts, combined, runLog, checkpoint);
    }
    else
    {
        combined.meta.generator     = {RESULT_GENERATOR_NAME, XPUPEAK_VERSION_STR};
        combined.meta.generatedAt   = isoTimestampUtc();
        combined.meta.runId         = newRunId();
        combined.meta.evidenceKey   = newEvidenceKey();
        for (const auto &be : backendRegistry())
            combined.meta.build.backends.push_back(buildBackendFor(be.id));
#ifdef XPUPEAK_BUILD_CONFIG
        combined.meta.build.config = XPUPEAK_BUILD_CONFIG;
#endif
        combined.meta.system        = probeSystem();
        combined.meta.invocation    = invocationFrom(opts, argc, mutableArgv.data());
    }
    // The sidecar: every entry on disk as it happens, for the native crash
    // the document never gets written after.  The app adopts one left behind
    // on its next launch (app/lib/src/services/run_history_store.dart).  And
    // the checkpoint of what finished, which a run left behind continues
    // from (include/common/run_checkpoint.h).
    if (opts.enableOutput)
    {
        runLog.openSidecar(opts.outputFile, /*append=*/resuming && resume.sidecarFound);
        checkpoint.open(opts.outputFile);
    }
    int status = 0;

    for (Backend b : opts.requestedButNotBuilt())
        XPUPEAK_LOG(Warning, "xpupeak: the %s backend is not in this build",
                    backendInfo(b).name);

    // --verbose: the catalog as the run saw it, in the file.  Already
    // enumerated once for the run screen, so the memoised probes make this
    // cheap here, unlike in the CLI.
    if (opts.verbose && opts.enableOutput)
        for (const auto &be : backendRegistry())
            if (opts.backendEnabled(be.id))
            {
                xpupeak::ScopedConsoleMute mute(
                    xpupeak::ScopedConsoleMute::Capture::Verbose, {},
                    /*stderrOnly=*/true);
                combined.inventory.push_back(be.enumerate());
            }

    // Held until the launch returns, as in the CLI.  The desktop app holds
    // nothing else, and a window being open does not count as activity to
    // any OS idle timer; on Android and iOS it is a no-op and the app keeps
    // the screen on itself (include/common/keep_awake.h).
    xpupeak::KeepAwake keepAwake;
    // And paused while the OS says the device is too hot (thermal.h) --
    // on a phone, where it matters.
    xpupeak::ThermalWatch thermalWatch;

    for (const auto &be : backendRegistry())
    {
        if (!opts.backendEnabled(be.id) || xpupeak::cancelRequested())
            continue;
        // Finished by the part of the run this one resumes.
        if (opts.resume.backendsDone.test(static_cast<size_t>(be.id)))
            continue;

        auto peak = be.create();
        // --verbose with -o: the base logger mirrors the metric rows onto the
        // document's log beside the backend/device/test headers it always
        // records, so a GUI-saved file reads as the run looked live.  The
        // sidecar gets them either way.
        const bool mirror = opts.verbose && opts.enableOutput;
        peak->log.reset(new LoggerFfi(on_event, user_data, mirror));
        peak->applyOptions(opts);
        status |= peak->runAll();
        combined.append(peak->log->doc);
        // Backstop for Core ML's compile cache, as in the CLI (see
        // include/common/coreml_cache.h); the app's own cache directory on
        // iOS and macOS is where it lands.
        xpupeak::purgeCoreMLCompileCache();
    }

    // Every device onto the processor it is a view of, before the file is
    // written, as in the CLI (include/common/hardware_link.h).
    xpupeak::linkHardware(combined);

    bool cancelled = xpupeak::cancelRequested();

    combined.meta.cancelled = cancelled;
    combined.meta.durationS = checkpoint.runningS();
    if (resuming)
        combined.meta.resumed.push_back(checkpoint.partRecord(combined));
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
        // A cancelled run keeps its checkpoint, so it can still be continued
        // (--resume): what it saved is its result until then.
        checkpoint.close(/*remove=*/saved && !cancelled);
        if (!saved)
            status |= 1;
    }

    int result = cancelled ? XPUPEAK_RUN_CANCELLED : status;

    emitDone(on_event, user_data, result, cancelled, combined.meta.durationS);
    g_running.store(false);
    return result;
}
