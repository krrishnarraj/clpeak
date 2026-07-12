#include "clpeak_ffi.h"
#include "logger_ffi.h"

#include <common/common.h>
#include <common/inventory.h>
#include <common/options.h>
#include <common/peak.h>
#include <common/result_store.h>
#include <version.h>

#ifdef ENABLE_OPENCL
#include <opencl/cl_peak.h>
#endif
#ifdef ENABLE_VULKAN
#include <vulkan/vk_peak.h>
#endif
#ifdef ENABLE_CUDA
#include <cuda/cuda_peak.h>
#endif
#ifdef ENABLE_ROCM
#include <rocm/rocm_peak.h>
#endif
#ifdef ENABLE_METAL
#include <metal/mtl_peak.h>
#endif
#ifdef ENABLE_ONEAPI
#include <oneapi/oneapi_peak.h>
#endif
#ifdef ENABLE_CPU
#include <cpu/cpu_peak.h>
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

// Same registration order as the CLI (src/cli/main.cpp).
struct BackendEntry
{
    const char *name;
    std::function<BackendInventory()> enumerate;
    std::function<std::unique_ptr<Peak>()> create;
    bool CliOptions::*skip;
};

std::vector<BackendEntry> buildBackends()
{
    std::vector<BackendEntry> out;
#ifdef ENABLE_CUDA
    out.push_back({"CUDA",
                   [] { return CudaPeak::enumerate(); },
                   [] { return std::unique_ptr<Peak>(new CudaPeak()); },
                   &CliOptions::skipCuda});
#endif
#ifdef ENABLE_ROCM
    out.push_back({"ROCm",
                   [] { return RocmPeak::enumerate(); },
                   [] { return std::unique_ptr<Peak>(new RocmPeak()); },
                   &CliOptions::skipRocm});
#endif
#ifdef ENABLE_METAL
    out.push_back({"Metal",
                   [] { return MetalPeak::enumerate(); },
                   [] { return std::unique_ptr<Peak>(new MetalPeak()); },
                   &CliOptions::skipMetal});
#endif
#ifdef ENABLE_ONEAPI
    out.push_back({"oneAPI",
                   [] { return OneapiPeak::enumerate(); },
                   [] { return std::unique_ptr<Peak>(new OneapiPeak()); },
                   &CliOptions::skipOneapi});
#endif
#ifdef ENABLE_VULKAN
    out.push_back({"Vulkan",
                   [] { return vkPeak::enumerate(); },
                   [] { return std::unique_ptr<Peak>(new vkPeak()); },
                   &CliOptions::skipVulkan});
#endif
#ifdef ENABLE_OPENCL
    out.push_back({"OpenCL",
                   [] { return clPeak::enumerate(); },
                   [] { return std::unique_ptr<Peak>(new clPeak()); },
                   &CliOptions::skipOpenCL});
#endif
#ifdef ENABLE_CPU
    out.push_back({"CPU",
                   [] { return CpuPeak::enumerate(); },
                   [] { return std::unique_ptr<Peak>(new CpuPeak()); },
                   &CliOptions::skipCpu});
#endif
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
    for (const auto &be : buildBackends())
        inv.push_back(be.enumerate());
    return copyString(inventoryToJson(inv));
}

void clpeak_free_string(char *s)
{
    std::free(s);
}

void clpeak_request_cancel(void)
{
    clpeak::requestCancel();
}

char *clpeak_load_result_file_json(const char *path)
{
    if (!path)
        return nullptr;
    ResultStore store = loadResultFile(path);
    if (store.empty())
        return nullptr;
    return copyString(resultsToJson(store));
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

    ResultStore combined;
    int status = 0;

    for (const auto &be : buildBackends())
    {
        if (opts.*(be.skip) || clpeak::cancelRequested())
            continue;

        auto peak = be.create();
        peak->log.reset(new LoggerFfi(on_event, user_data));
        peak->applyOptions(opts);
        status |= peak->runAll();
        combined.insert(combined.end(),
                        peak->log->results.begin(), peak->log->results.end());
    }

    // Centralized file dump, exactly like the CLI — also runs after a
    // cancellation so partial results get persisted.
    if (opts.enableJson && !saveJson(combined, opts.jsonFile))
        status |= 1;
    if (opts.enableCsv && !saveCsv(combined, opts.csvFile))
        status |= 1;
    if (opts.enableXml && !saveXml(combined, opts.xmlFile))
        status |= 1;

    bool cancelled = clpeak::cancelRequested();
    int result = cancelled ? CLPEAK_RUN_CANCELLED : status;

    emitDone(on_event, user_data, result, cancelled);
    g_running.store(false);
    return result;
}
