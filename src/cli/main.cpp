#include <common/peak.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/result_store.h>
#include <cli/logger_cli.h>
#include <functional>
#include <iostream>

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

static void mergeResults(ResultStore &dst, const ResultStore &src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

// A thin wrapper that captures everything we need per backend so the rest of
// main() can iterate instead of repeating #ifdef-guarded blocks.
struct BackendEntry
{
    const char *name;
    std::function<BackendInventory()> enumerate;
    std::function<void(const BackendInventory &, std::ostream &)> printInv;
    std::function<std::unique_ptr<Peak>()> create;
    bool CliOptions::*skip;
};

// Build the backend list once.  Each enabled backend registers its static
// enumerate / printInventory / factory lambdas here so that main() only
// has simple loops.
static std::vector<BackendEntry> buildBackends()
{
    std::vector<BackendEntry> out;
#ifdef ENABLE_CUDA
    out.push_back({
        "CUDA",
        []{ return CudaPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os){ CudaPeak::printInventory(inv, os); },
        []{ return std::make_unique<CudaPeak>(); },
        &CliOptions::skipCuda,
    });
#endif
#ifdef ENABLE_ROCM
    out.push_back({
        "ROCm",
        []{ return RocmPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os){ RocmPeak::printInventory(inv, os); },
        []{ return std::make_unique<RocmPeak>(); },
        &CliOptions::skipRocm,
    });
#endif
#ifdef ENABLE_METAL
    out.push_back({
        "Metal",
        []{ return MetalPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os){ MetalPeak::printInventory(inv, os); },
        []{ return std::make_unique<MetalPeak>(); },
        &CliOptions::skipMetal,
    });
#endif
#ifdef ENABLE_ONEAPI
    out.push_back({
        "oneAPI",
        []{ return OneapiPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os){ OneapiPeak::printInventory(inv, os); },
        []{ return std::make_unique<OneapiPeak>(); },
        &CliOptions::skipOneapi,
    });
#endif
#ifdef ENABLE_VULKAN
    out.push_back({
        "Vulkan",
        []{ return vkPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os){ vkPeak::printInventory(inv, os); },
        []{ return std::make_unique<vkPeak>(); },
        &CliOptions::skipVulkan,
    });
#endif
#ifdef ENABLE_OPENCL
    out.push_back({
        "OpenCL",
        []{ return clPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os){ clPeak::printInventory(inv, os); },
        []{ return std::make_unique<clPeak>(); },
        &CliOptions::skipOpenCL,
    });
#endif
    return out;
}

static std::vector<BackendInventory> enumerateAllBackends(
    const CliOptions &opts, const std::vector<BackendEntry> &backends)
{
    std::vector<BackendInventory> out;
    for (const auto &be : backends)
        if (!(opts.*(be.skip)))
            out.push_back(be.enumerate());
    return out;
}

int main(int argc, char **argv)
{
    CliOptions opts;
    parseCliOptions(argc, argv, opts);

    auto backends = buildBackends();

    // --list-devices: print every backend's inventory.
    if (opts.listDevices)
    {
        auto invs = enumerateAllBackends(opts, backends);
        for (const auto &inv : invs)
            for (const auto &be : backends)
                if (inv.backend == be.name)
                {
                    be.printInv(inv, std::cout);
                    break;
                }
        return 0;
    }

    ResultStore combined;

    // Run every enabled backend in order.  If a backend fails but at least
    // one preceding backend succeeded, we suppress the error so that a
    // single broken device doesn't mask results from healthy ones.
    bool anyPrecedingSucceeded = false;
    int  lastError = 0;

    for (const auto &be : backends)
    {
        if (opts.*(be.skip)) continue;

        auto peak = be.create();
        peak->log.reset(new LoggerCli(opts.compareFile));
        peak->applyOptions(opts);
        int status = peak->runAll();
        mergeResults(combined, peak->log->results);

        if (status != 0 && anyPrecedingSucceeded)
            status = 0;
        if (status == 0)
            anyPrecedingSucceeded = true;
        if (status != 0)
            lastError = status;
    }

    // Centralized file dump: one file per enabled format.
    if (opts.enableJson) saveJson(combined, opts.jsonFile);
    if (opts.enableCsv)  saveCsv (combined, opts.csvFile);
    if (opts.enableXml)  saveXml (combined, opts.xmlFile);

    return lastError;
}
