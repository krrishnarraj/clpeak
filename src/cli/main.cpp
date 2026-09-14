#include <common/peak.h>
#include <common/common.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/run_document.h>
#include <common/logger_text.h>
#include <common/host_info.h>
#include <version.h>
#include <chrono>
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
#ifdef ENABLE_CPU
#include <cpu/cpu_peak.h>
#endif
#ifdef ENABLE_ONNX
#include <onnx/onnx_peak.h>
#endif
#ifdef ENABLE_COREML
#include <coreml/coreml_peak.h>
#endif

// A thin wrapper that captures everything we need per backend so the rest of
// main() can iterate instead of repeating #ifdef-guarded blocks.  Name and
// flag come from the backend table in options.cpp.
struct BackendEntry
{
    Backend id;
    std::function<BackendInventory()> enumerate;
    std::function<void(const BackendInventory &, std::ostream &)> printInv;
    std::function<std::unique_ptr<Peak>()> create;
};

// Build the backend list once.  Each enabled backend registers its static
// enumerate / printInventory / factory lambdas here so that main() only
// has simple loops.
static std::vector<BackendEntry> buildBackends()
{
    std::vector<BackendEntry> out;
#ifdef ENABLE_CUDA
    out.push_back({
        Backend::Cuda,
        []
        { return CudaPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { CudaPeak::printInventory(inv, os); },
        []
        { return std::make_unique<CudaPeak>(); },
    });
#endif
#ifdef ENABLE_ROCM
    out.push_back({
        Backend::Rocm,
        []
        { return RocmPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { RocmPeak::printInventory(inv, os); },
        []
        { return std::make_unique<RocmPeak>(); },
    });
#endif
#ifdef ENABLE_METAL
    out.push_back({
        Backend::Metal,
        []
        { return MetalPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { MetalPeak::printInventory(inv, os); },
        []
        { return std::make_unique<MetalPeak>(); },
    });
#endif
#ifdef ENABLE_ONEAPI
    out.push_back({
        Backend::Oneapi,
        []
        { return OneapiPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { OneapiPeak::printInventory(inv, os); },
        []
        { return std::make_unique<OneapiPeak>(); },
    });
#endif
#ifdef ENABLE_VULKAN
    out.push_back({
        Backend::Vulkan,
        []
        { return vkPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { vkPeak::printInventory(inv, os); },
        []
        { return std::make_unique<vkPeak>(); },
    });
#endif
#ifdef ENABLE_OPENCL
    out.push_back({
        Backend::OpenCL,
        []
        { return clPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { clPeak::printInventory(inv, os); },
        []
        { return std::make_unique<clPeak>(); },
    });
#endif
#ifdef ENABLE_CPU
    out.push_back({
        Backend::Cpu,
        []
        { return CpuPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { CpuPeak::printInventory(inv, os); },
        []
        { return std::make_unique<CpuPeak>(); },
    });
#endif
#ifdef ENABLE_ONNX
    out.push_back({
        Backend::Onnx,
        []
        { return OnnxPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { OnnxPeak::printInventory(inv, os); },
        []
        { return std::make_unique<OnnxPeak>(); },
    });
#endif
#ifdef ENABLE_COREML
    out.push_back({
        Backend::Coreml,
        []
        { return CoreMLPeak::enumerate(); },
        [](const BackendInventory &inv, std::ostream &os)
        { CoreMLPeak::printInventory(inv, os); },
        []
        { return std::make_unique<CoreMLPeak>(); },
    });
#endif
    return out;
}

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

    auto backends = buildBackends();

    // A backend asked for by name that this binary does not carry: say so,
    // or the run reads as "no devices".
    for (Backend b : opts.requestedButNotBuilt())
        std::cout << "clpeak: the " << backendInfo(b).name
                  << " backend is not in this build\n";

    // --list-devices: print every enabled backend's inventory.
    if (opts.listDevices)
    {
        for (const auto &be : backends)
            if (opts.backendEnabled(be.id))
                be.printInv(be.enumerate(), std::cout);
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
