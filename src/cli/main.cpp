#include <common/peak.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/result_store.h>
#include <cli/logger_cli.h>
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

// Aggregate every backend not skipped in opts.  Lives here (not in common)
// because it needs every backend header.
static std::vector<BackendInventory> enumerateAllBackends(const CliOptions &opts)
{
    std::vector<BackendInventory> out;
#ifdef ENABLE_OPENCL
    if (!opts.skipOpenCL)
        out.push_back(clPeak::enumerate());
#endif
#ifdef ENABLE_VULKAN
    if (!opts.skipVulkan)
        out.push_back(vkPeak::enumerate());
#endif
#ifdef ENABLE_CUDA
    if (!opts.skipCuda)
        out.push_back(CudaPeak::enumerate());
#endif
#ifdef ENABLE_ROCM
    if (!opts.skipRocm)
        out.push_back(RocmPeak::enumerate());
#endif
#ifdef ENABLE_METAL
    if (!opts.skipMetal)
        out.push_back(MetalPeak::enumerate());
#endif
#ifdef ENABLE_ONEAPI
    if (!opts.skipOneapi)
        out.push_back(OneapiPeak::enumerate());
#endif
    return out;
}

int main(int argc, char **argv)
{
    CliOptions opts;
    parseCliOptions(argc, argv, opts);

    // --list-devices: print every backend's inventory.
    if (opts.listDevices)
    {
        auto invs = enumerateAllBackends(opts);
        for (const auto &inv : invs)
        {
#ifdef ENABLE_OPENCL
            if (inv.backend == "OpenCL")
                clPeak::printInventory(inv, std::cout);
#endif
#ifdef ENABLE_VULKAN
            if (inv.backend == "Vulkan")
                vkPeak::printInventory(inv, std::cout);
#endif
#ifdef ENABLE_CUDA
            if (inv.backend == "CUDA")
                CudaPeak::printInventory(inv, std::cout);
#endif
#ifdef ENABLE_ROCM
            if (inv.backend == "ROCm")
                RocmPeak::printInventory(inv, std::cout);
#endif
#ifdef ENABLE_METAL
            if (inv.backend == "Metal")
                MetalPeak::printInventory(inv, std::cout);
#endif
#ifdef ENABLE_ONEAPI
            if (inv.backend == "oneAPI")
                OneapiPeak::printInventory(inv, std::cout);
#endif
        }
        return 0;
    }

    ResultStore combined;

    int clStatus = 0;
#ifdef ENABLE_OPENCL
    if (!opts.skipOpenCL)
    {
        clPeak clObj;
        clObj.log.reset(new LoggerCli(opts.compareFile));
        clObj.applyOptions(opts);
        clStatus = clObj.runAll();
        mergeResults(combined, clObj.log->results);
    }
#endif

    int vkStatus = 0;
#ifdef ENABLE_VULKAN
    if (!opts.skipVulkan)
    {
        vkPeak vkObj;
        vkObj.log.reset(new LoggerCli(opts.compareFile));
        vkObj.applyOptions(opts);
        vkStatus = vkObj.runAll();
        mergeResults(combined, vkObj.log->results);
        if (vkStatus != 0 && !opts.skipOpenCL && clStatus == 0)
            vkStatus = 0;
    }
#endif

    int cuStatus = 0;
#ifdef ENABLE_CUDA
    if (!opts.skipCuda)
    {
        CudaPeak cuObj;
        cuObj.log.reset(new LoggerCli(opts.compareFile));
        cuObj.applyOptions(opts);
        cuStatus = cuObj.runAll();
        mergeResults(combined, cuObj.log->results);
        if (cuStatus != 0 && ((!opts.skipOpenCL && clStatus == 0) ||
                              (!opts.skipVulkan && vkStatus == 0)))
            cuStatus = 0;
    }
#endif

    int rocmStatus = 0;
#ifdef ENABLE_ROCM
    if (!opts.skipRocm)
    {
        RocmPeak rocmObj;
        rocmObj.log.reset(new LoggerCli(opts.compareFile));
        rocmObj.applyOptions(opts);
        rocmStatus = rocmObj.runAll();
        mergeResults(combined, rocmObj.log->results);
        if (rocmStatus != 0 && ((!opts.skipOpenCL && clStatus == 0) ||
                                (!opts.skipVulkan && vkStatus == 0) ||
                                (!opts.skipCuda   && cuStatus == 0)))
            rocmStatus = 0;
    }
#endif

    int mtlStatus = 0;
#ifdef ENABLE_METAL
    if (!opts.skipMetal)
    {
        MetalPeak mtlObj;
        mtlObj.log.reset(new LoggerCli(opts.compareFile));
        mtlObj.applyOptions(opts);
        mtlStatus = mtlObj.runAll();
        mergeResults(combined, mtlObj.log->results);
        if (mtlStatus != 0 &&
            ((!opts.skipOpenCL && clStatus  == 0) ||
             (!opts.skipVulkan && vkStatus  == 0) ||
             (!opts.skipCuda   && cuStatus  == 0) ||
             (!opts.skipRocm   && rocmStatus == 0)))
            mtlStatus = 0;
    }
#endif

    int oneapiStatus = 0;
#ifdef ENABLE_ONEAPI
    if (!opts.skipOneapi)
    {
        OneapiPeak oneObj;
        oneObj.log.reset(new LoggerCli(opts.compareFile));
        oneObj.applyOptions(opts);
        oneapiStatus = oneObj.runAll();
        mergeResults(combined, oneObj.log->results);
        if (oneapiStatus != 0 &&
            ((!opts.skipOpenCL && clStatus   == 0) ||
             (!opts.skipVulkan && vkStatus   == 0) ||
             (!opts.skipCuda   && cuStatus   == 0) ||
             (!opts.skipRocm   && rocmStatus == 0) ||
             (!opts.skipMetal  && mtlStatus  == 0)))
            oneapiStatus = 0;
    }
#endif

    // Centralized file dump: one file per enabled format.
    if (opts.enableJson) saveJson(combined, opts.jsonFile);
    if (opts.enableCsv)  saveCsv (combined, opts.csvFile);
    if (opts.enableXml)  saveXml (combined, opts.xmlFile);

    if (clStatus     != 0) return clStatus;
    if (vkStatus     != 0) return vkStatus;
    if (cuStatus     != 0) return cuStatus;
    if (rocmStatus   != 0) return rocmStatus;
    if (mtlStatus    != 0) return mtlStatus;
    return oneapiStatus;
}
