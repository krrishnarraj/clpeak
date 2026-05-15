#include <peak.h>
#include <opencl/cl_peak.h>
#include <options.h>
#include <inventory.h>
#include <result_store.h>
#include <iostream>

#ifdef ENABLE_VULKAN
#include <vk_peak.h>
#endif
#ifdef ENABLE_CUDA
#include <cuda_peak.h>
#endif
#ifdef ENABLE_METAL
#include <mtl_peak.h>
#endif

static void mergeResults(ResultStore &dst, const ResultStore &src)
{
    dst.insert(dst.end(), src.begin(), src.end());
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
            if (inv.backend == "OpenCL")
                clPeak::printInventory(inv, std::cout);
#ifdef ENABLE_VULKAN
            else if (inv.backend == "Vulkan")
                vkPeak::printInventory(inv, std::cout);
#endif
#ifdef ENABLE_CUDA
            else if (inv.backend == "CUDA")
                CudaPeak::printInventory(inv, std::cout);
#endif
#ifdef ENABLE_METAL
            else if (inv.backend == "Metal")
                MetalPeak::printInventory(inv, std::cout);
#endif
        }
        return 0;
    }

    ResultStore combined;

    int clStatus = 0;
    if (!opts.skipOpenCL)
    {
        clPeak clObj;
        clObj.applyOptions(opts);
        clStatus = clObj.runAll();
        mergeResults(combined, clObj.log->results);
    }

    int vkStatus = 0;
#ifdef ENABLE_VULKAN
    if (!opts.skipVulkan)
    {
        vkPeak vkObj;
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
        cuObj.applyOptions(opts);
        cuStatus = cuObj.runAll();
        mergeResults(combined, cuObj.log->results);
        if (cuStatus != 0 && ((!opts.skipOpenCL && clStatus == 0) ||
                              (!opts.skipVulkan && vkStatus == 0)))
            cuStatus = 0;
    }
#endif

    int mtlStatus = 0;
#ifdef ENABLE_METAL
    if (!opts.skipMetal)
    {
        MetalPeak mtlObj;
        mtlObj.applyOptions(opts);
        mtlStatus = mtlObj.runAll();
        mergeResults(combined, mtlObj.log->results);
        if (mtlStatus != 0 &&
            ((!opts.skipOpenCL && clStatus  == 0) ||
             (!opts.skipVulkan && vkStatus  == 0) ||
             (!opts.skipCuda   && cuStatus  == 0)))
            mtlStatus = 0;
    }
#endif

    // Centralized file dump: one file per enabled format.
    if (opts.enableJson) saveJson(combined, opts.jsonFile);
    if (opts.enableCsv)  saveCsv (combined, opts.csvFile);
    if (opts.enableXml)  saveXml (combined, opts.xmlFile);

    if (clStatus  != 0) return clStatus;
    if (vkStatus  != 0) return vkStatus;
    if (cuStatus  != 0) return cuStatus;
    return mtlStatus;
}
