#include <clpeak.h>
#include <inventory.h>
#include <options.h>
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

// Append `src` to the end of `dst` in arrival order.  Used to merge each
// backend's accumulated metrics into a single store before serialization,
// so a `--json-file out.json` run that hits multiple backends produces one
// file containing every row (rather than the last backend's rows
// overwriting the previous one).
static void mergeResults(ResultStore &dst, const ResultStore &src)
{
  dst.insert(dst.end(), src.begin(), src.end());
}

int main(int argc, char **argv)
{
  CliOptions opts;
  parseCliOptions(argc, argv, opts);

  // --list-devices: print every backend's inventory via the shared enumerator.
  if (opts.listDevices)
  {
    printInventory(enumerateAllBackends(opts), std::cout);
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
    // Vulkan failure (e.g. no loader or no physical devices) is non-fatal when
    // OpenCL ran successfully. Treat it as a warning.
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

  // Centralised dump: write one file per enabled format, containing rows
  // from every backend that ran.  Per-backend loggers no longer write
  // files (see clpeak.cpp::applyOptions).
  if (opts.enableJson) saveJson(combined, opts.jsonFile);
  if (opts.enableCsv)  saveCsv (combined, opts.csvFile);
  if (opts.enableXml)  saveXml (combined, opts.xmlFile);

  if (clStatus  != 0) return clStatus;
  if (vkStatus  != 0) return vkStatus;
  if (cuStatus  != 0) return cuStatus;
  return mtlStatus;
}
