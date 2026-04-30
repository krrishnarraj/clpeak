#include <clpeak.h>
#include <options.h>

#ifdef ENABLE_VULKAN
#include <vk_peak.h>
#endif
#ifdef ENABLE_CUDA
#include <cuda_peak.h>
#endif
#ifdef ENABLE_METAL
#include <mtl_peak.h>
#endif

int main(int argc, char **argv)
{
  CliOptions opts;
  parseCliOptions(argc, argv, opts);

  int clStatus = 0;
  if (!opts.skipOpenCL)
  {
    clPeak clObj;
    clObj.applyOptions(opts);
    clStatus = clObj.runAll();
  }

  int vkStatus = 0;
#ifdef ENABLE_VULKAN
  if (!opts.skipVulkan)
  {
    vkPeak vkObj;
    vkObj.applyOptions(opts);
    vkStatus = vkObj.runAll();
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
    if (mtlStatus != 0 &&
        ((!opts.skipOpenCL && clStatus  == 0) ||
         (!opts.skipVulkan && vkStatus  == 0) ||
         (!opts.skipCuda   && cuStatus  == 0)))
      mtlStatus = 0;
  }
#endif

  if (clStatus  != 0) return clStatus;
  if (vkStatus  != 0) return vkStatus;
  if (cuStatus  != 0) return cuStatus;
  return mtlStatus;
}
