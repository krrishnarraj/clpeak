#include <clpeak.h>
#include <cstring>

#ifdef ENABLE_VULKAN
#include <vk_peak.h>
#endif
#ifdef ENABLE_CUDA
#include <cuda_peak.h>
#endif

int main(int argc, char **argv)
{
  bool skipOpenCL = false;
  bool skipVulkan = false;
  bool skipCuda   = false;
  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--no-opencl") == 0)
      skipOpenCL = true;
    else if (strcmp(argv[i], "--no-vulkan") == 0)
      skipVulkan = true;
    else if (strcmp(argv[i], "--no-cuda") == 0)
      skipCuda = true;
  }

  int clStatus = 0;
  if (!skipOpenCL)
  {
    clPeak clObj;
    clObj.parseArgs(argc, argv);
    clStatus = clObj.runAll();
  }

  int vkStatus = 0;
#ifdef ENABLE_VULKAN
  if (!skipVulkan)
  {
    vkPeak vkObj;
    vkObj.parseArgs(argc, argv);
    vkStatus = vkObj.runAll();
    // Vulkan failure (e.g. no loader or no physical devices) is non-fatal when
    // OpenCL ran successfully. Treat it as a warning.
    if (vkStatus != 0 && !skipOpenCL && clStatus == 0)
      vkStatus = 0;
  }
#else
  (void)skipVulkan;
#endif

  int cuStatus = 0;
#ifdef ENABLE_CUDA
  if (!skipCuda)
  {
    CudaPeak cuObj;
    cuObj.parseArgs(argc, argv);
    cuStatus = cuObj.runAll();
    // Same non-fatal policy as Vulkan: a missing CUDA driver / no devices
    // shouldn't fail the run if another backend already produced numbers.
    if (cuStatus != 0 && ((!skipOpenCL && clStatus == 0) || (!skipVulkan && vkStatus == 0)))
      cuStatus = 0;
  }
#else
  (void)skipCuda;
#endif

  if (clStatus != 0) return clStatus;
  if (vkStatus != 0) return vkStatus;
  return cuStatus;
}
