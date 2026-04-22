#include <clpeak.h>
#include <cstring>

#ifdef ENABLE_VULKAN
#include <vk_peak.h>
#endif

int main(int argc, char **argv)
{
  bool skipOpenCL = false;
  bool skipVulkan = false;
  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--no-opencl") == 0)
      skipOpenCL = true;
    else if (strcmp(argv[i], "--no-vulkan") == 0)
      skipVulkan = true;
  }

  int clStatus = 0;
  if (!skipOpenCL)
  {
    clPeak clObj;
    clObj.parseArgs(argc, argv);
    clStatus = clObj.runAll();
  }

#ifdef ENABLE_VULKAN
  int vkStatus = 0;
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
  return (clStatus != 0) ? clStatus : vkStatus;
#else
  (void)skipVulkan;
  return clStatus;
#endif
}
