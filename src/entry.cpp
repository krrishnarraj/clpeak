#include <clpeak.h>
#include <cstring>

#ifdef ENABLE_VULKAN
#include <vk_peak.h>
#endif

int main(int argc, char **argv)
{
#ifdef ENABLE_VULKAN
  // Check if --vulkan flag is present
  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--vulkan") == 0)
    {
      vkPeak vkObj;
      vkObj.parseArgs(argc, argv);
      return vkObj.runAll();
    }
  }
#endif

  clPeak clObj;
  clObj.parseArgs(argc, argv);
  return clObj.runAll();
}
