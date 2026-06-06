#include "clpeak_ios_bridge.h"
#include "logger_ios.h"

#include <common/inventory.h>
#include <common/options.h>

#ifdef ENABLE_VULKAN
#include <vulkan/vk_peak.h>
#endif
#ifdef ENABLE_METAL
#include <metal/mtl_peak.h>
#endif
#ifdef ENABLE_CPU
#include <cpu/cpu_peak.h>
#endif

#include <version.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
std::vector<BackendInventory> enumerateAllBackends(const CliOptions &opts)
{
  std::vector<BackendInventory> out;
#ifdef ENABLE_CPU
  if (!opts.skipCpu)
    out.push_back(CpuPeak::enumerate());
#endif
#ifdef ENABLE_METAL
  if (!opts.skipMetal)
    out.push_back(MetalPeak::enumerate());
#endif
#ifdef ENABLE_VULKAN
  if (!opts.skipVulkan)
    out.push_back(vkPeak::enumerate());
#endif
  return out;
}

char *copyString(const std::string &value)
{
  char *out = static_cast<char *>(std::malloc(value.size() + 1));
  if (!out) return nullptr;
  std::memcpy(out, value.c_str(), value.size() + 1);
  return out;
}
}

char *clpeak_ios_copy_backend_catalog_json(void)
{
  CliOptions opts;
  return copyString(inventoryToJson(enumerateAllBackends(opts)));
}

void clpeak_ios_free_string(char *value)
{
  std::free(value);
}

const char *clpeak_ios_version(void)
{
  return CLPEAK_VERSION_STR;
}

int clpeak_ios_launch(int argc,
                      const char **argv,
                      ClpeakIOSCallbacks callbacks,
                      void *context)
{
  std::vector<char *> mutableArgv;
  mutableArgv.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; i++)
    mutableArgv.push_back(const_cast<char *>(argv[i]));

  CliOptions opts;
  parseCliOptions(argc, mutableArgv.data(), opts);

  int status = 0;

#ifdef ENABLE_CPU
  if (!opts.skipCpu)
  {
    CpuPeak cpuObj;
    cpuObj.log.reset(new LoggerIOS(callbacks, context));
    cpuObj.applyOptions(opts);
    status |= cpuObj.runAll();
  }
#endif

#ifdef ENABLE_METAL
  if (!opts.skipMetal)
  {
    MetalPeak mtlObj;
    mtlObj.log.reset(new LoggerIOS(callbacks, context));
    mtlObj.applyOptions(opts);
    status = mtlObj.runAll();
  }
#endif

#ifdef ENABLE_VULKAN
  if (!opts.skipVulkan)
  {
    vkPeak vkObj;
    vkObj.log.reset(new LoggerIOS(callbacks, context));
    vkObj.applyOptions(opts);
    int vkStatus = vkObj.runAll();
    status |= vkStatus;
  }
#endif

  return status;
}
