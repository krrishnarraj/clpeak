// Thin JNI wrapper around backend enumeration.  All real enumeration work
// lives in the backend classes (clPeak / vkPeak); this file just aggregates
// the per-backend inventories and serializes them to JSON for Kotlin.

#include <jni.h>

#include <common/inventory.h>
#include <common/options.h>
#include <opencl/cl_peak.h>
#ifdef ENABLE_VULKAN
#include <vulkan/vk_peak.h>
#endif

static std::vector<BackendInventory> enumerateAllBackends(const CliOptions &opts)
{
  std::vector<BackendInventory> out;
  if (!opts.skipOpenCL)
    out.push_back(clPeak::enumerate());
#ifdef ENABLE_VULKAN
  if (!opts.skipVulkan)
    out.push_back(vkPeak::enumerate());
#endif
  return out;
}

extern "C" JNIEXPORT jstring JNICALL
Java_kr_clpeak_BenchmarkRepository_nativeEnumerateBackends(JNIEnv *env, jobject)
{
  CliOptions opts;
  return env->NewStringUTF(inventoryToJson(enumerateAllBackends(opts)).c_str());
}
