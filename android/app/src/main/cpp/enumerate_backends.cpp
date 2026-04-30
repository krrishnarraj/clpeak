// Non-destructive backend/device enumeration exposed to Kotlin.
// Returns a JSON string consumed by BackendCatalog.kt. Keeping the schema
// flat avoids any third-party JSON dependency on the C++ side.
//
// IMPORTANT: We deliberately do NOT use the libopencl-stub here. The stub
// caches each wrapper's resolved function pointer in a `static` local on
// first call, but clPeak::runAll() calls stubOpenclReset() (dlclose) at the
// start of every run — invalidating the underlying library while leaving the
// cached function pointers dangling. If we touched the stub from this code
// path before a run, the run would crash inside cl::Platform::get when it
// hit a stale cached pointer.
//
// Instead, this file dlopen()s libOpenCL on its own private handle, dlsym()s
// just the few entry points it needs, and dlclose()s before returning. The
// stub never sees this activity; clpeak.cpp's stubOpenclReset() then loads
// the library fresh on first run, with no stale state.

#include <jni.h>

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <CL/cl.h>

#ifdef ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace {

std::string jsonEscape(const std::string &in)
{
  std::string out;
  out.reserve(in.size() + 2);
  for (char c : in)
  {
    switch (c)
    {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      default:
        if (static_cast<unsigned char>(c) < 0x20)
        {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        }
        else
        {
          out += c;
        }
    }
  }
  return out;
}

std::string trimNul(const std::vector<char> &buf)
{
  size_t n = buf.size();
  while (n > 0 && (buf[n - 1] == '\0' || buf[n - 1] == ' '))
    n--;
  return std::string(buf.data(), n);
}

// Private, throwaway dlopen of libOpenCL — see header comment.
struct OpenClLoader {
  void *handle = nullptr;

  using fn_GetPlatformIDs = cl_int(CL_API_CALL *)(cl_uint, cl_platform_id *, cl_uint *);
  using fn_GetPlatformInfo = cl_int(CL_API_CALL *)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
  using fn_GetDeviceIDs = cl_int(CL_API_CALL *)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
  using fn_GetDeviceInfo = cl_int(CL_API_CALL *)(cl_device_id, cl_device_info, size_t, void *, size_t *);

  fn_GetPlatformIDs  GetPlatformIDs  = nullptr;
  fn_GetPlatformInfo GetPlatformInfo = nullptr;
  fn_GetDeviceIDs    GetDeviceIDs    = nullptr;
  fn_GetDeviceInfo   GetDeviceInfo   = nullptr;

  static const char *kCandidates[];

  bool load()
  {
    for (const char **p = kCandidates; *p; ++p)
    {
      handle = dlopen(*p, RTLD_NOW | RTLD_LOCAL);
      if (handle) break;
    }
    if (!handle) return false;
    GetPlatformIDs  = (fn_GetPlatformIDs ) dlsym(handle, "clGetPlatformIDs");
    GetPlatformInfo = (fn_GetPlatformInfo) dlsym(handle, "clGetPlatformInfo");
    GetDeviceIDs    = (fn_GetDeviceIDs   ) dlsym(handle, "clGetDeviceIDs");
    GetDeviceInfo   = (fn_GetDeviceInfo  ) dlsym(handle, "clGetDeviceInfo");
    return GetPlatformIDs && GetPlatformInfo && GetDeviceIDs && GetDeviceInfo;
  }

  ~OpenClLoader()
  {
    if (handle) dlclose(handle);
  }
};

const char *OpenClLoader::kCandidates[] = {
  "libOpenCL.so",
  "/system/vendor/lib64/libOpenCL.so",
  "/system/lib64/libOpenCL.so",
  "/system/vendor/lib/libOpenCL.so",
  "/system/lib/libOpenCL.so",
  nullptr,
};

void appendOpenCl(std::ostringstream &out)
{
  OpenClLoader cl;
  if (!cl.load())
  {
    out << "\"opencl\":{\"available\":false,\"platforms\":[]}";
    return;
  }

  cl_uint platformCount = 0;
  cl_int  err           = cl.GetPlatformIDs(0, nullptr, &platformCount);
  if (err != CL_SUCCESS || platformCount == 0)
  {
    out << "\"opencl\":{\"available\":false,\"platforms\":[]}";
    return;
  }

  std::vector<cl_platform_id> platforms(platformCount);
  err = cl.GetPlatformIDs(platformCount, platforms.data(), nullptr);
  if (err != CL_SUCCESS)
  {
    out << "\"opencl\":{\"available\":false,\"platforms\":[]}";
    return;
  }

  out << "\"opencl\":{\"available\":true,\"platforms\":[";
  for (cl_uint p = 0; p < platformCount; ++p)
  {
    if (p) out << ",";
    size_t nameSize = 0;
    cl.GetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 0, nullptr, &nameSize);
    std::vector<char> nameBuf(nameSize);
    if (nameSize)
      cl.GetPlatformInfo(platforms[p], CL_PLATFORM_NAME, nameSize, nameBuf.data(), nullptr);

    out << "{\"index\":" << p
        << ",\"name\":\"" << jsonEscape(trimNul(nameBuf)) << "\""
        << ",\"devices\":[";

    cl_uint deviceCount = 0;
    err = cl.GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
    if (err == CL_SUCCESS && deviceCount > 0)
    {
      std::vector<cl_device_id> devices(deviceCount);
      cl.GetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
      for (cl_uint d = 0; d < deviceCount; ++d)
      {
        if (d) out << ",";
        size_t devNameSize = 0;
        cl.GetDeviceInfo(devices[d], CL_DEVICE_NAME, 0, nullptr, &devNameSize);
        std::vector<char> devNameBuf(devNameSize);
        if (devNameSize)
          cl.GetDeviceInfo(devices[d], CL_DEVICE_NAME, devNameSize, devNameBuf.data(), nullptr);
        out << "{\"index\":" << d
            << ",\"name\":\"" << jsonEscape(trimNul(devNameBuf)) << "\"}";
      }
    }
    out << "]}";
  }
  out << "]}";
}

#ifdef ENABLE_VULKAN
const char *vkDeviceTypeName(VkPhysicalDeviceType t)
{
  switch (t)
  {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
    default:                                     return "other";
  }
}
#endif

void appendVulkan(std::ostringstream &out)
{
#ifdef ENABLE_VULKAN
  VkApplicationInfo appInfo{};
  appInfo.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "clpeak-enumerate";
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo ci{};
  ci.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &appInfo;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
  {
    out << "\"vulkan\":{\"available\":false,\"devices\":[]}";
    return;
  }

  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devs(count);
  if (count) vkEnumeratePhysicalDevices(instance, &count, devs.data());

  out << "\"vulkan\":{\"available\":true,\"devices\":[";
  for (uint32_t i = 0; i < count; ++i)
  {
    if (i) out << ",";
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(devs[i], &props);
    out << "{\"index\":" << i
        << ",\"name\":\"" << jsonEscape(props.deviceName) << "\""
        << ",\"type\":\""  << vkDeviceTypeName(props.deviceType) << "\"}";
  }
  out << "]}";
  vkDestroyInstance(instance, nullptr);
#else
  out << "\"vulkan\":{\"available\":false,\"devices\":[]}";
#endif
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_kr_clpeak_BenchmarkRepository_nativeEnumerateBackends(JNIEnv *env, jobject)
{
  std::ostringstream out;
  out << "{";
  appendOpenCl(out);
  out << ",";
  appendVulkan(out);
  out << "}";
  return env->NewStringUTF(out.str().c_str());
}
