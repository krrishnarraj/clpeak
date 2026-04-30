// Non-destructive backend/device enumeration exposed to Kotlin.
// Returns a JSON string consumed by BackendCatalog.kt. Keeping the schema
// flat avoids any third-party JSON dependency on the C++ side.

#include <jni.h>

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

void appendOpenCl(std::ostringstream &out)
{
  cl_uint platformCount = 0;
  cl_int  err           = clGetPlatformIDs(0, nullptr, &platformCount);
  if (err != CL_SUCCESS || platformCount == 0)
  {
    out << "\"opencl\":{\"available\":false,\"platforms\":[]}";
    return;
  }

  std::vector<cl_platform_id> platforms(platformCount);
  err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
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
    clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, 0, nullptr, &nameSize);
    std::vector<char> nameBuf(nameSize);
    if (nameSize)
      clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, nameSize, nameBuf.data(), nullptr);

    out << "{\"index\":" << p
        << ",\"name\":\"" << jsonEscape(trimNul(nameBuf)) << "\""
        << ",\"devices\":[";

    cl_uint deviceCount = 0;
    err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
    if (err == CL_SUCCESS && deviceCount > 0)
    {
      std::vector<cl_device_id> devices(deviceCount);
      clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
      for (cl_uint d = 0; d < deviceCount; ++d)
      {
        if (d) out << ",";
        size_t devNameSize = 0;
        clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 0, nullptr, &devNameSize);
        std::vector<char> devNameBuf(devNameSize);
        if (devNameSize)
          clGetDeviceInfo(devices[d], CL_DEVICE_NAME, devNameSize, devNameBuf.data(), nullptr);
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
