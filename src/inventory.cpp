#include <inventory.h>
#include <options.h>

#include <ostream>
#include <sstream>
#include <string>

namespace
{

  std::string jsonEscape(const std::string &in)
  {
    std::string out;
    out.reserve(in.size() + 2);
    for (char c : in)
    {
      switch (c)
      {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
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

  void appendDeviceJson(std::ostream &os, const InventoryDevice &d)
  {
    os << "{\"index\":" << d.index
       << ",\"name\":\"" << jsonEscape(d.name) << "\""
       << ",\"type\":\"" << jsonEscape(d.typeStr) << "\"";
    if (!d.driverVersion.empty())
      os << ",\"driver\":\"" << jsonEscape(d.driverVersion) << "\"";
    if (!d.apiVersion.empty())
      os << ",\"api\":\"" << jsonEscape(d.apiVersion) << "\"";
    if (d.numComputeUnits)
      os << ",\"computeUnits\":" << d.numComputeUnits;
    if (d.maxClockMHz)
      os << ",\"clockMHz\":" << d.maxClockMHz;
    if (d.globalMemBytes)
      os << ",\"globalMemBytes\":" << d.globalMemBytes;
    if (d.maxAllocBytes)
      os << ",\"maxAllocBytes\":" << d.maxAllocBytes;
    if (d.hasFp16)
      os << ",\"fp16\":true";
    if (d.hasFp64)
      os << ",\"fp64\":true";
    os << "}";
  }

  void printOpenCLBackend(const BackendInventory &b, std::ostream &os)
  {
    os << "\n=== OpenCL backend ===\n";
    for (const auto &plat : b.platforms)
    {
      os << "Platform " << plat.index << ": " << plat.name << "\n";
      for (const auto &d : plat.devices)
      {
        os << "  Device " << d.index << ": " << d.name;
        if (!d.typeStr.empty())
          os << " [" << d.typeStr << "]";
        os << "\n";
        if (!d.driverVersion.empty())
          os << "    Driver    : " << d.driverVersion << "\n";
        if (d.numComputeUnits)
          os << "    CUs       : " << d.numComputeUnits << "\n";
        if (d.maxClockMHz)
          os << "    Clock     : " << d.maxClockMHz << " MHz\n";
        if (d.globalMemBytes)
          os << "    Global mem: " << (d.globalMemBytes / (1024 * 1024)) << " MB\n";
        if (d.maxAllocBytes)
          os << "    Max alloc : " << (d.maxAllocBytes / (1024 * 1024)) << " MB\n";
        os << "    FP16      : " << (d.hasFp16 ? "yes" : "no") << "\n";
        os << "    FP64      : " << (d.hasFp64 ? "yes" : "no") << "\n";
      }
    }
  }

  void printVulkanBackend(const BackendInventory &b, std::ostream &os)
  {
    os << "\n=== Vulkan backend ===\n";
    if (!b.available)
    {
      os << "Vulkan: failed to create instance or no devices found\n";
      return;
    }
    for (const auto &plat : b.platforms)
    {
      for (const auto &d : plat.devices)
      {
        os << "  Vulkan Device " << d.index << ": " << d.name;
        if (!d.typeStr.empty())
          os << " [" << d.typeStr << "]";
        os << "\n";
        if (!d.apiVersion.empty())
          os << "    API       : " << d.apiVersion << "\n";
      }
    }
  }

  void printCudaBackend(const BackendInventory &b, std::ostream &os)
  {
    os << "\n=== CUDA backend ===\n";
    if (!b.available)
    {
      os << "CUDA: driver init failed or no devices found\n";
      return;
    }
    for (const auto &plat : b.platforms)
    {
      for (const auto &d : plat.devices)
      {
        os << "  CUDA Device " << d.index << ": " << d.name;
        if (!d.typeStr.empty())
          os << " [" << d.typeStr << "]";
        os << "\n";
      }
    }
  }

  void printMetalBackend(const BackendInventory &b, std::ostream &os)
  {
    os << "\n=== Metal backend ===\n";
    if (!b.available)
    {
      os << "Metal: no devices found\n";
      return;
    }
    for (const auto &plat : b.platforms)
      for (const auto &d : plat.devices)
        os << "  Metal Device " << d.index << ": " << d.name << "\n";
  }

} // namespace

std::vector<BackendInventory> enumerateAllBackends(const CliOptions &opts)
{
  std::vector<BackendInventory> out;
  if (!opts.skipOpenCL)
    out.push_back(enumerateOpenCL());
#ifdef ENABLE_VULKAN
  if (!opts.skipVulkan)
    out.push_back(enumerateVulkan());
#endif
#ifdef ENABLE_CUDA
  if (!opts.skipCuda)
    out.push_back(enumerateCuda());
#endif
#ifdef ENABLE_METAL
  if (!opts.skipMetal)
    out.push_back(enumerateMetal());
#endif
  return out;
}

void printInventory(const std::vector<BackendInventory> &inv, std::ostream &os)
{
  for (const auto &b : inv)
  {
    if (b.backend == "OpenCL")
      printOpenCLBackend(b, os);
    else if (b.backend == "Vulkan")
      printVulkanBackend(b, os);
    else if (b.backend == "CUDA")
      printCudaBackend(b, os);
    else if (b.backend == "Metal")
      printMetalBackend(b, os);
  }
}

std::string inventoryToJson(const std::vector<BackendInventory> &inv)
{
  std::ostringstream os;
  os << "{\"backends\":[";
  for (size_t i = 0; i < inv.size(); ++i)
  {
    if (i)
      os << ",";
    const auto &b = inv[i];
    os << "{\"name\":\"" << jsonEscape(b.backend) << "\""
       << ",\"available\":" << (b.available ? "true" : "false")
       << ",\"platforms\":[";
    for (size_t p = 0; p < b.platforms.size(); ++p)
    {
      if (p)
        os << ",";
      const auto &plat = b.platforms[p];
      os << "{\"index\":" << plat.index
         << ",\"name\":\"" << jsonEscape(plat.name) << "\""
         << ",\"devices\":[";
      for (size_t d = 0; d < plat.devices.size(); ++d)
      {
        if (d)
          os << ",";
        appendDeviceJson(os, plat.devices[d]);
      }
      os << "]}";
    }
    os << "]}";
  }
  os << "]}";
  return os.str();
}
