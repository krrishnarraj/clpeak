#include <common/inventory.h>
#include <common/common.h>
#include <common/options.h>
#include <algorithm>
#include <cstdio>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>

namespace
{

  void appendDeviceJson(std::ostream &os, const InventoryDevice &d)
  {
    os << "{\"index\":" << d.index
       << ",\"name\":\"" << jsonEscape(d.name) << "\""
       << ",\"type\":\"" << jsonEscape(d.typeStr) << "\"";
    if (!d.arch.empty())
      os << ",\"arch\":\"" << jsonEscape(d.arch) << "\"";
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

  void appendStringArrayJson(std::ostream &os, const std::vector<std::string> &v)
  {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
      if (i)
        os << ",";
      os << "\"" << jsonEscape(v[i]) << "\"";
    }
    os << "]";
  }

  // The token --device takes for this device.
  std::string deviceToken(const BackendInventory &b, const InventoryDevice &d)
  {
    return std::string(backendInfo(b.id).flag) + ":" + std::to_string(d.index);
  }

  std::string memorySize(std::uint64_t bytes)
  {
    const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0);
    char buf[32];
    if (mb >= 1024.0)
      std::snprintf(buf, sizeof(buf), "%.1f GB", mb / 1024.0);
    else
      std::snprintf(buf, sizeof(buf), "%.0f MB", mb);
    return buf;
  }

  // What one compute unit is called on this kind of device.
  const char *unitWord(const std::string &typeStr)
  {
    if (typeStr == "CPU") return " threads";
    if (typeStr == "NPU") return " cores";
    return " CUs";
  }

  // Everything known about a device beyond its name and type, as one
  // comma-separated line.  Empty fields are simply absent.
  std::string deviceDetails(const BackendInventory &b, const InventoryPlatform &p,
                            const InventoryDevice &d)
  {
    std::vector<std::string> parts;
    // A backend with several platforms (OpenCL) says which one a device is
    // on; the single synthetic platform of every other backend says nothing.
    if (b.platforms.size() > 1)
      parts.push_back("platform " + p.name);
    if (!d.arch.empty())
      parts.push_back(d.arch);
    if (!d.apiVersion.empty())
      parts.push_back("API " + d.apiVersion);
    if (!d.driverVersion.empty())
      parts.push_back("driver " + d.driverVersion);
    if (d.numComputeUnits)
      parts.push_back(std::to_string(d.numComputeUnits) + unitWord(d.typeStr));
    if (d.maxClockMHz)
      parts.push_back(std::to_string(d.maxClockMHz) + " MHz");
    if (d.globalMemBytes)
      parts.push_back(memorySize(d.globalMemBytes));
    if (d.maxAllocBytes)
      parts.push_back("max alloc " + memorySize(d.maxAllocBytes));
    if (d.hasFp16)
      parts.push_back("fp16");
    if (d.hasFp64)
      parts.push_back("fp64");

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i)
    {
      if (i)
        out += ", ";
      out += parts[i];
    }
    return out;
  }

  std::string padded(const std::string &s, size_t width)
  {
    if (s.size() >= width)
      return s;
    return s + std::string(width - s.size(), ' ');
  }

} // namespace

std::string inventoryToJson(const std::vector<BackendInventory> &inv)
{
  std::ostringstream os;
  // Machine-readable interchange: JSON is locale-free, and the GUI decodes
  // this with the host toolkit's locale already installed (see fmtFloat in
  // src/ffi/logger_ffi.cpp).
  os.imbue(std::locale::classic());
  os << "{\"backends\":[";
  for (size_t i = 0; i < inv.size(); ++i)
  {
    if (i)
      os << ",";
    const auto &b = inv[i];
    os << "{\"name\":\"" << jsonEscape(backendInfo(b.id).name) << "\""
       << ",\"flag\":\"" << backendInfo(b.id).flag << "\""
       << ",\"available\":" << (b.available ? "true" : "false");
    if (!b.info.empty())
      os << ",\"info\":\"" << jsonEscape(b.info) << "\"";
    if (!b.unavailableReason.empty())
      os << ",\"reason\":\"" << jsonEscape(b.unavailableReason) << "\"";
    if (!b.notes.empty())
    {
      os << ",\"notes\":";
      appendStringArrayJson(os, b.notes);
    }
    os << ",\"platforms\":[";
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

void printInventory(const std::vector<BackendInventory> &inv, std::ostream &os,
                    bool showAbsent)
{
  // Column widths from the whole listing, so every backend's rows line up.
  size_t tokenWidth = 0, nameWidth = 0;
  for (const auto &b : inv)
    for (const auto &p : b.platforms)
      for (const auto &d : p.devices)
      {
        tokenWidth = std::max(tokenWidth, deviceToken(b, d).size());
        nameWidth  = std::max(nameWidth, d.name.size());
      }
  tokenWidth = std::max<size_t>(tokenWidth, 8) + 2;
  nameWidth  = std::min<size_t>(nameWidth, 40) + 2;

  os << "\n Devices, named as --device takes them (e.g. --device cuda:0,vulkan:1):\n";

  for (const auto &b : inv)
  {
    const BackendInfo &be = backendInfo(b.id);
    os << "\n " << be.name;
    if (!b.info.empty())
      os << "  " << b.info;
    os << "\n";

    if (!b.available)
    {
      os << "   not available";
      if (!b.unavailableReason.empty())
        os << ": " << b.unavailableReason;
      os << "\n";
      continue;
    }

    bool any = false;
    for (const auto &p : b.platforms)
      for (const auto &d : p.devices)
      {
        any = true;
        os << "   " << padded(deviceToken(b, d), tokenWidth)
           << padded(d.name, nameWidth);
        if (!d.typeStr.empty())
          os << "[" << d.typeStr << "]";
        const std::string details = deviceDetails(b, p, d);
        if (!details.empty())
          os << "  " << details;
        os << "\n";
      }
    if (!any)
      os << "   no usable devices\n";

    if (clpeak::verboseEnabled())
      for (const auto &n : b.notes)
        os << "   (" << n << ")\n";
  }

  if (showAbsent)
  {
    std::string absent;
    for (size_t i = 0; i < static_cast<size_t>(Backend::COUNT); ++i)
    {
      const BackendInfo &be = backendInfo(static_cast<Backend>(i));
      if (be.builtIn)
        continue;
      if (!absent.empty())
        absent += ", ";
      absent += be.name;
    }
    if (!absent.empty())
      os << "\n Not in this build: " << absent << "\n";
  }
  os << "\n";
}
