#include <common/inventory.h>
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

} // namespace

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
