#include <common/inventory.h>
#include <common/common.h>
#include <common/hardware_link.h>
#include <common/json_writer.h>
#include <common/options.h>
#include <common/run_document.h>
#include <common/vocabulary.h>
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>

namespace
{

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

  // A cache's size: "32 KB", "12 MB", "1.5 MB".
  std::string cacheSize(std::uint64_t bytes)
  {
    const char *units[] = {"B", "KB", "MB", "GB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 3)
    {
      v /= 1024.0;
      u++;
    }
    char buf[32];
    if (v == static_cast<double>(static_cast<std::uint64_t>(v)))
      std::snprintf(buf, sizeof(buf), "%llu %s", static_cast<unsigned long long>(v), units[u]);
    else
      std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    return buf;
  }

  // What one compute unit is called on this kind of device.
  const char *unitWord(const std::string &typeStr)
  {
    if (typeStr == "CPU") return " threads";
    if (typeStr == "NPU") return " cores";
    return " CUs";
  }

  std::string joined(const std::vector<std::string> &parts, const char *sep = ", ")
  {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i)
    {
      if (i)
        out += sep;
      out += parts[i];
    }
    return out;
  }

  // Everything known about a device beyond its name and type, as one
  // comma-separated line.  Empty fields are simply absent.
  std::string deviceDetails(const BackendInventory &b, const InventoryPlatform &p,
                            const InventoryDevice &d)
  {
    std::vector<std::string> parts;
    // The processor first: what --devices takes for every device on it.
    if (!d.processor.empty())
      parts.push_back("on " + d.processor);
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
    if (!d.origin.empty())
      parts.push_back(d.origin);
    return joined(parts);
  }

  std::string padded(const std::string &s, size_t width)
  {
    if (s.size() >= width)
      return s;
    return s + std::string(width - s.size(), ' ');
  }

  // ── The hardware view ──────────────────────────────────────────────────

  // The machine in one line: what a person calls it, then what the OS says
  // it is.  Never anything that names the unit or its owner: `system` holds
  // none of that (docs/format-v4.md, Never recorded).
  std::string systemLine(const SystemInfo &s)
  {
    std::vector<std::string> parts;
    const std::string title = !s.name.empty() ? s.name : !s.model.empty() ? s.model
                            : !s.board.empty() ? s.board : s.soc;
    std::vector<std::string> what;
    if (!s.formFactor.empty())
      what.push_back(s.formFactor + (s.formFactorDetail.empty() ? "" : " (" + s.formFactorDetail + ")"));
    if (s.isVirtual)
      what.push_back("virtual");
    if (s.translated)
      what.push_back("translated");
    std::string id = s.manufacturer;
    if (!s.model.empty() && s.model != title)
      id += (id.empty() ? "" : " ") + s.model;
    if (!id.empty())
      what.push_back(id);
    if (!s.board.empty() && s.board != title)
      what.push_back("board " + s.board);
    std::string os = s.os.distribution;
    if (os.empty())
    {
      std::vector<std::string> words;
      for (const std::string *w : {&s.os.name, &s.os.version})
        if (!w->empty())
          words.push_back(*w);
      os = joined(words, " ");
    }
    if (!os.empty())
      what.push_back(os);
    if (!s.arch.empty())
      what.push_back(s.arch);
    if (s.memoryBytes)
      what.push_back(memorySize(s.memoryBytes));
    return title.empty() ? joined(what) : title + (what.empty() ? "" : "; " + joined(what, "; "));
  }

  std::string cachesText(const std::vector<CacheInfo> &caches)
  {
    std::vector<std::string> parts;
    for (const CacheInfo &c : caches)
    {
      // l1d -> L1d, l2 -> L2, slc -> system cache.
      std::string level = c.level;
      if (level == "slc")
        level = "system cache";
      else if (!level.empty())
        level[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(level[0])));
      std::string text = level;
      if (c.sizeBytes)
        text += " " + cacheSize(c.sizeBytes);
      if (c.instances > 1)
        text += " x" + std::to_string(c.instances);
      parts.push_back(text);
    }
    return joined(parts);
  }

  // A CPU's facts: its ISA and counts, CPUID's family and model, its engines
  // and the caches the whole processor shares, then one line per cluster.
  void cpuLines(const CpuFacts &c, std::vector<std::string> &head,
                std::vector<std::string> &lines)
  {
    if (!c.isa.empty())
      head.push_back(c.isa);
    if (c.cores)
      head.push_back(std::to_string(c.cores) + (c.cores == 1 ? " core" : " cores"));
    if (c.threads && c.threads != c.cores)
      head.push_back(std::to_string(c.threads) + " threads");
    if (c.sockets > 1)
      head.push_back(std::to_string(c.sockets) + " sockets");
    if (c.cpuidFamily >= 0)
      head.push_back("CPUID family " + std::to_string(c.cpuidFamily) + " model " +
                     std::to_string(c.cpuidModel));
    for (const CpuEngine &e : c.engines)
    {
      std::string text = e.name.empty() ? e.kind + " engine" : e.name;
      std::vector<std::string> how;
      if (!e.scope.empty())
        how.push_back("one per " + e.scope);
      if (e.reachedBy == "library")
        how.push_back("through a library");
      else if (!e.reachedBy.empty())
        how.push_back("by " + e.reachedBy);
      if (!how.empty())
        text += " (" + joined(how) + ")";
      head.push_back(text);
    }
    if (!c.caches.empty())
      head.push_back("shared " + cachesText(c.caches));
    for (const CpuCluster &k : c.clusters)
    {
      std::vector<std::string> parts;
      std::string cores = std::to_string(k.cores) + (k.cores == 1 ? " core" : " cores");
      if (k.threadsPerCore > 1)
        cores += " x " + std::to_string(k.threadsPerCore) + " threads";
      parts.push_back(k.kind.empty() ? cores : k.kind + ": " + cores);
      if (k.maxMhz)
        parts.push_back("up to " + std::to_string(k.maxMhz) + " MHz");
      if (!k.microarch.empty())
        parts.push_back(k.microarch);
      if (!k.midr.empty())
        parts.push_back("MIDR " + k.midr);
      if (!k.caches.empty())
        parts.push_back(cachesText(k.caches));
      lines.push_back(joined(parts));
    }
  }

  // A GPU's, an NPU's or an accelerator's facts in one line.
  void unitFacts(const HardwareFacts &f, std::vector<std::string> &head)
  {
    if (f.integrated == 1)
      head.push_back("integrated");
    else if (f.integrated == 0)
      head.push_back("discrete");
    if (!f.arch.empty())
      head.push_back(f.arch);
    if (f.computeUnits)
    {
      const std::string unit = f.computeUnit.empty() ? "unit" : f.computeUnit;
      head.push_back(std::to_string(f.computeUnits) + " " + unit + (f.computeUnits == 1 ? "" : "s"));
    }
    if (f.clockMhz)
      head.push_back(std::to_string(f.clockMhz) + " MHz");
    if (f.memoryBytes)
      head.push_back(memorySize(f.memoryBytes));
    // A discrete part's PCI ids name its model; an integrated GPU's need
    // not (an M1 Pro's Vulkan device id is 64).
    if (f.integrated != 1 && !f.vendorId.empty() && !f.deviceId.empty())
      head.push_back("PCI " + f.vendorId + ":" + f.deviceId);
  }

  void printHardware(const std::vector<BackendInventory> &inv, const InventoryHardware &hw,
                     std::ostream &os)
  {
    const std::string system = systemLine(hw.system);
    if (!system.empty())
      os << "\n System: " << system << "\n";

    os << "\n Hardware, named as --devices takes it too (e.g. --devices gpu0: every\n"
          " device on that processor):\n";

    size_t idWidth = 4;
    for (const ProcessorInfo &p : hw.processors)
      idWidth = std::max(idWidth, p.id.size());
    idWidth += 2;
    const std::string under(3 + idWidth, ' ');

    auto printProcessor = [&](const ProcessorInfo &p) {
      std::vector<std::string> head, lines;
      if (p.kind == xpupeak::toString(xpupeak::ProcessorKind::Cpu))
        cpuLines(p.facts.cpu, head, lines);
      else
        unitFacts(p.facts, head);
      os << "   " << padded(p.id, idWidth) << (p.name.empty() ? "(unnamed)" : p.name);
      if (!p.kind.empty())
        os << "  [" << p.kind << "]";
      if (!head.empty())
        os << "  " << joined(head);
      os << "\n";
      for (const std::string &l : lines)
        os << under << l << "\n";
      std::vector<std::string> tokens;
      for (const BackendInventory &b : inv)
        for (const InventoryPlatform &plat : b.platforms)
          for (const InventoryDevice &d : plat.devices)
            if (d.processor == p.id)
              tokens.push_back(inventoryDeviceToken(b.id, d.index));
      os << under << (tokens.empty() ? "no device listed here reaches it"
                                     : "devices: " + joined(tokens, "  "))
         << "\n";
    };

    std::vector<bool> shown(hw.processors.size(), false);
    for (const PackageInfo &pkg : hw.packages)
    {
      os << "\n " << padded(pkg.id, idWidth + 2) << (pkg.name.empty() ? "(unnamed)" : pkg.name);
      if (!pkg.kind.empty())
        os << "  [" << pkg.kind << "]";
      os << "\n";
      for (size_t i = 0; i < hw.processors.size(); i++)
        if (hw.processors[i].package == pkg.id)
        {
          printProcessor(hw.processors[i]);
          shown[i] = true;
        }
    }
    // A processor in no package the listing names: none today.
    for (size_t i = 0; i < hw.processors.size(); i++)
      if (!shown[i])
        printProcessor(hw.processors[i]);
  }

} // namespace

std::string inventoryDeviceToken(Backend b, int index)
{
  return std::string(backendInfo(b).flag) + ":" + std::to_string(index);
}

InventoryHardware linkInventory(std::vector<BackendInventory> &inv)
{
  // The listing as a run's document would hold its devices, linked by the
  // run's own linker: the same rules, the same ids, the same --verbose lines.
  RunDocument doc;
  doc.meta.system = probeSystem();
  for (const BackendInventory &b : inv)
    for (const InventoryPlatform &p : b.platforms)
      for (const InventoryDevice &d : p.devices)
      {
        DeviceResult dev;
        dev.backend     = backendInfo(b.id).name;
        dev.platform    = p.name;
        dev.name        = d.name;
        dev.type        = d.type;
        dev.deviceIndex = d.index;
        dev.facts       = d.facts;
        dev.evidence    = d.evidence;
        doc.devices.push_back(std::move(dev));
      }
  xpupeak::linkHardware(doc, xpupeak::LinkHost::Probe);

  size_t i = 0;
  for (BackendInventory &b : inv)
    for (InventoryPlatform &p : b.platforms)
      for (InventoryDevice &d : p.devices)
        d.processor = doc.devices[i++].processor;

  InventoryHardware hw;
  hw.system     = std::move(doc.meta.system);
  hw.packages   = std::move(doc.packages);
  hw.processors = std::move(doc.processors);
  return hw;
}

bool resolveProcessorSelectors(const std::vector<BackendInventory> &inv,
                               const InventoryHardware &hw, CliOptions &opts,
                               std::string &error)
{
  for (const std::string &id : opts.deviceProcessors)
  {
    const auto known = std::find_if(hw.processors.begin(), hw.processors.end(),
                                    [&](const ProcessorInfo &p) { return p.id == id; });
    if (known == hw.processors.end())
    {
      std::vector<std::string> ids;
      for (const ProcessorInfo &p : hw.processors)
        ids.push_back(p.id);
      error = "--devices " + id + ": the backends that run reach no " + id +
              (ids.empty() ? "" : "; they reach " + joined(ids) +
                                      " (--list-devices with the same backend flags lists them)");
      return false;
    }
    std::vector<std::string> tokens;
    for (const BackendInventory &b : inv)
    {
      if (!opts.enabledBackends.test(static_cast<size_t>(b.id)))
        continue;
      for (const InventoryPlatform &p : b.platforms)
        for (const InventoryDevice &d : p.devices)
        {
          if (d.processor != id)
            continue;
          tokens.push_back(inventoryDeviceToken(b.id, d.index));
          const bool listed = std::any_of(opts.devices.begin(), opts.devices.end(),
                                          [&](const DeviceSelector &s) {
                                            return s.backend == b.id && s.index == d.index;
                                          });
          if (!listed)
            opts.devices.push_back({b.id, d.index});
        }
    }
    if (tokens.empty())
    {
      error = "--devices " + id + ": no device of a backend that runs is on " + id;
      return false;
    }
  }
  return true;
}

void writeInventoryBackends(JsonWriter &w, const std::vector<BackendInventory> &inv)
{
  for (const auto &b : inv)
  {
    w.beginObject();
    w.str("name", backendInfo(b.id).name);
    w.str("flag", backendInfo(b.id).flag);
    w.boolean("available", b.available);
    w.strIf("info", b.info);
    w.strIf("reason", b.unavailableReason);
    if (!b.notes.empty())
    {
      w.beginArray("notes");
      for (const auto &n : b.notes) w.rawString(n);
      w.endArray();
    }
    w.beginArray("platforms");
    for (const auto &plat : b.platforms)
    {
      w.beginObject();
      w.integer("index", plat.index);
      w.str("name", plat.name);
      w.beginArray("devices");
      for (const auto &d : plat.devices)
      {
        w.beginObject();
        w.integer("index", d.index);
        w.str("name", d.name);
        w.str("type", d.typeStr);
        w.strIf("arch", d.arch);
        w.strIf("driver", d.driverVersion);
        w.strIf("api", d.apiVersion);
        w.strIf("origin", d.origin);
        if (d.numComputeUnits) w.uint("compute_units", d.numComputeUnits);
        if (d.maxClockMHz)     w.uint("clock_mhz", d.maxClockMHz);
        if (d.globalMemBytes)  w.uint("global_mem_bytes", d.globalMemBytes);
        if (d.maxAllocBytes)   w.uint("max_alloc_bytes", d.maxAllocBytes);
        if (d.hasFp16)         w.boolean("fp16", true);
        if (d.hasFp64)         w.boolean("fp64", true);
        // The linker's answer, never its evidence: a listing's tokens are
        // made under a key of this process's and mean nothing outside it.
        w.strIf("processor", d.processor);
        w.endObject();
      }
      w.endArray();
      w.endObject();
    }
    w.endArray();
    w.endObject();
  }
}

std::string inventoryToJson(const std::vector<BackendInventory> &inv,
                            const InventoryHardware *hw)
{
  // Machine-readable interchange: compact, and locale-free by construction
  // (JsonWriter) -- the GUI decodes this with the host toolkit's locale
  // already installed.
  std::ostringstream os;
  JsonWriter w(os, /*compact=*/true);
  w.beginObject();
  w.beginArray("backends");
  writeInventoryBackends(w, inv);
  w.endArray();
  if (hw)
    writeHardwareJson(w, hw->packages, hw->processors);
  w.endObject();
  return os.str();
}

void printInventory(const std::vector<BackendInventory> &inv, std::ostream &os,
                    const InventoryHardware *hw)
{
  if (hw && !hw->empty())
    printHardware(inv, *hw, os);

  // Column widths from the whole listing, so every backend's rows line up.
  size_t tokenWidth = 0, nameWidth = 0;
  for (const auto &b : inv)
    for (const auto &p : b.platforms)
      for (const auto &d : p.devices)
      {
        tokenWidth = std::max(tokenWidth, inventoryDeviceToken(b.id, d.index).size());
        nameWidth  = std::max(nameWidth, d.name.size());
      }
  tokenWidth = std::max<size_t>(tokenWidth, 8) + 2;
  nameWidth  = std::min<size_t>(nameWidth, 40) + 2;

  os << "\n Devices, named as --devices takes them (e.g. --devices cuda:0,vulkan:1):\n";

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
        os << "   " << padded(inventoryDeviceToken(b.id, d.index), tokenWidth)
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

    if (xpupeak::verboseEnabled())
      for (const auto &n : b.notes)
        os << "   (" << n << ")\n";
  }

  os << "\n";
}
