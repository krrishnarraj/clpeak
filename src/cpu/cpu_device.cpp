#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include "cpu_kernels.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <unistd.h>
#include <cstdio>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define CLPEAK_CPU_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

// ---------------------------------------------------------------------------
// Small platform helpers
// ---------------------------------------------------------------------------

#if defined(__APPLE__)
static uint64_t sysctlU64(const char *name)
{
  uint64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) != 0)
    return 0;
  return v;
}
static std::string sysctlStr(const char *name)
{
  size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0)
    return {};
  std::string s(len, '\0');
  if (sysctlbyname(name, &s[0], &len, nullptr, 0) != 0)
    return {};
  if (!s.empty() && s.back() == '\0')
    s.pop_back();
  return s;
}
#endif

#if defined(__linux__)
// Read an entire small sysfs/procfs file into a string.
static std::string readFile(const char *path)
{
  FILE *f = std::fopen(path, "rb");
  if (!f)
    return {};
  std::string out;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
    out.append(buf, n);
  std::fclose(f);
  return out;
}
// Parse a sysfs cache size string like "32K" / "1024K" / "36608K" -> bytes.
static uint64_t parseCacheSize(const std::string &s)
{
  if (s.empty())
    return 0;
  uint64_t val = std::strtoull(s.c_str(), nullptr, 10);
  if (s.find('K') != std::string::npos || s.find('k') != std::string::npos)
    return val * 1024ull;
  if (s.find('M') != std::string::npos || s.find('m') != std::string::npos)
    return val * 1024ull * 1024ull;
  return val;
}
// Count the CPUs in a sysfs cpu-list string like "0-7" or "0-3,16-19".
static int countCpuList(const std::string &s)
{
  int total = 0;
  size_t i = 0;
  while (i < s.size())
  {
    // parse a number
    while (i < s.size() && !std::isdigit((unsigned char)s[i])) i++;
    if (i >= s.size()) break;
    long a = std::strtol(s.c_str() + i, nullptr, 10);
    while (i < s.size() && std::isdigit((unsigned char)s[i])) i++;
    if (i < s.size() && s[i] == '-')
    {
      i++;
      long b = std::strtol(s.c_str() + i, nullptr, 10);
      while (i < s.size() && std::isdigit((unsigned char)s[i])) i++;
      total += (int)(b - a + 1);
    }
    else
    {
      total += 1;
    }
  }
  return total;
}

static std::string firstLineValue(const std::string &cpuinfo, const char *key)
{
  size_t pos = cpuinfo.find(key);
  if (pos == std::string::npos)
    return {};
  size_t colon = cpuinfo.find(':', pos);
  if (colon == std::string::npos)
    return {};
  size_t eol = cpuinfo.find('\n', colon);
  std::string v = cpuinfo.substr(colon + 1, eol - colon - 1);
  size_t a = v.find_first_not_of(" \t");
  size_t b = v.find_last_not_of(" \t\r");
  if (a == std::string::npos)
    return {};
  return v.substr(a, b - a + 1);
}
#endif

#if defined(CLPEAK_CPU_X86)
static inline void cpuid(uint32_t leaf, uint32_t sub, uint32_t regs[4])
{
#if defined(_MSC_VER)
  int r[4];
  __cpuidex(r, (int)leaf, (int)sub);
  regs[0] = r[0];
  regs[1] = r[1];
  regs[2] = r[2];
  regs[3] = r[3];
#else
  __cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
#endif
}
static std::string x86Brand()
{
  uint32_t regs[4];
  cpuid(0x80000000u, 0, regs);
  if (regs[0] < 0x80000004u)
    return {};
  char brand[49] = {0};
  for (uint32_t i = 0; i < 3; i++)
  {
    cpuid(0x80000002u + i, 0, regs);
    std::memcpy(brand + i * 16 + 0, &regs[0], 4);
    std::memcpy(brand + i * 16 + 4, &regs[1], 4);
    std::memcpy(brand + i * 16 + 8, &regs[2], 4);
    std::memcpy(brand + i * 16 + 12, &regs[3], 4);
  }
  std::string s(brand);
  size_t a = s.find_first_not_of(' ');
  return a == std::string::npos ? std::string{} : s.substr(a);
}
static std::string x86Vendor()
{
  uint32_t regs[4];
  cpuid(0, 0, regs);
  char v[13] = {0};
  std::memcpy(v + 0, &regs[1], 4);
  std::memcpy(v + 4, &regs[3], 4);
  std::memcpy(v + 8, &regs[2], 4);
  return std::string(v);
}
#endif

// ---------------------------------------------------------------------------
// ARM MIDR_EL1 -> human-readable CPU name.  Many ARM machines expose no
// marketing brand string (server VMs, Windows-on-ARM), but MIDR is
// architecturally mandatory: implementer byte [31:24] + part number [15:4].
// Decoding uses a lookup table for KNOWN cores, but degrades gracefully on
// unknown ones — the implementer byte alone names the vendor, and an unknown
// part renders as "<Vendor> CPU (part 0x###)", never worse than the old
// "Unknown CPU"/"Linux CPU" fallbacks.  Heterogeneous chips list each distinct
// core with its count, e.g. "4x Cortex-X925 + 6x Cortex-A725".
// Linux + Windows only: macOS always has the sysctl brand string.
// ---------------------------------------------------------------------------
#if (defined(__aarch64__) || defined(_M_ARM64)) && (defined(__linux__) || defined(_WIN32))
static const char *armImplementerName(unsigned imp)
{
  switch (imp)
  {
  case 0x41: return "Arm";
  case 0x42: return "Broadcom";
  case 0x43: return "Cavium";
  case 0x46: return "Fujitsu";
  case 0x48: return "HiSilicon";
  case 0x4e: return "NVIDIA";
  case 0x50: return "Applied Micro";
  case 0x51: return "Qualcomm";
  case 0x53: return "Samsung";
  case 0x56: return "Marvell";
  case 0x61: return "Apple";
  case 0x69: return "Intel";
  case 0x6d: return "Microsoft";
  case 0xc0: return "Ampere";
  default:   return nullptr;
  }
}

static const char *armPartName(unsigned imp, unsigned part)
{
  if (imp == 0x41)  // Arm Ltd designs
    switch (part)
    {
    case 0xd03: return "Cortex-A53";     case 0xd04: return "Cortex-A35";
    case 0xd05: return "Cortex-A55";     case 0xd07: return "Cortex-A57";
    case 0xd08: return "Cortex-A72";     case 0xd09: return "Cortex-A73";
    case 0xd0a: return "Cortex-A75";     case 0xd0b: return "Cortex-A76";
    case 0xd0c: return "Neoverse N1";    case 0xd0d: return "Cortex-A77";
    case 0xd40: return "Neoverse V1";    case 0xd41: return "Cortex-A78";
    case 0xd44: return "Cortex-X1";      case 0xd46: return "Cortex-A510";
    case 0xd47: return "Cortex-A710";    case 0xd48: return "Cortex-X2";
    case 0xd49: return "Neoverse N2";    case 0xd4b: return "Cortex-A78C";
    case 0xd4d: return "Cortex-A715";    case 0xd4e: return "Cortex-X3";
    case 0xd4f: return "Neoverse V2";    case 0xd80: return "Cortex-A520";
    case 0xd81: return "Cortex-A720";    case 0xd82: return "Cortex-X4";
    case 0xd84: return "Neoverse V3";    case 0xd85: return "Cortex-X925";
    case 0xd87: return "Cortex-A725";    case 0xd88: return "Cortex-A520AE";
    case 0xd8e: return "Neoverse N3";
    default: return nullptr;
    }
  if (imp == 0x51)  // Qualcomm custom cores (Kryo parts fall through to generic)
    switch (part)
    {
    case 0x001: return "Oryon";
    default: return nullptr;
    }
  if (imp == 0x4e)  // NVIDIA custom cores (Grace uses Arm Neoverse V2 above)
    switch (part)
    {
    case 0x003: return "Denver 2";
    case 0x004: return "Carmel";
    default: return nullptr;
    }
  if (imp == 0x6d)  // Microsoft (Azure Cobalt: Neoverse-N2-based, own implementer)
    switch (part)
    {
    case 0xd49: return "Azure Cobalt 100";
    default: return nullptr;
    }
  if (imp == 0xc0)  // Ampere
    switch (part)
    {
    case 0xac3: case 0xac4: return "AmpereOne";
    default: return nullptr;
    }
  if (imp == 0x48 && part == 0xd01) return "TaiShan V110";   // Kunpeng 920
  if (imp == 0x46 && part == 0x001) return "A64FX";
  return nullptr;
}

// Compose a name from the distinct MIDR values of all cores (first-seen order).
static std::string armCpuNameFromMidrs(const std::vector<uint64_t> &midrs,
                                       std::string &vendorOut)
{
  struct Kind { unsigned imp, part; int count; };
  std::vector<Kind> kinds;
  for (uint64_t m : midrs)
  {
    if (!m) continue;
    unsigned imp = (unsigned)((m >> 24) & 0xFF), part = (unsigned)((m >> 4) & 0xFFF);
    bool found = false;
    for (auto &k : kinds)
      if (k.imp == imp && k.part == part) { k.count++; found = true; break; }
    if (!found) kinds.push_back({imp, part, 1});
  }
  if (kinds.empty())
    return {};
  if (vendorOut.empty())
    if (const char *v = armImplementerName(kinds[0].imp)) vendorOut = v;
  std::string name;
  for (const auto &k : kinds)
  {
    if (!name.empty()) name += " + ";
    // Only prefix per-kind core counts on heterogeneous chips; the homogeneous
    // count is already in the "Cores" device property.
    if (kinds.size() > 1) name += std::to_string(k.count) + "x ";
    if (const char *p = armPartName(k.imp, k.part))
      name += p;
    else
    {
      char buf[48];
      const char *v = armImplementerName(k.imp);
      if (v) std::snprintf(buf, sizeof(buf), "%s CPU (part 0x%03x)", v, k.part);
      else   std::snprintf(buf, sizeof(buf), "ARM CPU (impl 0x%02x, part 0x%03x)", k.imp, k.part);
      name += buf;
    }
  }
  return name;
}

#if defined(__linux__)
// Per-CPU MIDR from sysfs (exposed by the arm64 kernel since 4.7); falls back
// to the "CPU implementer" / "CPU part" pairs in /proc/cpuinfo (present per
// processor block even when there is no "model name" on ARM).
static std::vector<uint64_t> collectMidrs(const std::string &cpuinfo)
{
  std::vector<uint64_t> v;
  for (int cpu = 0; cpu < 4096; cpu++)
  {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/regs/identification/midr_el1", cpu);
    std::string s = readFile(path);
    if (s.empty())
      break;
    v.push_back(std::strtoull(s.c_str(), nullptr, 16));
  }
  if (!v.empty())
    return v;
  // cpuinfo fallback: each "CPU part" line pairs with the most recent
  // "CPU implementer" line in its processor block.
  uint64_t imp = 0;
  size_t pos = 0;
  while (pos < cpuinfo.size())
  {
    size_t eol = cpuinfo.find('\n', pos);
    std::string line = cpuinfo.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
    if (line.rfind("CPU implementer", 0) == 0)
    {
      size_t c = line.find(':');
      if (c != std::string::npos) imp = std::strtoull(line.c_str() + c + 1, nullptr, 16);
    }
    else if (line.rfind("CPU part", 0) == 0)
    {
      size_t c = line.find(':');
      if (c != std::string::npos)
      {
        uint64_t part = std::strtoull(line.c_str() + c + 1, nullptr, 16);
        v.push_back((imp << 24) | ((part & 0xFFF) << 4));
      }
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return v;
}
#elif defined(_WIN32)
// Windows exports each core's (sanitised) MIDR_EL1 as the REG_QWORD "CP 4000"
// under CentralProcessor\<n> — same mechanism as the ID-register feature probe
// in cpu_dispatch.cpp.
static std::vector<uint64_t> collectMidrs()
{
  std::vector<uint64_t> v;
  for (int cpu = 0; cpu < 4096; cpu++)
  {
    char key[80];
    std::snprintf(key, sizeof(key),
                  "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\%d", cpu);
    uint64_t midr = 0;
    DWORD sz = sizeof(midr);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, key, "CP 4000", RRF_RT_REG_QWORD,
                     nullptr, &midr, &sz) != ERROR_SUCCESS)
      break;
    v.push_back(midr);
  }
  return v;
}
#endif
#endif // (aarch64 || ARM64) && (linux || windows)

// ---------------------------------------------------------------------------
// ISA capability + name, from the RUNTIME feature probe (cpu_dispatch.cpp), so
// these reflect the host the binary is actually running on — not the build host.
// ---------------------------------------------------------------------------
static void detectIsa(cpu_device_info_t &info)
{
  const clpeak_cpu::CpuFeatures &f = clpeak_cpu::cpuFeatures();
  info.hasAVX2    = f.avx2;
  info.hasFMA     = f.fma;
  info.hasAVX512  = f.avx512f;
  info.hasNEON    = f.neon;
  info.hasFP16    = f.fp16 || f.avx512fp16;
  info.hasFP16FML = f.fp16fml;
  info.hasBF16    = f.bf16 || f.avx512bf16 || f.svebf16 || f.avx10_2_512;
  info.hasInt8DP  = f.dotprod || f.avx512vnni || f.avxvnni || f.avxvnniint8 || f.sve;
  info.hasAVXVNNI = f.avxvnni || f.avxvnniint8;
  info.hasAMX     = f.amx_int8 || f.amx_bf16 || f.amx_fp16 || f.amx_tf32 || f.amx_fp8;
  info.hasSVE     = f.sve;
  info.hasSVE2    = f.sve2;
  info.sveVLBytes = clpeak_cpu::sveVLBytes();
  info.hasSME     = f.sme;
  info.hasSME2    = f.sme2;
  info.smeSVLBytes = clpeak_cpu::smeSVLBytes();
  info.isaName    = clpeak_cpu::isaName();
  // Report the active SVE vector length alongside the ISA name, e.g.
  // "SVE2 (VL=256b)" -- it's the defining knob for SVE peak throughput.
  if (info.sveVLBytes > 0)
    info.isaName += " (VL=" + std::to_string(info.sveVLBytes * 8) + "b)";
  // SME rides alongside the vector ISA (it's a separate streaming engine, not
  // the "widest" vector ISA), with its streaming VL: "NEON + SME2 (SVL=512b)".
  if (info.smeSVLBytes > 0)
    info.isaName += std::string(" + ") + (info.hasSME2 ? "SME2" : "SME") +
                    " (SVL=" + std::to_string(info.smeSVLBytes * 8) + "b)";
}

// ---------------------------------------------------------------------------
void detectCpuInfo(cpu_device_info_t &info)
{
  info.logicalCores = (int)std::thread::hardware_concurrency();
  if (info.logicalCores < 1)
    info.logicalCores = 1;

  detectIsa(info);

#if defined(__APPLE__)
  info.name = sysctlStr("machdep.cpu.brand_string");
  if (info.name.empty())
    info.name = "Apple CPU";
  info.vendor = sysctlStr("machdep.cpu.vendor");
  info.physicalCores = (int)sysctlU64("hw.physicalcpu");
  info.perfCores = (int)sysctlU64("hw.perflevel0.physicalcpu");
  info.effCores = (int)sysctlU64("hw.perflevel1.physicalcpu");
  info.l1dCacheBytes = sysctlU64("hw.perflevel0.l1dcachesize");
  info.l2CacheBytes = sysctlU64("hw.perflevel0.l2cachesize");
  if (!info.l1dCacheBytes)
    info.l1dCacheBytes = sysctlU64("hw.l1dcachesize");
  if (!info.l2CacheBytes)
    info.l2CacheBytes = sysctlU64("hw.l2cachesize");
  // On Apple Silicon, L1d is per P-core; L2 is a shared cluster cache.
  // Report total L1d = per-core x physicalCores; L2 total = L2 as-is.
  if (info.l1dCacheBytes && info.physicalCores > 0)
    info.l1dTotalBytes = info.l1dCacheBytes * (uint64_t)info.physicalCores;
  info.l2TotalBytes = info.l2CacheBytes;
  info.l3CacheBytes = sysctlU64("hw.l3cachesize");
  info.totalMemBytes = sysctlU64("hw.memsize");
  uint64_t hz = sysctlU64("hw.cpufrequency_max");
  info.clockMHz = (int)(hz / 1000000ull);
  if (info.vendor.empty() && info.name.rfind("Apple", 0) == 0)
    info.vendor = "Apple";

#elif defined(__linux__)
  std::string cpuinfo = readFile("/proc/cpuinfo");
  info.name = firstLineValue(cpuinfo, "model name");
  if (info.name.empty())
    info.name = firstLineValue(cpuinfo, "Hardware");
  info.vendor = firstLineValue(cpuinfo, "vendor_id");
#if defined(CLPEAK_CPU_X86)
  if (info.name.empty())
    info.name = x86Brand();
  if (info.vendor.empty())
    info.vendor = x86Vendor();
#endif
#if defined(__aarch64__)
  // ARM machines usually have no "model name"; decode MIDR instead.
  if (info.name.empty())
    info.name = armCpuNameFromMidrs(collectMidrs(cpuinfo), info.vendor);
#endif
  if (info.name.empty())
    info.name = "Linux CPU";

  // Physical cores: "cpu cores" (per socket) * distinct physical ids.
  {
    int coresPerSocket = std::atoi(firstLineValue(cpuinfo, "cpu cores").c_str());
    int sockets = 0;
    size_t p = 0;
    while ((p = cpuinfo.find("physical id", p)) != std::string::npos)
    {
      sockets++;
      p += 11;
    }
    // physical id lines repeat per logical CPU; count distinct is overkill —
    // approximate sockets as 1 when the field is present at all.
    if (sockets > 0)
      sockets = 1;
    info.physicalCores = coresPerSocket > 0 ? coresPerSocket * (sockets ? sockets : 1) : 0;
  }

  // Cache sizes from sysfs (index0..3: level + type + size).
  for (int idx = 0; idx < 4; idx++)
  {
    char base[128];
    std::snprintf(base, sizeof(base), "/sys/devices/system/cpu/cpu0/cache/index%d/", idx);
    std::string lvl = readFile((std::string(base) + "level").c_str());
    std::string type = readFile((std::string(base) + "type").c_str());
    std::string size = readFile((std::string(base) + "size").c_str());
    if (lvl.empty())
      continue;
    int level = std::atoi(lvl.c_str());
    uint64_t bytes = parseCacheSize(size);
    bool isData = type.rfind("Data", 0) == 0 || type.rfind("Unified", 0) == 0;
    // For each cache level, read shared_cpu_list to compute the number of
    // instances: instances = logicalCores / cpus_sharing_one_instance.
    auto computeInstances = [&](const char *b) -> int {
      int cpusShared = countCpuList(readFile((std::string(b) + "shared_cpu_list").c_str()));
      if (cpusShared <= 0) return 0;
      int inst = info.logicalCores / cpusShared;
      return inst < 1 ? 1 : inst;
    };
    if (level == 1 && isData && type.rfind("Data", 0) == 0)
    {
      info.l1dCacheBytes = bytes;
      int inst = computeInstances(base);
      if (inst > 0) info.l1dTotalBytes = bytes * (uint64_t)inst;
    }
    else if (level == 2 && isData)
    {
      info.l2CacheBytes = bytes;
      int inst = computeInstances(base);
      if (inst > 0) info.l2TotalBytes = bytes * (uint64_t)inst;
    }
    else if (level == 3 && isData)
    {
      info.l3CacheBytes = bytes;   // one instance (per-CCX/CCD on AMD)
      int inst = computeInstances(base);
      if (inst > 0) info.l3TotalBytes = bytes * (uint64_t)inst;
    }
  }
#if defined(_SC_LEVEL1_DCACHE_SIZE)
  if (!info.l1dCacheBytes && sysconf(_SC_LEVEL1_DCACHE_SIZE) > 0)
    info.l1dCacheBytes = (uint64_t)sysconf(_SC_LEVEL1_DCACHE_SIZE);
  if (!info.l2CacheBytes && sysconf(_SC_LEVEL2_CACHE_SIZE) > 0)
    info.l2CacheBytes = (uint64_t)sysconf(_SC_LEVEL2_CACHE_SIZE);
  if (!info.l3CacheBytes && sysconf(_SC_LEVEL3_CACHE_SIZE) > 0)
    info.l3CacheBytes = (uint64_t)sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
  {
    long pages = sysconf(_SC_PHYS_PAGES);
    long psz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && psz > 0)
      info.totalMemBytes = (uint64_t)pages * (uint64_t)psz;
  }
  {
    std::string khz = readFile("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (!khz.empty())
      info.clockMHz = (int)(std::strtoull(khz.c_str(), nullptr, 10) / 1000ull);
  }

#elif defined(_WIN32)
  // The struct default is "Unknown CPU" (non-empty), which used to make every
  // name.empty() fallback below dead code on ARM64 -- clear it so the registry
  // and MIDR paths actually run.
  info.name.clear();
#if defined(CLPEAK_CPU_X86)
  info.name = x86Brand();
  info.vendor = x86Vendor();
#endif
  if (info.name.empty())
  {
    // ARM64 has no CPUID; the registry carries the marketing name (and the
    // x86 path uses this as a fallback too if CPUID yields nothing).
    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &key) == ERROR_SUCCESS)
    {
      char buf[128];
      DWORD sz = sizeof(buf) - 1, type = 0;
      if (RegQueryValueExA(key, "ProcessorNameString", nullptr, &type,
                           (LPBYTE)buf, &sz) == ERROR_SUCCESS && type == REG_SZ)
      {
        buf[sz] = '\0';   // registry strings are not guaranteed NUL-terminated
        info.name = buf;
      }
      RegCloseKey(key);
    }
  }
#if defined(_M_ARM64) || defined(__aarch64__)
  // No marketing string in the registry either -> decode the per-core MIDRs
  // the kernel exports as "CP 4000".
  if (info.name.empty())
    info.name = armCpuNameFromMidrs(collectMidrs(), info.vendor);
#endif
  if (info.name.empty())
    info.name = "Windows CPU";
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms))
    info.totalMemBytes = ms.ullTotalPhys;
  // Cache + physical core topology via GetLogicalProcessorInformationEx.
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
  if (len)
  {
    std::vector<char> buf(len);
    auto *p = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
    if (GetLogicalProcessorInformationEx(RelationAll, p, &len))
    {
      char *ptr = buf.data();
      char *end = ptr + len;
      int physical = 0;
      while (ptr < end)
      {
        auto *e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(ptr);
        if (e->Relationship == RelationProcessorCore)
          physical++;
        else if (e->Relationship == RelationCache)
        {
          const auto &c = e->Cache;
          if (c.Type == CacheData || c.Type == CacheUnified)
          {
            if (c.Level == 1 && c.Type == CacheData)
            {
              info.l1dCacheBytes  = c.CacheSize;
              info.l1dTotalBytes += c.CacheSize;
            }
            else if (c.Level == 2)
            {
              info.l2CacheBytes  = c.CacheSize;
              info.l2TotalBytes += c.CacheSize;
            }
            else if (c.Level == 3)
            {
              info.l3CacheBytes  = c.CacheSize;       // one instance
              info.l3TotalBytes += c.CacheSize;       // sum across instances
            }
          }
        }
        ptr += e->Size;
      }
      info.physicalCores = physical;
    }
  }
#else
  info.name = "Unknown CPU";
#endif

  if (info.physicalCores <= 0)
    info.physicalCores = info.logicalCores;

  // Sane fallbacks so the cache benchmarks always have a working-set target.
  if (!info.l1dCacheBytes)
    info.l1dCacheBytes = 32ull * 1024;
  if (!info.l2CacheBytes)
    info.l2CacheBytes = 512ull * 1024;
  if (!info.l3CacheBytes)
    info.l3CacheBytes = 8ull * 1024 * 1024;
  // If totals couldn't be determined, fall back to the per-core size (no breakdown shown).
  if (info.l1dTotalBytes < info.l1dCacheBytes)
    info.l1dTotalBytes = info.l1dCacheBytes;
  if (info.l2TotalBytes < info.l2CacheBytes)
    info.l2TotalBytes = info.l2CacheBytes;
  if (info.l3TotalBytes < info.l3CacheBytes)
    info.l3TotalBytes = info.l3CacheBytes;
}

#endif // ENABLE_CPU
