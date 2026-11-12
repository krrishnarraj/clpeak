#ifdef ENABLE_CPU

// _GNU_SOURCE before the first libc header, for glibc's cpu_set_t (see
// thread_pool.cpp).
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <cpu/cpu_peak.h>
#include "cpu_kernels.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <common/vocabulary.h>

#if defined(__APPLE__)
#include <mach/machine.h>   // CPUFAMILY_ARM_*
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(__linux__)
#include <climits>
#include <dirent.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define XPUPEAK_CPU_X86 1
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
// The CPU ids in a sysfs cpu-list string like "0-7" or "0-3,16-19".
static std::vector<int> parseCpuList(const std::string &s)
{
  std::vector<int> ids;
  size_t i = 0;
  while (i < s.size())
  {
    while (i < s.size() && !std::isdigit((unsigned char)s[i])) i++;
    if (i >= s.size()) break;
    long a = std::strtol(s.c_str() + i, nullptr, 10);
    while (i < s.size() && std::isdigit((unsigned char)s[i])) i++;
    long b = a;
    if (i < s.size() && s[i] == '-')
    {
      i++;
      b = std::strtol(s.c_str() + i, nullptr, 10);
      while (i < s.size() && std::isdigit((unsigned char)s[i])) i++;
    }
    for (long c = a; c <= b && c < 65536; c++)
      ids.push_back((int)c);
  }
  return ids;
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

#if defined(XPUPEAK_CPU_X86)
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
  // Qualcomm custom cores (Kryo parts fall through to generic).  The kernel's
  // compatibles count by part ("qcom,oryon-2-2" is part 0x002), Qualcomm by
  // generation: part 0x002 is the 3rd Gen Oryon of the Snapdragon X2 Elite
  // and the 8 Elite Gen 5.
  if (imp == 0x51)
    switch (part)
    {
    case 0x001: return "Oryon";
    case 0x002: return "Oryon Gen 3";
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

// Compose a name from the distinct cores (first-seen order).  A kind of core
// is its implementer, part AND variant: Qualcomm tells its Oryon core types
// apart by the variant alone -- an 8 Elite Gen 5 has six part 0x002 variant
// 2 cores and two variant 3 primes, a Snapdragon X2 Elite variants 2 and 1 --
// so keyed on the part, the chip read as eight alike.  Kinds that share a
// name then carry their top clock (`maxMHz`, by CPU number, 0 = unknown)
// where it tells them apart, else their variant as Arm writes it ("r3").
static std::string armCpuNameFromMidrs(const std::vector<uint64_t> &midrs,
                                       const std::vector<int> &maxMHz,
                                       std::string &vendorOut)
{
  struct Kind { unsigned imp, part, variant; int count, mhz; std::string name; };
  std::vector<Kind> kinds;
  for (size_t cpu = 0; cpu < midrs.size(); cpu++)
  {
    const uint64_t m = midrs[cpu];
    if (!m) continue;
    const unsigned imp = (unsigned)((m >> 24) & 0xFF), variant = (unsigned)((m >> 20) & 0xF),
                   part = (unsigned)((m >> 4) & 0xFFF);
    const int mhz = cpu < maxMHz.size() ? maxMHz[cpu] : 0;
    Kind *k = nullptr;
    for (auto &x : kinds)
      if (x.imp == imp && x.part == part && x.variant == variant) { k = &x; break; }
    if (!k)
    {
      kinds.push_back({imp, part, variant, 0, 0, {}});
      k = &kinds.back();
    }
    k->count++;
    k->mhz = std::max(k->mhz, mhz);
  }
  if (kinds.empty())
    return {};
  if (vendorOut.empty())
    if (const char *v = armImplementerName(kinds[0].imp)) vendorOut = v;
  for (auto &k : kinds)
  {
    if (const char *p = armPartName(k.imp, k.part))
    {
      k.name = p;
      continue;
    }
    char buf[48];
    const char *v = armImplementerName(k.imp);
    if (v) std::snprintf(buf, sizeof(buf), "%s CPU (part 0x%03x)", v, k.part);
    else   std::snprintf(buf, sizeof(buf), "ARM CPU (impl 0x%02x, part 0x%03x)", k.imp, k.part);
    k.name = buf;
  }
  std::vector<std::string> suffix(kinds.size());
  for (size_t i = 0; i < kinds.size(); i++)
  {
    std::vector<size_t> same;
    for (size_t j = 0; j < kinds.size(); j++)
      if (kinds[j].name == kinds[i].name) same.push_back(j);
    if (same.size() < 2)
      continue;
    bool byClock = true;
    for (size_t a : same)
      for (size_t b : same)
        byClock = byClock && kinds[a].mhz > 0 && (a == b || kinds[a].mhz != kinds[b].mhz);
    char buf[24];
    if (byClock) std::snprintf(buf, sizeof(buf), " (%.2f GHz)", kinds[i].mhz / 1000.0);
    else         std::snprintf(buf, sizeof(buf), " r%u", kinds[i].variant);
    suffix[i] = buf;
  }
  std::string name;
  for (size_t i = 0; i < kinds.size(); i++)
  {
    if (!name.empty()) name += " + ";
    // Only prefix per-kind core counts on heterogeneous chips; the homogeneous
    // count is already in the "Cores" device property.
    if (kinds.size() > 1) name += std::to_string(kinds[i].count) + "x ";
    name += kinds[i].name + suffix[i];
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
  // "CPU implementer" and "CPU variant" lines in its processor block.
  uint64_t imp = 0, variant = 0;
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
    else if (line.rfind("CPU variant", 0) == 0)
    {
      size_t c = line.find(':');
      if (c != std::string::npos) variant = std::strtoull(line.c_str() + c + 1, nullptr, 16);
    }
    else if (line.rfind("CPU part", 0) == 0)
    {
      size_t c = line.find(':');
      if (c != std::string::npos)
      {
        uint64_t part = std::strtoull(line.c_str() + c + 1, nullptr, 16);
        v.push_back((imp << 24) | ((variant & 0xF) << 20) | ((part & 0xFFF) << 4));
      }
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  return v;
}

// Each CPU's top clock in MHz by CPU number, 0 where cpufreq does not say.
static std::vector<int> collectMaxMHz(size_t cpus)
{
  std::vector<int> v(cpus, 0);
  for (size_t cpu = 0; cpu < cpus; cpu++)
  {
    char path[96];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%zu/cpufreq/cpuinfo_max_freq", cpu);
    v[cpu] = (int)(std::strtoull(readFile(path).c_str(), nullptr, 10) / 1000);
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
// The processor's facts (cpu_device_info_t::facts).  Linux and Windows
// describe each online CPU and every cache instance the OS lists; the
// clusters -- one per kind of core -- their kinds, and which caches the whole
// processor shares are drawn from that here, the same way for both.  Apple's
// performance levels come grouped already (detectCpuInfo).
// ---------------------------------------------------------------------------
namespace {

// One logical CPU, as its platform describes it.
struct TopoCpu
{
  int id = -1;
  long long core = -1;      // its physical core, unique across packages; -1 unknown
  uint64_t rank = 0;        // the OS's ranking: higher is faster
  int maxMHz = 0;           // 0 where the OS does not say
  uint64_t midr = 0;        // Arm: MIDR_EL1; 0 unknown
  std::string tier;         // what makes two CPUs the same kind of core
  std::string about;        // what the OS says of it, for the --verbose dump
  std::string cache[4];     // the instances it reads: [0] L1d, [1] L1i, [2] L2, [3] L3
};

// One cache instance the OS lists, and the CPUs that read it.
struct TopoCache
{
  int slot = 0;             // as TopoCpu::cache
  uint64_t size = 0;        // 0: listed without one
  unsigned line = 0;        // 0: not given
  std::vector<int> cpus;
};
using TopoCaches = std::map<std::string, TopoCache>;

const xpupeak::CacheLevel kSlotLevel[4] = {xpupeak::CacheLevel::L1d, xpupeak::CacheLevel::L1i,
                                           xpupeak::CacheLevel::L2, xpupeak::CacheLevel::L3};
const char *const kSlotName[4] = {"L1d", "L1i", "L2", "L3"};

} // anonymous namespace

// "64 KB", "12 MB"; "unsized" for a level the OS lists without a size.
static std::string topoBytes(uint64_t b)
{
  char buf[32];
  if (!b)
    return "unsized";
  if (b >= (1ull << 20) && !(b % (1ull << 20)))
    std::snprintf(buf, sizeof(buf), "%llu MB", (unsigned long long)(b >> 20));
  else if (b >= 1024 && !(b % 1024))
    std::snprintf(buf, sizeof(buf), "%llu KB", (unsigned long long)(b >> 10));
  else
    std::snprintf(buf, sizeof(buf), "%llu B", (unsigned long long)b);
  return buf;
}

// CPU ids as ranges: "0-3,8".
[[maybe_unused]] static std::string cpuRanges(std::vector<int> ids)
{
  std::sort(ids.begin(), ids.end());
  std::string out;
  for (size_t i = 0; i < ids.size();)
  {
    size_t j = i;
    while (j + 1 < ids.size() && ids[j + 1] == ids[j] + 1)
      j++;
    out += (out.empty() ? "" : ",") + std::to_string(ids[i]);
    if (j > i)
      out += "-" + std::to_string(ids[j]);
    i = j + 1;
  }
  return out;
}

// The size of each cache `c` reads, as a kind of core's signature: "/32768/...".
[[maybe_unused]] static std::string cacheSignature(const TopoCpu &c, const TopoCaches &caches)
{
  std::string out;
  for (int s = 0; s < 4; s++)
  {
    auto it = c.cache[s].empty() ? caches.end() : caches.find(c.cache[s]);
    out += "/" + std::to_string(it == caches.end() ? 0 : it->second.size);
  }
  return out;
}

// The microarchitecture a MIDR names, where the part table knows it.  x86
// gets none: its CPUID family, model and stepping are stored as they are,
// and naming them is a reader's table, as a model id's name is.
[[maybe_unused]] static std::string microarchOf(uint64_t midr)
{
#if (defined(__aarch64__) || defined(_M_ARM64)) && (defined(__linux__) || defined(_WIN32))
  if (const char *p = armPartName((unsigned)((midr >> 24) & 0xFF), (unsigned)((midr >> 4) & 0xFFF)))
    return p;
#endif
  (void)midr;
  return {};
}

// The clusters' kinds, fastest first.  None for a CPU of one kind of core.
// By rank otherwise: the fastest is `performance` and the slowest
// `efficiency`.  On Arm a prime core sits above the big ones -- the fastest of
// three or more kinds, or the faster of two bins of one core (an 8 Elite's
// Oryon) -- and on x86 a third, slowest kind is `low_power` (Intel's LP
// E-cores), the kinds between it and the fastest `efficiency`.
[[maybe_unused]] static void assignKinds(std::vector<CpuCluster> &cl,
                                         const std::vector<uint64_t> &midr, bool arm)
{
  using xpupeak::ClusterKind;
  const size_t n = cl.size();
  if (n < 2)
    return;
  auto set = [&](size_t i, ClusterKind k) { cl[i].kind = xpupeak::toString(k); };
  auto core = [](uint64_t m) { return m & 0xFF00FFF0ull; };   // implementer + part
  for (size_t i = 0; i < n; i++)
    set(i, ClusterKind::Performance);
  set(n - 1, ClusterKind::Efficiency);
  if (arm)
  {
    const bool oneCore = n == 2 && midr[0] && core(midr[0]) == core(midr[1]);
    if (n >= 3 || oneCore)
      set(0, ClusterKind::Prime);
    if (oneCore)
      set(1, ClusterKind::Performance);
  }
  else if (n >= 3)
  {
    for (size_t i = 1; i + 1 < n; i++)
      set(i, ClusterKind::Efficiency);
    set(n - 1, ClusterKind::LowPower);
  }
}

// The --verbose lines for the facts' clusters, after those for what they
// were drawn from: one per cluster, and the caches the processor shares.
static void logClusters(cpu_device_info_t &info)
{
  const CpuFacts &f = info.facts;
  auto cacheList = [](const std::vector<CacheInfo> &list) {
    std::string out;
    for (const CacheInfo &e : list)
    {
      out += std::string(out.empty() ? "" : ", ") + e.level + " " + topoBytes(e.sizeBytes);
      if (e.instances)
        out += " x" + std::to_string(e.instances);
    }
    return out.empty() ? std::string("none") : out;
  };
  for (size_t k = 0; k < f.clusters.size(); k++)
  {
    const CpuCluster &c = f.clusters[k];
    std::string line = "cluster " + std::to_string(k) + ": " +
                       (c.kind.empty() ? std::string("one kind of core") : c.kind) + ", " +
                       std::to_string(c.cores) + " cores x " + std::to_string(c.threadsPerCore) +
                       (c.threadsPerCore == 1 ? " thread" : " threads");
    if (c.maxMhz)
      line += ", " + std::to_string(c.maxMhz) + " MHz";
    if (!c.microarch.empty())
      line += ", " + c.microarch;
    if (!c.midr.empty())
      line += " (midr " + c.midr + ")";
    info.topologyLines.push_back(line + "; caches " + cacheList(c.caches));
  }
  info.topologyLines.push_back("caches the whole processor shares: " + cacheList(f.caches));
}

// The facts' clusters, cores, threads and caches from the CPUs and cache
// instances the OS lists, and the --verbose lines that show both.
[[maybe_unused]] static void buildClusters(cpu_device_info_t &info,
                                           const std::vector<TopoCpu> &cpus,
                                           const TopoCaches &caches, bool arm)
{
  CpuFacts &f = info.facts;
  if (cpus.empty())
    return;

  // The kinds of core, fastest first: by the best rank among each one's CPUs,
  // the first seen winning a tie.
  struct Tier
  {
    std::vector<size_t> members;
    uint64_t rank = 0;
  };
  std::vector<Tier> tiers;
  std::map<std::string, size_t> tierIndex;
  for (size_t i = 0; i < cpus.size(); i++)
  {
    auto it = tierIndex.find(cpus[i].tier);
    if (it == tierIndex.end())
    {
      it = tierIndex.emplace(cpus[i].tier, tiers.size()).first;
      tiers.emplace_back();
    }
    Tier &t = tiers[it->second];
    t.members.push_back(i);
    t.rank = std::max(t.rank, cpus[i].rank);
  }
  std::vector<size_t> order(tiers.size());
  for (size_t i = 0; i < order.size(); i++)
    order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return tiers[a].rank > tiers[b].rank; });

  f.clusters.clear();
  std::vector<uint64_t> midrs;
  std::map<int, size_t> clusterOf;   // CPU id -> cluster
  for (size_t k = 0; k < order.size(); k++)
  {
    const Tier &t = tiers[order[k]];
    CpuCluster c;
    std::set<long long> cores;
    uint64_t midr = 0;
    int mhz = 0;
    for (size_t i : t.members)
    {
      const TopoCpu &u = cpus[i];
      cores.insert(u.core >= 0 ? u.core : -1 - (long long)u.id);   // unknown: a core of its own
      mhz = std::max(mhz, u.maxMHz);
      if (!midr)
        midr = u.midr;
      clusterOf[u.id] = k;
    }
    c.cores          = (unsigned)cores.size();
    c.threadsPerCore = (unsigned)((t.members.size() + cores.size() - 1) / cores.size());
    c.maxMhz         = (unsigned)mhz;
    if (midr)
    {
      char hex[16];
      std::snprintf(hex, sizeof(hex), "%08llx", (unsigned long long)(midr & 0xFFFFFFFFull));
      c.midr      = hex;
      c.microarch = microarchOf(midr);
    }
    f.clusters.push_back(c);
    midrs.push_back(midr);
  }
  assignKinds(f.clusters, midrs, arm);

  // An instance is its cluster's own, or the whole processor's: read by more
  // than one kind of core, or by every CPU there is.  Instances alike are
  // counted together, smaller levels first.
  std::vector<const TopoCache *> ordered;
  for (const auto &kv : caches)
    ordered.push_back(&kv.second);
  std::stable_sort(ordered.begin(), ordered.end(), [](const TopoCache *a, const TopoCache *b) {
    return a->slot != b->slot ? a->slot < b->slot : a->size > b->size;
  });
  auto add = [](std::vector<CacheInfo> &list, const TopoCache &c) {
    const std::string level = xpupeak::toString(kSlotLevel[c.slot]);
    for (CacheInfo &e : list)
      if (e.level == level && e.sizeBytes == c.size && e.lineBytes == c.line)
      {
        e.instances++;
        return;
      }
    CacheInfo e;
    e.level     = level;
    e.sizeBytes = c.size;
    e.instances = 1;
    e.lineBytes = c.line;
    list.push_back(e);
  };
  f.caches.clear();
  for (const TopoCache *c : ordered)
  {
    std::set<size_t> kinds;
    size_t online = 0;
    for (int id : c->cpus)
    {
      auto it = clusterOf.find(id);
      if (it == clusterOf.end())
        continue;
      kinds.insert(it->second);
      online++;
    }
    if (kinds.empty())
      continue;
    const bool whole = kinds.size() > 1 || (online == cpus.size() && cpus.size() > 1);
    add(whole ? f.caches : f.clusters[*kinds.begin()].caches, *c);
  }

  f.threads = (unsigned)cpus.size();
  f.cores   = 0;
  for (const CpuCluster &c : f.clusters)
    f.cores += c.cores;

  // What the OS said of each CPU, then what was drawn from it.
  for (const TopoCpu &t : cpus)
  {
    std::string caches_;
    for (int s = 0; s < 4; s++)
    {
      auto it = t.cache[s].empty() ? caches.end() : caches.find(t.cache[s]);
      if (it == caches.end())
        continue;
      caches_ += std::string(caches_.empty() ? "" : ", ") + kSlotName[s] + " " +
                 topoBytes(it->second.size);
      if (it->second.cpus.size() > 1)
        caches_ += " [" + cpuRanges(it->second.cpus) + "]";
    }
    info.topologyLines.push_back("cpu" + std::to_string(t.id) + ": " + t.about + "; " +
                                 (caches_.empty() ? std::string("no caches listed") : caches_));
  }
  logClusters(info);
}

// ---------------------------------------------------------------------------
// Per-CPU topology: which CPU is fastest, and which cache instance every CPU
// reads at each level.  The OS ranks the cores -- Linux `cpu_capacity`, else
// the highest maximum clock; Windows `EfficiencyClass` -- and XpuPeak only
// orders them.  Ties keep the lowest id, so a homogeneous machine still runs
// its single-thread rows on cpu0.  macOS has no hard affinity to pin with and
// describes its cores by performance level instead (detectCpuInfo).
// ---------------------------------------------------------------------------
#if defined(__linux__) || defined(_WIN32)
namespace {

// One cache instance on the chip, keyed by whatever identifies it to the OS:
// the CPUs sharing it in sysfs, a device-tree phandle, a GLPI group mask.
struct CacheInstance
{
  int level = 0;
  uint64_t size = 0;           // 0 = listed without a size
  std::vector<int> cpus;       // the CPUs that read it
};

// One CPU before ordering: its rank and the instance it reads at L1d / L2 / L3.
struct CpuRecord
{
  int id = -1;
  uint64_t rank = 0;
  int maxMHz = 0;
  std::string key[4];          // [1] L1d, [2] L2, [3] L3; empty = none listed
};

} // anonymous namespace

// Order the usable CPUs fastest first into info.cores, keep every cache
// instance the OS listed on info.cacheInstances with the cores pointing at
// theirs, and derive the per-core and ST sizes from them.  `usable` is what
// this process may run on; the instance lists cover every CPU.
static void applyTopology(cpu_device_info_t &info, const std::vector<CpuRecord> &recs,
                          const std::map<std::string, CacheInstance> &inst,
                          const std::vector<int> &usable)
{
  info.cacheInstances.clear();
  std::map<std::string, int> index;
  for (const auto &kv : inst)
  {
    cpu_cache_instance_t c;
    c.level = kv.second.level;
    c.bytes = kv.second.size;
    c.cpus  = kv.second.cpus;
    index[kv.first] = (int)info.cacheInstances.size();
    info.cacheInstances.push_back(std::move(c));
  }

  auto isUsable = [&](int id) {
    return std::find(usable.begin(), usable.end(), id) != usable.end();
  };
  info.cores.clear();
  for (const CpuRecord &r : recs)
  {
    if (!isUsable(r.id)) continue;
    cpu_core_t c;
    c.id     = r.id;
    c.rank   = r.rank;
    c.maxMHz = r.maxMHz;
    for (int lvl = 1; lvl <= 3; lvl++)
    {
      auto it = r.key[lvl].empty() ? index.end() : index.find(r.key[lvl]);
      if (it != index.end())
        c.cacheInstance[lvl] = it->second;
    }
    info.cores.push_back(c);
  }
  std::stable_sort(info.cores.begin(), info.cores.end(),
                   [](const cpu_core_t &a, const cpu_core_t &b) {
                     return a.rank != b.rank ? a.rank > b.rank : a.id < b.id;
                   });
  if (info.cores.empty())
    return;
  if (info.cores.front().maxMHz > 0)
    info.clockMHz = info.cores.front().maxMHz;
  refreshCacheSizes(info);
}

// Name the ST core for the header when the cores are not all alike: the part
// name where the platform has one, then its CPU number ("Cortex-X4, cpu7").
static void nameStCore(cpu_device_info_t &info, const std::string &part)
{
  if (info.cores.size() < 2 || info.cores.front().rank == info.cores.back().rank)
    return;
  info.stCore = (part.empty() ? std::string() : part + ", ") + "cpu" +
                std::to_string(info.cores.front().id);
}

#if (defined(__aarch64__) || defined(_M_ARM64))
static std::string armPartOf(const std::vector<uint64_t> &midrs, int cpu)
{
  if (cpu < 0 || (size_t)cpu >= midrs.size())
    return {};
  const uint64_t m = midrs[(size_t)cpu];
  const char *p = armPartName((unsigned)((m >> 24) & 0xFF), (unsigned)((m >> 4) & 0xFFF));
  return p ? p : "";
}
#endif
#endif // linux || windows

#if defined(__linux__)
// A device-tree cell -- a 32-bit big-endian property value -- or 0 if absent.
static uint32_t dtCell(const std::string &path)
{
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f)
    return 0;
  unsigned char b[4];
  const size_t n = std::fread(b, 1, sizeof(b), f);
  std::fclose(f);
  return n == 4 ? (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3] : 0;
}

// Every node under `dir`, `depth` levels down, by phandle.  Cache nodes live
// under /cpus: beside the cpu nodes, or inside the node of the cpu they serve.
static void dtPhandles(const std::string &dir, int depth, std::map<uint32_t, std::string> &out)
{
  uint32_t ph = dtCell(dir + "/phandle");
  if (!ph)
    ph = dtCell(dir + "/linux,phandle");
  if (ph)
    out[ph] = dir;
  if (depth == 0)
    return;
  DIR *d = opendir(dir.c_str());
  if (!d)
    return;
  while (dirent *e = readdir(d))
  {
    if (e->d_name[0] == '.')
      continue;
    const std::string child = dir + "/" + e->d_name;
    struct stat st;
    if (stat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
      dtPhandles(child, depth - 1, out);
  }
  closedir(d);
}

// The device tree's sized caches for `cpu`: its node's d-cache-size, then each
// next-level-cache node's cache-size.  The kernel builds sysfs from this same
// tree, so it only adds something where sysfs listed a level without a size
// or listed nothing -- but it is the last place to look before deciding a
// level is not there.
namespace {
struct DtCache
{
  int level;
  uint64_t size;
  std::string key;
};
} // anonymous namespace
static std::vector<DtCache> dtCaches(int cpu, const std::map<uint32_t, std::string> &phandles)
{
  std::vector<DtCache> out;
  char link[96];
  std::snprintf(link, sizeof(link), "/sys/devices/system/cpu/cpu%d/of_node", cpu);
  char real[PATH_MAX];
  if (!realpath(link, real))
    return out;
  const std::string node = real;
  if (uint32_t sz = dtCell(node + "/d-cache-size"))
    out.push_back({1, sz, "dt:l1d:cpu" + std::to_string(cpu)});
  int level = 1;
  uint32_t next = dtCell(node + "/next-level-cache");
  for (int hop = 0; next && hop < 4; hop++)
  {
    auto it = phandles.find(next);
    if (it == phandles.end())
      break;
    const uint32_t lvl = dtCell(it->second + "/cache-level");
    level = lvl ? (int)lvl : level + 1;
    if (uint32_t sz = dtCell(it->second + "/cache-size"))
      out.push_back({level, sz, "dt:" + std::to_string(next)});
    next = dtCell(it->second + "/next-level-cache");
  }
  return out;
}

// Linux / Android: rank every online CPU, and read the cache instances out of
// each CPU's own sysfs rather than cpu0's -- on a phone cpu0 is a little core,
// with a little core's caches.  `midrs` (arm64) names the ST core.
static void linuxTopology(cpu_device_info_t &info, const std::vector<uint64_t> &midrs)
{
  std::vector<int> online = parseCpuList(readFile("/sys/devices/system/cpu/online"));
  if (online.empty())
    for (int c = 0; c < info.logicalCores; c++)
      online.push_back(c);

  // What this process may run on -- an Android app's cpuset, taskset, a
  // container -- since a worker pinned anywhere else is refused.
  std::vector<int> usable;
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0)
    for (int c : online)
      if (c < CPU_SETSIZE && CPU_ISSET(c, &set))
        usable.push_back(c);
  if (usable.empty())
    usable = online;

  std::map<std::string, CacheInstance> inst;
  std::vector<CpuRecord> recs;
  std::map<uint32_t, std::string> phandles;
  bool phandlesRead = false;
  // For the facts, what else the OS says of each CPU -- its core, its MIDR,
  // which PMU claims it on an Intel hybrid -- beside `recs`, and of the cache
  // instances: their line sizes, and the instruction caches the rows never
  // use.
  struct CpuExtra
  {
    long long core = -1;
    int package = -1;
    uint64_t midr = 0;
    std::string l1i;
  };
  std::vector<CpuExtra> extras;
  TopoCaches l1i;
  std::map<std::string, unsigned> lineOf;
  const std::vector<int> pmuCore = parseCpuList(readFile("/sys/devices/cpu_core/cpus"));
  const std::vector<int> pmuAtom = parseCpuList(readFile("/sys/devices/cpu_atom/cpus"));
  auto dt = [&](int cpu) {
    if (!phandlesRead)
    {
      dtPhandles("/sys/firmware/devicetree/base/cpus", 3, phandles);
      phandlesRead = true;
    }
    return dtCaches(cpu, phandles);
  };

  for (int cpu : online)
  {
    char base[64];
    std::snprintf(base, sizeof(base), "/sys/devices/system/cpu/cpu%d/", cpu);
    const std::string b = base;
    CpuRecord r;
    r.id = cpu;
    // cpu_capacity is the scheduler's own measure (arm64 / riscv: 1024 = the
    // biggest core at its top clock); x86 has none, and its hybrid parts
    // rank by maximum clock instead, as do favoured cores.
    const uint64_t capacity =
        std::strtoull(readFile((b + "cpu_capacity").c_str()).c_str(), nullptr, 10);
    const uint64_t maxKHz =
        std::strtoull(readFile((b + "cpufreq/cpuinfo_max_freq").c_str()).c_str(), nullptr, 10);
    r.rank   = (capacity << 32) | (maxKHz & 0xffffffffull);
    r.maxMHz = (int)(maxKHz / 1000);

    CpuExtra x;
    {
      const std::string coreId = readFile((b + "topology/core_id").c_str());
      const std::string pkgId  = readFile((b + "topology/physical_package_id").c_str());
      const long core = coreId.empty() ? -1 : std::strtol(coreId.c_str(), nullptr, 10);
      x.package = pkgId.empty() ? -1 : (int)std::strtol(pkgId.c_str(), nullptr, 10);
      if (core >= 0)
        x.core = ((long long)std::max(x.package, 0) << 32) | (long long)core;
#if defined(__aarch64__)
      // Read per CPU: the list collectMidrs() keeps stops at the first CPU
      // without a register file, and an Android phone takes cores offline.
      const std::string m = readFile((b + "regs/identification/midr_el1").c_str());
      x.midr = !m.empty() ? std::strtoull(m.c_str(), nullptr, 16)
                          : ((size_t)cpu < midrs.size() ? midrs[(size_t)cpu] : 0);
#endif
    }

    uint64_t sizes[4] = {0, 0, 0, 0};
    bool unsized = false;
    for (int idx = 0; idx < 16; idx++)
    {
      const std::string ip = b + "cache/index" + std::to_string(idx) + "/";
      const std::string lvl = readFile((ip + "level").c_str());
      if (lvl.empty())
        break;
      const int level = std::atoi(lvl.c_str());
      const std::string type = readFile((ip + "type").c_str());
      const bool data = type.rfind("Data", 0) == 0;
      std::string shared = readFile((ip + "shared_cpu_list").c_str());
      shared.erase(shared.find_last_not_of(" \t\r\n") + 1);
      const std::string sharers = shared.empty() ? "cpu" + std::to_string(cpu) : shared;
      const unsigned line = (unsigned)std::atoi(readFile((ip + "coherency_line_size").c_str()).c_str());
      // The instruction cache is a fact only: no row reads it.
      if (level == 1 && type.rfind("Instruction", 0) == 0)
      {
        x.l1i = "L1i:" + sharers;
        TopoCache &ic = l1i[x.l1i];
        ic.slot = 1;
        if (!ic.size)
          ic.size = parseCacheSize(readFile((ip + "size").c_str()));
        if (!ic.line)
          ic.line = line;
        ic.cpus.push_back(cpu);
        continue;
      }
      if (level < 1 || level > 3 || !(data || (level > 1 && type.rfind("Unified", 0) == 0)))
        continue;
      r.key[level] = "L" + std::to_string(level) + ":" + sharers;
      if (line)
        lineOf[r.key[level]] = line;
      sizes[level] = parseCacheSize(readFile((ip + "size").c_str()));
      unsized = unsized || !sizes[level];
    }
    // arm64 lists a level without its `size` when the device tree handed to
    // the kernel had none; some kernels list no caches at all.
    if (unsized || r.key[1].empty())
      for (const DtCache &d : dt(cpu))
        if (d.level >= 1 && d.level <= 3 && !sizes[d.level])
        {
          if (r.key[d.level].empty())
            r.key[d.level] = d.key;
          sizes[d.level] = d.size;
        }
    for (int lvl = 1; lvl <= 3; lvl++)
    {
      if (r.key[lvl].empty())
        continue;
      CacheInstance &ci = inst[r.key[lvl]];
      ci.level = lvl;
      if (!ci.size)
        ci.size = sizes[lvl];
      ci.cpus.push_back(cpu);
    }
    recs.push_back(r);
    extras.push_back(x);
  }

  // No CPU listed an L3.  That is a real answer on most phones and every
  // Apple-designed core, but only once the device tree describes none either.
  bool anyL3 = false;
  for (const auto &kv : inst)
    anyL3 = anyL3 || kv.second.level == 3;
  if (!anyL3)
    for (CpuRecord &r : recs)
      for (const DtCache &d : dt(r.id))
        if (d.level == 3)
        {
          r.key[3] = d.key;
          CacheInstance &ci = inst[d.key];
          ci.level = 3;
          ci.size  = d.size;
          ci.cpus.push_back(r.id);
        }

  // The facts, from every online CPU and every instance listed, before any
  // size is measured.  A kind of core is its MIDR and the OS's capacity and
  // clock on Arm, where a faster bin of one core is a cluster of its own; on
  // x86 the PMU that claims it (Intel's P- and E-cores), since favoured cores
  // of one kind boost higher without being another kind.  The caches it
  // reads tell the rest apart (AMD's compact cores).
  {
#if defined(__aarch64__)
    const bool arm = true;
#else
    const bool arm = false;
#endif
    TopoCaches caches = l1i;
    for (const auto &kv : inst)
    {
      TopoCache &c = caches[kv.first];
      c.slot = kv.second.level == 1 ? 0 : kv.second.level;
      c.size = kv.second.size;
      auto ln = lineOf.find(kv.first);
      c.line = ln == lineOf.end() ? 0 : ln->second;
      c.cpus = kv.second.cpus;
    }
    std::vector<TopoCpu> cpus;
    std::set<int> packages;
    for (size_t i = 0; i < recs.size(); i++)
    {
      const CpuRecord &r = recs[i];
      const CpuExtra &x = extras[i];
      TopoCpu t;
      t.id     = r.id;
      t.core   = x.core;
      t.rank   = r.rank;
      t.maxMHz = r.maxMHz;
      t.midr   = x.midr;
      t.cache[0] = r.key[1];
      t.cache[1] = x.l1i;
      t.cache[2] = r.key[2];
      t.cache[3] = r.key[3];
      if (x.package >= 0)
        packages.insert(x.package);
      const bool pCore = std::find(pmuCore.begin(), pmuCore.end(), r.id) != pmuCore.end();
      const bool eCore = std::find(pmuAtom.begin(), pmuAtom.end(), r.id) != pmuAtom.end();
      char about[160];
      std::snprintf(about, sizeof(about), "package %d core %lld, capacity %llu, %d MHz", x.package,
                    x.core >= 0 ? (x.core & 0xffffffffll) : -1ll,
                    (unsigned long long)(r.rank >> 32), r.maxMHz);
      t.about = about;
      if (x.midr)
      {
        char m[32];
        std::snprintf(m, sizeof(m), ", midr %08llx", (unsigned long long)(x.midr & 0xFFFFFFFFull));
        t.about += m;
      }
      if (pCore || eCore)
        t.about += pCore ? ", P-core" : ", E-core";
      char tier[64];
      if (arm)
        std::snprintf(tier, sizeof(tier), "%llx/%llu", (unsigned long long)(x.midr & 0xFF0FFFF0ull),
                      (unsigned long long)r.rank);
      else
        std::snprintf(tier, sizeof(tier), "pmu%d", pCore ? 1 : eCore ? 2 : 0);
      t.tier = tier + cacheSignature(t, caches);
      cpus.push_back(t);
    }
    info.sockets = (int)packages.size();
    buildClusters(info, cpus, caches, arm);
  }

  applyTopology(info, recs, inst, usable);

  std::string part;
  if (!info.cores.empty())
  {
#if defined(__aarch64__)
    part = armPartOf(midrs, info.cores.front().id);
#else
    // Intel hybrid: the core PMU lists the P-cores.
    const std::vector<int> pcores = parseCpuList(readFile("/sys/devices/cpu_core/cpus"));
    if (std::find(pcores.begin(), pcores.end(), info.cores.front().id) != pcores.end())
      part = "P-core";
#endif
  }
  (void)midrs;
  nameStCore(info, part);
}

#elif defined(_WIN32)
namespace {
// One GLPI cache record.
struct WinCache
{
  int level;
  uint64_t size;
  WORD group;
  KAFFINITY mask;
  bool instruction;   // an L1i: a fact only, which no row reads
  unsigned line;
};
// One GLPI core record: the logical processors that are its threads.
struct WinCore
{
  WORD group;
  KAFFINITY mask;
};
} // anonymous namespace

// Windows: GLPI's cache records carry the mask of the processors sharing each
// instance, and CPU sets rank the processors (EfficiencyClass: higher is
// faster).  Group 0 only, which is all the pool's pinning addresses: a machine
// with several processor groups keeps the pool's old order, and its facts
// have no clusters.
static void windowsTopology(cpu_device_info_t &info, const std::vector<WinCache> &caches,
                            const std::vector<WinCore> &cores, const std::vector<uint64_t> &midrs)
{
  if (GetActiveProcessorGroupCount() != 1)
    return;
  DWORD_PTR procMask = 0, sysMask = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask) || !procMask)
    return;

  int effClass[64] = {0};
  ULONG len = 0;
  GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
  if (len)
  {
    std::vector<char> buf(len);
    if (GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data()),
                                   len, &len, GetCurrentProcess(), 0))
      for (char *p = buf.data(); p < buf.data() + len;)
      {
        auto *e = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION *>(p);
        if (e->Size == 0)
          break;
        if (e->Type == CpuSetInformation && e->CpuSet.Group == 0 &&
            e->CpuSet.LogicalProcessorIndex < 64)
          effClass[e->CpuSet.LogicalProcessorIndex] = e->CpuSet.EfficiencyClass;
        p += e->Size;
      }
  }

  std::vector<CpuRecord> recs;
  std::vector<int> usable;
  for (int lp = 0; lp < 64; lp++)
  {
    if (!(sysMask & ((DWORD_PTR)1 << lp)))
      continue;
    CpuRecord r;
    r.id   = lp;
    r.rank = (uint64_t)effClass[lp];
    recs.push_back(r);
    if (procMask & ((DWORD_PTR)1 << lp))
      usable.push_back(lp);
  }
  std::map<std::string, CacheInstance> inst;
  for (const WinCache &c : caches)
  {
    if (c.group != 0 || c.level < 1 || c.level > 3 || c.instruction)
      continue;
    char key[48];
    std::snprintf(key, sizeof(key), "L%d:%llx", c.level, (unsigned long long)c.mask);
    CacheInstance &ci = inst[key];
    ci.level = c.level;
    ci.size  = c.size;
    for (CpuRecord &r : recs)
      if (c.mask & ((KAFFINITY)1 << r.id))
      {
        r.key[c.level] = key;
        ci.cpus.push_back(r.id);
      }
  }
  // The facts, from every processor of group 0: a kind of core is its
  // efficiency class, its MIDR on Arm, and the caches it reads.  GLPI gives
  // no clock.
  {
#if defined(_M_ARM64) || defined(__aarch64__)
    const bool arm = true;
#else
    const bool arm = false;
#endif
    TopoCaches tc;
    for (const WinCache &c : caches)
    {
      if (c.group != 0 || c.level < 1 || c.level > 3)
        continue;
      char key[48];
      std::snprintf(key, sizeof(key), "%c%d:%llx", c.instruction ? 'i' : 'd', c.level,
                    (unsigned long long)c.mask);
      TopoCache &t = tc[key];
      t.slot = c.instruction ? 1 : (c.level == 1 ? 0 : c.level);   // only L1i is collected
      t.size = c.size;
      t.line = c.line;
      t.cpus.clear();
      for (const CpuRecord &r : recs)
        if (c.mask & ((KAFFINITY)1 << r.id))
          t.cpus.push_back(r.id);
    }
    std::vector<TopoCpu> cpus;
    for (const CpuRecord &r : recs)
    {
      TopoCpu t;
      t.id   = r.id;
      t.rank = r.rank;
      for (size_t k = 0; k < cores.size(); k++)
        if (cores[k].group == 0 && (cores[k].mask & ((KAFFINITY)1 << r.id)))
          t.core = (long long)k;
#if defined(_M_ARM64) || defined(__aarch64__)
      if ((size_t)r.id < midrs.size())
        t.midr = midrs[(size_t)r.id];
#endif
      for (const auto &kv : tc)
        if (std::find(kv.second.cpus.begin(), kv.second.cpus.end(), r.id) != kv.second.cpus.end())
          t.cache[kv.second.slot] = kv.first;
      char about[96];
      std::snprintf(about, sizeof(about), "core %lld, efficiency class %llu", t.core,
                    (unsigned long long)r.rank);
      t.about = about;
      if (t.midr)
      {
        char m[32];
        std::snprintf(m, sizeof(m), ", midr %08llx", (unsigned long long)(t.midr & 0xFFFFFFFFull));
        t.about += m;
      }
      char tier[64];
      std::snprintf(tier, sizeof(tier), "%llu/%llx", (unsigned long long)r.rank,
                    (unsigned long long)(t.midr & 0xFF0FFFF0ull));
      t.tier = tier + cacheSignature(t, tc);
      cpus.push_back(t);
    }
    buildClusters(info, cpus, tc, arm);
  }

  applyTopology(info, recs, inst, usable);

  std::string part;
#if defined(_M_ARM64) || defined(__aarch64__)
  if (!info.cores.empty())
    part = armPartOf(midrs, info.cores.front().id);
#endif
  (void)midrs;
  nameStCore(info, part);
}
#endif

// ---------------------------------------------------------------------------
// ISA capability + name, from the RUNTIME feature probe (cpu_dispatch.cpp), so
// these reflect the host the binary is actually running on — not the build host.
// ---------------------------------------------------------------------------
static void detectIsa(cpu_device_info_t &info)
{
  const xpupeak_cpu::CpuFeatures &f = xpupeak_cpu::cpuFeatures();
  info.hasAVX2    = f.avx2;
  info.hasFMA     = f.fma;
  info.hasAVX512  = f.avx512f;
  info.hasNEON    = f.neon;
  info.hasFP16    = f.fp16 || f.avx512fp16;
  info.hasFP16FML = f.fp16fml;
  info.hasBF16    = f.bf16 || f.avx512bf16 || f.svebf16 || f.avx10_2_512;
  info.hasInt8DP  = f.dotprod || f.avx512vnni || f.avxvnni || f.avxvnniint8 || f.sve;
  info.hasAVXVNNI = f.avxvnni || f.avxvnniint8;
  info.hasAMX     = f.amx_int8 || f.amx_bf16 || f.amx_fp16 || f.amx_fp8;
  info.hasSVE     = f.sve;
  info.hasSVE2    = f.sve2;
  info.sveVLBytes = xpupeak_cpu::sveVLBytes();
  info.hasSME     = f.sme;
  info.hasSME2    = f.sme2;
  info.smeSVLBytes = xpupeak_cpu::smeSVLBytes();
  info.emulatedX86OnArm = f.emulatedX86OnArm;
  info.isaName    = xpupeak_cpu::isaName();
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
// The per-core and ST cache fields and the instance lists, from
// info.cacheInstances -- the OS's sizes, and what probeCacheSizes() measured
// where it gave none.  A platform that lists no instances (macOS, Windows)
// sets those fields itself.
static void deriveCacheFields(cpu_device_info_t &info)
{
  if (info.cacheInstances.empty())
    return;
  std::vector<uint64_t> *lists[4] = {nullptr, &info.l1dSizes, &info.l2Sizes, &info.l3Sizes};
  bool *measured[4] = {nullptr, &info.l1dMeasured, &info.l2Measured, &info.l3Measured};
  for (int lvl = 1; lvl <= 3; lvl++)
  {
    lists[lvl]->clear();
    *measured[lvl] = false;
  }
  for (const cpu_cache_instance_t &c : info.cacheInstances)
  {
    if (!c.bytes || c.level < 1 || c.level > 3)
      continue;
    lists[c.level]->push_back(c.bytes);
    *measured[c.level] = *measured[c.level] || c.measured;
  }
  for (cpu_core_t &core : info.cores)
  {
    uint64_t *bytes[4] = {nullptr, &core.l1dBytes, &core.l2Bytes, &core.l3Bytes};
    int *sharers[4]    = {nullptr, &core.l1dSharers, &core.l2Sharers, &core.l3Sharers};
    for (int lvl = 1; lvl <= 3; lvl++)
    {
      const int i = core.cacheInstance[lvl];
      if (i < 0)
        continue;
      const cpu_cache_instance_t &c = info.cacheInstances[(size_t)i];
      *bytes[lvl] = c.bytes;
      // The pool's workers that read it: an SMT sibling counts.
      int n = 0;
      for (int id : c.cpus)
        for (const cpu_core_t &u : info.cores)
          if (u.id == id)
          {
            n++;
            break;
          }
      *sharers[lvl] = std::max(n, 1);
    }
  }
  if (info.cores.empty())
    return;
  // The ST core's -- or a size the platform found some other way (sysconf),
  // until an instance has one.
  const cpu_core_t &fast = info.cores.front();
  if (fast.l1dBytes) info.l1dCacheBytes = fast.l1dBytes;
  if (fast.l2Bytes)  info.l2CacheBytes  = fast.l2Bytes;
  if (fast.l3Bytes)  info.l3CacheBytes  = fast.l3Bytes;
  info.l1dUnsized = fast.cacheInstance[1] >= 0 && !info.l1dCacheBytes;
  info.l2Unsized  = fast.cacheInstance[2] >= 0 && !info.l2CacheBytes;
  info.l3Unsized  = fast.cacheInstance[3] >= 0 && !info.l3CacheBytes;
}

// Totals are the sum of every instance the OS listed.  Where it listed
// none, a platform branch may have left a total of its own (Windows with
// several processor groups), and the floors below cover the rest.
static void cacheTotals(cpu_device_info_t &info)
{
  auto sum = [](const std::vector<uint64_t> &v) {
    uint64_t t = 0;
    for (uint64_t x : v) t += x;
    return t;
  };
  if (!info.l1dSizes.empty()) info.l1dTotalBytes = sum(info.l1dSizes);
  if (!info.l2Sizes.empty())  info.l2TotalBytes  = sum(info.l2Sizes);
  if (!info.l3Sizes.empty())  info.l3TotalBytes  = sum(info.l3Sizes);
  // If totals couldn't be determined, fall back to the per-core size (no breakdown shown).
  if (info.l1dTotalBytes < info.l1dCacheBytes)
    info.l1dTotalBytes = info.l1dCacheBytes;
  if (info.l2TotalBytes < info.l2CacheBytes)
    info.l2TotalBytes = info.l2CacheBytes;
  if (info.l3TotalBytes < info.l3CacheBytes)
    info.l3TotalBytes = info.l3CacheBytes;
}

void refreshCacheSizes(cpu_device_info_t &info)
{
  deriveCacheFields(info);
  cacheTotals(info);
}

// ---------------------------------------------------------------------------
// The facts every platform gives the same way.
// ---------------------------------------------------------------------------

// The processor's instruction set, as `system.arch` spells it.
static const char *archWord()
{
#if defined(__aarch64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
  return "x86";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#elif defined(__riscv) && __riscv_xlen == 64
  return "riscv64";
#else
  return "";
#endif
}

#if defined(__APPLE__)
#if defined(__aarch64__)
// Apple's AMX: the matrix unit beside each cluster, reached only through
// Accelerate, that every Apple core family since the A13's has -- M1 to M3
// among them.  From M4 the probe sees the same unit as SME instead.
static bool appleAmx()
{
  switch ((uint32_t)sysctlU64("hw.cpufamily"))
  {
  case 0:
  case CPUFAMILY_ARM_SWIFT:
  case CPUFAMILY_ARM_CYCLONE:
  case CPUFAMILY_ARM_TYPHOON:
  case CPUFAMILY_ARM_TWISTER:
  case CPUFAMILY_ARM_HURRICANE:
  case CPUFAMILY_ARM_MONSOON_MISTRAL:
  case CPUFAMILY_ARM_VORTEX_TEMPEST:
    return false;
  default:
    return true;
  }
}
#endif

// Apple's facts: a cluster per performance level, as the OS groups its cores
// -- there is no per-CPU topology to draw them from, and Apple silicon gives
// no clock -- with each level's L1s and its L2s, one per `cpusperl2` CPUs.
// An Intel Mac has no levels and gets one cluster of the sizes it reports.
static void appleFacts(cpu_device_info_t &info)
{
  CpuFacts &f = info.facts;
  const unsigned line = (unsigned)sysctlU64("hw.cachelinesize");
  auto cache = [line](xpupeak::CacheLevel level, uint64_t size, unsigned instances) {
    CacheInfo c;
    c.level     = xpupeak::toString(level);
    c.sizeBytes = size;
    c.instances = instances;
    c.lineBytes = line;
    return c;
  };
  const uint64_t nLevels = sysctlU64("hw.nperflevels");
  for (uint64_t lvl = 0; lvl < nLevels; lvl++)
  {
    auto key = [lvl](const char *name) {
      char k[64];
      std::snprintf(k, sizeof(k), "hw.perflevel%llu.%s", (unsigned long long)lvl, name);
      return std::string(k);
    };
    const std::string name = sysctlStr(key("name").c_str());
    const unsigned cores   = (unsigned)sysctlU64(key("physicalcpu").c_str());
    const unsigned threads = (unsigned)sysctlU64(key("logicalcpu").c_str());
    const unsigned perL2   = (unsigned)sysctlU64(key("cpusperl2").c_str());
    const uint64_t l1d = sysctlU64(key("l1dcachesize").c_str());
    const uint64_t l1i = sysctlU64(key("l1icachesize").c_str());
    const uint64_t l2  = sysctlU64(key("l2cachesize").c_str());
    CpuCluster c;
    // The OS names its levels; a CPU of one level is one kind of core.
    if (nLevels > 1)
    {
      if (name == "Performance")     c.kind = xpupeak::toString(xpupeak::ClusterKind::Performance);
      else if (name == "Efficiency") c.kind = xpupeak::toString(xpupeak::ClusterKind::Efficiency);
    }
    c.cores          = cores;
    c.threadsPerCore = cores ? (threads + cores - 1) / cores : 0;
    if (l1d)
      c.caches.push_back(cache(xpupeak::CacheLevel::L1d, l1d, cores));
    if (l1i)
      c.caches.push_back(cache(xpupeak::CacheLevel::L1i, l1i, cores));
    if (l2)
      c.caches.push_back(cache(xpupeak::CacheLevel::L2, l2,
                               perL2 ? (threads + perL2 - 1) / perL2 : 0));
    f.clusters.push_back(c);
    info.topologyLines.push_back(
        "perflevel" + std::to_string(lvl) + " " + name + ": " + std::to_string(cores) +
        " cores, " + std::to_string(threads) + " threads; L1d " + topoBytes(l1d) + ", L1i " +
        topoBytes(l1i) + ", L2 " + topoBytes(l2) + " per " + std::to_string(perL2) + " CPUs");
  }
  if (!nLevels)
  {
    CpuCluster c;
    const unsigned cores = (unsigned)sysctlU64("hw.physicalcpu");
    const unsigned threads = (unsigned)sysctlU64("hw.logicalcpu");
    c.cores          = cores;
    c.threadsPerCore = cores ? (threads + cores - 1) / cores : 0;
    c.maxMhz         = (unsigned)(sysctlU64("hw.cpufrequency_max") / 1000000ull);
    if (const uint64_t l1d = sysctlU64("hw.l1dcachesize"))
      c.caches.push_back(cache(xpupeak::CacheLevel::L1d, l1d, 0));
    if (const uint64_t l1i = sysctlU64("hw.l1icachesize"))
      c.caches.push_back(cache(xpupeak::CacheLevel::L1i, l1i, 0));
    if (const uint64_t l2 = sysctlU64("hw.l2cachesize"))
      c.caches.push_back(cache(xpupeak::CacheLevel::L2, l2, 0));
    if (const uint64_t l3 = sysctlU64("hw.l3cachesize"))
      f.caches.push_back(cache(xpupeak::CacheLevel::L3, l3, 0));
    f.clusters.push_back(c);
  }
  f.cores   = (unsigned)sysctlU64("hw.physicalcpu");
  f.threads = (unsigned)sysctlU64("hw.logicalcpu");
  info.sockets = (int)sysctlU64("hw.packages");
  logClusters(info);
}
#endif

// What every platform's facts share: the ISA, the probe's features, the
// matrix engines, CPUID, and the totals where no topology gave them.
static void commonFacts(cpu_device_info_t &info)
{
  CpuFacts &f = info.facts;
  f.isa = archWord();
  if (info.sockets > 0)
    f.sockets = (unsigned)info.sockets;
  if (!f.threads)
    f.threads = (unsigned)info.logicalCores;
  if (!f.cores)
    f.cores = (unsigned)info.physicalCores;

#if defined(XPUPEAK_CPU_X86)
  {
    uint32_t r[4];
    cpuid(0, 0, r);
    if (r[0] >= 1)
    {
      f.cpuidVendor = x86Vendor();
      cpuid(1, 0, r);
      const uint32_t eax = r[0];
      int family = (int)((eax >> 8) & 0xF), model = (int)((eax >> 4) & 0xF);
      if (family == 0xF)
        family += (int)((eax >> 20) & 0xFF);
      if (family == 0x6 || family >= 0xF)
        model += (int)((eax >> 16) & 0xF) << 4;
      f.cpuidFamily   = family;
      f.cpuidModel    = model;
      f.cpuidStepping = (int)(eax & 0xF);
    }
  }
#endif

  // In the vocabulary's order, from the runtime probe: what the running CPU
  // and OS together allow.
  using F = xpupeak::CpuFeature;
  const xpupeak_cpu::CpuFeatures &x = xpupeak_cpu::cpuFeatures();
  const std::pair<F, bool> has[] = {
      {F::Sse2, x.sse2}, {F::Ssse3, x.ssse3}, {F::Sse4_1, x.sse41}, {F::Sse4_2, x.sse42},
      {F::Avx, x.avx}, {F::Avx2, x.avx2}, {F::Fma, x.fma},
      {F::Avx512f, x.avx512f}, {F::Avx512bw, x.avx512bw}, {F::Avx512vl, x.avx512vl},
      {F::Avx512dq, x.avx512dq}, {F::Avx512Vnni, x.avx512vnni}, {F::Avx512Bf16, x.avx512bf16},
      {F::Avx512Fp16, x.avx512fp16}, {F::AvxVnni, x.avxvnni}, {F::AvxVnniInt8, x.avxvnniint8},
      {F::AvxVnniInt16, x.avxvnniint16}, {F::Avx10, x.avx10}, {F::Avx10_2, x.avx10_2_512},
      {F::AmxTile, x.amx_tile}, {F::AmxInt8, x.amx_int8}, {F::AmxBf16, x.amx_bf16},
      {F::AmxFp16, x.amx_fp16}, {F::AmxFp8, x.amx_fp8}, {F::Vaes, x.vaes},
      {F::Neon, x.neon}, {F::Fp16, x.fp16}, {F::Fhm, x.fp16fml}, {F::Dotprod, x.dotprod},
      {F::Bf16, x.bf16}, {F::I8mm, x.i8mm}, {F::Sve, x.sve}, {F::Sve2, x.sve2},
      {F::Sme, x.sme}, {F::Sme2, x.sme2}, {F::SmeF64f64, x.smeF64F64}, {F::Fp8dot4, x.fp8dot4},
      {F::Aes, x.aes}, {F::Sha256, x.sha256}, {F::Sha512, x.sha512}, {F::Crc32, x.crc32},
  };
  for (const auto &h : has)
    if (h.second)
      f.features.push_back(xpupeak::toString(h.first));

  // A matrix engine is the CPU's, not a processor of its own: Intel's AMX
  // tiles sit in every core; SME, and Apple's AMX before it, is a unit per
  // cluster, so its all-core rate does not scale with cores.
  auto engine = [&](const char *name, xpupeak::EngineScope scope, xpupeak::EngineReach reach) {
    CpuEngine e;
    e.kind      = xpupeak::toString(xpupeak::EngineKind::Matrix);
    e.name      = name;
    e.scope     = xpupeak::toString(scope);
    e.reachedBy = xpupeak::toString(reach);
    f.engines.push_back(e);
  };
  if (x.amx_tile)
    engine("AMX", xpupeak::EngineScope::Core, xpupeak::EngineReach::Instructions);
  if (x.sme)
    engine(x.sme2 ? "SME2" : "SME", xpupeak::EngineScope::Cluster,
           xpupeak::EngineReach::Instructions);
#if defined(__APPLE__) && defined(__aarch64__)
  else if (appleAmx())
    engine("AMX", xpupeak::EngineScope::Cluster, xpupeak::EngineReach::Library);
#endif
}

// ---------------------------------------------------------------------------
void detectCpuInfo(cpu_device_info_t &info)
{
  info.facts = CpuFacts{};
  info.topologyLines.clear();
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
  // The single-thread rows run on a P-core -- macOS has no affinity to pin
  // with, but its scheduler gives a lone busy thread one (the fp32 ST row
  // reads a P-core's rate run after run) -- so the per-instance sizes are
  // perf level 0's.
  info.l1dCacheBytes = sysctlU64("hw.perflevel0.l1dcachesize");
  info.l2CacheBytes = sysctlU64("hw.perflevel0.l2cachesize");
  if (!info.l1dCacheBytes)
    info.l1dCacheBytes = sysctlU64("hw.l1dcachesize");
  if (!info.l2CacheBytes)
    info.l2CacheBytes = sysctlU64("hw.l2cachesize");
  // Every instance, per perf level: an L1d per core and an L2 per cluster.
  // The L2 has to be counted over the clusters inside each level as well:
  // hw.perflevel0.l2cachesize is ONE cluster, so an M1 Pro (2 P-clusters of
  // 12 MB + 1 E-cluster of 4 MB = 28 MB) reports 12 from that sysctl alone.
  // The aggregate is what sizes the STREAM arrays in bandwidth.cpp, and
  // understating it by 2.3x is how a "DRAM" row ends up partly cached.  The
  // L1d differs by level too: an M1 Pro's E-cores have half a P-core's.
  {
    const uint64_t nLevels = sysctlU64("hw.nperflevels");
    for (uint64_t lvl = 0; lvl < nLevels; lvl++)
    {
      auto field = [lvl](const char *name) {
        char key[64];
        std::snprintf(key, sizeof(key), "hw.perflevel%llu.%s", (unsigned long long)lvl, name);
        return sysctlU64(key);
      };
      const uint64_t cores = field("physicalcpu"), l1d = field("l1dcachesize");
      const uint64_t l2 = field("l2cachesize"), per = field("cpusperl2");
      if (l1d)
        info.l1dSizes.insert(info.l1dSizes.end(), (size_t)cores, l1d);
      if (l2 && cores && per)
        info.l2Sizes.insert(info.l2Sizes.end(), (size_t)((cores + per - 1) / per), l2);
    }
  }
  // Intel Macs have no perf levels: an L1d per core, and the one L2 reported.
  if (info.l1dSizes.empty() && info.l1dCacheBytes && info.physicalCores > 0)
    info.l1dSizes.assign((size_t)info.physicalCores, info.l1dCacheBytes);
  if (info.l2Sizes.empty() && info.l2CacheBytes)
    info.l2Sizes.push_back(info.l2CacheBytes);
  info.l3CacheBytes = sysctlU64("hw.l3cachesize");
  if (info.l3CacheBytes)
    info.l3Sizes.push_back(info.l3CacheBytes);
  info.totalMemBytes = sysctlU64("hw.memsize");
  uint64_t hz = sysctlU64("hw.cpufrequency_max");
  info.clockMHz = (int)(hz / 1000000ull);
  if (info.vendor.empty() && info.name.rfind("Apple", 0) == 0)
    info.vendor = "Apple";
  appleFacts(info);

#elif defined(__linux__)
  std::string cpuinfo = readFile("/proc/cpuinfo");
  info.name = firstLineValue(cpuinfo, "model name");
  if (info.name.empty())
    info.name = firstLineValue(cpuinfo, "Hardware");
  info.vendor = firstLineValue(cpuinfo, "vendor_id");
#if defined(XPUPEAK_CPU_X86)
  if (info.name.empty())
    info.name = x86Brand();
  if (info.vendor.empty())
    info.vendor = x86Vendor();
#endif
  std::vector<uint64_t> midrs;
#if defined(__aarch64__)
  // ARM machines usually have no "model name"; decode MIDR instead.  The
  // decode also fills the vendor from the implementer byte when cpuinfo had
  // no vendor_id (always the case on ARM), even if the name came from
  // elsewhere.
  {
    midrs = collectMidrs(cpuinfo);
    std::string midrName = armCpuNameFromMidrs(midrs, collectMaxMHz(midrs.size()), info.vendor);
    if (info.name.empty())
      info.name = midrName;
  }
#endif
  if (info.name.empty())
    info.name = "Linux CPU";

  // Physical cores: "cpu cores" (per socket) * distinct physical ids.
  {
    int coresPerSocket = std::atoi(firstLineValue(cpuinfo, "cpu cores").c_str());
    int sockets = 0;
    size_t p = 0;
    std::unordered_set<std::string> socket_ids;
    while ((p = cpuinfo.find("physical id", p)) != std::string::npos)
    {
      p += 11;
      socket_ids.insert(cpuinfo.substr(p, cpuinfo.find("\n", p) - p));
    }
    sockets = (int)socket_ids.size();
    info.physicalCores = coresPerSocket > 0 ? coresPerSocket * (sockets ? sockets : 1) : 0;
  }

  // Rank the cores, order the pool fastest first, and read every CPU's own
  // caches; the per-instance sizes and the clock are then the ST core's.
  linuxTopology(info, midrs);
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

#elif defined(_WIN32)
  // The struct default is "Unknown CPU" (non-empty), which used to make every
  // name.empty() fallback below dead code on ARM64 -- clear it so the registry
  // and MIDR paths actually run.
  info.name.clear();
#if defined(XPUPEAK_CPU_X86)
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
  std::vector<uint64_t> midrs;
#if defined(_M_ARM64) || defined(__aarch64__)
  // Decode the per-core MIDRs the kernel exports as "CP 4000": the name is a
  // fallback for when the registry has no marketing string, but the vendor
  // comes from the implementer byte either way (Windows-on-ARM registry
  // entries carry a name like "Cobalt 100" yet no vendor).
  {
    midrs = collectMidrs();
    std::string midrName = armCpuNameFromMidrs(midrs, {}, info.vendor);
    if (info.name.empty())
      info.name = midrName;
  }
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
      int physical = 0, packages = 0;
      std::vector<WinCache> caches;
      std::vector<WinCore> coreMasks;
      while (ptr < end)
      {
        auto *e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(ptr);
        if (e->Relationship == RelationProcessorCore)
        {
          physical++;
          if (e->Processor.GroupCount >= 1)
            coreMasks.push_back({e->Processor.GroupMask[0].Group, e->Processor.GroupMask[0].Mask});
        }
        else if (e->Relationship == RelationProcessorPackage)
          packages++;
        else if (e->Relationship == RelationCache)
        {
          const auto &c = e->Cache;
          if (c.Type == CacheInstruction && c.Level == 1)
            caches.push_back({1, (uint64_t)c.CacheSize, c.GroupMask.Group, c.GroupMask.Mask,
                              true, (unsigned)c.LineSize});
          if (c.Type == CacheData || c.Type == CacheUnified)
          {
            if (c.Level <= 3 && (c.Level > 1 || c.Type == CacheData))
              caches.push_back({(int)c.Level, (uint64_t)c.CacheSize, c.GroupMask.Group,
                                c.GroupMask.Mask, false, (unsigned)c.LineSize});
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
      info.sockets       = packages;
      // Rank the processors, order the pool fastest first, and point the
      // per-instance sizes at the ST core's caches rather than whichever
      // record came last.
      windowsTopology(info, caches, coreMasks, midrs);
    }
  }
#else
  info.name = "Unknown CPU";
#endif

  if (info.physicalCores <= 0)
    info.physicalCores = info.logicalCores;

  // No fallback sizes.  A 32 KB / 512 KB guess once stood in for an L1d / L2
  // the OS gave no size for -- an Android kernel whose device tree has none
  // lists its caches without one -- and the cache rows then measured a working
  // set sized for a cache that was not the phone's.  The same went for an
  // invented 8 MB L3: plenty of CPUs genuinely have none (every Apple Silicon
  // part, Snapdragon X, most phone SoCs), and the made-up one printed an "L3"
  // that does not exist, reported an L3 *faster* than the L2 next to it, and
  // undersized the STREAM arrays.  A size stays 0 when unknown -- with
  // l1dUnsized / l2Unsized / l3Unsized when the level is listed without one --
  // until probeCacheSizes() measures it, and the rows that would need one it
  // could not measure skip (cacheSizeGap).
  cacheTotals(info);
  commonFacts(info);
}

const char *cacheSizeGap(const cpu_device_info_t &info, int level)
{
  // The level's own size first, so a CPU with no L3 says so whatever else the
  // OS withheld; then the levels beneath.  A size the OS withheld that the
  // latency sweep could not find either says so.
  const bool p = info.cacheProbed;
  // A listing with nothing past the L1d is one the OS cut short -- no core
  // has an L1 alone -- not a CPU without an L2 or an L3: an Android kernel
  // whose device tree has no cache nodes lists the L1 its CPU reports.
  bool pastL1 = info.cacheInstances.empty();
  for (const cpu_cache_instance_t &c : info.cacheInstances)
    pastL1 = pastL1 || c.level >= 2;
  if (level >= 3 && !info.l3CacheBytes)
    return !info.l3Unsized ? (pastL1 ? "no L3 on this CPU" : "the OS lists no cache past the L1d")
           : p ? "the OS lists an L3 but gives no size for it, and measuring found no step for it"
               : "the OS lists an L3 but gives no size for it";
  if (level >= 2 && !info.l2CacheBytes && (level == 2 || info.l2Unsized))
    return !info.l2Unsized ? (pastL1 ? "the OS lists no L2" : "the OS lists no cache past the L1d")
           : p ? "the OS lists an L2 but gives no size for it, and measuring found no step for it"
               : "the OS lists an L2 but gives no size for it";
  // Every CPU has an L1d, so one with no size is unknown, listed or not.
  if (!info.l1dCacheBytes)
    return !info.l1dUnsized ? "the OS gives no L1d size"
           : p ? "the OS lists an L1d but gives no size for it, and measuring found no step for it"
               : "the OS lists an L1d but gives no size for it";
  return nullptr;
}

#endif // ENABLE_CPU
