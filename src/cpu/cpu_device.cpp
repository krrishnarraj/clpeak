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
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
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
// Per-CPU topology: which CPU is fastest, and which cache instance every CPU
// reads at each level.  The OS ranks the cores -- Linux `cpu_capacity`, else
// the highest maximum clock; Windows `EfficiencyClass` -- and clpeak only
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

// Order the usable CPUs fastest first into info.cores, point the ST fields at
// the first of them, and list every sized instance for the header.  `usable`
// is what this process may run on; the instance lists cover every CPU.
static void applyTopology(cpu_device_info_t &info, const std::vector<CpuRecord> &recs,
                          const std::map<std::string, CacheInstance> &inst,
                          const std::vector<int> &usable)
{
  for (const auto &kv : inst)
  {
    const CacheInstance &c = kv.second;
    if (!c.size) continue;
    if (c.level == 1)      info.l1dSizes.push_back(c.size);
    else if (c.level == 2) info.l2Sizes.push_back(c.size);
    else if (c.level == 3) info.l3Sizes.push_back(c.size);
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
    uint64_t *bytes[4] = {nullptr, &c.l1dBytes, &c.l2Bytes, &c.l3Bytes};
    int *sharers[4]    = {nullptr, &c.l1dSharers, &c.l2Sharers, &c.l3Sharers};
    for (int lvl = 1; lvl <= 3; lvl++)
    {
      auto it = r.key[lvl].empty() ? inst.end() : inst.find(r.key[lvl]);
      if (it == inst.end()) continue;
      *bytes[lvl] = it->second.size;
      int n = 0;
      for (int id : it->second.cpus)
        if (isUsable(id)) n++;
      *sharers[lvl] = std::max(n, 1);
    }
    info.cores.push_back(c);
  }
  std::stable_sort(info.cores.begin(), info.cores.end(),
                   [](const cpu_core_t &a, const cpu_core_t &b) {
                     return a.rank != b.rank ? a.rank > b.rank : a.id < b.id;
                   });
  if (info.cores.empty())
    return;

  const cpu_core_t &fast = info.cores.front();
  info.l1dCacheBytes = fast.l1dBytes;
  info.l2CacheBytes  = fast.l2Bytes;
  info.l3CacheBytes  = fast.l3Bytes;
  if (fast.maxMHz > 0)
    info.clockMHz = fast.maxMHz;
  for (const CpuRecord &r : recs)
    if (r.id == fast.id && !r.key[3].empty() && !fast.l3Bytes)
      info.l3Unsized = true;
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
      if (level < 1 || level > 3 || !(data || (level > 1 && type.rfind("Unified", 0) == 0)))
        continue;
      std::string shared = readFile((ip + "shared_cpu_list").c_str());
      shared.erase(shared.find_last_not_of(" \t\r\n") + 1);
      r.key[level] = "L" + std::to_string(level) + ":" +
                     (shared.empty() ? "cpu" + std::to_string(cpu) : shared);
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
};
} // anonymous namespace

// Windows: GLPI's cache records carry the mask of the processors sharing each
// instance, and CPU sets rank the processors (EfficiencyClass: higher is
// faster).  Group 0 only, which is all the pool's pinning addresses: a machine
// with several processor groups keeps the pool's old order.
static void windowsTopology(cpu_device_info_t &info, const std::vector<WinCache> &caches,
                            const std::vector<uint64_t> &midrs)
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
    if (c.group != 0 || c.level < 1 || c.level > 3)
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
  info.hasAMX     = f.amx_int8 || f.amx_bf16 || f.amx_fp16 || f.amx_fp8;
  info.hasSVE     = f.sve;
  info.hasSVE2    = f.sve2;
  info.sveVLBytes = clpeak_cpu::sveVLBytes();
  info.hasSME     = f.sme;
  info.hasSME2    = f.sme2;
  info.smeSVLBytes = clpeak_cpu::smeSVLBytes();
  info.emulatedX86OnArm = f.emulatedX86OnArm;
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
  std::vector<uint64_t> midrs;
#if defined(__aarch64__)
  // ARM machines usually have no "model name"; decode MIDR instead.  The
  // decode also fills the vendor from the implementer byte when cpuinfo had
  // no vendor_id (always the case on ARM), even if the name came from
  // elsewhere.
  {
    midrs = collectMidrs(cpuinfo);
    std::string midrName = armCpuNameFromMidrs(midrs, info.vendor);
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
  std::vector<uint64_t> midrs;
#if defined(_M_ARM64) || defined(__aarch64__)
  // Decode the per-core MIDRs the kernel exports as "CP 4000": the name is a
  // fallback for when the registry has no marketing string, but the vendor
  // comes from the implementer byte either way (Windows-on-ARM registry
  // entries carry a name like "Cobalt 100" yet no vendor).
  {
    midrs = collectMidrs();
    std::string midrName = armCpuNameFromMidrs(midrs, info.vendor);
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
      int physical = 0;
      std::vector<WinCache> caches;
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
            if (c.Level <= 3 && (c.Level > 1 || c.Type == CacheData))
              caches.push_back({(int)c.Level, (uint64_t)c.CacheSize, c.GroupMask.Group,
                                c.GroupMask.Mask});
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
      // Rank the processors, order the pool fastest first, and point the
      // per-instance sizes at the ST core's caches rather than whichever
      // record came last.
      windowsTopology(info, caches, midrs);
    }
  }
#else
  info.name = "Unknown CPU";
#endif

  if (info.physicalCores <= 0)
    info.physicalCores = info.logicalCores;

  // Totals are the sum of every instance the OS listed.  Where it listed
  // none, a platform branch may have left a total of its own (Windows with
  // several processor groups), and the floors below cover the rest.
  auto sum = [](const std::vector<uint64_t> &v) {
    uint64_t t = 0;
    for (uint64_t x : v) t += x;
    return t;
  };
  if (!info.l1dSizes.empty()) info.l1dTotalBytes = sum(info.l1dSizes);
  if (!info.l2Sizes.empty())  info.l2TotalBytes  = sum(info.l2Sizes);
  if (!info.l3Sizes.empty())  info.l3TotalBytes  = sum(info.l3Sizes);

  // Sane fallbacks so the cache benchmarks always have a working-set target.
  // They are assumptions, not sizes: an Android kernel whose device tree gives
  // none lists its caches without one, and a 32 KB / 512 KB printed in the
  // header there read as the phone's.  The header leaves them out.
  if (!info.l1dCacheBytes)
  {
    info.l1dCacheBytes = 32ull * 1024;
    info.l1dAssumed = true;
  }
  if (!info.l2CacheBytes)
  {
    info.l2CacheBytes = 512ull * 1024;
    info.l2Assumed = true;
  }
  // L3 gets NO fallback: plenty of CPUs genuinely have none (every Apple
  // Silicon part, Snapdragon X, most phone SoCs), and inventing 8 MB there was
  // worse than reporting nothing.  It printed an "L3" that does not exist,
  // pointed the L3 cache-bandwidth and latency rows at a working set that
  // still fits in L2 (M1 Pro reported an L3 *faster* than its own L2), and
  // sized the STREAM arrays off 8 MB instead of the ~28 MB really there.
  // Zero means "no L3" -- or, with l3Unsized, an L3 the OS gives no size for;
  // the rows that need one skip as Unsupported either way.
  // If totals couldn't be determined, fall back to the per-core size (no breakdown shown).
  if (info.l1dTotalBytes < info.l1dCacheBytes)
    info.l1dTotalBytes = info.l1dCacheBytes;
  if (info.l2TotalBytes < info.l2CacheBytes)
    info.l2TotalBytes = info.l2CacheBytes;
  if (info.l3TotalBytes < info.l3CacheBytes)
    info.l3TotalBytes = info.l3CacheBytes;
}

#endif // ENABLE_CPU
