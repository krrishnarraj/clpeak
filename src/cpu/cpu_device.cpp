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
  info.hasBF16    = f.bf16 || f.avx512bf16;
  info.hasInt8DP  = f.dotprod || f.avx512vnni;
  info.hasAMX     = f.amx_int8 || f.amx_bf16;
  info.isaName    = clpeak_cpu::isaName();
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
    if (level == 1 && isData && type.rfind("Data", 0) == 0)
      info.l1dCacheBytes = bytes;
    else if (level == 2 && isData)
      info.l2CacheBytes = bytes;
    else if (level == 3 && isData)
    {
      info.l3CacheBytes = bytes;   // one instance (per-CCX/CCD on AMD)
      // Aggregate L3 = per-instance size x number of L3 instances.  The L3 is
      // shared by `cpusPerL3` logical CPUs, so instances = logicalCores / that.
      int cpusPerL3 = countCpuList(readFile((std::string(base) + "shared_cpu_list").c_str()));
      if (cpusPerL3 > 0)
      {
        int instances = info.logicalCores / cpusPerL3;
        if (instances < 1) instances = 1;
        info.l3TotalBytes = bytes * (uint64_t)instances;
      }
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
#if defined(CLPEAK_CPU_X86)
  info.name = x86Brand();
  info.vendor = x86Vendor();
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
              info.l1dCacheBytes = c.CacheSize;
            else if (c.Level == 2)
              info.l2CacheBytes = c.CacheSize;
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
  // If the number of L3 instances couldn't be determined, assume a single
  // shared LLC (correct for Intel client/Apple; conservative elsewhere).
  if (info.l3TotalBytes < info.l3CacheBytes)
    info.l3TotalBytes = info.l3CacheBytes;
}

#endif // ENABLE_CPU
