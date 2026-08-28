#include <common/common.h>
#include <atomic>
#include <cstring>

namespace clpeak {
static bool g_verbose = false;
bool verboseEnabled()   { return g_verbose; }
void setVerbose(bool on) { g_verbose = on; }

static std::atomic<bool> g_cancelRequested{false};
void requestCancel()   { g_cancelRequested.store(true, std::memory_order_relaxed); }
bool cancelRequested() { return g_cancelRequested.load(std::memory_order_relaxed); }
void resetCancel()     { g_cancelRequested.store(false, std::memory_order_relaxed); }
}

// The global-bandwidth working set has to be big enough that re-reading it
// misses the device's last-level cache.  Every backend warms the buffer up and
// then re-reads that same buffer for the whole timed phase, so whatever stays
// cache-resident is counted as memory traffic it never was: a Ryzen 7 5700X3D
// (96 MB of V-Cache) read 70-99 GBPS through the old 128 MB CPU-device default,
// twice the 47.6 GBPS its dual-channel DDR4-3600 actually delivers.  Eight times
// the cache leaves the residual hit rate in the low percent.  The ceiling is
// only a guard against a driver reporting nonsense: the working set reaches it
// solely on a device reporting half a gigabyte of cache, which is a machine
// with the memory to spare, and each backend still clamps to its own allocation
// budget (maxAllocSize / 2, totalGlobalMem / 4) underneath.
static const uint64_t GLOBAL_BW_CACHE_ESCAPE     = 8;
static const uint64_t GLOBAL_BW_MAX_WORKING_SET  = 4ULL << 30;

benchmark_config_t benchmark_config_t::forDevice(DeviceType type,
                                                 uint64_t lastLevelCacheBytes)
{
    benchmark_config_t cfg;
    if (type == DeviceType::Cpu) {
        // 512 MB, same as the GPU default: a CPU device's "global memory" is
        // system DRAM sitting behind an LLC that is now routinely 32-128 MB, and
        // the old 128 MB did not clear it.  It is only the fallback for a
        // runtime that reports no cache size -- the escape rule below is what
        // sizes this on a device that does.
        cfg.globalBWMaxSize   = 1 << 29;
        cfg.computeWgsPerCU   = 512;
        cfg.computeDPWgsPerCU = 256;
        cfg.transferBWMaxSize = 1 << 27;
    } else {  // Gpu / Accelerator
        cfg.globalBWMaxSize   = 1 << 29;
        cfg.computeWgsPerCU   = 2048;
        cfg.computeDPWgsPerCU = 512;
        cfg.transferBWMaxSize = 1 << 29;
    }

    if (lastLevelCacheBytes) {
        uint64_t escape = lastLevelCacheBytes * GLOBAL_BW_CACHE_ESCAPE;
        if (escape > GLOBAL_BW_MAX_WORKING_SET)
            escape = GLOBAL_BW_MAX_WORKING_SET;
        if (escape > cfg.globalBWMaxSize)
            cfg.globalBWMaxSize = escape;
    }

    cfg.targetTimeUs       = DEFAULT_TARGET_TIME_US;
    cfg.kernelLatencyIters = 2000;
    return cfg;
}

unsigned int pickIters(double per_iter_us, unsigned int target_us,
                       unsigned int forced, unsigned int max_iters)
{
  if (forced) return forced;
  if (target_us == 0) target_us = 5000000; // 5s legacy default
  if (per_iter_us < 1.0) per_iter_us = 1.0;
  double want = (double)target_us / per_iter_us;
  if (want < 1.0)               want = 1.0;
  if (want > (double)max_iters) want = (double)max_iters;
  return (unsigned int)want;
}

std::string jsonEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s)
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

void populate(float *ptr, uint64_t N)
{
    // Use pseudo-random data to defeat hardware memory compression (some GPUs
    // transparently compress buffers, inflating apparent bandwidth when the
    // content is predictable/compressible).
    uint32_t state = 0xDEADBEEF;
    for (uint64_t i = 0; i < N; i++)
    {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Reinterpret bits as float; mask off sign+exponent high bit to avoid
        // NaN/Inf (keep exponent in [1,127] range so values are finite).
        uint32_t bits = (state & 0x7F7FFFFF) | 0x00800000;
        float val;
        memcpy(&val, &bits, sizeof(val));
        ptr[i] = val;
    }
}

// ---------------------------------------------------------------------------
// System memory
// ---------------------------------------------------------------------------

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

namespace clpeak {

uint64_t systemMemoryBytes()
{
#if defined(_WIN32)
  MEMORYSTATUSEX ms;
  ms.dwLength = sizeof(ms);
  if (GlobalMemoryStatusEx(&ms))
    return (uint64_t)ms.ullTotalPhys;
  return 0;
#elif defined(__APPLE__)
  uint64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname("hw.memsize", &v, &len, nullptr, 0) == 0)
    return v;
  return 0;
#else
  const long pages = sysconf(_SC_PHYS_PAGES);
  const long psz   = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && psz > 0)
    return (uint64_t)pages * (uint64_t)psz;
  return 0;
#endif
}

uint64_t memoryBudget(uint64_t ceiling, unsigned fraction)
{
  if (fraction == 0)
    fraction = 1;
  const uint64_t total = systemMemoryBytes();
  if (!total)
    return ceiling;
  const uint64_t share = total / fraction;
  return share < ceiling ? share : ceiling;
}

} // namespace clpeak
