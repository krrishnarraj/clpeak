#include <common/common.h>
#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstring>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define CLPEAK_WRITE _write
#else
#include <unistd.h>
#define CLPEAK_WRITE write
#endif

namespace clpeak {
static bool g_verbose = false;
bool verboseEnabled()   { return g_verbose; }
void setVerbose(bool on) { g_verbose = on; }

// ── Diagnostics route ──────────────────────────────────────────────────────

const char *logLevelString(LogLevel level)
{
    switch (level)
    {
    case LogLevel::Error:   return "error";
    case LogLevel::Warning: return "warning";
    case LogLevel::Info:    return "info";
    case LogLevel::Debug:   return "debug";
    }
    return "info";
}

LogLevel logLevelFromString(const std::string &s)
{
    if (s == "error")   return LogLevel::Error;
    if (s == "warning") return LogLevel::Warning;
    if (s == "debug")   return LogLevel::Debug;
    return LogLevel::Info;
}

static LogSink *g_logSink = nullptr;
void     setLogSink(LogSink *sink) { g_logSink = sink; }
LogSink *logSink()                 { return g_logSink; }

static std::atomic<int> g_realStderrFd{-1};
void setRealStderrFd(int fd) { g_realStderrFd.store(fd, std::memory_order_relaxed); }

void stderrWrite(const std::string &text)
{
    (void)fflush(stderr);
    const int fd = g_realStderrFd.load(std::memory_order_relaxed);
    if (fd >= 0)
    {
        // A capture (console_mute.h) owns fd 2 right now; the saved copy is
        // the terminal.  Unbuffered by construction, like stderr itself.
        size_t done = 0;
        while (done < text.size())
        {
            const auto n = CLPEAK_WRITE(fd, text.data() + done,
                                        static_cast<unsigned>(text.size() - done));
            if (n <= 0) break;
            done += static_cast<size_t>(n);
        }
        return;
    }
    (void)fputs(text.c_str(), stderr);
    (void)fflush(stderr);
}

void logMessage(LogLevel level, const std::string &source, std::string message)
{
    // Trailing newlines and spaces are the terminal's business, not the
    // message's: every renderer adds its own line ending, and a stored entry
    // should not carry one.  Leading whitespace stays -- some sites indent
    // to show nesting.
    while (!message.empty() &&
           (message.back() == '\n' || message.back() == '\r' ||
            message.back() == ' '  || message.back() == '\t'))
        message.pop_back();
    if (message.empty()) return;

    if (g_logSink)
    {
        g_logSink->onLog(level, source, message);
        return;
    }
    // No run in progress (--list-devices, the GUI enumerating its catalog):
    // the terminal is all there is.  Debug lines stay gated on --verbose as
    // they always were.
    if (level == LogLevel::Debug && !g_verbose) return;
    stderrWrite(message + "\n");
}

void logf(LogLevel level, const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    va_list copy;
    va_copy(copy, args);
    const int n = vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    std::string message;
    if (n > 0)
    {
        std::vector<char> buf(static_cast<size_t>(n) + 1);
        vsnprintf(buf.data(), buf.size(), fmt, args);
        message.assign(buf.data(), static_cast<size_t>(n));
    }
    va_end(args);
    logMessage(level, "", std::move(message));
}

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

// Fraction of a discrete device's own memory to use when the reported cache
// size is missing or known to be incomplete.  Two APIs need it.  AMD's MALL /
// Infinity Cache sits in front of memory but is excluded from what HIP and
// OpenCL report as the cache -- 96 MB on an RX 7900 XTX, 256 MB on MI300X --
// so the reported figure escapes an L2 that was never the last level, and the
// working set falls back to the 512 MB floor: twice the MALL on MI300X, not
// eight times it.  Vulkan has no cache query at all and would sit on that same
// floor everywhere.  Board memory is the one number both APIs do report, and
// across current discrete parts the last level runs about 1/256 of it, so
// memory/32 reproduces the 8x escape -- on an RTX 5090 (128 MB L2, 32 GB) the
// two rules agree on 1 GB exactly.  It is a proxy, not a measurement: it only
// ever raises the working set, and the ceiling below still caps it.
static const uint64_t GLOBAL_BW_MEM_FRACTION     = 32;

benchmark_config_t benchmark_config_t::forDevice(DeviceType type,
                                                 uint64_t lastLevelCacheBytes,
                                                 uint64_t dedicatedMemBytes)
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

    uint64_t escape = std::max(lastLevelCacheBytes * GLOBAL_BW_CACHE_ESCAPE,
                               dedicatedMemBytes / GLOBAL_BW_MEM_FRACTION);
    if (escape) {
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
#include <dirent.h>
#include <sys/resource.h>
#else
#include <unistd.h>
#include <dirent.h>
#include <sys/resource.h>
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

bool openFileDescriptors(unsigned long &used, unsigned long &limit)
{
  used = 0;
  limit = 0;
#if defined(_WIN32)
  return false;
#else
  // The kernel's own list of this process's descriptors, one entry each:
  // /proc/self/fd on Linux, /dev/fd on the BSDs and macOS.  Counting them
  // costs one directory read, where probing every descriptor below the
  // limit would cost a syscall per number (a million on macOS).
  const char *dirs[] = {"/proc/self/fd", "/dev/fd"};
  bool counted = false;
  for (const char *d : dirs)
  {
    DIR *dir = opendir(d);
    if (!dir)
      continue;
    unsigned long n = 0;
    while (struct dirent *e = readdir(dir))
      if (e->d_name[0] != '.')
        n++;
    closedir(dir);
    // opendir held one descriptor of its own while we counted.
    used = n > 0 ? n - 1 : 0;
    counted = true;
    break;
  }
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY)
    limit = (unsigned long)rl.rlim_cur;
  return counted;
#endif
}

} // namespace clpeak
