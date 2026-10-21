#ifdef ENABLE_CPU

// _GNU_SOURCE must precede the first libc header for glibc to expose cpu_set_t /
// CPU_SET; harmless on Bionic (Android).  Defined before cpu_peak.h, which pulls
// in <thread>/<mutex> and therefore libc headers.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <cpu/cpu_peak.h>

#include <cstring>

#if defined(__linux__)
#include <cerrno>
#include <sched.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Pin the calling thread to a single logical core; returns 0, or the OS error
// when the core was refused (an Android cpuset that withholds it, a CPU taken
// offline).  Best-effort: keeps the per-core cache / bandwidth measurements
// stable and the single-thread rows on the core they name.  macOS (especially
// Apple Silicon) does not expose hard affinity, so it is a no-op there and the
// scheduler is trusted to keep a busy thread resident.
static int pinToCore(int core)
{
#if defined(__linux__)
  // sched_setaffinity(0, ...) pins the calling thread and works on both glibc
  // and Bionic (Android), unlike the glibc-only pthread_setaffinity_np.
  if (core < 0 || core >= CPU_SETSIZE)
    return EINVAL;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0 ? 0 : errno;
#elif defined(_WIN32)
  if (core < 0 || core >= 64)
    return 0;   // beyond group 0: left to the scheduler, as it always was
  return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1ull << core) ? 0
                                                                            : (int)GetLastError();
#else
  (void)core;   // macOS / other: advisory only.
  return 0;
#endif
}

CpuThreadPool::CpuThreadPool(int maxThreads, std::vector<int> cpuIds)
    : nMax(maxThreads < 1 ? 1 : maxThreads), pinIds(std::move(cpuIds)),
      pinError(new std::atomic<int>[(size_t)(maxThreads < 1 ? 1 : maxThreads)])
{
  for (int t = 0; t < nMax; t++)
    pinError[(size_t)t].store(0, std::memory_order_relaxed);
  workers.reserve(nMax);
  for (int t = 0; t < nMax; t++)
    workers.emplace_back([this, t] { workerLoop(t); });
}

CpuThreadPool::~CpuThreadPool()
{
  {
    std::unique_lock<std::mutex> lk(mtx);
    stop = true;
    generation++;          // wake everyone so they observe `stop`
  }
  cvStart.notify_all();
  for (auto &w : workers)
    if (w.joinable())
      w.join();
}

int CpuThreadPool::pinTarget(int tid) const
{
  return (size_t)tid < pinIds.size() ? pinIds[(size_t)tid] : tid;
}

std::vector<std::string> CpuThreadPool::pinFailures() const
{
  std::vector<std::string> out;
  for (int t = 0; t < nMax; t++)
  {
    int err = pinError[(size_t)t].load(std::memory_order_relaxed);
    if (!err)
      continue;
#if defined(__linux__)
    const std::string why = std::strerror(err);
#else
    const std::string why = "error " + std::to_string(err);
#endif
    out.push_back("cpu" + std::to_string(pinTarget(t)) + " (" + why + ")");
  }
  return out;
}

void CpuThreadPool::workerLoop(int tid)
{
  uint64_t myGen = 0;

  for (;;)
  {
    const std::function<void(int)> *localJob = nullptr;
    {
      std::unique_lock<std::mutex> lk(mtx);
      cvStart.wait(lk, [&] { return stop || generation != myGen; });
      if (stop)
        return;
      myGen = generation;
      if (tid < activeCount)
        localJob = job;     // valid until this dispatch completes (run() blocks)
    }

    if (localJob)
    {
      // Pin at every job, not once: a phone that pauses a core migrates its
      // threads off and can leave them unpinned, and a single-thread row must
      // not quietly measure whatever core the scheduler chose instead.  A
      // no-op syscall when the thread is already where it belongs.
      pinError[(size_t)tid].store(pinToCore(pinTarget(tid)), std::memory_order_relaxed);
      (*localJob)(tid);
      std::unique_lock<std::mutex> lk(mtx);
      if (--remaining == 0)
        cvDone.notify_one();
    }
  }
}

void CpuThreadPool::run(int n, const std::function<void(int)> &body)
{
  if (n < 1) n = 1;
  if (n > nMax) n = nMax;

  std::unique_lock<std::mutex> lk(mtx);
  job         = &body;
  activeCount = n;
  remaining   = n;
  generation++;
  cvStart.notify_all();
  cvDone.wait(lk, [&] { return remaining == 0; });
  job = nullptr;
}

#endif // ENABLE_CPU
