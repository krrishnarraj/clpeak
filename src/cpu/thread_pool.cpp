#ifdef ENABLE_CPU

// _GNU_SOURCE must precede the first libc header for glibc to expose cpu_set_t /
// CPU_SET; harmless on Bionic (Android).  Defined before cpu_peak.h, which pulls
// in <thread>/<mutex> and therefore libc headers.
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <cpu/cpu_peak.h>

#if defined(__linux__)
#include <sched.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Pin the calling thread to a single logical core.  Best-effort: keeps the
// per-core cache / bandwidth measurements stable.  macOS (especially Apple
// Silicon) does not expose hard affinity, so it is a no-op there and the
// scheduler is trusted to keep a busy thread resident.
static void pinToCore(int core)
{
#if defined(__linux__)
  // sched_setaffinity(0, ...) pins the calling thread and works on both glibc
  // and Bionic (Android), unlike the glibc-only pthread_setaffinity_np.
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  sched_setaffinity(0, sizeof(set), &set);
#elif defined(_WIN32)
  if (core < 64)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1ull << core);
#else
  (void)core;   // macOS / other: advisory only.
#endif
}

CpuThreadPool::CpuThreadPool(int maxThreads)
    : nMax(maxThreads < 1 ? 1 : maxThreads)
{
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

void CpuThreadPool::workerLoop(int tid)
{
  pinToCore(tid);
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
