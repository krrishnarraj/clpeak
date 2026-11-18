#ifdef ENABLE_ONNX

// _GNU_SOURCE must precede the first libc header for glibc to expose
// cpu_set_t and the CPU_* macros; harmless on Bionic (Android).
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "onnx_threads.h"
#include "onnx_model.h"
#include "onnx_session.h"

#include <onnx/onnx_peak.h>

#include <algorithm>
#include <string>
#include <vector>

#if defined(__linux__)
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sched.h>
#include <sys/types.h>
#endif

namespace
{

#if defined(__linux__)

  // This process's threads, by id, sorted.
  std::vector<pid_t> threadIds()
  {
    std::vector<pid_t> ids;
    if (DIR *d = opendir("/proc/self/task"))
    {
      while (const dirent *e = readdir(d))
      {
        char *end = nullptr;
        const long id = std::strtol(e->d_name, &end, 10);
        if (end != e->d_name && *end == '\0' && id > 0)
          ids.push_back((pid_t)id);
      }
      closedir(d);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
  }

  std::vector<int> cpusIn(const cpu_set_t &s)
  {
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; c++)
      if (CPU_ISSET(c, &s))
        cpus.push_back(c);
    return cpus;
  }

  // "0-3,16-19" for a sorted list.
  std::string cpuRanges(const std::vector<int> &cpus)
  {
    std::string out;
    for (size_t i = 0; i < cpus.size();)
    {
      size_t j = i;
      while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1)
        j++;
      if (!out.empty())
        out += ",";
      out += std::to_string(cpus[i]);
      if (j > i)
        out += (j == i + 1 ? "," : "-") + std::to_string(cpus[j]);
      i = j + 1;
    }
    return out;
  }

  bool sameCpus(const cpu_set_t &a, const cpu_set_t &b)
  {
    for (int c = 0; c < CPU_SETSIZE; c++)
      if (CPU_ISSET(c, &a) != CPU_ISSET(c, &b))
        return false;
    return true;
  }

#endif

} // namespace

OnnxCallerPin::OnnxCallerPin(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
#if defined(__linux__)
  if (ep.deviceType != DeviceType::Cpu)
    return;
  cpu_set_t own;
  CPU_ZERO(&own);
  if (sched_getaffinity(0, sizeof own, &own) != 0)
    return;
  const char *key = ep.providerKey.c_str();

  // Every session builds its own pool with the same defaults, so one trivial
  // session shows where the runtime puts the threads of every session on
  // this provider.  It creates them while it builds the session, and pins
  // each as it creates it.
  const std::vector<pid_t> before = threadIds();
  OnnxSessionResult s = onnxCreateSession(rt, ep, onnxTrivialMulModel(),
                                          /*keepConstantsUnfolded=*/false,
                                          /*profile=*/false,
                                          /*keepQdqUnfused=*/false,
                                          /*verifyPlacement=*/false);
  if (!s.session)
  {
    XPUPEAK_VLOG("onnx[%s]: no session to find the thread pool from (%s); the "
                 "calling thread is left to the scheduler\n",
                 key, s.error.empty() ? "no reason given" : s.error.c_str());
    return;
  }
  // A thread whose mask is this thread's inherited it: the runtime pinned it
  // nowhere.
  int created = 0, pinned = 0;
  cpu_set_t taken;
  CPU_ZERO(&taken);
  for (pid_t id : threadIds())
  {
    if (std::binary_search(before.begin(), before.end(), id))
      continue;
    created++;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(id, sizeof mask, &mask) != 0 || sameCpus(mask, own))
      continue;
    pinned++;
    for (int c = 0; c < CPU_SETSIZE; c++)
      if (CPU_ISSET(c, &mask))
        CPU_SET(c, &taken);
  }
  rt.api->ReleaseSession(s.session);

  const std::vector<int> mine = cpusIn(own);
  std::vector<int> left;
  for (int c : mine)
    if (!CPU_ISSET(c, &taken))
      left.push_back(c);
  if (pinned == 0)
  {
    XPUPEAK_VLOG("onnx[%s]: ONNX Runtime pinned none of the %d threads its "
                 "pool created; the calling thread is left to the scheduler\n",
                 key, created);
    return;
  }
  const std::string takenList = cpuRanges(cpusIn(taken));
  if (left.empty())
  {
    XPUPEAK_LOG(Info,
                "onnx[%s]: ONNX Runtime pinned %d threads of its pool to CPUs "
                "%s, which covers every CPU the calling thread may use (%s), "
                "so that thread shares a core with one of them",
                key, pinned, takenList.c_str(), cpuRanges(mine).c_str());
    return;
  }
  // Only free CPUs already: nothing to move.
  if (left.size() == mine.size())
  {
    XPUPEAK_VLOG("onnx[%s]: ONNX Runtime pinned %d threads of its pool to CPUs "
                 "%s, none of which the calling thread may use (%s)\n",
                 key, pinned, takenList.c_str(), cpuRanges(mine).c_str());
    return;
  }

  cpu_set_t pin;
  CPU_ZERO(&pin);
  for (int c : left)
    CPU_SET(c, &pin);
  if (sched_setaffinity(0, sizeof pin, &pin) != 0)
  {
    XPUPEAK_LOG(Info,
                "onnx[%s]: ONNX Runtime pinned %d threads of its pool to CPUs "
                "%s; the calling thread could not be pinned to %s, the CPUs "
                "they leave (%s), so it may share a core with one of them",
                key, pinned, takenList.c_str(), cpuRanges(left).c_str(),
                std::strerror(errno));
    return;
  }
  restore = mine;
  XPUPEAK_LOG(Info,
              "onnx[%s]: ONNX Runtime pinned %d threads of its pool to CPUs %s "
              "and left %s for the thread that calls it, which runs this "
              "provider's tests there",
              key, pinned, takenList.c_str(), cpuRanges(left).c_str());
#else
  (void)rt;
  (void)ep;
#endif
}

OnnxCallerPin::~OnnxCallerPin()
{
#if defined(__linux__)
  if (restore.empty())
    return;
  cpu_set_t own;
  CPU_ZERO(&own);
  for (int c : restore)
    CPU_SET(c, &own);
  (void)sched_setaffinity(0, sizeof own, &own);
#endif
}

#endif // ENABLE_ONNX
