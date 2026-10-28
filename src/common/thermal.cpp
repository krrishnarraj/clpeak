#include <common/thermal.h>
#include <common/common.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#if defined(__ANDROID__)
#include <dlfcn.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if defined(__APPLE__) && TARGET_OS_IPHONE
// NSProcessInfo.thermalState as an int (thermal_ios.mm).
int clpeakIosThermalState();
#endif

namespace clpeak {
namespace {

// Android's ATHERMAL_STATUS_* values, the scale iOS's states map onto.
enum Heat : int
{
  kUnknown = -1,
  kNone = 0,
  kLight,
  kModerate,
  kSevere,
  kCritical,
  kEmergency,
  kShutdown
};

// Severe is where Android's UX is "largely impacted" and iOS's "serious";
// light, where neither throttles enough to matter.
constexpr int kPauseAt = kSevere;
constexpr int kResumeAt = kLight;
constexpr double kMaxPauseS = 300.0;
constexpr auto kPoll = std::chrono::seconds(2);
constexpr auto kSlice = std::chrono::milliseconds(100);

const char *heatName(int h)
{
  static const char *const names[] = {"none", "light", "moderate", "severe", "critical", "emergency", "shutdown"};
  return (h >= kNone && h <= kShutdown) ? names[h] : "unknown";
}

int readHeat()
{
#if defined(__ANDROID__)
  struct AThermalManager;
  using Acquire = AThermalManager *(*)();
  using Status = int (*)(AThermalManager *);
  static AThermalManager *manager = nullptr;
  static Status status = nullptr;
  static bool opened = false;
  if (!opened)
  {
    opened = true;
    if (void *lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL))
    {
      auto acquire = reinterpret_cast<Acquire>(dlsym(lib, "AThermal_acquireManager"));
      status = reinterpret_cast<Status>(dlsym(lib, "AThermal_getCurrentThermalStatus"));
      if (acquire && status)
        manager = acquire();
    }
  }
  if (!manager)
    return kUnknown;
  const int s = status(manager);
  return s < kNone ? kUnknown : std::min(s, (int)kShutdown);
#elif defined(__APPLE__) && TARGET_OS_IPHONE
  switch (clpeakIosThermalState())
  {
  case 0: return kNone;      // nominal
  case 1: return kLight;     // fair
  case 2: return kSevere;    // serious
  case 3: return kCritical;  // critical
  default: return kUnknown;
  }
#else
  return kUnknown;
#endif
}

std::atomic<bool> watching{false};
int lastHeat = kUnknown;   // the run's thread is the only one at a boundary

// clpeak's own lines, so the CLI prints the pause inline like any warning.
// Each starts "thermal status ", which is how the app tells a pause
// (a warning) from its end (BenchmarkService.coolingDown).
void note(LogLevel level, const std::string &message)
{
  logMessage(level, "", message);
}

std::string seconds(double s)
{
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.0f s", s);
  return buf;
}

} // namespace

ThermalWatch::ThermalWatch()
{
  lastHeat = kUnknown;
  watching = true;
}

ThermalWatch::~ThermalWatch()
{
  watching = false;
}

void thermalGate(const std::function<void()> &beforePausing)
{
  if (!watching)
    return;
  int h = readHeat();
  if (h == kUnknown)
    return;
  if (h != lastHeat)
    note(LogLevel::Info, std::string("thermal status ") + heatName(h));
  lastHeat = h;
  if (h < kPauseAt)
    return;

  if (beforePausing)
    beforePausing();
  note(LogLevel::Warning, std::string("thermal status ") + heatName(h) + ": pausing until the device cools to " +
                              heatName(kResumeAt) + ", for at most " + seconds(kMaxPauseS));
  const auto t0 = std::chrono::steady_clock::now();
  auto waitedS = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  while (h > kResumeAt && !cancelRequested() && waitedS() < kMaxPauseS)
  {
    for (auto slept = std::chrono::milliseconds(0); slept < kPoll && !cancelRequested(); slept += kSlice)
      std::this_thread::sleep_for(kSlice);
    const int now = readHeat();
    if (now != kUnknown)
      h = now;
  }
  lastHeat = h;
  if (cancelRequested())
    return;
  if (h <= kResumeAt)
    note(LogLevel::Info, std::string("thermal status ") + heatName(h) + " after " + seconds(waitedS()) +
                             ": resuming");
  else
    note(LogLevel::Info, std::string("thermal status still ") + heatName(h) + " after " + seconds(waitedS()) +
                             ": resuming anyway");
}

} // namespace clpeak
