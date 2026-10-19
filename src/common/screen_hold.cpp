#include <common/screen_hold.h>
#include <common/common.h>
#include <common/run_log.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace clpeak {
namespace {

std::atomic<bool> onScreen{true};
std::atomic<bool> paused{false};
constexpr auto kSlice = std::chrono::milliseconds(100);

// What the run waits for at a boundary, or null when it goes on.
const char *holdReason()
{
  if (paused.load())
    return "held until the run is resumed";
  if (!onScreen.load())
    return "held until the app is back on screen";
  return nullptr;
}

} // namespace

void setHostOnScreen(bool on)
{
  if (onScreen.exchange(on) == on)
    return;
  // Called on the app's thread, not the run's, so these go on the run's log
  // by the route that is safe from there; outside a run they go nowhere.
  if (on)
    RunLog::hostNote(LogLevel::Info, "app", "the app is back on screen");
  else
    RunLog::hostNote(LogLevel::Warning, "app",
                     "the app left the screen: the run holds at its next reading until the app is back, "
                     "and the phone may end the app meanwhile");
}

void setRunPaused(bool on)
{
  if (paused.exchange(on) == on)
    return;
  if (on)
    RunLog::hostNote(LogLevel::Info, "app",
                     "the run was paused: it holds at its next reading until it is resumed");
  else
    RunLog::hostNote(LogLevel::Info, "app", "the run was resumed after a pause");
}

void holdGate()
{
  // The run's own line, in the scope of the test it stopped in -- again
  // when what it waits for changes.
  const char *said = nullptr;
  while (!cancelRequested())
  {
    const char *why = holdReason();
    if (!why)
      return;
    if (why != said)
      logMessage(LogLevel::Info, "", why);
    said = why;
    std::this_thread::sleep_for(kSlice);
  }
}

} // namespace clpeak
