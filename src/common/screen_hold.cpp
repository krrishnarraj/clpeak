#include <common/screen_hold.h>
#include <common/common.h>
#include <common/run_log.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace clpeak {
namespace {

std::atomic<bool> onScreen{true};
constexpr auto kSlice = std::chrono::milliseconds(100);

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

void screenGate()
{
  if (onScreen.load() || cancelRequested())
    return;
  // The run's own line, in the scope of the test it stopped in.
  logMessage(LogLevel::Info, "", "held until the app is back on screen");
  while (!onScreen.load() && !cancelRequested())
    std::this_thread::sleep_for(kSlice);
}

} // namespace clpeak
