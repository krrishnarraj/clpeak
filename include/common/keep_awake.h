#ifndef CLPEAK_KEEP_AWAKE_H
#define CLPEAK_KEEP_AWAKE_H

#include <memory>

// Holds the machine out of idle sleep for as long as it lives: one per run,
// in the CLI's run loop and in the GUI bridge's alike.
//
// A run is minutes to hours of compute with no keyboard or mouse input, which
// is exactly what an OS idle timer waits for -- none of them counts CPU or GPU
// load as activity.  A machine that sleeps partway through ends the run (a
// Windows Modern Standby laptop suspends every desktop process), and one that
// resumes it has a test whose timings straddle the gap.
//
// Only idle sleep is held off: closing the lid or choosing Sleep still sleeps
// the machine.  Per platform:
//
//   Windows  a power request, system-required -- and display-required on a
//            Modern Standby machine, which on battery honours a system-only
//            request for just five minutes past the sleep timeout.  Listed by
//            `powercfg /requests`.
//   macOS    an IOKit PreventUserIdleSystemSleep assertion; the display may
//            still sleep.  Listed by `pmset -g assertions`.
//   Linux    a logind sleep inhibitor (`systemd-inhibit --list`), which holds
//            off logind's own idle action; and the desktop's, since a desktop
//            suspends on idle by its own inhibitors -- GNOME and its forks by
//            their session manager's, KDE and Xfce by
//            org.freedesktop.PowerManagement's.  Inside Flatpak, the Inhibit
//            portal instead.
//   Android and iOS  nothing: the app holds the screen on for itself
//            (app/lib/src/services/screen_wake.dart).
//
// Best-effort throughout.  A platform that refuses costs the run a --verbose
// line and nothing else, and what was taken is released with this object --
// or by the OS when the process dies, so a crashed run cannot leave the
// machine unable to sleep.

namespace clpeak {

class KeepAwake
{
public:
    KeepAwake();
    ~KeepAwake();

    KeepAwake(const KeepAwake &) = delete;
    KeepAwake &operator=(const KeepAwake &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace clpeak

#endif // CLPEAK_KEEP_AWAKE_H
