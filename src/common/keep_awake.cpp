#include <common/keep_awake.h>
#include <common/common.h>

#if defined(_WIN32)
#define CLPEAK_KEEP_AWAKE_WINDOWS 1
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_OSX
#define CLPEAK_KEEP_AWAKE_MACOS 1
#endif
#elif defined(__linux__) && !defined(__ANDROID__)
#define CLPEAK_KEEP_AWAKE_LINUX 1
#endif

#if defined(CLPEAK_KEEP_AWAKE_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <powrprof.h>
#elif defined(CLPEAK_KEEP_AWAKE_MACOS)
#include <IOKit/pwr_mgt/IOPMLib.h>
#elif defined(CLPEAK_KEEP_AWAKE_LINUX)
#include <common/dynlib.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(CLPEAK_KEEP_AWAKE_WINDOWS)

namespace clpeak {

struct KeepAwake::Impl
{
    HANDLE request = INVALID_HANDLE_VALUE;
    bool   system  = false;
    bool   display = false;
};

KeepAwake::KeepAwake() : impl_(new Impl)
{
    // What `powercfg /requests` prints under the process; the call copies it.
    wchar_t reason[] = L"clpeak benchmark run in progress";
    REASON_CONTEXT context = {};
    context.Version                   = POWER_REQUEST_CONTEXT_VERSION;
    context.Flags                     = POWER_REQUEST_CONTEXT_SIMPLE_STRING;
    context.Reason.SimpleReasonString = reason;
    impl_->request = PowerCreateRequest(&context);
    if (impl_->request == INVALID_HANDLE_VALUE)
    {
        CLPEAK_VLOG("keep-awake: PowerCreateRequest failed (error %lu), so the "
                    "machine may sleep during the run\n", GetLastError());
        return;
    }

    DWORD error = 0;
    impl_->system = PowerSetRequest(impl_->request, PowerRequestSystemRequired) != 0;
    if (!impl_->system)
        error = GetLastError();

    // Modern Standby -- every Snapdragon laptop, most current x86 ones --
    // honours a system-required request on battery for only five minutes past
    // the sleep timeout (PowerSetRequest's documentation), and then suspends
    // every desktop process.  A display request is not cut short, and Windows
    // 11 does not sleep a machine whose display is held on, so there the run
    // keeps the display lit.  A traditional (S3) machine honours the system
    // request for as long as it is held, and its display keeps its own timeout.
    SYSTEM_POWER_CAPABILITIES caps = {};
    const bool modernStandby = GetPwrCapabilities(&caps) && caps.AoAc;
    if (modernStandby)
    {
        impl_->display = PowerSetRequest(impl_->request, PowerRequestDisplayRequired) != 0;
        if (!impl_->display && !error)
            error = GetLastError();
    }

    if (impl_->system || impl_->display)
        CLPEAK_VLOG("keep-awake: holding a %s power request%s\n",
                    impl_->system && impl_->display ? "system + display"
                    : impl_->system                 ? "system"
                                                    : "display",
                    modernStandby ? " (Modern Standby)" : "");
    if (error)
        CLPEAK_VLOG("keep-awake: PowerSetRequest failed (error %lu), so the "
                    "machine may sleep during the run\n", error);
}

KeepAwake::~KeepAwake()
{
    if (impl_->request == INVALID_HANDLE_VALUE)
        return;
    if (impl_->display)
        PowerClearRequest(impl_->request, PowerRequestDisplayRequired);
    if (impl_->system)
        PowerClearRequest(impl_->request, PowerRequestSystemRequired);
    CloseHandle(impl_->request);
}

} // namespace clpeak

#elif defined(CLPEAK_KEEP_AWAKE_MACOS)

namespace clpeak {

struct KeepAwake::Impl
{
    IOPMAssertionID assertion = kIOPMNullAssertionID;
};

KeepAwake::KeepAwake() : impl_(new Impl)
{
    // The assertion `caffeinate -i` takes: it holds on battery as on mains
    // (PreventSystemSleep, `-s`, is mains-only), and leaves the display to
    // sleep on its own timeout.  The name is what `pmset -g assertions` lists.
    const IOReturn r = IOPMAssertionCreateWithName(
        kIOPMAssertionTypePreventUserIdleSystemSleep, kIOPMAssertionLevelOn,
        CFSTR("clpeak benchmark run in progress"), &impl_->assertion);
    if (r != kIOReturnSuccess)
    {
        impl_->assertion = kIOPMNullAssertionID;
        CLPEAK_VLOG("keep-awake: IOPMAssertionCreateWithName failed (0x%x), so "
                    "the machine may sleep during the run\n",
                    static_cast<unsigned>(r));
        return;
    }
    CLPEAK_VLOG("keep-awake: holding a PreventUserIdleSystemSleep assertion\n");
}

KeepAwake::~KeepAwake()
{
    if (impl_->assertion != kIOPMNullAssertionID)
        IOPMAssertionRelease(impl_->assertion);
}

} // namespace clpeak

#elif defined(CLPEAK_KEEP_AWAKE_LINUX)

namespace {

// sd-bus, resolved at run time from libsystemd or elogind's copy of it:
// linking it would stop the binary from starting wherever it is absent
// (containers, distributions without systemd), and a run is merely better off
// with it.  These are its public declarations, ABI-stable since systemd 221.
struct sd_bus;
struct sd_bus_message;
struct sd_bus_error
{
    const char *name;
    const char *message;
    int         needFree;
};

struct SdBus
{
    int (*openSystem)(sd_bus **) = nullptr;
    int (*openUser)(sd_bus **) = nullptr;
    int (*callMethod)(sd_bus *, const char *, const char *, const char *,
                      const char *, sd_bus_error *, sd_bus_message **,
                      const char *, ...) = nullptr;
    int (*readMessage)(sd_bus_message *, const char *, ...) = nullptr;
    sd_bus_message *(*unrefMessage)(sd_bus_message *) = nullptr;
    void (*freeError)(sd_bus_error *) = nullptr;
    sd_bus *(*closeBus)(sd_bus *) = nullptr;
    // systemd 240+.  Without it, a call to a service that never answers waits
    // out sd-bus's 25 s default.
    int (*setTimeout)(sd_bus *, uint64_t) = nullptr;
    bool ok = false;
};

const SdBus &sdBus()
{
    static const SdBus api = [] {
        SdBus b;
        void *lib = clpeak::dynOpen({"libsystemd.so.0", "libelogind.so.0"});
        if (!lib)
            return b;
        bool ok = true;
#define CLPEAK_SD_SYM(member, name)                                             \
    b.member = reinterpret_cast<decltype(b.member)>(clpeak::dynSym(lib, name)); \
    ok = ok && (b.member != nullptr)
        CLPEAK_SD_SYM(openSystem, "sd_bus_open_system");
        CLPEAK_SD_SYM(openUser, "sd_bus_open_user");
        CLPEAK_SD_SYM(callMethod, "sd_bus_call_method");
        CLPEAK_SD_SYM(readMessage, "sd_bus_message_read");
        CLPEAK_SD_SYM(unrefMessage, "sd_bus_message_unref");
        CLPEAK_SD_SYM(freeError, "sd_bus_error_free");
        CLPEAK_SD_SYM(closeBus, "sd_bus_flush_close_unref");
#undef CLPEAK_SD_SYM
        b.setTimeout = reinterpret_cast<decltype(b.setTimeout)>(
            clpeak::dynSym(lib, "sd_bus_set_method_call_timeout"));
        b.ok = ok;
        return b;
    }();
    return api;
}

const char *const kWho = "clpeak";
const char *const kWhy = "benchmark run in progress";
constexpr unsigned kInhibitSuspend = 4;  // the GNOME session and portal flag

// Every service asked here answers at once or is not there to answer.
sd_bus *openBus(const SdBus &sd, bool user, std::string &why)
{
    sd_bus *bus = nullptr;
    const int r = user ? sd.openUser(&bus) : sd.openSystem(&bus);
    if (r < 0)
    {
        why = std::strerror(-r);
        return nullptr;
    }
    if (sd.setTimeout)
        sd.setTimeout(bus, 5ull * 1000 * 1000);  // microseconds
    return bus;
}

// What every call here does with its outcome: drop the reply it has no use
// for, and on failure say why.
bool settle(const SdBus &sd, int r, sd_bus_message *reply, sd_bus_error &error,
            std::string &why)
{
    if (reply)
        sd.unrefMessage(reply);
    if (r >= 0)
        return true;
    why = error.message ? error.message : error.name ? error.name : std::strerror(-r);
    sd.freeError(&error);
    return false;
}

// logind's sleep inhibitor, a descriptor held until closed.  "block-weak"
// (systemd 257+) refuses idle suspend and other users' requests but not this
// user's own; that is what "block" meant until 257 made it refuse the user
// too, and what an older logind -- which rejects the new mode -- still gives
// "block".
int takeLogindInhibitor(const SdBus &sd, std::string &mode, std::string &why)
{
    sd_bus *bus = openBus(sd, false, why);
    if (!bus)
        return -1;
    int fd = -1;
    for (const char *m : {"block-weak", "block"})
    {
        sd_bus_error error = {};
        sd_bus_message *reply = nullptr;
        const int r = sd.callMethod(bus, "org.freedesktop.login1",
                                    "/org/freedesktop/login1",
                                    "org.freedesktop.login1.Manager", "Inhibit",
                                    &error, &reply, "ssss", "sleep", kWho, kWhy, m);
        if (r < 0)
        {
            settle(sd, r, reply, error, why);
            continue;
        }
        // The descriptor belongs to the reply: keep a copy that outlives it.
        int replyFd = -1;
        const int rr = sd.readMessage(reply, "h", &replyFd);
        if (rr < 0)
            why = std::strerror(-rr);
        else if ((fd = fcntl(replyFd, F_DUPFD_CLOEXEC, 3)) < 0)
            why = std::strerror(errno);
        else
            mode = m;
        sd.unrefMessage(reply);
        break;
    }
    sd.closeBus(bus);
    return fd;
}

std::string joined(const std::vector<std::string> &items)
{
    std::string s;
    for (const auto &i : items)
        s += (s.empty() ? "" : ", ") + i;
    return s;
}

} // namespace

namespace clpeak {

// logind's inhibitor lives on a descriptor, the desktop's on the session-bus
// connection that asked for them; closing either ends what it holds, whether
// the destructor does it or the process exiting does.
struct KeepAwake::Impl
{
    int     logindFd = -1;
    sd_bus *session  = nullptr;
};

KeepAwake::KeepAwake() : impl_(new Impl)
{
    const SdBus &sd = sdBus();
    if (!sd.ok)
    {
        CLPEAK_VLOG("keep-awake: no libsystemd or libelogind to ask logind or "
                    "the desktop with, so the machine may sleep during the run\n");
        return;
    }
    std::vector<std::string> held, missed;
    std::string why;

    // Inside Flatpak neither logind nor the desktop's services are reachable,
    // and the Inhibit portal is how a sandboxed app asks.
    const bool flatpak = access("/.flatpak-info", F_OK) == 0;
    if (!flatpak)
    {
        std::string mode;
        impl_->logindFd = takeLogindInhibitor(sd, mode, why);
        if (impl_->logindFd >= 0)
            held.push_back("logind sleep inhibitor (" + mode + ")");
        else
            missed.push_back("logind (" + why + ")");
    }

    // The desktop's own idle suspend, asked in its own terms and for suspend
    // only, so the screen still blanks and locks as configured.  GNOME's
    // decides by its session manager's inhibitors alone; KDE's reads logind's
    // only in "block" mode.  A session normally has one of the two services,
    // and the missing one answers ServiceUnknown at once.
    sd_bus *bus = openBus(sd, true, why);
    if (!bus)
        missed.push_back("session bus (" + why + ")");
    else
    {
        const size_t before = held.size();
        sd_bus_error error = {};
        sd_bus_message *reply = nullptr;
        if (flatpak)
        {
            const int r = sd.callMethod(bus, "org.freedesktop.portal.Desktop",
                                        "/org/freedesktop/portal/desktop",
                                        "org.freedesktop.portal.Inhibit", "Inhibit",
                                        &error, &reply, "sua{sv}", "", kInhibitSuspend,
                                        1u, "reason", "s", kWhy);
            if (settle(sd, r, reply, error, why))
                held.push_back("Inhibit portal");
            else
                missed.push_back("Inhibit portal (" + why + ")");
        }
        else
        {
            int r = sd.callMethod(bus, "org.gnome.SessionManager",
                                  "/org/gnome/SessionManager",
                                  "org.gnome.SessionManager", "Inhibit", &error,
                                  &reply, "susu", kWho, 0u, kWhy, kInhibitSuspend);
            if (settle(sd, r, reply, error, why))
                held.push_back("GNOME session inhibitor");
            else
                missed.push_back("GNOME session manager (" + why + ")");

            error = {};
            reply = nullptr;
            r = sd.callMethod(bus, "org.freedesktop.PowerManagement",
                              "/org/freedesktop/PowerManagement/Inhibit",
                              "org.freedesktop.PowerManagement.Inhibit", "Inhibit",
                              &error, &reply, "ss", kWho, kWhy);
            if (settle(sd, r, reply, error, why))
                held.push_back("org.freedesktop.PowerManagement inhibitor");
            else
                missed.push_back("org.freedesktop.PowerManagement (" + why + ")");
        }
        if (held.size() > before)
            impl_->session = bus;
        else
            sd.closeBus(bus);
    }

    if (!held.empty())
        CLPEAK_VLOG("keep-awake: holding %s\n", joined(held).c_str());
    if (!missed.empty())
        CLPEAK_VLOG("keep-awake: not held: %s\n", joined(missed).c_str());
    if (held.empty())
        CLPEAK_VLOG("keep-awake: nothing held, so the machine may sleep during "
                    "the run\n");
}

KeepAwake::~KeepAwake()
{
    if (impl_->session)
        sdBus().closeBus(impl_->session);
    if (impl_->logindFd >= 0)
        close(impl_->logindFd);
}

} // namespace clpeak

#else

namespace clpeak {

// Android and iOS: the app holds the screen on itself (screen_wake.dart).
struct KeepAwake::Impl
{
};

KeepAwake::KeepAwake() {}

KeepAwake::~KeepAwake() = default;

} // namespace clpeak

#endif
