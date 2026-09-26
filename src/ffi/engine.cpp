// clpeak_engine_main: the desktop app's engine process (clpeak_ffi.h has the
// protocol).
//
// The desktop app never enumerates or runs in its own process.  That process
// holds a GUI toolkit, a GL driver and whatever those load, and a vendor
// runtime loaded beside them can break on it.  The case that moved this out:
// on a Radeon 890M with a user-local ROCm, HIP's first stream compiles its
// blit kernels with LLVM, and inside the GUI that compile failed LLVM's own
// verifier ("Attribute list does not match Module context!") while the CLI
// was fine.  Mesa's radeonsi, drawing the app's window, had already loaded
// the system LLVM.  A process of its own gives each catalog and run what the
// CLI has: no toolkit, a main-thread stack, a runtime setup loaded fresh, and
// a native crash that ends the run instead of the app.

#include "clpeak_ffi.h"
#include "launch.h"

#include <common/common.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace
{

// The app reads events off stdout, one JSON document per line, so nothing
// else may reach it.  The sink takes stdout for itself and points the
// process's stdout at stderr, where a library's stray printf lands instead,
// and where the app forwards it to its own stderr.
class EventSink
{
public:
    bool open()
    {
        std::fflush(stdout);
#ifdef _WIN32
        HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        if (out == nullptr || out == INVALID_HANDLE_VALUE)
            return false;
        // Not inheritable: a process some runtime starts must not hold the
        // app's pipe open past this one's exit.
        if (!DuplicateHandle(GetCurrentProcess(), out, GetCurrentProcess(),
                             &handle_, 0, FALSE, DUPLICATE_SAME_ACCESS))
            return false;
        SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
        // Closes the CRT's copy of the original, the last inheritable one.
        (void)_dup2(_fileno(stderr), _fileno(stdout));
        return true;
#else
        // Close-on-exec for the same reason as above.
        fd_ = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, 3);
        if (fd_ < 0)
            return false;
        if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
        {
            close(fd_);
            fd_ = -1;
            return false;
        }
        // A write to an app that has gone must fail, not kill the process
        // before -o has saved what ran.
        std::signal(SIGPIPE, SIG_IGN);
        return true;
#endif
    }

    // One event, whole, as one line.  Vendor runtimes emit `log` events from
    // their own threads, hence the lock.
    void emit(const char *json)
    {
        std::string line(json);
        line += '\n';
        std::lock_guard<std::mutex> lock(mutex_);
        if (broken_)
            return;
        if (!writeAll(line.data(), line.size()))
        {
            // The app is gone: stop at the next test boundary, which also
            // saves what ran (-o).
            broken_ = true;
            clpeak::requestCancel();
        }
    }

private:
    bool writeAll(const char *p, size_t n)
    {
        while (n > 0)
        {
#ifdef _WIN32
            DWORD wrote = 0;
            const DWORD chunk = n > 0x40000000u ? 0x40000000u : static_cast<DWORD>(n);
            if (!WriteFile(handle_, p, chunk, &wrote, nullptr))
                return false;
#else
            const ssize_t wrote = write(fd_, p, n);
            if (wrote < 0)
            {
                if (errno == EINTR)
                    continue;
                return false;
            }
#endif
            p += wrote;
            n -= static_cast<size_t>(wrote);
        }
        return true;
    }

    std::mutex mutex_;
    bool broken_ = false;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

void onEvent(void *userData, char *json)
{
    if (!json)
        return;
    static_cast<EventSink *>(userData)->emit(json);
    clpeak_free_string(json);
}

// Cancellation, from the app: a "cancel" line, or stdin closing.  A closed
// stdin means the app has gone, and without it the run would go on alone to
// the end.  Either way the run stops at the next test boundary with -o
// saving what ran, as a cancel in the app's own process did.
void watchStdin()
{
    char line[256];
    while (std::fgets(line, sizeof line, stdin))
        if (std::strncmp(line, "cancel", 6) == 0)
            clpeak::requestCancel();
    clpeak::requestCancel();
}

std::string take(char *s, const char *fallback)
{
    std::string out = s ? s : fallback;
    clpeak_free_string(s);
    return out;
}

// Everything this process was for is on disk and down the pipe by now, so it
// ends without static destructors or library finalizers: tearing down
// runtimes is where ONNX Runtime and LiteRT crash, and a crash report for a
// run that finished would say the opposite of what happened.
[[noreturn]] void endProcess(int code)
{
    std::fflush(stderr);
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#endif
    _exit(code);
}

int usage(const char *why)
{
    std::fprintf(stderr,
                 "clpeak-engine: %s\n"
                 "usage: clpeak-engine <clpeak_ffi library> <catalog|launch> "
                 "[--set-onnx-library PATH] [--set-onnx-ep [!]NAME=PATH]... "
                 "[--set-onnx-winml PATH] [--set-litert-library PATH] "
                 "[-- run arguments]\n",
                 why);
    return 64;  // EX_USAGE
}

} // namespace

int clpeak_engine_main(int argc, const char **argv)
{
    if (argc < 1 || !argv || !argv[0])
        return usage("no mode given");
    const std::string mode = argv[0];
    if (mode != "catalog" && mode != "launch")
        return usage(("unknown mode '" + mode + "'").c_str());

    // The runtime setup, through the same setters the in-process app calls,
    // so it stays out of the run's argv -- which the saved document records.
    std::string epSpec;
    int i = 1;
    for (; i < argc; i++)
    {
        const char *a = argv[i];
        if (!std::strcmp(a, "--"))
        {
            i++;
            break;
        }
        if (i + 1 >= argc)
            return usage((std::string("missing value for ") + a).c_str());
        const char *v = argv[++i];
        if (!std::strcmp(a, "--set-onnx-library"))
            clpeak_set_onnx_library(v);
        else if (!std::strcmp(a, "--set-onnx-ep"))
            epSpec += (epSpec.empty() ? "" : "\n") + std::string(v);
        else if (!std::strcmp(a, "--set-onnx-winml"))
            clpeak_set_onnx_winml(1, v);
        else if (!std::strcmp(a, "--set-litert-library"))
            clpeak_set_litert_library(v);
        else
            return usage((std::string("unknown option ") + a).c_str());
    }
    if (!epSpec.empty())
        clpeak_set_onnx_ep_libraries(epSpec.c_str());

    EventSink sink;
    if (!sink.open())
    {
        std::fprintf(stderr, "clpeak-engine: no stdout to send events on\n");
        return 70;  // EX_SOFTWARE
    }

    if (mode == "catalog")
    {
        // The statuses after the enumeration, which is what loads the
        // runtimes they describe.
        std::string doc = "{\"t\":\"catalog\",\"catalog\":";
        doc += take(clpeak_copy_backend_catalog_json(), "{\"backends\":[]}");
        doc += ",\"onnx\":" + take(clpeak_copy_onnx_status_json(), "{}");
        doc += ",\"litert\":" + take(clpeak_copy_litert_status_json(), "{}");
        doc += "}";
        sink.emit(doc.c_str());
        endProcess(0);
    }

    std::thread(watchStdin).detach();
    std::vector<const char *> run{"clpeak"};
    for (; i < argc; i++)
        run.push_back(argv[i]);
    const int rc = clpeakLaunch(static_cast<int>(run.size()), run.data(),
                                &onEvent, &sink, /*resetCancel=*/false);
    // The status itself travels on the `done` event; the exit code only says
    // whether there was one worth reading.
    endProcess(rc == 0 ? 0 : 1);
}
