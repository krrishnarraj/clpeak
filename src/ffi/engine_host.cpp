// xpupeak-engine: the desktop app's engine process -- a loader for
// xpupeak_engine_main (xpupeak_ffi.h), and nothing else.
//
//   xpupeak-engine <xpupeak_ffi library> <catalog|launch> [...]
//
// The app names the library it resolved for itself, so this binary needs no
// rpath per layout (build tree, app bundle, `flutter run` dev loop) and no
// link-time dependency on the library.

#include <cstdio>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#include <string>
#include <vector>
#else
#include <dlfcn.h>
#endif

typedef int (*EngineMain)(int argc, const char **argv);

static const char kUsage[] =
    "usage: xpupeak-engine <xpupeak_ffi library> <catalog|launch> [...]\n";

#ifdef _WIN32

static std::string utf8(const wchar_t *w)
{
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1)
        return std::string();
    std::string s(static_cast<size_t>(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}

// A GUI-subsystem binary: started from the app, a console one would open a
// console window for every catalog and run.  stdin/stdout/stderr are still
// the pipes the app hands over.
int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
    int argc = 0;
    LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!wargv || argc < 3)
    {
        std::fputs(kUsage, stderr);
        return 64;
    }
    // The library's own dependencies resolve from its directory.
    HMODULE lib = LoadLibraryExW(wargv[1], nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!lib)
    {
        std::fprintf(stderr, "xpupeak-engine: cannot load %s (error %lu)\n",
                     utf8(wargv[1]).c_str(), GetLastError());
        return 69;
    }
    EngineMain engineMain =
        reinterpret_cast<EngineMain>(GetProcAddress(lib, "xpupeak_engine_main"));
    if (!engineMain)
    {
        std::fprintf(stderr, "xpupeak-engine: %s has no xpupeak_engine_main\n",
                     utf8(wargv[1]).c_str());
        return 69;
    }
    // xpupeak takes UTF-8, as it does from the app's own FFI calls.
    std::vector<std::string> args;
    for (int i = 2; i < argc; i++)
        args.push_back(utf8(wargv[i]));
    LocalFree(wargv);
    std::vector<const char *> argv;
    for (const auto &a : args)
        argv.push_back(a.c_str());
    return engineMain(static_cast<int>(argv.size()), argv.data());
}

#else

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fputs(kUsage, stderr);
        return 64;
    }
    // Global, as the CLI has its runtimes: nothing in the engine should see
    // a narrower symbol scope than the binary it stands in for.
    void *lib = dlopen(argv[1], RTLD_LAZY | RTLD_GLOBAL);
    if (!lib)
    {
        std::fprintf(stderr, "xpupeak-engine: cannot load %s: %s\n", argv[1], dlerror());
        return 69;
    }
    EngineMain engineMain = reinterpret_cast<EngineMain>(dlsym(lib, "xpupeak_engine_main"));
    if (!engineMain)
    {
        std::fprintf(stderr, "xpupeak-engine: %s has no xpupeak_engine_main\n", argv[1]);
        return 69;
    }
    return engineMain(argc - 2, const_cast<const char **>(argv + 2));
}

#endif
