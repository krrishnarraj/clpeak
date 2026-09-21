#ifdef ENABLE_LITERT

#include "litert_runtime.h"

#include <common/common.h>
#include <common/dynlib.h>

#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <system_error>

#include "litert/c/internal/litert_runtime_c_api.h"   // LITERT_RUNTIME_ABI_VERSION

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

static std::mutex g_mutex;
static LitertRuntime g_rt;
static bool g_loaded = false;
static bool g_attempted = false;
static std::string g_loadError;
static std::string g_override;   // --litert-lib / clpeak_set_litert_library
static std::string g_npuDir;     // --litert-npu-dir
static std::string g_npuStage;   // clpeak_set_litert_npu_stage_dir (Android)
static std::string g_npuResolved; // the vendor directory staged there, if any

// Every loaded runtime stays mapped for the life of the process and is
// remembered under its override key ("" for the default search), so a
// settings screen that switches back reuses the handle.
static std::map<std::string, LitertRuntime> g_cache;

// The file a loaded handle came from, when the platform can say.  A bare
// soname resolves out of the APK's lib dir on Android and out of the pip
// wheel on desktops; the directory is where LiteRT expects to find its
// accelerator and dispatch libraries, so it has to be known.
static std::string resolvePath(void *lib, void *anySymbol)
{
#ifdef _WIN32
  (void)anySymbol;
  char buf[MAX_PATH * 4];
  const DWORD n = GetModuleFileNameA(reinterpret_cast<HMODULE>(lib), buf, sizeof buf);
  if (n == 0 || n >= sizeof buf)
    return "";
  return std::string(buf, n);
#else
  (void)lib;
  Dl_info info;
  if (anySymbol && dladdr(anySymbol, &info) && info.dli_fname)
    return info.dli_fname;
  return "";
#endif
}

static std::string dirOf(const std::string &path)
{
  const size_t s = path.find_last_of("/\\");
  if (s == std::string::npos)
    return "";
  return path.substr(0, s);
}

static bool resolveApi(void *lib, LitertApi &api, const char *what)
{
  std::string missing;
#define CLPEAK_LITERT_RESOLVE_REQ(name)                                        \
  api.name = reinterpret_cast<decltype(&::name)>(clpeak::dynSym(lib, #name));   \
  if (!api.name)                                                              \
    missing += missing.empty() ? #name : std::string(", ") + #name;
  CLPEAK_LITERT_REQUIRED(CLPEAK_LITERT_RESOLVE_REQ)
#undef CLPEAK_LITERT_RESOLVE_REQ
#define CLPEAK_LITERT_RESOLVE_OPT(name)                                        \
  api.name = reinterpret_cast<decltype(&::name)>(clpeak::dynSym(lib, #name));
  CLPEAK_LITERT_OPTIONAL(CLPEAK_LITERT_RESOLVE_OPT)
#undef CLPEAK_LITERT_RESOLVE_OPT
  if (!missing.empty())
  {
    g_loadError = std::string(what) + " lacks " + missing +
                  " -- not a LiteRT runtime this build can drive (compiled "
                  "against ABI " LITERT_RUNTIME_ABI_VERSION ")";
    return false;
  }
  return true;
}

static void loadRuntime()
{
  auto cached = g_cache.find(g_override);
  if (cached != g_cache.end())
  {
    g_rt = cached->second;
    g_loaded = true;
    return;
  }

  // A library the user named is the library to measure, and nothing else
  // will do: falling through to whatever else is installed would report one
  // runtime's numbers under another's name.  Absolute before the loader
  // sees it: a relative module path finds the file itself but not the
  // sibling libraries beside it (see clpeak::absoluteModulePath).
  const char *named = g_override.empty() ? nullptr : g_override.c_str();
  const std::string absNamed =
      named ? clpeak::absoluteModulePath(named) : std::string();

  void *lib = nullptr;
  if (named)
  {
    lib = clpeak::dynOpen({absNamed.c_str()});
    if (!lib)
      g_loadError = std::string("could not load LiteRT from '") + absNamed + "'";
  }
  else
  {
    lib = clpeak::dynOpen({
#if defined(_WIN32)
        "libLiteRt.dll",
        "LiteRt.dll",
#elif defined(__APPLE__) && TARGET_OS_IPHONE
        // iOS: the app's Frameworks directory, where the Runner's embed phase
        // puts Google's libLiteRt.dylib next to its Metal accelerator
        // (tools/build_ios_native.sh stages both).  dyld expands
        // @executable_path in a dlopen path, and dladdr below turns it into
        // the real directory the accelerator is then searched in.  Nothing
        // outside the signed bundle would load, so this is the whole list.
        "@executable_path/Frameworks/libLiteRt.dylib",
#elif defined(__APPLE__)
        "libLiteRt.dylib",
        "@executable_path/libLiteRt.dylib",
        "@loader_path/libLiteRt.dylib",
        "/opt/homebrew/lib/libLiteRt.dylib",
        "/usr/local/lib/libLiteRt.dylib",
#else
        // Android: the bare soname resolves out of the APK's lib dir.
        "libLiteRt.so",
#endif
    });
    if (!lib)
      g_loadError = "LiteRT library (libLiteRt) not found";
  }
  if (!lib)
    return;

  // A file that lacks the required entry points stays mapped like any
  // other handle.  This is the path that found the rule in common/dynlib.h:
  // the ai-edge-litert 2.2.0 wheel's libLiteRt.so was refused here, dlclosed,
  // and took the process down at exit through the static destructors it had
  // registered and could not retire.
  LitertRuntime cur;
  if (!resolveApi(lib, cur.api, named ? named : "libLiteRt"))
    return;

  cur.lib = lib;
  cur.path = named ? absNamed : resolvePath(lib, reinterpret_cast<void *>(cur.api.LiteRtCreateEnvironment));
  if (cur.path.empty())
    cur.path = resolvePath(lib, reinterpret_cast<void *>(cur.api.LiteRtCreateEnvironment));
  cur.libraryDir = dirOf(cur.path);
  cur.abiVersion = LITERT_RUNTIME_ABI_VERSION;

  // Deliberately never dlclosed: see the header.
  g_rt = cur;
  g_cache[g_override] = cur;
  g_loaded = true;
}

void litertSetLibraryOverride(const std::string &path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (path == g_override)
    return;
  g_override = path;
  g_loaded = false;
  g_attempted = false;
  g_rt = LitertRuntime{};
  g_loadError.clear();
}

void litertSetNpuDirOverride(const std::string &dir)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (dir.empty())
  {
    g_npuDir.clear();
    return;
  }
  // Absolute: LiteRT's own loader lists and loads from this directory, and
  // a relative one would name a different place after any directory change.
  // A directory is always a filesystem path (never a loader search token),
  // so this applies even to a bare name.
  std::error_code ec;
  const std::string abs = std::filesystem::absolute(dir, ec).string();
  g_npuDir = ec ? dir : abs;
}

std::string litertNpuDirOverride()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_npuDir;
}

void litertSetNpuStageDir(const std::string &dir)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_npuStage = dir;
}

std::string litertNpuStageDir()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_npuStage;
}

void litertSetNpuResolvedDir(const std::string &dir)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_npuResolved = dir;
}

std::string litertNpuDir()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_npuDir.empty())
    return g_npuDir;
  if (!g_npuResolved.empty())
    return g_npuResolved;
  return g_loaded ? g_rt.libraryDir : std::string();
}

const LitertRuntime *litertRuntime()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_attempted)
  {
    g_attempted = true;
    loadRuntime();
  }
  return g_loaded ? &g_rt : nullptr;
}

std::string litertLoadDiagnostic()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_loadError;
}

std::string litertStatusText(const LitertRuntime &rt, LiteRtStatus st)
{
  const char *s = rt.api.LiteRtGetStatusString ? rt.api.LiteRtGetStatusString(st) : nullptr;
  if (s && *s)
    return s;
  return "LiteRT status " + std::to_string((int)st);
}

#endif // ENABLE_LITERT
