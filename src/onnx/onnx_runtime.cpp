#ifdef ENABLE_ONNX

#include "onnx_runtime.h"

#include <common/common.h>
#include <common/dynlib.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Oldest OrtApi we are prepared to speak.  Every entry point this backend
// calls existed well before this; requesting downwards from ORT_API_VERSION
// lets a binary built against a new header run on an older installed runtime.
static const uint32_t kMinApiVersion = 17;  // ONNX Runtime 1.17 (2024)

static std::mutex g_mutex;
static bool       g_attempted = false;  // a failed search is not retried
static std::string g_loadError;         // why the last attempt failed

// The runtime ortRuntime() hands out, once one has loaded.  Written by the
// first successful load and never again: from then on the setup is fixed
// (onnx_runtime.h), so a pointer to it stays valid and unchanged for the
// life of the process.
static OrtRuntime g_loaded;
static const OrtRuntime *g_rt = nullptr;

// The runtime setup (onnx_runtime.h): what was last asked for, and what is
// in effect -- the same until a runtime has loaded (g_rt), `active` frozen
// after.  The library is ignored on a statically linked build.
struct Setup
{
  std::string library;
  bool winml = false;
  std::string winmlPath;
};
static Setup g_requested;
static Setup g_active;
static uint64_t g_winmlGeneration = 1;

void onnxSetWinml(bool enabled, const std::string &path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_requested.winml     = enabled;
  g_requested.winmlPath = path;
  if (g_rt || (g_active.winml == enabled && g_active.winmlPath == path))
    return;
  g_active.winml     = enabled;
  g_active.winmlPath = path;
  g_winmlGeneration++;
  // The catalog's directory steers the default search (winmlDefaultRuntime),
  // so a runtime that did not load may load now: look again.
  g_attempted = false;
  g_loadError.clear();
}

bool onnxWinmlEnabled()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_active.winml;
}

std::string onnxWinmlPathHint()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_active.winmlPath;
}

uint64_t onnxWinmlGeneration()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_winmlGeneration;
}

#ifdef _WIN32
// Let a runtime loaded by path find the DLLs it ships with.  A dependency
// linked at build time is resolved beside the DLL that needs it, but ONNX
// Runtime *delay-loads* DirectML.dll (and d3d12/dxgi), and a delay-load is
// a plain LoadLibrary by name when the provider is first attached -- exe
// directory, system, current directory, PATH, never the runtime's own
// directory.  So a DirectML-capable onnxruntime.dll picked from a package
// directory (the Windows ML runtime, the DirectML NuGet) attached its
// provider, went looking for DirectML.dll, and a delay-load with nothing
// to find is a structured exception, not a refusal: the process died in
// the viability probe.  SetDllDirectory puts the runtime's directory in
// the current-directory slot of that search, for the whole process, which
// is where every other backend's libraries live anyway (system32).
static void searchBesideRuntime(const std::string &path)
{
  const size_t slash = path.find_last_of("/\\");
  if (slash == std::string::npos)
    return;
  // Absolute, so a directory given relative to where clpeak was started
  // still names the same place after the current directory changes.
  std::error_code ec;
  const std::filesystem::path dir =
      std::filesystem::absolute(path.substr(0, slash), ec);
  if (!ec)
    SetDllDirectoryA(dir.string().c_str());
}

// Absolute filesystem path of the default runtime, bypassing the loader's
// already-loaded-module cache.  LoadLibrary("onnxruntime.dll") with a bare
// name returns the first-loaded module of that basename -- after one or more
// custom DLLs (usually also named onnxruntime.dll) are mapped that is a
// custom build, not the system one, so "Use default" would stick on (or fall
// back to) a previous pick.  SearchPath searches the filesystem only, so the
// absolute path it returns loads the file it names.  Empty when no system
// runtime is on the search path.
static std::string resolveWindowsDefaultAbsolute()
{
  std::string buf(32768, '\0');
  DWORD n = SearchPathA(NULL, "onnxruntime.dll", NULL,
                        static_cast<DWORD>(buf.size()), buf.data(), NULL);
  if (n == 0 || n >= buf.size())
    return "";
  buf.resize(n);
  return buf;
}
#endif

// Fill `out` from an OrtApiBase, whether it came from dlsym or a direct call.
// Returns false when the runtime serves no API version this build can speak.
static bool adoptApiBase(const OrtApiBase *base, OrtRuntime &out)
{
  if (!base)
    return false;

  // Ask for the highest API this runtime can actually serve.  ORT numbers its
  // API after its own minor version (1.23.x serves API 23), so the version
  // string gives the right answer in one call -- and asking for anything
  // higher makes ORT print "The requested API version [N] is not available"
  // to the console, once per attempt, below any log level we control.  Simply
  // counting down from ORT_API_VERSION produced a wall of those lines against
  // every older runtime.
  const char *verStr = base->GetVersionString();
  uint32_t wanted = ORT_API_VERSION;
  if (verStr)
  {
    unsigned major = 0, minor = 0;
#ifdef _MSC_VER
    if (sscanf_s(verStr, "%u.%u", &major, &minor) == 2 && major == 1 &&
        minor < wanted)
#else
    if (sscanf(verStr, "%u.%u", &major, &minor) == 2 && major == 1 &&
        minor < wanted)
#endif
      wanted = minor;
  }

  const OrtApi *api = nullptr;
  uint32_t version = wanted;
  for (; version >= kMinApiVersion; version--)
  {
    api = base->GetApi(version);
    if (api)
      break;
  }
  if (!api)
  {
    g_loadError = "onnxruntime " + std::string(base->GetVersionString()) +
                  " exposes no OrtApi in [" + std::to_string(kMinApiVersion) +
                  ", " + std::to_string((unsigned)ORT_API_VERSION) + "]";
    return false;
  }

  out.base          = base;
  out.api           = api;
  out.apiVersion    = version;
  out.versionString = base->GetVersionString();
  return true;
}

#ifdef CLPEAK_ONNX_STATIC

// Statically linked (iOS): the runtime is already in the binary, and there
// is no library to name.
static void loadRuntime()
{
  if (adoptApiBase(OrtGetApiBase(), g_loaded))
    g_rt = &g_loaded;
  else
    g_loadError = "the linked-in ONNX Runtime exposes no usable OrtApi";
}

void onnxSetLibraryOverride(const std::string &) {}

#else

// Every handle a load attempt opened, usable runtime or not: none is ever
// closed (common/dynlib.h), and on Windows a bare-name load can hand one of
// them back (loadRuntime).
static std::vector<void *> g_opened;

#ifdef _WIN32
// The onnxruntime.dll beside the Windows ML catalog DLL, when the catalog is
// on, was given its DLL or directory, and nobody named a runtime.  Microsoft's
// package puts the two side by side, and that runtime is the one its
// providers were built and tested against -- so it is the default
// candidate then, ahead of whatever else the search path holds.  Empty
// otherwise, or when there is no runtime beside the catalog.
static std::string winmlDefaultRuntime(const Setup &setup)
{
  if (!setup.winml)
    return "";
  std::string hint = setup.winmlPath;
  if (hint.empty())
    return "";
  const DWORD attrs = GetFileAttributesA(hint.c_str());
  const bool isDir = attrs != INVALID_FILE_ATTRIBUTES &&
                     (attrs & FILE_ATTRIBUTE_DIRECTORY);
  if (!isDir)
  {
    const size_t slash = hint.find_last_of("/\\");
    hint = slash == std::string::npos ? "" : hint.substr(0, slash);
  }
  if (hint.empty())
    return "";
  const std::string beside = hint + "\\onnxruntime.dll";
  if (GetFileAttributesA(beside.c_str()) == INVALID_FILE_ATTRIBUTES)
    return "";
  // Absolute: a relative module path finds the file itself against the
  // current directory but not the sibling libraries beside it, so the same
  // file fails to load by relative path and succeeds by absolute one.
  // Every other path LoadLibrary gets here is absolute already.
  std::error_code ec;
  const std::string absBeside = std::filesystem::absolute(beside, ec).string();
  return ec ? beside : absBeside;
}
#endif

// The file a loaded handle came from, when the platform can say.  A bare
// soname resolves out of the APK's lib dir on Android and out of a wheel or
// Homebrew prefix on desktops; reporting it is what lets a settings screen
// show which runtime is actually loaded, the way the LiteRT backend does.
static std::string resolveLoadedPath(void *lib, void *anySymbol)
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

static bool opened(void *lib)
{
  return std::find(g_opened.begin(), g_opened.end(), lib) != g_opened.end();
}

static void loadRuntime()
{
  // A library the user named -- by --onnx-lib or by the FFI setter -- is the
  // library to measure, and nothing else will do.  The conventional names are
  // searched only when nobody named one: quietly falling back would report a
  // different runtime's version and a different runtime's numbers under the
  // name of the one that was asked for, which is the one mistake this setting
  // exists to prevent.
  const char *named = g_active.library.empty() ? nullptr : g_active.library.c_str();
  // Absolute before the loader sees it: a relative module path finds the
  // file itself but not the sibling libraries beside it, so the same file
  // fails by relative path and loads by absolute one (see
  // clpeak::absoluteModulePath).
  const std::string absNamed =
      named ? clpeak::absoluteModulePath(named) : std::string();
  // The default search names no file, except when the Windows ML catalog
  // steered it to the runtime beside itself: that one is recorded, so the
  // status can say which runtime a catalog-driven run measured.
  std::string steered;

  void *lib = nullptr;
  if (named)
  {
#ifdef _WIN32
    searchBesideRuntime(absNamed);
#endif
    lib = clpeak::dynOpen({absNamed.c_str()});
    if (!lib)
      g_loadError = std::string("could not load onnxruntime from '") +
                    absNamed + "'";
  }
  else
  {
    // The absolute Homebrew/local paths matter on macOS: /opt/homebrew/lib is
    // not on the default dlopen search path.  On Android the bare soname is
    // what resolves -- a packaged runtime lands in the APK's read-only lib
    // dir, which is on the linker path.
#ifdef _WIN32
    // The runtime beside the Windows ML catalog first (see
    // winmlDefaultRuntime), then the system's.
    std::string absDefault = winmlDefaultRuntime(g_active);
    steered = absDefault;
    if (absDefault.empty())
      absDefault = resolveWindowsDefaultAbsolute();
    else
      searchBesideRuntime(absDefault);
    if (!absDefault.empty())
    {
      // Absolute path: loads the file it names even when a DLL of the same
      // basename is already mapped.  A bare "onnxruntime.dll" here would
      // hand back that DLL instead.
      lib = clpeak::dynOpen({absDefault.c_str()});
      if (!lib)
        g_loadError = std::string("could not load onnxruntime from '") +
                      absDefault + "'";
    }
    else
    {
      lib = clpeak::dynOpen({"onnxruntime.dll"});
      if (!lib)
      {
        g_loadError = "onnxruntime library not found";
      }
      else if (opened(lib))
      {
        // No system runtime on the filesystem search path, yet a bare load
        // succeeded: a DLL of that name an earlier attempt mapped, shining
        // through (see resolveWindowsDefaultAbsolute), not a default.  This
        // call's extra reference on it is simply kept: the module stays
        // mapped for the life of the process anyway.
        g_loadError = "onnxruntime library not found";
        lib = nullptr;
      }
    }
#else
    lib = clpeak::dynOpen({
#if defined(__APPLE__)
        "libonnxruntime.dylib",
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
#else
        "libonnxruntime.so",
        "libonnxruntime.so.1",
#endif
    });
    if (!lib)
      g_loadError = "onnxruntime library not found";
#endif
  }
  if (!lib)
    return;
  if (!opened(lib))
    g_opened.push_back(lib);

  // A file that turns out not to be a usable ONNX Runtime stays mapped like
  // any other handle: unloading a library whose constructors have run is
  // what common/dynlib.h explains is never safe.
  void *sym = clpeak::dynSym(lib, "OrtGetApiBase");
  auto getBase = reinterpret_cast<const OrtApiBase *(ORT_API_CALL *)()>(sym);
  if (!getBase)
  {
    g_loadError = std::string(named ? named : "onnxruntime") +
                  " exports no OrtGetApiBase -- not an ONNX Runtime library";
    return;
  }

  OrtRuntime cur;
  if (!adoptApiBase(getBase(), cur))
    return;

  // Deliberately not dlclosed for the rest of the process: see the note on
  // onnxSetLibraryOverride() in the header.
  cur.lib = lib;
  if (named)
  {
    cur.path = absNamed;
  }
  else
  {
    // Found by name: resolve where it came from so a settings screen can
    // show the file, the way the LiteRT backend does.  The Windows ML
    // steered path is the fallback when the platform cannot say.
    cur.path = resolveLoadedPath(lib, sym);
    if (cur.path.empty())
      cur.path = steered;
  }
  // The setup is fixed from here on (onnx_runtime.h).
  g_loaded = std::move(cur);
  g_rt = &g_loaded;
}

void onnxSetLibraryOverride(const std::string &path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_requested.library = path;
  // Once a runtime has loaded, the choice waits for the next start
  // (onnxPendingSetup); until then it applies at the next ortRuntime().
  if (g_rt || g_active.library == path)
    return;
  g_active.library = path;
  g_attempted = false;
  g_loadError.clear();
}

#endif // CLPEAK_ONNX_STATIC

bool onnxPendingSetup(OnnxPendingSetup &out)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_rt)
    return false;   // nothing loaded: a choice applies as soon as it is made
  // The loaded file named by another spelling -- or named now, where the
  // default search found it -- is no change.
  bool library = g_requested.library != g_active.library;
  if (library && !g_requested.library.empty() &&
      clpeak::sameModulePath(g_requested.library, g_loaded.path))
    library = false;
  // A catalog that is off has no folder that matters.
  const bool winml =
      g_requested.winml != g_active.winml ||
      (g_requested.winml && !clpeak::sameModulePath(g_requested.winmlPath, g_active.winmlPath));
  if (!library && !winml)
    return false;
  out.library   = g_requested.library;
  out.winml     = g_requested.winml;
  out.winmlPath = g_requested.winmlPath;
  return true;
}

const OrtRuntime *ortRuntime()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_attempted)
  {
    g_attempted = true;
    loadRuntime();
  }
  return g_rt;
}

std::string onnxLoadDiagnostic()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_loadError;
}

#endif // ENABLE_ONNX
