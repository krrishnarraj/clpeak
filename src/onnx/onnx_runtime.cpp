#ifdef ENABLE_ONNX

#include "onnx_runtime.h"

#include <common/common.h>
#include <common/dynlib.h>

#include <cstdio>
#include <map>
#include <mutex>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

// Oldest OrtApi we are prepared to speak.  Every entry point this backend
// calls existed well before this; requesting downwards from ORT_API_VERSION
// lets a binary built against a new header run on an older installed runtime.
static const uint32_t kMinApiVersion = 17;  // ONNX Runtime 1.17 (2024)

static std::mutex g_mutex;
static OrtRuntime g_rt;
static bool       g_loaded    = false;
static bool       g_attempted = false;  // a failed search is not retried
static std::string g_loadError;         // why the last attempt failed

// Every successfully loaded runtime stays mapped for the life of the process
// (unloading is unsafe) and is remembered under its override key ("" for the
// default search).  Switching back to a previously used library then reuses
// its handle instead of mapping the file a second time.
static std::map<std::string, OrtRuntime> g_cache;  // key = g_override value

#ifdef _WIN32
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

// Statically linked (iOS): the runtime is already in the binary.
static void loadRuntime()
{
  g_rt.lib  = nullptr;
  g_rt.path.clear();
  g_loaded  = adoptApiBase(OrtGetApiBase(), g_rt);
  if (!g_loaded)
    g_loadError = "the linked-in ONNX Runtime exposes no usable OrtApi";
}

void onnxSetLibraryOverride(const std::string &) {}

#else

static std::string g_override;  // --onnx-lib / clpeak_set_onnx_library

static void loadRuntime()
{
  // A repeat pick reuses the still-mapped handle: no second mapping of the
  // same file, and (on Windows) no chance for a bare-name search to alias a
  // different file of the same basename.
  auto cached = g_cache.find(g_override);
  if (cached != g_cache.end())
  {
    g_rt     = cached->second;
    g_loaded = true;
    return;
  }

  // A library the user named -- by --onnx-lib or by the FFI setter -- is the
  // library to measure, and nothing else will do.  The conventional names are
  // searched only when nobody named one: quietly falling back would report a
  // different runtime's version and a different runtime's numbers under the
  // name of the one that was asked for, which is the one mistake this setting
  // exists to prevent.
  const char *named = g_override.empty() ? nullptr : g_override.c_str();

  void *lib = nullptr;
  if (named)
  {
    lib = clpeak::dynOpen({named});
    if (!lib)
      g_loadError = std::string("could not load onnxruntime from '") + named +
                    "'";
  }
  else
  {
    // The absolute Homebrew/local paths matter on macOS: /opt/homebrew/lib is
    // not on the default dlopen search path.  On Android the bare soname is
    // what resolves -- a packaged runtime lands in the APK's read-only lib
    // dir, which is on the linker path.
#ifdef _WIN32
    const std::string absDefault = resolveWindowsDefaultAbsolute();
    if (!absDefault.empty())
    {
      // Absolute path: loads the file it names even when custom DLLs of the
      // same basename are already mapped.  A bare "onnxruntime.dll" here
      // would hand back the first-loaded custom build instead.
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
      else
      {
        // No system runtime on the filesystem search path, yet a bare load
        // succeeded: that is an already-mapped custom DLL shining through
        // (see resolveWindowsDefaultAbsolute), not a default.  Adopting it
        // would report the previous pick's version under the default's name.
        for (const auto &kv : g_cache)
        {
          if (kv.second.lib == lib)
          {
            clpeak::dynClose(lib);  // drop this call's ref; the leaked one stays
            g_loadError = "onnxruntime library not found";
            lib = nullptr;
            break;
          }
        }
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

  auto getBase = reinterpret_cast<const OrtApiBase *(ORT_API_CALL *)()>(
      clpeak::dynSym(lib, "OrtGetApiBase"));
  if (!getBase)
  {
    g_loadError = std::string(named ? named : "onnxruntime") +
                  " exports no OrtGetApiBase -- not an ONNX Runtime library";
    clpeak::dynClose(lib);
    return;
  }

  OrtRuntime cur;
  if (!adoptApiBase(getBase(), cur))
  {
    clpeak::dynClose(lib);
    return;
  }

  // Deliberately not dlclosed for the rest of the process: see the note on
  // onnxSetLibraryOverride() in the header.
  cur.lib  = lib;
  cur.path = named ? named : "";
  g_rt     = cur;
  g_cache[g_override] = cur;
  g_loaded = true;
}

void onnxSetLibraryOverride(const std::string &path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (path == g_override)
    return;
  g_override = path;
  // Force the next ortRuntime() to search again.  The previously loaded
  // handle stays mapped; g_rt is simply repointed once the new one loads.
  g_loaded    = false;
  g_attempted = false;
  g_rt        = OrtRuntime{};
  g_loadError.clear();
}

#endif // CLPEAK_ONNX_STATIC

const OrtRuntime *ortRuntime()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_attempted)
  {
    g_attempted = true;
    loadRuntime();
  }
  return g_loaded ? &g_rt : nullptr;
}

std::string onnxLoadDiagnostic()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_loadError;
}

#endif // ENABLE_ONNX
