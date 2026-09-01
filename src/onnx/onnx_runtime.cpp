#ifdef ENABLE_ONNX

#include "onnx_runtime.h"

#include <common/common.h>
#include <common/dynlib.h>

#include <cstdio>
#include <mutex>
#include <string>

// Oldest OrtApi we are prepared to speak.  Every entry point this backend
// calls existed well before this; requesting downwards from ORT_API_VERSION
// lets a binary built against a new header run on an older installed runtime.
static const uint32_t kMinApiVersion = 17;  // ONNX Runtime 1.17 (2024)

static std::mutex g_mutex;
static OrtRuntime g_rt;
static bool       g_loaded    = false;
static bool       g_attempted = false;  // a failed search is not retried
static std::string g_loadError;         // why the last attempt failed

// Fill g_rt from an OrtApiBase, whether it came from dlsym or a direct call.
// Returns false when the runtime serves no API version this build can speak.
static bool adoptApiBase(const OrtApiBase *base)
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

  g_rt.base          = base;
  g_rt.api           = api;
  g_rt.apiVersion    = version;
  g_rt.versionString = base->GetVersionString();
  return true;
}

#ifdef CLPEAK_ONNX_STATIC

// Statically linked (iOS): the runtime is already in the binary.
static void loadRuntime()
{
  g_rt.lib  = nullptr;
  g_rt.path.clear();
  g_loaded  = adoptApiBase(OrtGetApiBase());
  if (!g_loaded)
    g_loadError = "the linked-in ONNX Runtime exposes no usable OrtApi";
}

void onnxSetLibraryOverride(const std::string &) {}

#else

static std::string g_override;  // --onnx-lib / clpeak_set_onnx_library

static void loadRuntime()
{
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
    lib = clpeak::dynOpen({
#if defined(_WIN32)
        "onnxruntime.dll",
#elif defined(__APPLE__)
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

  if (!adoptApiBase(getBase()))
  {
    clpeak::dynClose(lib);
    return;
  }

  // Deliberately not dlclosed for the rest of the process: see the note on
  // onnxSetLibraryOverride() in the header.
  g_rt.lib  = lib;
  g_rt.path = named ? named : "";
  g_loaded  = true;
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
