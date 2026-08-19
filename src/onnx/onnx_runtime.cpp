#ifdef ENABLE_ONNX

#include "onnx_runtime.h"

#include <common/common.h>
#include <common/dynlib.h>

#include <cstdlib>
#include <mutex>

// Oldest OrtApi we are prepared to speak.  Every entry point this backend
// calls existed well before this; requesting downwards from ORT_API_VERSION
// lets a binary built against a new header run on an older installed runtime.
static const uint32_t kMinApiVersion = 17;  // ONNX Runtime 1.17 (2024)

static OrtRuntime g_rt;
static bool       g_loaded = false;
static std::once_flag g_once;

static void loadRuntime()
{
  // Explicit override first, then the platform's conventional names.  The
  // absolute Homebrew/local paths matter on macOS: /opt/homebrew/lib is not
  // on the default dlopen search path.
  const char *env = std::getenv("CLPEAK_ONNXRUNTIME_LIB");
  void *lib = clpeak::dynOpen({
      env,
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
    return;

  auto getBase = reinterpret_cast<const OrtApiBase *(ORT_API_CALL *)()>(
      clpeak::dynSym(lib, "OrtGetApiBase"));
  if (!getBase)
  {
    clpeak::dynClose(lib);
    return;
  }

  const OrtApiBase *base = getBase();
  if (!base)
  {
    clpeak::dynClose(lib);
    return;
  }

  // Ask for the newest API we were built against, then walk down so an older
  // installed runtime still works (GetApi returns null for unknown versions).
  const OrtApi *api = nullptr;
  uint32_t version = ORT_API_VERSION;
  for (; version >= kMinApiVersion; version--)
  {
    api = base->GetApi(version);
    if (api)
      break;
  }
  if (!api)
  {
    if (clpeak::verboseEnabled())
      fprintf(stderr, "clpeak: onnxruntime %s exposes no OrtApi in [%u, %u]\n",
              base->GetVersionString(), kMinApiVersion,
              (unsigned)ORT_API_VERSION);
    clpeak::dynClose(lib);
    return;
  }

  g_rt.lib           = lib;
  g_rt.base          = base;
  g_rt.api           = api;
  g_rt.apiVersion    = version;
  g_rt.versionString = base->GetVersionString();
  g_loaded = true;
}

const OrtRuntime *ortRuntime()
{
  std::call_once(g_once, loadRuntime);
  return g_loaded ? &g_rt : nullptr;
}

#endif // ENABLE_ONNX
