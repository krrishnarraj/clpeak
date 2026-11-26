#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <common/dynlib.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

static std::mutex g_mutex;
static WebgpuRuntime g_rt;
static bool g_loaded = false;
static bool g_attempted = false;
static std::string g_loadError;
static std::string g_override;

static bool resolveApi(void *lib, WebgpuApi &api, const char *what)
{
  std::string missing;
#define CLPEAK_WEBGPU_RESOLVE_REQ(name)                                       \
  api.name = reinterpret_cast<decltype(&::name)>(clpeak::dynSym(lib, #name)); \
  if (!api.name)                                                              \
    missing += missing.empty() ? #name : std::string(", ") + #name;
  CLPEAK_WEBGPU_REQUIRED(CLPEAK_WEBGPU_RESOLVE_REQ)
#undef CLPEAK_WEBGPU_RESOLVE_REQ
#define CLPEAK_WEBGPU_RESOLVE_OPT(name) \
  api.name = reinterpret_cast<decltype(&::name)>(clpeak::dynSym(lib, #name));
  CLPEAK_WEBGPU_OPTIONAL(CLPEAK_WEBGPU_RESOLVE_OPT)
#undef CLPEAK_WEBGPU_RESOLVE_OPT
  if (!missing.empty())
  {
    g_loadError = std::string(what) + " lacks " + missing +
                  " -- not a compatible WebGPU (webgpu.h) runtime";
    return false;
  }
  return true;
}

static void loadRuntime()
{
  const char *named = g_override.empty() ? nullptr : g_override.c_str();
  const std::string absNamed =
      named ? clpeak::absoluteModulePath(named) : std::string();

  void *lib = nullptr;
  if (named)
  {
    lib = clpeak::dynOpen({absNamed.c_str()});
    if (!lib)
      g_loadError = std::string("could not load WebGPU library from '") + absNamed + "'";
  }
  else
  {
    // Check a generic libwebgpu soname first, then known webgpu.h
    // implementations (e.g., wgpu-native, Dawn).  Any other implementation can
    // be selected explicitly via --webgpu-lib.
    lib = clpeak::dynOpen({
#if defined(_WIN32)
        "webgpu.dll",
        "libwebgpu.dll",
        "wgpu_native.dll",
        "webgpu_dawn.dll",
        "libwebgpu_dawn.dll",
#elif defined(__APPLE__)
        "libwebgpu.dylib",
        "@executable_path/libwebgpu.dylib",
        "@loader_path/libwebgpu.dylib",
        "/opt/homebrew/lib/libwebgpu.dylib",
        "/usr/local/lib/libwebgpu.dylib",
        "libwgpu_native.dylib",
        "/opt/homebrew/lib/libwgpu_native.dylib",
        "libwebgpu_dawn.dylib",
        "@executable_path/libwebgpu_dawn.dylib",
        "@loader_path/libwebgpu_dawn.dylib",
        "/opt/homebrew/lib/libwebgpu_dawn.dylib",
        "/usr/local/lib/libwebgpu_dawn.dylib",
#else
        "libwebgpu.so",
        "libwgpu_native.so",
        "libwebgpu_dawn.so",
#endif
    });
    if (!lib)
      g_loadError = "WebGPU library not found";
  }
  if (!lib)
    return;

  WebgpuRuntime cur;
  if (!resolveApi(lib, cur.api, named ? named : "WebGPU library"))
    return;

  cur.lib = lib;
  cur.path = named ? absNamed
                   : clpeak::dynPath(lib, reinterpret_cast<void *>(cur.api.wgpuCreateInstance));

  // Deliberately never dlclosed: see include/common/dynlib.h.
  g_rt = cur;
  g_loaded = true;
}

void webgpuSetLibraryOverride(const std::string &path)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_loaded || path == g_override)
    return;
  g_override = path;
  g_attempted = false;
  g_loadError.clear();
}

std::string webgpuLoadDiagnostic()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_loadError;
}

const WebgpuRuntime *webgpuRuntime()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_attempted)
  {
    g_attempted = true;
    loadRuntime();
  }
  return g_loaded ? &g_rt : nullptr;
}

bool wgpuWaitOne(const WebgpuRuntime &rt, WGPUInstance inst, bool timedWaitAny,
                 WGPUFuture future, const std::atomic<bool> *done)
{
  if (done && done->load(std::memory_order_acquire))
    return true;
  if (future.id == 0)
    return false;
  if (timedWaitAny && rt.api.wgpuInstanceWaitAny)
  {
    WGPUFutureWaitInfo wi = WGPU_FUTURE_WAIT_INFO_INIT;
    wi.future = future;
    WGPUWaitStatus ws = rt.api.wgpuInstanceWaitAny(inst, 1, &wi, UINT64_MAX);
    return ws == WGPUWaitStatus_Success && wi.completed;
  }
  if (!done)
    return false;
  // Fallback when TimedWaitAny is unavailable: pump instance events until done.
  while (!done->load(std::memory_order_acquire))
  {
    rt.api.wgpuInstanceProcessEvents(inst);
    if (!done->load(std::memory_order_acquire))
      std::this_thread::yield();
  }
  return true;
}

#endif // ENABLE_WEBGPU
