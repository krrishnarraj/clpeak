#ifdef ENABLE_ONNX

#include "onnx_winml.h"
#include "onnx_plugin.h"
#include "onnx_runtime.h"

#include <common/common.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <filesystem>
#endif

namespace
{

std::mutex g_mutex;
OnnxWinmlResolution g_res;
uint64_t g_resGeneration = 0;     // config generation the memo answers for
const void *g_resRuntime = nullptr;

} // namespace

#ifdef _WIN32

// The flat C surface of Microsoft.Windows.AI.MachineLearning.dll this file
// uses, transcribed from WinMLEpCatalog.h (Microsoft.Windows.AI.MachineLearning
// 2.3.42; STDAPI is HRESULT __stdcall).  Nothing is linked: every entry point
// is resolved by name.
namespace
{

typedef struct WinMLEpCatalog *WinMLEpCatalogHandle;
typedef struct WinMLEp *WinMLEpHandle;

enum WinMLEpReadyState
{
  WinMLEpReadyState_Ready      = 0,
  WinMLEpReadyState_NotReady   = 1,
  WinMLEpReadyState_NotPresent = 2,
};

enum WinMLEpCertification
{
  WinMLEpCertification_Unknown     = 0,
  WinMLEpCertification_Certified   = 1,
  WinMLEpCertification_Uncertified = 2,
};

struct WinMLEpInfo
{
  const char *name;
  const char *version;
  const char *packageFamilyName;
  const char *libraryPath;
  const char *packageRootPath;
  WinMLEpReadyState readyState;
  WinMLEpCertification certification;
};

typedef BOOL(__stdcall *WinMLEpEnumCallback)(WinMLEpHandle ep,
                                            const WinMLEpInfo *info,
                                            void *context);

typedef HRESULT(__stdcall *PfnCatalogCreate)(WinMLEpCatalogHandle *);
typedef void(__stdcall *PfnCatalogRelease)(WinMLEpCatalogHandle);
typedef HRESULT(__stdcall *PfnCatalogEnumProviders)(WinMLEpCatalogHandle,
                                                         WinMLEpEnumCallback,
                                                         void *);
typedef HRESULT(__stdcall *PfnEpEnsureReady)(WinMLEpHandle);
typedef HRESULT(__stdcall *PfnEpGetLibraryPathSize)(WinMLEpHandle, size_t *);
typedef HRESULT(__stdcall *PfnEpGetLibraryPath)(WinMLEpHandle, size_t, char *,
                                                     size_t *);

struct WinmlApi
{
  PfnCatalogCreate        catalogCreate        = nullptr;
  PfnCatalogRelease       catalogRelease       = nullptr;
  PfnCatalogEnumProviders catalogEnumProviders = nullptr;
  PfnEpEnsureReady        epEnsureReady        = nullptr;
  PfnEpGetLibraryPathSize epGetLibraryPathSize = nullptr;
  PfnEpGetLibraryPath     epGetLibraryPath     = nullptr;
};

std::string hresultText(HRESULT hr)
{
  char code[32];
  snprintf(code, sizeof code, "0x%08lX", (unsigned long)hr);
  char *msg = nullptr;
  const DWORD n = FormatMessageA(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr, (DWORD)hr, 0, reinterpret_cast<char *>(&msg), 0, nullptr);
  std::string text = code;
  if (n && msg)
  {
    std::string m(msg, n);
    while (!m.empty() && (m.back() == '\n' || m.back() == '\r' || m.back() == ' ' || m.back() == '.'))
      m.pop_back();
    if (!m.empty())
      text += " (" + m + ")";
  }
  if (msg)
    LocalFree(msg);
  return text;
}

std::string moduleDirectory(HMODULE mod)
{
  std::string buf(32768, '\0');
  const DWORD n = GetModuleFileNameA(mod, &buf[0], (DWORD)buf.size());
  if (n == 0 || n >= buf.size())
    return "";
  buf.resize(n);
  const size_t s = buf.find_last_of("/\\");
  return s == std::string::npos ? "" : buf.substr(0, s);
}

const char *kDllName = "Microsoft.Windows.AI.MachineLearning.dll";

// Where the catalog DLL is looked for, in order: the path named (the DLL
// itself or its directory), beside the loaded onnxruntime (the NuGet
// package puts both in runtimes/win-<arch>/native), beside the executable,
// then the loader's own search.  Absolute candidates load with their own
// directory first on the dependency search, as the loader does for an
// executable's own DLLs.
HMODULE loadCatalogDll(const OrtRuntime *rt, const std::string &hint,
                       std::string &pathOut, std::string &error)
{
  std::vector<std::string> candidates;
  if (!hint.empty())
  {
    std::error_code ec;
    if (std::filesystem::is_directory(hint, ec))
      candidates.push_back(hint + "\\" + kDllName);
    else
      candidates.push_back(hint);
  }
  else
  {
    if (rt && rt->lib)
    {
      const std::string dir = moduleDirectory(reinterpret_cast<HMODULE>(rt->lib));
      if (!dir.empty())
        candidates.push_back(dir + "\\" + kDllName);
    }
    const std::string exeDir = moduleDirectory(nullptr);
    if (!exeDir.empty())
      candidates.push_back(exeDir + "\\" + kDllName);
  }

  for (const auto &candidate : candidates)
  {
    std::error_code ec;
    if (!std::filesystem::exists(candidate, ec))
      continue;
    // Absolute: LOAD_WITH_ALTERED_SEARCH_PATH is undefined for a relative
    // path, and a relative --onnx-winml is the ordinary spelling.
    const std::string c = std::filesystem::absolute(candidate, ec).string();
    HMODULE h = LoadLibraryExA(c.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (h)
    {
      pathOut = c;
      return h;
    }
    error = "could not load " + c + ": " + hresultText(HRESULT_FROM_WIN32(GetLastError()));
    return nullptr;
  }
  if (hint.empty())
  {
    if (HMODULE h = LoadLibraryA(kDllName))
    {
      pathOut = moduleDirectory(h) + "\\" + kDllName;
      return h;
    }
    error = std::string(kDllName) +
            " not found beside the loaded onnxruntime, beside clpeak or on the "
            "DLL search path (tool/fetch_winml.ps1 fetches it, or name it "
            "with --onnx-winml)";
  }
  else
  {
    error = "no " + std::string(kDllName) + " at " + hint;
  }
  return nullptr;
}

bool resolveApi(HMODULE h, WinmlApi &api, std::string &error)
{
  struct Entry { const char *name; void **slot; };
  const Entry entries[] = {
      {"WinMLEpCatalogCreate",        reinterpret_cast<void **>(&api.catalogCreate)},
      {"WinMLEpCatalogRelease",       reinterpret_cast<void **>(&api.catalogRelease)},
      {"WinMLEpCatalogEnumProviders", reinterpret_cast<void **>(&api.catalogEnumProviders)},
      {"WinMLEpEnsureReady",          reinterpret_cast<void **>(&api.epEnsureReady)},
      {"WinMLEpGetLibraryPathSize",   reinterpret_cast<void **>(&api.epGetLibraryPathSize)},
      {"WinMLEpGetLibraryPath",       reinterpret_cast<void **>(&api.epGetLibraryPath)},
  };
  for (const auto &e : entries)
  {
    *e.slot = reinterpret_cast<void *>(GetProcAddress(h, e.name));
    if (!*e.slot)
    {
      error = std::string(kDllName) + " exports no " + e.name +
              " -- not the Windows ML runtime this build knows (2.3)";
      return false;
    }
  }
  return true;
}

struct EnumState
{
  std::vector<std::pair<WinMLEpHandle, OnnxWinmlProvider>> found;
};

BOOL __stdcall collectProvider(WinMLEpHandle ep, const WinMLEpInfo *info, void *ctx)
{
  auto *state = static_cast<EnumState *>(ctx);
  if (!info || !info->name)
    return TRUE;
  OnnxWinmlProvider p;
  p.name          = info->name;
  p.version       = info->version ? info->version : "";
  p.packageFamily = info->packageFamilyName ? info->packageFamilyName : "";
  p.libraryPath   = info->libraryPath ? info->libraryPath : "";
  p.certified     = info->certification == WinMLEpCertification_Certified;
  p.ready         = info->readyState == WinMLEpReadyState_Ready;
  p.installed     = info->readyState != WinMLEpReadyState_NotPresent;
  if (info->readyState == WinMLEpReadyState_NotPresent)
    p.error = "not installed";
  else if (info->readyState == WinMLEpReadyState_NotReady)
    p.error = "installed, not yet added to this process";
  state->found.emplace_back(ep, std::move(p));
  return TRUE;
}

std::string libraryPathOf(const WinmlApi &api, WinMLEpHandle ep, std::string &error)
{
  size_t size = 0;
  HRESULT hr = api.epGetLibraryPathSize(ep, &size);
  if (FAILED(hr) || size == 0)
  {
    error = "no library path: " + hresultText(hr);
    return "";
  }
  std::string buf(size, '\0');
  size_t used = 0;
  hr = api.epGetLibraryPath(ep, buf.size(), &buf[0], &used);
  if (FAILED(hr))
  {
    error = "no library path: " + hresultText(hr);
    return "";
  }
  buf.resize(used ? used : size);
  while (!buf.empty() && buf.back() == '\0')
    buf.pop_back();
  return buf;
}

void resolveWindows(const OrtRuntime *rt, const std::string &hint,
                    OnnxWinmlResolution &res)
{
  HMODULE h = loadCatalogDll(rt, hint, res.dllPath, res.error);
  if (!h)
    return;
  WinmlApi api;
  if (!resolveApi(h, api, res.error))
    return;

  WinMLEpCatalogHandle catalog = nullptr;
  HRESULT hr = api.catalogCreate(&catalog);
  if (FAILED(hr) || !catalog)
  {
    res.error = "WinMLEpCatalogCreate failed: " + hresultText(hr) +
                " -- the catalog needs Windows 11 24H2 (build 26100) or newer";
    return;
  }

  EnumState state;
  hr = api.catalogEnumProviders(catalog, collectProvider, &state);
  if (FAILED(hr))
  {
    res.error = "WinMLEpCatalogEnumProviders failed: " + hresultText(hr);
    api.catalogRelease(catalog);
    return;
  }

  for (auto &entry : state.found)
  {
    WinMLEpHandle ep = entry.first;
    OnnxWinmlProvider &p = entry.second;
    if (!p.certified)
    {
      // Windows ML itself registers only certified providers; an
      // uncertified one is listed so the status can say it exists.
      p.error = "not certified; not registered";
      res.providers.push_back(p);
      continue;
    }
    if (!p.ready)
    {
      // A download from the Store, of tens to hundreds of megabytes, on
      // the first run only.  Said out loud, since the run sits here until
      // it is done.  An installed provider that is merely not yet part of
      // this process is added in a moment, and that is only worth a
      // verbose line.
      if (!p.installed)
        clpeak::logMessage(clpeak::LogLevel::Warning, "",
                           "ONNX: Windows ML is installing the " + p.name +
                               (p.version.empty() ? "" : " " + p.version) +
                               " execution provider from the Microsoft Store; "
                               "this can take a few minutes");
      else
        CLPEAK_VLOG("onnx: Windows ML adding the installed %s provider to this process\n",
                    p.name.c_str());
      hr = api.epEnsureReady(ep);
      if (FAILED(hr))
      {
        p.error = "install failed: " + hresultText(hr);
        res.providers.push_back(p);
        continue;
      }
      p.ready = true;
      p.error.clear();
    }
    std::string err;
    const std::string path = libraryPathOf(api, ep, err);
    if (path.empty())
    {
      p.ready = false;
      p.error = err;
    }
    else
    {
      p.libraryPath = path;
    }
    res.providers.push_back(p);
  }
  api.catalogRelease(catalog);
}

} // namespace

#endif // _WIN32

const OnnxWinmlResolution *onnxWinmlResolved(const OrtRuntime *rt)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const void *rtKey = rt ? rt->lib : nullptr;
  if (g_resGeneration == onnxEpConfigGeneration() && g_resRuntime == rtKey)
    return &g_res;
  return nullptr;
}

const OnnxWinmlResolution &onnxWinmlResolve(const OrtRuntime *rt,
                                            const std::string &hint)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const uint64_t gen = onnxEpConfigGeneration();
  const void *rtKey = rt ? rt->lib : nullptr;
  if (g_resGeneration == gen && g_resRuntime == rtKey)
    return g_res;
  g_res = OnnxWinmlResolution();
  g_resGeneration = gen;
  g_resRuntime = rtKey;
#ifdef _WIN32
  resolveWindows(rt, hint, g_res);
  for (const auto &p : g_res.providers)
    CLPEAK_VLOG("onnx: Windows ML provider %s %s: %s%s\n", p.name.c_str(),
                p.version.c_str(),
                p.ready ? p.libraryPath.c_str() : "not ready",
                p.error.empty() ? "" : (" (" + p.error + ")").c_str());
  if (!g_res.error.empty())
    CLPEAK_VLOG("onnx: Windows ML catalog: %s\n", g_res.error.c_str());
#else
  (void)rt;
  (void)hint;
  g_res.error = "the Windows ML execution-provider catalog is a Windows 11 feature";
#endif
  return g_res;
}

#endif // ENABLE_ONNX
