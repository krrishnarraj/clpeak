#ifdef ENABLE_ONNX

#include "onnx_plugin.h"
#include "onnx_session.h"
#include "onnx_winml.h"

#include <common/common.h>
#include <common/dynlib.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

// Plugin execution providers need the OrtApi entries that arrived with
// ONNX Runtime 1.22 (RegisterExecutionProviderLibrary, GetEpDevices,
// SessionOptionsAppendExecutionProvider_V2).  An older runtime hands out a
// shorter table, so the slots must not even be read below this.
static const uint32_t kMinPluginApiVersion = 22;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

namespace
{

std::mutex g_cfgMutex;
std::vector<OnnxEpLibrary> g_libs;
bool g_winmlEnabled = false;
std::string g_winmlPath;
uint64_t g_generation = 1;

// What the current environment registered, and against which generation.
std::mutex g_envMutex;
std::vector<OnnxEpLibraryStatus> g_status;
std::set<std::string> g_builtinEpNames;   // provider names listed before registering
const OrtEnv *g_statusEnv = nullptr;
uint64_t g_statusGeneration = 0;          // the configuration g_status answers for

std::string describeType(OrtHardwareDeviceType t)
{
  switch (t)
  {
  case OrtHardwareDeviceType_CPU: return "CPU";
  case OrtHardwareDeviceType_GPU: return "GPU";
  case OrtHardwareDeviceType_NPU: return "NPU";
  }
  return "Unknown";
}

DeviceType classOf(OrtHardwareDeviceType t)
{
  switch (t)
  {
  case OrtHardwareDeviceType_CPU: return DeviceType::Cpu;
  case OrtHardwareDeviceType_GPU: return DeviceType::Gpu;
  case OrtHardwareDeviceType_NPU: return DeviceType::Accelerator;
  }
  return DeviceType::Unknown;
}

int classRank(DeviceType t)
{
  switch (t)
  {
  case DeviceType::Accelerator: return 0;
  case DeviceType::Gpu:         return 1;
  case DeviceType::Cpu:         return 2;
  default:                      return 3;
  }
}

void appendPairs(const OrtRuntime &rt, const OrtKeyValuePairs *kvps,
                 std::vector<std::pair<std::string, std::string>> &out)
{
  if (!kvps)
    return;
  const char *const *keys = nullptr;
  const char *const *vals = nullptr;
  size_t n = 0;
  rt.api->GetKeyValuePairs(kvps, &keys, &vals, &n);
  for (size_t i = 0; i < n; i++)
    if (keys && keys[i] && vals && vals[i])
      out.emplace_back(keys[i], vals[i]);
}

} // namespace

uint64_t onnxEpConfigGeneration()
{
  std::lock_guard<std::mutex> lock(g_cfgMutex);
  return g_generation;
}

void onnxSetEpLibraries(std::vector<OnnxEpLibrary> libs)
{
  // Absolute before the runtime's own loader sees them: a relative plugin
  // path has the same sibling-resolution problem as a relative runtime one
  // (see clpeak::absoluteModulePath).  Covers the CLI and the FFI setter;
  // the catalog-appended libraries are absolute already.
  for (auto &lib : libs)
    lib.path = clpeak::absoluteModulePath(lib.path.c_str());
  std::lock_guard<std::mutex> lock(g_cfgMutex);
  bool same = libs.size() == g_libs.size();
  for (size_t i = 0; same && i < libs.size(); i++)
    same = libs[i].name == g_libs[i].name && libs[i].path == g_libs[i].path &&
           libs[i].named == g_libs[i].named;
  if (same)
    return;
  g_libs = std::move(libs);
  g_generation++;
}

const std::vector<OnnxEpLibrary> &onnxEpLibraries()
{
  std::lock_guard<std::mutex> lock(g_cfgMutex);
  return g_libs;
}

void onnxSetWinml(bool enabled, const std::string &path)
{
  {
    std::lock_guard<std::mutex> lock(g_cfgMutex);
    if (enabled == g_winmlEnabled && path == g_winmlPath)
      return;
    g_winmlEnabled = enabled;
    g_winmlPath    = path;
    g_generation++;
  }
  // The catalog's directory steers the runtime's default search.
  onnxRuntimeRecheck();
}

bool onnxWinmlEnabled()
{
  std::lock_guard<std::mutex> lock(g_cfgMutex);
  return g_winmlEnabled;
}

std::string onnxWinmlPathHint()
{
  std::lock_guard<std::mutex> lock(g_cfgMutex);
  return g_winmlPath;
}

std::vector<OnnxEpLibrary> onnxEffectiveEpLibraries(const OrtRuntime &rt)
{
  std::vector<OnnxEpLibrary> libs;
  bool winml = false;
  std::string hint;
  {
    std::lock_guard<std::mutex> lock(g_cfgMutex);
    libs  = g_libs;
    winml = g_winmlEnabled;
    hint  = g_winmlPath;
  }
  if (!winml)
    return libs;

  // The catalog's providers join the configured ones.  A provider the
  // catalog lists but could not make ready carries its reason in the
  // resolution (reported by the status and the run notes) and registers
  // nothing here.  One entry per registration name: a library named on the
  // command line as well wins, since that path is the one asked for.
  const OnnxWinmlResolution &res = onnxWinmlResolve(&rt, hint);
  for (const auto &p : res.providers)
  {
    if (!p.ready || p.libraryPath.empty())
      continue;
    bool dup = false;
    for (const auto &l : libs)
      if (l.name == p.name)
        dup = true;
    if (dup)
      continue;
    OnnxEpLibrary lib;
    lib.name  = p.name;
    lib.path  = p.libraryPath;
    lib.named = true;
    libs.push_back(std::move(lib));
  }
  return libs;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void onnxRegisterEpLibraries(const OrtRuntime &rt, OrtEnv *env)
{
  const std::vector<OnnxEpLibrary> libs = onnxEffectiveEpLibraries(rt);

  std::lock_guard<std::mutex> lock(g_envMutex);
  g_status.clear();
  g_builtinEpNames.clear();
  g_statusEnv = env;
  g_statusGeneration = onnxEpConfigGeneration();
  if (libs.empty())
    return;

  if (rt.apiVersion < kMinPluginApiVersion)
  {
    for (const auto &lib : libs)
    {
      OnnxEpLibraryStatus st;
      st.lib   = lib;
      st.error = "plugin execution providers need ONNX Runtime 1." +
                 std::to_string(kMinPluginApiVersion) +
                 " or newer; this runtime is " + rt.versionString;
      g_status.push_back(std::move(st));
    }
    return;
  }

  // The provider names present before any plugin: the built-in providers
  // that implement device enumeration (the CPU provider always, DirectML
  // on Windows, ...).  A name that appears only after registering belongs
  // to a plugin.  Names, not pointers: the runtime may rebuild its list.
  {
    const OrtEpDevice *const *devs = nullptr;
    size_t n = 0;
    if (OrtStatus *st = rt.api->GetEpDevices(env, &devs, &n))
      rt.api->ReleaseStatus(st);
    else
      for (size_t i = 0; i < n; i++)
        if (const char *name = rt.api->EpDevice_EpName(devs[i]))
          g_builtinEpNames.insert(name);
  }

  for (const auto &lib : libs)
  {
    OnnxEpLibraryStatus st;
    st.lib = lib;
    OrtStatus *status = nullptr;
#ifdef _WIN32
    // ORTCHAR_T is wchar_t on Windows; the path arrived as UTF-8.
    std::wstring wide;
    {
      const int n = MultiByteToWideChar(CP_UTF8, 0, lib.path.c_str(),
                                        (int)lib.path.size(), nullptr, 0);
      if (n > 0)
      {
        wide.resize((size_t)n);
        MultiByteToWideChar(CP_UTF8, 0, lib.path.c_str(), (int)lib.path.size(),
                            &wide[0], n);
      }
    }
    status = rt.api->RegisterExecutionProviderLibrary(env, lib.name.c_str(),
                                                      wide.c_str());
#else
    status = rt.api->RegisterExecutionProviderLibrary(env, lib.name.c_str(),
                                                      lib.path.c_str());
#endif
    st.registered = (status == nullptr);
    if (status)
      st.error = onnxStatusText(rt, status);
    CLPEAK_VLOG("onnx: plugin library %s (%s): %s\n", lib.name.c_str(),
                lib.path.c_str(),
                st.registered ? "registered" : st.error.c_str());
    g_status.push_back(std::move(st));
  }
}

std::vector<OnnxEpLibraryStatus> onnxEpLibraryStatus()
{
  // A configuration changed since the environment registered is not yet
  // answered for: nothing, rather than the previous set's answers under
  // the new set's names.
  std::lock_guard<std::mutex> lock(g_envMutex);
  if (g_statusGeneration != onnxEpConfigGeneration())
    return {};
  return g_status;
}

std::string onnxEpLibraryPath(const std::string &registrationName)
{
  std::lock_guard<std::mutex> lock(g_envMutex);
  for (const auto &st : g_status)
    if (st.registered && st.lib.name == registrationName)
      return st.lib.path;
  return std::string();
}

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

std::vector<onnx_ep_info_t> onnxPluginDevices(const OrtRuntime &rt)
{
  std::vector<onnx_ep_info_t> out;
  if (rt.apiVersion < kMinPluginApiVersion)
    return out;
  if (onnxEffectiveEpLibraries(rt).empty())
  {
    // A set that just became empty leaves the previous environment, with
    // its registrations, standing until something rebuilds it -- and with
    // every probe memoized nothing might, so the status would go on
    // answering for libraries nobody asks for any more.  Rebuild it now.
    bool stale = false;
    {
      std::lock_guard<std::mutex> lock(g_envMutex);
      stale = g_statusEnv && g_statusGeneration != onnxEpConfigGeneration();
    }
    if (stale)
      (void)onnxEnv(rt);
    return out;
  }

  // Creating the environment is what registers the libraries (see
  // onnxEnv); enumeration is the first thing to need one when plugins are
  // configured.
  OrtEnv *env = onnxEnv(rt);
  if (!env)
    return out;

  std::lock_guard<std::mutex> lock(g_envMutex);
  bool any = false;
  for (const auto &st : g_status)
    any = any || st.registered;
  if (!any || g_statusEnv != env)
    return out;

  const OrtEpDevice *const *devs = nullptr;
  size_t n = 0;
  if (OrtStatus *st = rt.api->GetEpDevices(env, &devs, &n))
  {
    CLPEAK_VLOG("onnx: GetEpDevices failed: %s\n", onnxStatusText(rt, st).c_str());
    return out;
  }

  // Which registration a provider name belongs to.  The name a plugin's
  // factory reports is usually its registration name (QNN requires it);
  // a lone registered library owns whatever else appeared.
  std::string soleLibrary;
  int registered = 0;
  for (const auto &st : g_status)
    if (st.registered)
    {
      registered++;
      soleLibrary = st.lib.name;
    }
  if (registered != 1)
    soleLibrary.clear();

  std::map<std::string, int> seen;   // providerKey + type -> count, for labels
  for (size_t i = 0; i < n; i++)
  {
    const OrtEpDevice *d = devs[i];
    const char *epName = rt.api->EpDevice_EpName(d);
    if (!epName || !*epName || g_builtinEpNames.count(epName))
      continue;

    onnx_ep_info_t ep;
    ep.providerKey = epName;
    ep.epDevicePtr = d;
    for (const auto &st : g_status)
      if (st.registered && st.lib.name == ep.providerKey)
        ep.library = st.lib.name;
    if (ep.library.empty())
      ep.library = soleLibrary;

    const OrtHardwareDevice *hw = rt.api->EpDevice_Device(d);
    const OrtHardwareDeviceType hwType =
        hw ? rt.api->HardwareDevice_Type(hw) : OrtHardwareDeviceType_CPU;
    ep.typeStr    = describeType(hwType);
    ep.deviceType = classOf(hwType);
    if (hw)
    {
      if (const char *v = rt.api->HardwareDevice_Vendor(hw))
        ep.vendor = v;
      appendPairs(rt, rt.api->HardwareDevice_Metadata(hw), ep.hardware);
    }
    if (ep.vendor.empty())
      if (const char *v = rt.api->EpDevice_EpVendor(d))
        ep.vendor = v;
    appendPairs(rt, rt.api->EpDevice_EpMetadata(d), ep.hardware);

    // A plugin serving several devices of one kind (two GPUs) needs the
    // label to tell them apart: it keys the probe memos and the details.
    const int nth = ++seen[ep.providerKey + '\x1f' + ep.typeStr];
    ep.epDevice = ep.typeStr + (nth > 1 ? "#" + std::to_string(nth) : "");

    // Named as clpeak names the provider when it knows it and the device
    // is the one its table means (QNN's HTP row is "Hexagon NPU", not its
    // GPU backend); otherwise from what the runtime reported, which is
    // exact if less pretty: "QNN (Qualcomm GPU)".
    std::string display, tableType;
    DeviceType tableClass = DeviceType::Unknown;
    if (onnxEpTableEntry(ep.providerKey, display, tableType, tableClass) &&
        tableType == ep.typeStr)
    {
      ep.displayName = display;
    }
    else
    {
      std::string shortName = ep.providerKey;
      const std::string suffix = "ExecutionProvider";
      if (shortName.size() > suffix.size() &&
          shortName.compare(shortName.size() - suffix.size(), suffix.size(),
                            suffix) == 0)
        shortName.erase(shortName.size() - suffix.size());
      ep.displayName = shortName + " (" +
                       (ep.vendor.empty() ? "" : ep.vendor + " ") +
                       ep.typeStr + ")";
    }
    if (nth > 1)
      ep.displayName += " #" + std::to_string(nth);

    out.push_back(std::move(ep));
  }

  std::stable_sort(out.begin(), out.end(),
                   [](const onnx_ep_info_t &a, const onnx_ep_info_t &b) {
                     return classRank(a.deviceType) < classRank(b.deviceType);
                   });
  return out;
}

std::string onnxAppendPluginDevice(
    const OrtRuntime &rt, OrtSessionOptions *so, const onnx_ep_info_t &ep,
    const std::vector<std::pair<std::string, std::string>> &kv)
{
  if (!ep.epDevicePtr)
    return "no OrtEpDevice behind " + ep.providerKey;
  OrtEnv *env = onnxEnv(rt);
  if (!env)
    return "onnxruntime environment creation failed: " + onnxEnvError();
  std::vector<const char *> keys, vals;
  for (const auto &p : kv)
  {
    keys.push_back(p.first.c_str());
    vals.push_back(p.second.c_str());
  }
  const OrtEpDevice *devs[1] = {ep.epDevicePtr};
  return onnxStatusText(rt, rt.api->SessionOptionsAppendExecutionProvider_V2(
                                so, env, devs, 1, keys.data(), vals.data(),
                                keys.size()));
}

#endif // ENABLE_ONNX
