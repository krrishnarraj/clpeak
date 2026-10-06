#ifdef ENABLE_ONNX

#include "onnx_plugin.h"
#include "onnx_session.h"
#include "onnx_winml.h"

#include <common/common.h>
#include <common/console_mute.h>
#include <common/dynlib.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <mutex>
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
uint64_t g_generation = 1;

// One library on the environment: how it fared, where it came from, and
// the devices registering it added.  Those are the runtime's own
// OrtEpDevices, owned by the registration until it is unregistered (the
// list GetEpDevices returns is rebuilt then; the devices are not), so they
// say exactly which library serves a device.  The provider name a device
// reports does not: a library registers under whatever name it was given
// (OpenVINO's ignores it), can serve more than one provider name
// ("OpenVINOExecutionProvider.AUTO" beside "OpenVINOExecutionProvider"),
// and two libraries can serve the same one.
struct Registration
{
  OnnxEpLibraryStatus st;
  bool catalog = false;                       // resolved by the Windows ML catalog
  std::vector<const OrtEpDevice *> devices;   // in the runtime's order
};

// A provider the Windows ML catalog made ready, as a library to register.
struct CatalogLibrary
{
  OnnxEpLibrary lib;
  std::string version;   // the catalog's, for the status and notes
};

// What the environment has registered, and against which generation.
std::mutex g_envMutex;
std::vector<Registration> g_regs;         // the configured libraries, then the catalog's
const OrtEnv *g_statusEnv = nullptr;
uint64_t g_statusGeneration = 0;          // the configuration g_regs answers for

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

static std::vector<OnnxEpLibrary> configuredLibraries()
{
  std::lock_guard<std::mutex> lock(g_cfgMutex);
  return g_libs;
}

// The providers the Windows ML catalog made ready, when it is on.  One it
// lists but could not make ready carries its reason in the resolution
// (reported by the status and the run notes) and registers nothing.
static std::vector<CatalogLibrary> catalogLibraries(const OrtRuntime &rt)
{
  std::vector<CatalogLibrary> out;
  if (!onnxWinmlEnabled())
    return out;
  const OnnxWinmlResolution &res = onnxWinmlResolve(&rt, onnxWinmlPathHint());
  for (const auto &p : res.providers)
  {
    if (!p.ready || p.libraryPath.empty())
      continue;
    CatalogLibrary c;
    c.lib.name  = p.name;
    c.lib.path  = p.libraryPath;
    c.lib.named = true;
    c.version   = p.version;
    out.push_back(std::move(c));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

// The devices the environment lists now.
static std::vector<const OrtEpDevice *> epDevices(const OrtRuntime &rt, OrtEnv *env)
{
  const OrtEpDevice *const *devs = nullptr;
  size_t n = 0;
  if (OrtStatus *st = rt.api->GetEpDevices(env, &devs, &n))
  {
    CLPEAK_VLOG("onnx: GetEpDevices failed: %s\n", onnxStatusText(rt, st).c_str());
    return {};
  }
  return std::vector<const OrtEpDevice *>(devs, devs + n);
}

// Whether a library bundled on the off-chance (`named == false`) is missing
// from the package.  Only the Android app adds one -- Qualcomm's QNN plugin,
// which an APK carries only when tools/fetch_android_npu.sh staged
// Qualcomm's libraries for it -- by bare file name, which the runtime
// resolves beside itself, so that is where it is looked for.  Without the
// check a build without it asked the runtime to load it anyway, and the
// loader's "not found" read like a broken package.
static bool bundledLibraryMissing(const OrtRuntime &rt, const OnnxEpLibrary &lib)
{
  if (lib.named || rt.path.empty() || lib.path.find_first_of("/\\") != std::string::npos)
    return false;
  std::error_code ec;
  const bool there =
      std::filesystem::exists(std::filesystem::path(rt.path).parent_path() / lib.path, ec);
  return !there && !ec;
}

// Register one library on `env`: its status, registered or the runtime's
// reason, and the devices registering it added.
static Registration registerLibrary(const OrtRuntime &rt, OrtEnv *env,
                                    const OnnxEpLibrary &lib, bool catalog)
{
  Registration r;
  r.st.lib  = lib;
  r.catalog = catalog;
  if (bundledLibraryMissing(rt, lib))
  {
    r.st.error = "not packaged with this app";
    CLPEAK_VLOG("onnx: plugin library %s (%s): %s\n", lib.name.c_str(), lib.path.c_str(),
                r.st.error.c_str());
    return r;
  }
  const std::vector<const OrtEpDevice *> before = epDevices(rt, env);
  OrtStatus *status = nullptr;
  {
    // A vendor DLL can print below any log level; the status keeps the reason.
    clpeak::ScopedConsoleMute mute;
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
  }
  r.st.registered = (status == nullptr);
  if (status)
    r.st.error = onnxStatusText(rt, status);
  else
    for (const OrtEpDevice *d : epDevices(rt, env))
      if (std::find(before.begin(), before.end(), d) == before.end())
        r.devices.push_back(d);
  CLPEAK_VLOG("onnx: plugin library %s (%s): %s\n", lib.name.c_str(),
              lib.path.c_str(),
              r.st.registered ? "registered" : r.st.error.c_str());
  return r;
}

static void unregisterLibrary(const OrtRuntime &rt, OrtEnv *env, const Registration &r)
{
  if (!r.st.registered)
    return;
  OrtStatus *status = rt.api->UnregisterExecutionProviderLibrary(env, r.st.lib.name.c_str());
  const std::string err = status ? onnxStatusText(rt, status) : std::string();
  CLPEAK_VLOG("onnx: plugin library %s (%s): unregistered%s%s\n", r.st.lib.name.c_str(),
              r.st.lib.path.c_str(), err.empty() ? "" : ": ", err.c_str());
}

static bool sameLibrary(const OnnxEpLibrary &a, const OnnxEpLibrary &b)
{
  return a.name == b.name && a.path == b.path;
}

void onnxSyncEpLibraries(const OrtRuntime &rt, OrtEnv *env)
{
  std::vector<OnnxEpLibrary> libs;
  uint64_t generation = 0;
  {
    std::lock_guard<std::mutex> cfg(g_cfgMutex);
    libs       = g_libs;
    generation = g_generation;
  }

  std::unique_lock<std::mutex> lock(g_envMutex);
  if (g_statusEnv != env)
  {
    g_regs.clear();
    g_statusEnv = env;
  }

  if (rt.apiVersion < kMinPluginApiVersion)
  {
    std::vector<OnnxEpLibrary> all = libs;
    lock.unlock();   // resolving can install a provider (below)
    for (const auto &c : catalogLibraries(rt))
      all.push_back(c.lib);
    lock.lock();
    g_regs.clear();
    for (const auto &lib : all)
    {
      Registration r;
      r.st.lib   = lib;
      r.st.error = "plugin execution providers need ONNX Runtime 1." +
                   std::to_string(kMinPluginApiVersion) +
                   " or newer; this runtime is " + rt.versionString;
      g_regs.push_back(std::move(r));
    }
    g_statusGeneration = generation;
    return;
  }

  // The configured libraries go in ahead of the catalog's, as they do in a
  // new process -- which every desktop enumeration and run is.  Whether one
  // stands for a catalog provider (below) is decided by the providers it
  // serves, which only registering it tells; and a dependency is resolved
  // by module name, so one a catalog provider had already loaded is the one
  // it would get, whatever folder it came from.  A library new to this
  // sync therefore sends the catalog's registrations out first, and those
  // still wanted come back after it.
  bool newConfigured = false;
  for (const auto &lib : libs)
  {
    bool had = false;
    for (const auto &r : g_regs)
      had = had || (!r.catalog && sameLibrary(r.st.lib, lib));
    newConfigured = newConfigured || !had;
  }

  // What stays as it is: the same registration name from the same file,
  // with the answer it had -- registered, or refused (asking again would
  // only repeat it; removing and re-adding a library does ask again).
  // Everything else registered on the environment is unregistered, which
  // unloads it.
  std::vector<Registration> kept;
  for (auto &r : g_regs)
  {
    bool want = r.catalog && !newConfigured;
    for (const auto &lib : libs)
      want = want || (!r.catalog && sameLibrary(r.st.lib, lib));
    if (want)
      kept.push_back(std::move(r));
    else
      unregisterLibrary(rt, env, r);
  }
  g_regs.clear();

  auto takeKept = [&](const OnnxEpLibrary &lib, bool catalog, Registration &out) {
    for (auto it = kept.begin(); it != kept.end(); ++it)
      if (it->catalog == catalog && sameLibrary(it->st.lib, lib))
      {
        out = std::move(*it);
        kept.erase(it);
        return true;
      }
    return false;
  };

  // In the configured order, the kept answers and the new ones.
  for (const auto &lib : libs)
  {
    Registration r;
    if (takeKept(lib, false, r))
    {
      r.st.lib.named = lib.named;
      r.st.replaces.clear();
      g_regs.push_back(std::move(r));
    }
    else
      g_regs.push_back(registerLibrary(rt, env, lib, false));
  }
  const size_t configured = g_regs.size();

  // Resolving the catalog can install a provider from the Store, which
  // takes minutes the first time: not with the status locked.  Nothing
  // reads the half-synced set meanwhile -- the status is not answered for
  // until the generation is set below, and everything else that reads it
  // goes through onnxEnv(), which this sync is holding.
  lock.unlock();
  const std::vector<CatalogLibrary> catalog = catalogLibraries(rt);
  lock.lock();

  // The catalog's providers, in its order, except one a configured library
  // stands for: registered under its name, or serving a provider of that
  // name.  The library named is the one asked for (the catalog fills in
  // what was not), and two builds of one provider in a process share what
  // loaded first: their dependencies, resolved by module name, and the
  // environment's shared allocator, keyed by memory-info name -- ORT skips
  // creating the second copy's (seen with two builds of its example plugin).
  for (const auto &c : catalog)
  {
    size_t by = configured;
    for (size_t i = 0; i < configured && by == configured; i++)
    {
      if (g_regs[i].st.lib.name == c.lib.name)
        by = i;
      for (const OrtEpDevice *d : g_regs[i].devices)
      {
        const char *name = rt.api->EpDevice_EpName(d);
        if (name && c.lib.name == name)
          by = i;
      }
    }
    Registration r;
    const bool had = takeKept(c.lib, true, r);
    if (by < configured)
    {
      g_regs[by].st.replaces.push_back(c.lib.name +
                                       (c.version.empty() ? "" : " " + c.version));
      if (had)
        unregisterLibrary(rt, env, r);
      continue;
    }
    g_regs.push_back(had ? std::move(r) : registerLibrary(rt, env, c.lib, true));
  }
  // The catalog cannot change within a process, so nothing should be left.
  for (const auto &r : kept)
    unregisterLibrary(rt, env, r);
  g_statusGeneration = generation;
}

std::vector<OnnxEpLibraryStatus> onnxEpLibraryStatus()
{
  // A configuration changed since the environment registered is not yet
  // answered for: nothing, rather than the previous set's answers under
  // the new set's names.
  std::lock_guard<std::mutex> lock(g_envMutex);
  if (g_statusGeneration != onnxEpConfigGeneration())
    return {};
  std::vector<OnnxEpLibraryStatus> out;
  for (const auto &r : g_regs)
    out.push_back(r.st);
  return out;
}

std::string onnxEpLibraryPath(const std::string &registrationName)
{
  std::lock_guard<std::mutex> lock(g_envMutex);
  for (const auto &r : g_regs)
    if (r.st.registered && r.st.lib.name == registrationName)
      return r.st.lib.path;
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
  // Whether anything is wanted, without resolving the catalog: that is the
  // sync's to do, after the configured libraries have registered.
  if (configuredLibraries().empty() && !onnxWinmlEnabled())
  {
    // A set that just became empty leaves the environment's registrations
    // standing until something syncs it -- and with every probe memoized
    // nothing might, so the libraries would stay loaded and the status go
    // on answering for libraries nobody asks for any more.  Sync it now.
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
  if (g_statusEnv != env)
    return out;

  // Each device under the registration that added it, in the order the
  // libraries were configured (then the catalog's).
  std::vector<std::pair<const OrtEpDevice *, std::string>> devices;
  for (const auto &r : g_regs)
    for (const OrtEpDevice *d : r.devices)
      devices.emplace_back(d, r.st.lib.name);

  std::map<std::string, int> seen;   // providerKey + type -> count, for labels
  for (const auto &dl : devices)
  {
    const OrtEpDevice *d = dl.first;
    const char *epName = rt.api->EpDevice_EpName(d);
    if (!epName || !*epName)
      continue;

    onnx_ep_info_t ep;
    ep.providerKey = epName;
    ep.epDevicePtr = d;
    ep.library     = dl.second;

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
    if (const OrtKeyValuePairs *md = rt.api->EpDevice_EpMetadata(d))
    {
      // The version the plugin reports for itself, if any.  Read here, with
      // provenance, because `hardware` below merges both metadata maps and
      // a "version" there could equally describe the silicon.
      const char *const *keys = nullptr;
      const char *const *vals = nullptr;
      size_t n = 0;
      rt.api->GetKeyValuePairs(md, &keys, &vals, &n);
      for (size_t k = 0; k < n; k++)
      {
        if (k == 0)
          CLPEAK_VLOG("onnx: plugin %s EP metadata (%lu entries):\n",
                      ep.providerKey.c_str(), (unsigned long)n);
        CLPEAK_VLOG("onnx: plugin %s metadata %s=%s\n", ep.providerKey.c_str(),
                    keys && keys[k] ? keys[k] : "?",
                    vals && vals[k] ? vals[k] : "?");
        if (ep.pluginVersion.empty() && keys && keys[k] && vals && vals[k] &&
            std::string(keys[k]) == "version")
        {
          const std::string v = vals[k];
          const size_t b = v.find_first_not_of(" \t");
          if (b != std::string::npos)
            ep.pluginVersion = v.substr(b, v.find_last_not_of(" \t") - b + 1);
        }
      }
    }
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
