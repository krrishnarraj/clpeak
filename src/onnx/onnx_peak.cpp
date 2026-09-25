#ifdef ENABLE_ONNX

#include <onnx/onnx_peak.h>
#include "onnx_runtime.h"
#include "onnx_plugin.h"
#include "onnx_probe.h"
#include "onnx_session.h"
#include "onnx_winml.h"

#include <common/coreml_cache.h>
#include <common/options.h>

#include <algorithm>
#include <ostream>

// ---------------------------------------------------------------------------
// Execution-provider table
// ---------------------------------------------------------------------------
// Maps ORT registration names to how clpeak presents them.  Providers not in
// this table are still listed (name passed through) so a new EP shows up
// rather than being silently hidden; it just carries Unknown type until the
// table learns it.  The Azure EP is the one deliberate exclusion: it proxies
// inference to a remote service, and clpeak measures local silicon only.

struct EpTableEntry
{
  const char *key;
  const char *display;
  const char *type;      // "NPU" / "GPU" / "CPU"
  DeviceType  deviceType;
};

static const EpTableEntry kEpTable[] = {
  // NPU-class providers (vendor AI runtimes).  OpenVINO is not listed
  // here: one EP fronts NPU, GPU and CPU behind its `device_type` option
  // (see onnxAvailableEps below), so a single table row cannot name it.
  {"QNNExecutionProvider",        "Qualcomm QNN (Hexagon NPU)",   "NPU", DeviceType::Accelerator},
  {"VitisAIExecutionProvider",    "AMD Vitis AI (XDNA NPU)",      "NPU", DeviceType::Accelerator},
  {"CoreMLExecutionProvider",     "Apple CoreML (Neural Engine)", "NPU", DeviceType::Accelerator},
  {"NnapiExecutionProvider",      "Android NNAPI",                "NPU", DeviceType::Accelerator},

  // GPU providers
  {"NvTensorRTRTXExecutionProvider", "NVIDIA TensorRT for RTX",   "GPU", DeviceType::Gpu},
  {"TensorrtExecutionProvider",   "NVIDIA TensorRT",              "GPU", DeviceType::Gpu},
  {"CUDAExecutionProvider",       "NVIDIA CUDA",                  "GPU", DeviceType::Gpu},
  {"MIGraphXExecutionProvider",   "AMD MIGraphX",                 "GPU", DeviceType::Gpu},
  {"ROCMExecutionProvider",       "AMD ROCm",                     "GPU", DeviceType::Gpu},
  {"DmlExecutionProvider",        "DirectML",                     "GPU", DeviceType::Gpu},
  {"WebGpuExecutionProvider",     "WebGPU (Dawn)",                "GPU", DeviceType::Gpu},

  // CPU providers
  {"DnnlExecutionProvider",       "oneDNN (Dnnl)",                "CPU", DeviceType::Cpu},
  {"XnnpackExecutionProvider",    "XNNPACK",                      "CPU", DeviceType::Cpu},
  {"CPUExecutionProvider",        "ONNX Runtime CPU",             "CPU", DeviceType::Cpu},
};

static const EpTableEntry *epLookup(const std::string &key)
{
  for (const auto &e : kEpTable)
    if (key == e.key)
      return &e;
  return nullptr;
}

bool onnxEpTableEntry(const std::string &providerKey, std::string &display,
                      std::string &typeStr, DeviceType &deviceType)
{
  const EpTableEntry *t = epLookup(providerKey);
  if (!t)
    return false;
  display    = t->display;
  typeStr    = t->type;
  deviceType = t->deviceType;
  return true;
}

// The notes a run and a listing both make about the plugin libraries: a
// library someone named that would not register is said in normal output,
// like a named runtime that would not load; one the app bundled on the
// off-chance (`named == false`) only under --verbose, where a registered
// library is also confirmed with its path.  The Windows ML catalog reports
// the same way: its failures out loud, since --onnx-winml asked for it.
static void pluginNotes(const OrtRuntime &rt,
                        std::vector<std::string> &loud,
                        std::vector<std::string> &quiet)
{
  const std::vector<onnx_ep_info_t> devices = onnxPluginDevices(rt);
  for (const auto &st : onnxEpLibraryStatus())
  {
    if (!st.registered)
    {
      (st.lib.named ? loud : quiet)
          .push_back("the " + st.lib.name + " plugin library (" + st.lib.path +
                     ") did not register: " + st.error);
      continue;
    }
    // A plugin enumerates the hardware it can serve, and on a machine
    // without that hardware a registered library offers nothing -- which
    // is the answer, not a fault, but one worth a line for the person who
    // named the library.
    bool any = false;
    for (const auto &d : devices)
      any = any || d.library == st.lib.name;
    if (any)
      quiet.push_back("registered the " + st.lib.name + " plugin library from " +
                      st.lib.path);
    else
      (st.lib.named ? loud : quiet)
          .push_back("the " + st.lib.name + " plugin library (" + st.lib.path +
                     ") registered but offers no device on this machine");
  }
  if (onnxWinmlEnabled())
  {
    const OnnxWinmlResolution &res = onnxWinmlResolve(&rt, onnxWinmlPathHint());
    if (!res.error.empty())
      loud.push_back("Windows ML: " + res.error);
    for (const auto &p : res.providers)
      if (!p.ready)
        loud.push_back("Windows ML: the " + p.name +
                       (p.version.empty() ? "" : " " + p.version) +
                       " execution provider is not usable: " + p.error);
    if (res.error.empty() && res.providers.empty())
      loud.push_back("Windows ML: the catalog lists no execution provider for "
                     "this machine");
  }
  // A setup chosen after this runtime loaded: asked for and not in effect,
  // which is said out loud like a library that did not register.
  OnnxPendingSetup pending;
  if (onnxPendingSetup(pending))
  {
    std::string setup =
        pending.library.empty() ? std::string("the default search") : pending.library;
    // Windows ML only where it is part of the difference.
    if (pending.winml || onnxWinmlEnabled())
      setup += pending.winml ? ", Windows ML on" + (pending.winmlPath.empty()
                                                        ? std::string()
                                                        : " at " + pending.winmlPath)
                             : std::string(", Windows ML off");
    loud.push_back("the runtime setup chosen since ONNX Runtime " + rt.versionString +
                   " loaded (" + setup +
                   ") takes effect the next time clpeak starts: one process keeps "
                   "the runtime it loaded first");
  }
}

std::vector<onnx_ep_info_t> onnxAvailableEps(const OrtRuntime &rt)
{
  // Plugin providers first: each is here because someone asked for it, and
  // its NPU or GPU belongs ahead of the built-in CPU providers.  A plugin's
  // own CPU-class device (QNN's reference backend, when it is enabled) goes
  // after every built-in accelerator, where a CPU belongs.
  std::vector<onnx_ep_info_t> out, pluginCpus;
  for (auto &ep : onnxPluginDevices(rt))
    (ep.deviceType == DeviceType::Cpu ? pluginCpus : out).push_back(std::move(ep));

  char **providers = nullptr;
  int    count     = 0;
  OrtStatus *st = rt.api->GetAvailableProviders(&providers, &count);
  if (st)
  {
    rt.api->ReleaseStatus(st);
    out.insert(out.end(), pluginCpus.begin(), pluginCpus.end());
    return out;
  }

  for (int i = 0; i < count; i++)
  {
    std::string key = providers[i] ? providers[i] : "";
    if (key.empty() || key == "AzureExecutionProvider")
      continue;

    // OpenVINO selects its hardware per session through `device_type`,
    // so one provider registration is three benchmarkable devices.  Each
    // carries its target in epDevice; session creation passes it through
    // (see onnx_session.cpp).  Targets with no hardware behind them are
    // filtered by onnxUsableEps below -- an Arc dGPU box with no NPU
    // lists GPU and CPU, not a phantom NPU.
    if (key == "OpenVINOExecutionProvider")
    {
      static const struct { const char *dev; const char *display; const char *type; DeviceType dt; } kOv[] = {
        {"NPU", "Intel OpenVINO (NPU)", "NPU", DeviceType::Accelerator},
        {"GPU", "Intel OpenVINO (GPU)", "GPU", DeviceType::Gpu},
        {"CPU", "Intel OpenVINO (CPU)", "CPU", DeviceType::Cpu},
      };
      for (const auto &t : kOv)
      {
        onnx_ep_info_t ep;
        ep.providerKey = key;
        ep.epDevice    = t.dev;
        ep.displayName = t.display;
        ep.typeStr     = t.type;
        ep.deviceType  = t.dt;
        out.push_back(std::move(ep));
      }
      continue;
    }

    onnx_ep_info_t ep;
    ep.providerKey = key;
    if (const EpTableEntry *t = epLookup(key))
    {
      ep.displayName = t->display;
      ep.typeStr     = t->type;
      ep.deviceType  = t->deviceType;
    }
    else
    {
      ep.displayName = key;   // unrecognised EP: show it as-is
    }
    out.push_back(std::move(ep));
  }
  // Documented as never failing, but it is ORT_API2_STATUS so it still
  // returns one; release it rather than leaking the status object.
  if (OrtStatus *rel = rt.api->ReleaseAvailableProviders(providers, count))
    rt.api->ReleaseStatus(rel);

  // GetAvailableProviders returns default-priority order (accelerators
  // before CPU), which is also the order we want to benchmark in.  Keep it,
  // with a plugin's CPU-class devices after the built-in accelerators and
  // before the built-in CPU providers.
  auto firstCpu = std::find_if(out.begin(), out.end(), [](const onnx_ep_info_t &e) {
    return !e.epDevicePtr && e.deviceType == DeviceType::Cpu;
  });
  out.insert(firstCpu, pluginCpus.begin(), pluginCpus.end());
  return out;
}

// Providers the box can actually run, in the same order.  This is what
// listing and benchmarking both consume; the raw capability list above is
// only the input.  Viability answers are memoized per runtime and target,
// so the second caller in a process pays nothing.
std::vector<onnx_ep_info_t> onnxUsableEps(
    const OrtRuntime &rt,
    std::vector<std::pair<onnx_ep_info_t, std::string>> *skipped)
{
  std::vector<onnx_ep_info_t> out;
  for (const auto &ep : onnxAvailableEps(rt))
  {
    std::string reason;
    if (onnxEpViable(rt, ep, reason))
    {
      out.push_back(ep);
    }
    else
    {
      CLPEAK_VLOG("onnx: %s not usable (%s), skipping\n",
                  ep.displayName.c_str(), reason.c_str());
      if (skipped)
        skipped->emplace_back(ep, reason);
    }
  }
  return out;
}

OnnxRuntimeStatus onnxRuntimeStatus()
{
  OnnxRuntimeStatus st;
#ifdef CLPEAK_ONNX_STATIC
  st.linkedIn = true;
#endif
  if (const OrtRuntime *rt = ortRuntime())
  {
    st.available = true;
    st.version   = rt->versionString;
    st.path      = rt->path;
  }
  else
  {
    st.error = onnxLoadDiagnostic();
    if (st.error.empty())
      st.error = "onnxruntime library not found";
  }
  // What the environment registered; a library configured since is not in
  // it until the next enumeration or run syncs the set onto it (settings
  // screens refresh after enumerating).
  st.epLibraries  = onnxEpLibraryStatus();
  st.winmlEnabled = onnxWinmlEnabled();
  if (st.winmlEnabled && st.available)
  {
    // Only what the last enumeration or run found: resolving the catalog
    // here could install a provider, and a status query must not.  Nothing
    // resolved yet leaves both empty -- pending, like the plugin-library
    // status -- so a settings screen shows it as such rather than an error.
    if (const OnnxWinmlResolution *res = onnxWinmlResolved(ortRuntime()))
    {
      st.winmlPath  = res->dllPath;
      st.winmlError = res->error;
    }
  }
  OnnxPendingSetup pending;
  st.pending = onnxPendingSetup(pending);
  if (st.pending)
  {
    st.pendingLibrary   = pending.library;
    st.pendingWinml     = pending.winml;
    st.pendingWinmlPath = pending.winmlPath;
  }
  return st;
}

// ---------------------------------------------------------------------------
// OnnxPeak
// ---------------------------------------------------------------------------

OnnxPeak::OnnxPeak() = default;
OnnxPeak::~OnnxPeak() = default;

int OnnxPeak::runAll()
{
  const OrtRuntime *rt = ortRuntime();
  if (!rt)
  {
    std::string why = onnxLoadDiagnostic();
    log->note("ONNX: " +
              (why.empty() ? std::string("onnxruntime library not found") : why) +
              "\n");
    return 0;   // absent runtime is not an error, like a missing GPU driver
  }

  std::vector<std::pair<onnx_ep_info_t, std::string>> skipped;
  auto eps = onnxUsableEps(*rt, &skipped);
  // Verbose only: a provider that cannot run here is the normal case for a
  // missing accelerator (OpenVINO NPU with no NPU, a declining NNAPI), not
  // a warning, so the default output stays a device table.
  for (const auto &sk : skipped)
    CLPEAK_VLOG("ONNX: skipping %s (%s)\n", sk.first.displayName.c_str(),
                sk.second.c_str());
  // The plugin libraries and the Windows ML catalog: what someone asked for
  // and did not get is said in normal output (pluginNotes explains which).
  {
    std::vector<std::string> loud, quiet;
    pluginNotes(*rt, loud, quiet);
    for (const auto &n : loud)
      log->note("ONNX: " + n + "\n");
    for (const auto &n : quiet)
      CLPEAK_VLOG("ONNX: %s\n", n.c_str());
  }
  if (eps.empty())
  {
    log->note("ONNX: no execution providers available\n");
    return 0;
  }

  // A stock onnxruntime is built CPU-only, and its provider list says so.
  // Without this note the backend looks broken on a machine with an obvious
  // GPU or NPU in it -- the accelerator is fine, the runtime just cannot
  // reach it.
  bool hasAccelerator = false;
  for (const auto &ep : eps)
    if (ep.deviceType == DeviceType::Accelerator || ep.deviceType == DeviceType::Gpu)
      hasAccelerator = true;
  if (!hasAccelerator)
    log->note("ONNX: this onnxruntime build exposes CPU providers only -- "
              "install a GPU/NPU-enabled build and point --onnx-lib at it "
              "to benchmark accelerators\n");

  auto backendScope = log->beginBackend("ONNX");

  for (int idx = 0; idx < (int)eps.size(); idx++)
  {
    if (clpeak::cancelRequested())
      break;
    if (!isDeviceSelected(idx))
      continue;

    const onnx_ep_info_t &ep = eps[idx];

    benchmark_config_t cfg = benchmark_config_t::forDevice(ep.deviceType);
    cfg.targetTimeUs = targetTimeUs;

    std::vector<DeviceProp> details = {
        {"Execution provider", ep.providerKey},
        {"Type", ep.typeStr.empty() ? "Unknown" : ep.typeStr},
        {"ONNX Runtime", rt->versionString},
    };
    if (ep.epDevicePtr)
    {
      // A plugin provider: what the runtime says about the silicon behind
      // this device -- the vendor and whatever metadata the provider
      // attached (a SoC model, a device name), which is the only inventory
      // an NPU offers.  Bus trivia the provider also reports (PCI ids,
      // device indices, its own path and version) is not shown.
      static const char *const kHidden[] = {
          "Device", "card_idx", "Discrete", "pci_bus_id",
          "cuda_compute_capability", "cuda_device_id", "version",
          "library_path", "LUID", "DxgiAdapterNumber",
          "DxgiHighPerformanceIndex", "DxgiVideoMemory", "ov_device",
          "ov_meta_device", "AVAILABLE_DEVICES", "DEVICE_ARCHITECTURE",
          "DEVICE_GOPS", "DEVICE_LUID", "DEVICE_PCI_INFO", "DEVICE_TYPE",
          "DEVICE_UUID", "FULL_DEVICE_NAME", "GPU_DEVICE_ID",
          "GPU_DEVICE_MAX_ALLOC_MEM_SIZE", "GPU_DEVICE_TOTAL_MEM_SIZE",
          "GPU_EXECUTION_UNITS_COUNT", "GPU_MEMORY_STATISTICS",
          "GPU_UARCH_VERSION", "MAX_BATCH_SIZE", "OPTIMAL_BATCH_SIZE",
          "OPTIMIZATION_CAPABILITIES", "RANGE_FOR_ASYNC_INFER_REQUESTS",
          "RANGE_FOR_STREAMS", "nv_ep_ort_api_version",
      };
      if (!ep.vendor.empty())
        details.push_back({"Vendor", ep.vendor});
      for (const auto &kv : ep.hardware)
      {
        bool hidden = false;
        for (const char *h : kHidden)
          if (kv.first == h)
          {
            hidden = true;
            break;
          }
        if (!hidden)
          details.push_back({kv.first, kv.second});
      }
    }

    auto deviceScope = backendScope.beginDevice({
        ep.displayName,
        "",   // platform defaults to "ONNX"
        rt->versionString,
        details,
        -1,
        idx,
        ep.deviceType,
    });
    currentDeviceScope = &deviceScope;

    // A device lost under a previous provider is not this one's problem.
    onnxClearDeviceLost();

    // Global tiny probe once per EP: learn which dtypes this EP can
    // actually run at 64^3 before paying 1024^3 (QNN HTP: 0.5s vs 33s).
    // Subsequent runGemm/runConv etc consult the cache instead of
    // re-probing per variant.
    if (isAllowed(Benchmark::Gemm) || isAllowed(Benchmark::Conv) ||
        isAllowed(Benchmark::NumericError) || isAllowed(Benchmark::TransformerBlock))
    {
      (void)onnxProbeGemmCache(*rt, ep);
    }
    // Fresh folding record for this EP even when gemm itself is filtered
    // out: otherwise a stale entry from an earlier run in the same process
    // would suppress numeric-error rows that have no paired rate to stay in
    // step with.  runGemm clears again and repopulates when it runs.
    if (isAllowed(Benchmark::Gemm) || isAllowed(Benchmark::NumericError))
      onnxClearGemmFolded(ep);

    // Only a provider with a device of its own can lose one, and the latch is
    // process-wide: ORT's device-lost callback can still fire for the GPU that
    // just died while the CPU provider is running, and that is the GPU's news,
    // not the CPU's.
    const bool canLoseDevice = (ep.deviceType != DeviceType::Cpu);
    auto deviceLost = [&] { return canLoseDevice && onnxDeviceLost(); };

    // Run one benchmark, unless the device has already been lost -- see
    // onnxDeviceLost().  The probe above can lose it before the first test:
    // on a Pixel 7a the 32^3 gemm probe hangs the GPU and the driver resets
    // it, so the whole provider is gone before a single row is measured.
    auto phase = [&](Benchmark b, int (OnnxPeak::*fn)(const OrtRuntime &,
                                                      const onnx_ep_info_t &,
                                                      benchmark_config_t &)) {
      if (!isAllowed(b) || clpeak::cancelRequested() || deviceLost())
        return;
      (this->*fn)(*rt, ep, cfg);
    };

    // ---- Compute (FLOPS + OPS) ---------------------------
    phase(Benchmark::Gemm, &OnnxPeak::runGemm);
    phase(Benchmark::Conv, &OnnxPeak::runConv);
    // ---- Phase 3: what the speed rows above cost in accuracy -------------
    phase(Benchmark::NumericError, &OnnxPeak::runNumericError);
    // ---- Phase 4: AI composite (whole transformer block) -----------------
    phase(Benchmark::TransformerBlock, &OnnxPeak::runBlock);
    // ---- Phase 5: bandwidth ----------------------------------------------
    phase(Benchmark::Activation, &OnnxPeak::runActivation);
    phase(Benchmark::TensorBW, &OnnxPeak::runTensorBandwidth);
    phase(Benchmark::TransferBW, &OnnxPeak::runTransferBandwidth);
    // ---- Phase 6: latency ------------------------------------------------
    phase(Benchmark::KernelLatency, &OnnxPeak::runDispatchLatency);

    // One line that says what happened, in place of the dozens of rows each
    // remaining test would otherwise have filed against a device that cannot
    // answer.  An error, not a warning: nothing was measured here.
    if (deviceLost())
      CLPEAK_LOG(Error,
                 "ONNX %s: the device was lost during the run (the driver reset "
                 "it, and the provider cannot recover); no further tests were "
                 "attempted on it and any rows already filed are not measurements",
                 ep.displayName.c_str());

    currentDeviceScope = nullptr;

    // The CoreML execution provider leaves every model it compiled, weights
    // included, in Core ML's compile cache, which nothing evicts; see
    // include/common/coreml_cache.h.  A no-op everywhere else.
    clpeak::purgeCoreMLCompileCache();

    // What a plugin provider holds on the environment goes with its device.
    // Registering a plugin library makes ORT create a shared allocator for
    // each of its devices on the OrtEnv (ORT 1.23+), from the factory's own
    // implementation -- for a GPU provider a device-memory pool -- and that
    // allocator outlives every session.  Sessions are what clpeak releases;
    // the pool is what a provider that runs after it sees.  On an RTX 5060
    // (ORT 1.30, the CUDA provider as a plugin), every TensorRT graph with a
    // matmul or a convolution failed to build after the plugin's tests, and
    // the built-in CUDA provider after that lost its largest points, where
    // the same providers run clean alone -- the signature of a device whose
    // memory is spoken for.  Creating the shared allocator again replaces
    // the one registration made (the API's own "create/replace"), so the
    // old one is destroyed, returning what it held, and the device is left
    // exactly as registration left it for a later run.  A memory type the
    // factory does not provide is skipped; a refusal is logged, not fatal.
    if (ep.epDevicePtr && rt->apiVersion >= 23 && rt->api->CreateSharedAllocator &&
        rt->api->EpDevice_MemoryInfo)
    {
      if (OrtEnv *env = onnxEnv(*rt))
      {
        const OrtDeviceMemoryType kinds[] = {OrtDeviceMemoryType_DEFAULT,
                                             OrtDeviceMemoryType_HOST_ACCESSIBLE};
        for (OrtDeviceMemoryType kind : kinds)
        {
          if (!rt->api->EpDevice_MemoryInfo(ep.epDevicePtr, kind))
            continue;
          OrtStatus *st = rt->api->CreateSharedAllocator(env, ep.epDevicePtr, kind,
                                                         OrtDeviceAllocator, nullptr, nullptr);
          if (st)
            CLPEAK_VLOG("onnx[%s]: could not replace the shared %s allocator: %s\n",
                        ep.displayName.c_str(),
                        kind == OrtDeviceMemoryType_DEFAULT ? "device" : "host-accessible",
                        onnxStatusText(*rt, st).c_str());
          else
            CLPEAK_VLOG("onnx[%s]: replaced the shared %s allocator, releasing what it held\n",
                        ep.displayName.c_str(),
                        kind == OrtDeviceMemoryType_DEFAULT ? "device" : "host-accessible");
        }
      }
    }

    // The descriptor budget after each provider: a runtime that leaks them
    // (LiteRT's WebGPU accelerator does) starves everything that follows,
    // and this is the line that shows it happening.
    {
      unsigned long fds = 0, limit = 0;
      if (clpeak::openFileDescriptors(fds, limit))
        CLPEAK_VLOG("onnx[%s]: %lu open file descriptors%s\n", ep.displayName.c_str(), fds,
                    limit ? (" (limit " + std::to_string(limit) + ")").c_str() : "");
    }
  }

  return 0;
}

BackendInventory OnnxPeak::enumerate()
{
  BackendInventory inv;
  inv.id = kBackend;

  const OrtRuntime *rt = ortRuntime();
  if (!rt)
  {
    const std::string why = onnxLoadDiagnostic();
    inv.unavailableReason = why.empty() ? "onnxruntime library not found" : why;
    return inv;
  }
  inv.info = "ONNX Runtime " + rt->versionString;

  // Capabilities, not viability: a backend with providers that all fail
  // the probe still counts as available (with an empty device list), so
  // --list-devices reports the runtime rather than "library not found".
  if (onnxAvailableEps(*rt).empty())
  {
    inv.unavailableReason = "the runtime registers no execution providers";
    return inv;
  }
  inv.available = true;

  // Providers the runtime names but nothing here can run, with the reason.
  // Answers are memoized, so asking again costs nothing.
  std::vector<std::pair<onnx_ep_info_t, std::string>> skipped;
  auto eps = onnxUsableEps(*rt, &skipped);
  for (const auto &sk : skipped)
    inv.notes.push_back("skipping " + sk.first.displayName + ": " + sk.second);
  // A plugin library someone named that did not register, or a Windows ML
  // failure, is a fact about this listing and not a verbose-only note:
  // it rides the info line, where a missing runtime's reason would be.
  {
    std::vector<std::string> loud, quiet;
    pluginNotes(*rt, loud, quiet);
    for (const auto &n : loud)
      inv.info += "; " + n;
    for (const auto &n : quiet)
      inv.notes.push_back(n);
  }

  InventoryPlatform plat;
  plat.index = 0;
  plat.name  = "ONNX Runtime";

  for (int i = 0; i < (int)eps.size(); i++)
  {
    InventoryDevice dev;
    dev.index         = i;
    dev.name          = eps[i].displayName;
    dev.typeStr       = eps[i].typeStr;
    plat.devices.push_back(std::move(dev));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

#endif // ENABLE_ONNX
