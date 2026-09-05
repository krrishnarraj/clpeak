#ifdef ENABLE_ONNX

#include <onnx/onnx_peak.h>
#include "onnx_runtime.h"
#include "onnx_probe.h"

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

std::vector<onnx_ep_info_t> onnxAvailableEps(const OrtRuntime &rt)
{
  std::vector<onnx_ep_info_t> out;

  char **providers = nullptr;
  int    count     = 0;
  OrtStatus *st = rt.api->GetAvailableProviders(&providers, &count);
  if (st)
  {
    rt.api->ReleaseStatus(st);
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
  // before CPU), which is also the order we want to benchmark in.  Keep it.
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
#if defined(__ANDROID__)
    // The WebGPU execution provider is not offered on Android.  Its Dawn
    // backend shares the process GPU with the app's own UI, where a lost
    // device is unrecoverable -- and its Android support is experimental:
    // every matmul fails there (Mali-G710 never completes the buffer
    // download), and one fp16 run never returned at all, wedging the
    // phone's GPU until reboot.  Other OSes keep the provider.  Filtered
    // here rather than at enumeration so listing and runs agree, and with
    // a reason so the device list says why it is missing.
    if (ep.providerKey == "WebGpuExecutionProvider")
    {
      static const std::string reason =
          "the WebGPU execution provider is disabled on Android: it shares "
          "the app's GPU, where a lost device cannot be recovered, and its "
          "Android support is experimental";
      CLPEAK_VLOG("onnx: %s not usable (%s), skipping\n",
                  ep.displayName.c_str(), reason.c_str());
      if (skipped)
        skipped->emplace_back(ep, reason);
      continue;
    }
#endif
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
  return st;
}

// ---------------------------------------------------------------------------
// OnnxPeak
// ---------------------------------------------------------------------------

OnnxPeak::OnnxPeak() = default;
OnnxPeak::~OnnxPeak() = default;

void OnnxPeak::applyOptions(const CliOptions &opts)
{
  Peak::applyOptions(opts);
  deviceIndices = opts.onnxDeviceIndices;
}

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
    if (!deviceIndices.empty() &&
        std::find(deviceIndices.begin(), deviceIndices.end(), idx) == deviceIndices.end())
      continue;

    const onnx_ep_info_t &ep = eps[idx];

    benchmark_config_t cfg = benchmark_config_t::forDevice(ep.deviceType);
    cfg.targetTimeUs = targetTimeUs;

    std::vector<DeviceProp> details = {
        {"Execution provider", ep.providerKey},
        {"Type", ep.typeStr.empty() ? "Unknown" : ep.typeStr},
        {"ONNX Runtime", rt->versionString},
    };
    // The OpenVINO target is part of what was measured (NPU vs GPU vs
    // CPU are different silicon behind one provider name), so record it
    // alongside the provider rather than leaving rows to guess.
    if (!ep.epDevice.empty())
      details.push_back({"OpenVINO device", ep.epDevice});

    auto deviceScope = backendScope.beginDevice({
        ep.displayName,
        "",   // platform defaults to "ONNX"
        rt->versionString,
        details,
        -1,
        idx,
    });
    currentDeviceScope = &deviceScope;

    // Global tiny probe once per EP: learn which dtypes this EP can
    // actually run at 64^3 before paying 1024^3 (QNN HTP: 0.5s vs 33s).
    // Subsequent runGemm/runConv etc consult the cache instead of
    // re-probing per variant.
    if (isAllowed(Benchmark::OnnxGemm) || isAllowed(Benchmark::OnnxConv) ||
        isAllowed(Benchmark::OnnxNumericError) || isAllowed(Benchmark::OnnxBlock))
    {
      (void)onnxProbeGemmCache(*rt, ep);
    }

    // ---- Compute (FLOPS + OPS) ---------------------------
    if (isAllowed(Benchmark::OnnxGemm))
      runGemm(*rt, ep, cfg);

    if (isAllowed(Benchmark::OnnxConv))
      runConv(*rt, ep, cfg);

    // ---- Phase 3: what the speed rows above cost in accuracy -------------
    if (isAllowed(Benchmark::OnnxNumericError))
      runNumericError(*rt, ep, cfg);

    // ---- Phase 4: AI composite (whole transformer block) -----------------
    if (isAllowed(Benchmark::OnnxBlock))
      runBlock(*rt, ep, cfg);

    // ---- Phase 5: bandwidth ----------------------------------------------
    if (isAllowed(Benchmark::OnnxActivation))
      runActivation(*rt, ep, cfg);
    if (isAllowed(Benchmark::OnnxTensorBW))
      runTensorBandwidth(*rt, ep, cfg);
    if (isAllowed(Benchmark::OnnxTransferBW))
      runTransferBandwidth(*rt, ep, cfg);

    // ---- Phase 6: latency ------------------------------------------------
    if (isAllowed(Benchmark::OnnxDispatchLatency))
      runDispatchLatency(*rt, ep, cfg);

    currentDeviceScope = nullptr;
  }

  return 0;
}

BackendInventory OnnxPeak::enumerate()
{
  BackendInventory inv;
  inv.backend = "ONNX";

  const OrtRuntime *rt = ortRuntime();
  if (!rt)
    return inv;

  // Capabilities, not viability: a backend with providers that all fail
  // the probe still counts as available (with an empty device list), so
  // --list-devices reports the runtime rather than "library not found".
  if (onnxAvailableEps(*rt).empty())
    return inv;
  inv.available = true;

  auto eps = onnxUsableEps(*rt);

  InventoryPlatform plat;
  plat.index = 0;
  plat.name  = "ONNX Runtime " + rt->versionString;

  for (int i = 0; i < (int)eps.size(); i++)
  {
    InventoryDevice dev;
    dev.index         = i;
    dev.name          = eps[i].displayName;
    dev.typeStr       = eps[i].typeStr;
    dev.driverVersion = rt->versionString;
    plat.devices.push_back(std::move(dev));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

void OnnxPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
  os << "\n=== ONNX backend ===\n";
  if (!b.available)
  {
    std::string why = onnxLoadDiagnostic();
    os << "ONNX: "
       << (why.empty() ? std::string("onnxruntime library not found") : why)
       << "\n";
    return;
  }
  for (const auto &plat : b.platforms)
  {
    os << plat.name << "\n";
    for (const auto &d : plat.devices)
    {
      os << "  ONNX Device " << d.index << ": " << d.name;
      if (!d.typeStr.empty())
        os << " [" << d.typeStr << "]";
      os << "\n";
    }
  }
  // Providers the runtime names but nothing here can run: named with the
  // reason under --verbose only, so a missing NPU reads as absent hardware
  // rather than a detection failure when debugging, while the default
  // listing stays a device table.  Answers are memoized, so this re-probe
  // after enumerate() costs nothing.
  if (clpeak::verboseEnabled())
    if (const OrtRuntime *rt = ortRuntime())
    {
      std::vector<std::pair<onnx_ep_info_t, std::string>> skipped;
      (void)onnxUsableEps(*rt, &skipped);
      for (const auto &sk : skipped)
        os << "  (skipping " << sk.first.displayName << ": " << sk.second
           << ")\n";
    }
}

#endif // ENABLE_ONNX
