#ifdef ENABLE_ONNX

#include <onnx/onnx_peak.h>
#include "onnx_runtime.h"
#include "onnx_probe.h"

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

    // ---- Compute (FLOPS + OPS) ---------------------------
    if (isAllowed(Benchmark::Gemm))
      runGemm(*rt, ep, cfg);

    if (isAllowed(Benchmark::Conv))
      runConv(*rt, ep, cfg);

    // ---- Phase 3: what the speed rows above cost in accuracy -------------
    if (isAllowed(Benchmark::NumericError))
      runNumericError(*rt, ep, cfg);

    // ---- Phase 4: AI composite (whole transformer block) -----------------
    if (isAllowed(Benchmark::TransformerBlock))
      runBlock(*rt, ep, cfg);

    // ---- Phase 5: bandwidth ----------------------------------------------
    if (isAllowed(Benchmark::Activation))
      runActivation(*rt, ep, cfg);
    if (isAllowed(Benchmark::TensorBW))
      runTensorBandwidth(*rt, ep, cfg);
    if (isAllowed(Benchmark::TransferBW))
      runTransferBandwidth(*rt, ep, cfg);

    // ---- Phase 6: latency ------------------------------------------------
    if (isAllowed(Benchmark::KernelLatency))
      runDispatchLatency(*rt, ep, cfg);

    currentDeviceScope = nullptr;

    // The CoreML execution provider leaves every model it compiled, weights
    // included, in Core ML's compile cache, which nothing evicts; see
    // include/common/coreml_cache.h.  A no-op everywhere else.
    clpeak::purgeCoreMLCompileCache();
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
