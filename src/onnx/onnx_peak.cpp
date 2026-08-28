#ifdef ENABLE_ONNX

#include <onnx/onnx_peak.h>
#include "onnx_runtime.h"

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
  // NPU-class providers (vendor AI runtimes)
  {"QNNExecutionProvider",        "Qualcomm QNN (Hexagon NPU)",   "NPU", DeviceType::Accelerator},
  {"OpenVINOExecutionProvider",   "Intel OpenVINO",               "NPU", DeviceType::Accelerator},
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

  auto eps = onnxAvailableEps(*rt);
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
              "install a GPU/NPU-enabled build and point "
              "CLPEAK_ONNXRUNTIME_LIB at it to benchmark accelerators\n");

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

    auto deviceScope = backendScope.beginDevice({
        ep.displayName,
        "",   // platform defaults to "ONNX"
        rt->versionString,
        {
            {"Execution provider", ep.providerKey},
            {"Type", ep.typeStr.empty() ? "Unknown" : ep.typeStr},
            {"ONNX Runtime", rt->versionString},
        },
        -1,
        idx,
    });
    currentDeviceScope = &deviceScope;

    // ---- Phase 1: floating-point compute --------------------------------
    if (isAllowedAs(Benchmark::OnnxGemm, Category::FpCompute))
      runGemm(*rt, ep, cfg, Category::FpCompute);

    if (isAllowed(Benchmark::OnnxConv))
      runConv(*rt, ep, cfg);

    // ---- Phase 2: integer compute (int8 QDQ) -----------------------------
    if (isAllowedAs(Benchmark::OnnxGemm, Category::IntCompute))
      runGemm(*rt, ep, cfg, Category::IntCompute);

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

  auto eps = onnxAvailableEps(*rt);
  if (eps.empty())
    return inv;
  inv.available = true;

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
}

#endif // ENABLE_ONNX
