#ifdef ENABLE_COREML

#include <coreml/coreml_peak.h>
#include "coreml_internal.h"
#include "coreml_session.h"

#include <common/coreml_cache.h>
#include <common/options.h>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <ostream>
#include <sstream>

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------
// Core ML reports its compute devices since macOS 14 / iOS 17, and its
// compute plan -- the per-operation placement this backend's honesty rests on
// -- since 14.4 / 17.4.  Below that the backend declines to run rather than
// publish Neural Engine rows it cannot verify.

std::string coremlOsVersionString()
{
  NSOperatingSystemVersion v = NSProcessInfo.processInfo.operatingSystemVersion;
  std::ostringstream ss;
#if TARGET_OS_IPHONE
  ss << "iOS ";
#else
  ss << "macOS ";
#endif
  ss << v.majorVersion << "." << v.minorVersion;
  if (v.patchVersion)
    ss << "." << v.patchVersion;
  return ss.str();
}

int coremlOsMajorVersion()
{
  return (int)NSProcessInfo.processInfo.operatingSystemVersion.majorVersion;
}

int coremlSpecVersion()
{
  if (@available(macOS 26.0, iOS 26.0, *))
    return 10;
  if (@available(macOS 15.0, iOS 18.0, *))
    return 9;
  return 8;
}

std::vector<coreml_device_info_t> coremlDevices(std::string *why)
{
  std::vector<coreml_device_info_t> out;
  bool planApi = false;
  if (@available(macOS 14.4, iOS 17.4, *))
    planApi = true;
  if (!planApi)
  {
    if (why)
      *why = "the Core ML backend needs macOS 14.4 / iOS 17.4 for its compute "
             "plan, which is what proves an operation ran on the device named";
    return out;
  }
  @autoreleasepool
  {
    int gpuIndex = 0;
    std::vector<coreml_device_info_t> gpus, cpus;
    for (id<MLComputeDeviceProtocol> d in MLAllComputeDevices())
    {
      coreml_device_info_t dev;
      dev.kind = coremlKindOf(d);
      switch (dev.kind)
      {
      case CoremlDeviceKind::NeuralEngine:
        dev.coreCount = (int)[(MLNeuralEngineComputeDevice *)d totalCoreCount];
        dev.displayName = "Apple Neural Engine";
        dev.typeStr = "NPU";
        dev.deviceType = DeviceType::Accelerator;
        out.push_back(dev);   // accelerators first
        break;
      case CoremlDeviceKind::Gpu:
      {
        id<MTLDevice> mtl = [(MLGPUComputeDevice *)d metalDevice];
        dev.gpuIndex = gpuIndex++;
        dev.gpuName = mtl && mtl.name ? mtl.name.UTF8String : "GPU";
        dev.displayName = "GPU via Core ML (" + dev.gpuName + ")";
        dev.typeStr = "GPU";
        dev.deviceType = DeviceType::Gpu;
        gpus.push_back(dev);
        break;
      }
      case CoremlDeviceKind::Cpu:
        dev.displayName = "CPU via Core ML";
        dev.typeStr = "CPU";
        dev.deviceType = DeviceType::Cpu;
        cpus.push_back(dev);
        break;
      }
    }
    out.insert(out.end(), gpus.begin(), gpus.end());
    out.insert(out.end(), cpus.begin(), cpus.end());
  }
  if (out.empty() && why)
    *why = "Core ML reports no compute devices";
  return out;
}

// ---------------------------------------------------------------------------
// CoreMLPeak
// ---------------------------------------------------------------------------

CoreMLPeak::CoreMLPeak() = default;
CoreMLPeak::~CoreMLPeak() = default;

int CoreMLPeak::runAll()
{
  std::string why;
  auto devs = coremlDevices(&why);
  if (devs.empty())
  {
    log->note("Core ML: " + why + "\n");
    return 0;
  }

  // Whatever earlier runs left in Core ML's compile cache -- a crashed run's
  // models, or the ONNX backend's CoreML sessions -- goes before the first
  // model of this one is built.
  clpeak::purgeCoreMLCompileCache();

  auto backendScope = log->beginBackend("CoreML");
  const std::string os = coremlOsVersionString();
  const int spec = coremlSpecVersion();

  for (int idx = 0; idx < (int)devs.size(); idx++)
  {
    if (clpeak::cancelRequested())
      break;
    if (!isDeviceSelected(idx))
      continue;

    const coreml_device_info_t &dev = devs[idx];

    benchmark_config_t cfg = benchmark_config_t::forDevice(dev.deviceType);
    cfg.targetTimeUs = targetTimeUs;

    const char *units = dev.kind == CoremlDeviceKind::NeuralEngine ? "cpuAndNeuralEngine"
                        : dev.kind == CoremlDeviceKind::Gpu        ? "cpuAndGPU"
                                                                   : "cpuOnly";
    std::vector<DeviceProp> details = {
        {"Compute units", units},
        {"Type", dev.typeStr},
        {"Newest model spec", "version " + std::to_string(spec) + " (" + coremlOpsetName(spec) + ")"},
    };
    if (dev.coreCount > 0)
      details.push_back({"Neural Engine cores", std::to_string(dev.coreCount)});

    auto deviceScope = backendScope.beginDevice({
        dev.displayName,
        "",   // platform defaults to the backend name
        os,
        details,
        -1,
        idx,
    });
    currentDeviceScope = &deviceScope;

    // ---- Compute (FLOPS + OPS) ---------------------------------------------
    if (isAllowed(Benchmark::Gemm))
      runGemm(dev, cfg);
    if (isAllowed(Benchmark::Conv))
      runConv(dev, cfg);

    // ---- What the speed rows cost in accuracy ------------------------------
    if (isAllowed(Benchmark::NumericError))
      runNumericError(dev, cfg);

    // ---- AI composite (whole transformer block) ----------------------------
    if (isAllowed(Benchmark::TransformerBlock))
      runBlock(dev, cfg);

    // ---- Bandwidth ---------------------------------------------------------
    if (isAllowed(Benchmark::Activation))
      runActivation(dev, cfg);
    if (isAllowed(Benchmark::TensorBW))
      runTensorBandwidth(dev, cfg);
    if (isAllowed(Benchmark::TransferBW))
      runTransferBandwidth(dev, cfg);

    // ---- Latency -----------------------------------------------------------
    if (isAllowed(Benchmark::KernelLatency))
      runDispatchLatency(dev, cfg);

    currentDeviceScope = nullptr;
  }

  return 0;
}

BackendInventory CoreMLPeak::enumerate()
{
  BackendInventory inv;
  inv.id = kBackend;
  inv.info = coremlOsVersionString();

  std::string why;
  auto devs = coremlDevices(&why);
  if (devs.empty())
  {
    inv.unavailableReason = why.empty() ? "no compute devices" : why;
    return inv;
  }
  inv.available = true;

  InventoryPlatform plat;
  plat.index = 0;
  plat.name = "Core ML";
  for (int i = 0; i < (int)devs.size(); i++)
  {
    InventoryDevice d;
    d.index = i;
    d.name = devs[i].displayName;
    d.typeStr = devs[i].typeStr;
    d.numComputeUnits = devs[i].coreCount > 0 ? (unsigned)devs[i].coreCount : 0;
    plat.devices.push_back(std::move(d));
  }
  inv.platforms.push_back(std::move(plat));
  return inv;
}

#endif // ENABLE_COREML
