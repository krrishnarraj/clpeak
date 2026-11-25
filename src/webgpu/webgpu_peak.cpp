#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/options.h>

#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct InstanceHolder
{
  const WebgpuRuntime *rt = nullptr;
  std::vector<std::shared_ptr<WebgpuDevice::CallbackState>> callbackStates;
  WGPUInstance instance = nullptr;
  bool timedWaitAny = false;
  std::vector<WGPUAdapter> adapters;

  ~InstanceHolder()
  {
    if (!rt)
      return;
    for (WGPUAdapter a : adapters)
      if (a)
        rt->api.wgpuAdapterRelease(a);
    adapters.clear();
    if (instance)
    {
      rt->api.wgpuInstanceRelease(instance);
      instance = nullptr;
    }
    // Release callback states only after wgpuInstanceRelease drains/cancels
    // all callbacks tied to this instance.
    callbackStates.clear();
  }

  bool init(const WebgpuRuntime *runtime)
  {
    rt = runtime;
    // Try requesting TimedWaitAny when supported for zero-spin blocking waits;
    // fall back to a default instance + ProcessEvents otherwise.
    const bool hasTimedWaitFeature =
        rt->api.wgpuInstanceWaitAny &&
        rt->api.wgpuHasInstanceFeature &&
        rt->api.wgpuHasInstanceFeature(WGPUInstanceFeatureName_TimedWaitAny) != 0;
    if (hasTimedWaitFeature)
    {
      WGPUInstanceFeatureName feat = WGPUInstanceFeatureName_TimedWaitAny;
      WGPUInstanceLimits limits = WGPU_INSTANCE_LIMITS_INIT;
      limits.timedWaitAnyMaxCount = 1;
      WGPUInstanceDescriptor desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
      desc.requiredFeatureCount = 1;
      desc.requiredFeatures = &feat;
      desc.requiredLimits = &limits;
      instance = rt->api.wgpuCreateInstance(&desc);
      if (instance)
        timedWaitAny = true;
    }
    if (!instance)
    {
      WGPUInstanceDescriptor desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
      instance = rt->api.wgpuCreateInstance(&desc);
      timedWaitAny = false;
    }
    if (!instance)
      return false;

    // Probe adapters via standard requestAdapter: high-performance GPU,
    // low-power GPU, and fallback (CPU), deduplicating identical adapters.
    struct Probe
    {
      WGPUPowerPreference power;
      WGPUBool fallback;
    };
    const Probe probes[] = {
      {WGPUPowerPreference_HighPerformance, WGPU_FALSE},
      {WGPUPowerPreference_LowPower, WGPU_FALSE},
      {WGPUPowerPreference_Undefined, WGPU_TRUE},
    };

    struct AdapterKey
    {
      uint32_t vendorID;
      uint32_t deviceID;
      uint32_t backendType;
      uint32_t adapterType;
      std::string device;
      bool operator==(const AdapterKey &o) const
      {
        return vendorID == o.vendorID && deviceID == o.deviceID &&
               backendType == o.backendType && adapterType == o.adapterType &&
               device == o.device;
      }
    };
    std::vector<AdapterKey> seen;

    for (const Probe &p : probes)
    {
      WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
      opts.powerPreference = p.power;
      opts.forceFallbackAdapter = p.fallback;

      struct ReqState
      {
        std::atomic<bool> done{false};
        WGPURequestAdapterStatus status = WGPURequestAdapterStatus_Error;
        WGPUAdapter adapter = nullptr;
      };
      auto [req, ud] = wgpuMakeCallbackState<ReqState>();

      WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
      cb.mode = timedWaitAny ? WGPUCallbackMode_WaitAnyOnly
                             : WGPUCallbackMode_AllowProcessEvents;
      cb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adp,
                       WGPUStringView, void *ud1, void *)
      {
        auto s = wgpuTakeCallbackState<ReqState>(ud1);
        s->status = status;
        s->adapter = adp;
        s->done.store(true, std::memory_order_release);
      };
      cb.userdata1 = ud;

      WGPUFuture f = rt->api.wgpuInstanceRequestAdapter(instance, &opts, cb);
      bool waited = wgpuWaitOne(*rt, instance, timedWaitAny, f, &req->done);
      if (!waited || !req->done.load(std::memory_order_acquire) ||
          req->status != WGPURequestAdapterStatus_Success || !req->adapter)
        continue;

      AdapterKey key = {};
      WGPUAdapterInfo ai = WGPU_ADAPTER_INFO_INIT;
      if (rt->api.wgpuAdapterGetInfo(req->adapter, &ai) == WGPUStatus_Success)
      {
        key.vendorID = ai.vendorID;
        key.deviceID = ai.deviceID;
        key.backendType = static_cast<uint32_t>(ai.backendType);
        key.adapterType = static_cast<uint32_t>(ai.adapterType);
        key.device = wgpuToStdString(ai.device);
        rt->api.wgpuAdapterInfoFreeMembers(ai);
      }

      bool dup = false;
      for (const auto &k : seen)
        if (k == key) { dup = true; break; }
      if (dup)
      {
        rt->api.wgpuAdapterRelease(req->adapter);
        continue;
      }
      seen.push_back(std::move(key));
      adapters.push_back(req->adapter);
    }

    return true;
  }
};

} // namespace

WebgpuPeak::WebgpuPeak() = default;
WebgpuPeak::~WebgpuPeak() = default;

float WebgpuPeak::runKernel(WebgpuDevice &dev,
                            WGPUComputePipelineImpl *pipeline,
                            WGPUBindGroupImpl *bindGroup,
                            uint32_t groupCountX,
                            unsigned int targetTimeUsLocal,
                            unsigned int forcedIters,
                            bool serialize)
{
  const WebgpuRuntime &rt = *dev.rt;

  auto runBatch = [&](unsigned int n) -> float
  {
    WGPUCommandEncoderDescriptor encDesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder enc = rt.api.wgpuDeviceCreateCommandEncoder(dev.device, &encDesc);

    if (serialize)
    {
      // Separate compute passes ensure execution/memory serialization between
      // consecutive passes over the same bandwidth buffers/textures.
      for (unsigned int i = 0; i < n; i++)
      {
        WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUComputePassEncoder pass = rt.api.wgpuCommandEncoderBeginComputePass(enc, &passDesc);
        rt.api.wgpuComputePassEncoderSetPipeline(pass, pipeline);
        if (bindGroup)
          rt.api.wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
        rt.api.wgpuComputePassEncoderDispatchWorkgroups(pass, groupCountX, 1, 1);
        rt.api.wgpuComputePassEncoderEnd(pass);
        rt.api.wgpuComputePassEncoderRelease(pass);
      }
    }
    else
    {
      WGPUComputePassDescriptor passDesc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
      WGPUComputePassEncoder pass = rt.api.wgpuCommandEncoderBeginComputePass(enc, &passDesc);
      rt.api.wgpuComputePassEncoderSetPipeline(pass, pipeline);
      if (bindGroup)
        rt.api.wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
      for (unsigned int i = 0; i < n; i++)
        rt.api.wgpuComputePassEncoderDispatchWorkgroups(pass, groupCountX, 1, 1);
      rt.api.wgpuComputePassEncoderEnd(pass);
      rt.api.wgpuComputePassEncoderRelease(pass);
    }

    WGPUCommandBufferDescriptor cbDesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer cmdBuf = rt.api.wgpuCommandEncoderFinish(enc, &cbDesc);
    rt.api.wgpuCommandEncoderRelease(enc);

    auto start = std::chrono::high_resolution_clock::now();
    bool ok = dev.submitAndWait(cmdBuf);
    auto end = std::chrono::high_resolution_clock::now();
    rt.api.wgpuCommandBufferRelease(cmdBuf);

    if (!ok)
      return -1.0f;
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  };

  // Phase 1: untimed warmup.
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    if (runBatch(1) < 0.0f)
      return -1.0f;
  }

  // Phase 2: timed calibration probe.
  float probeUs = runBatch(1);
  if (probeUs < 0.0f)
    return -1.0f;
  if (probeUs <= 0.0f)
    probeUs = 1.0f;

  // Phase 3: timed run with calibrated iteration count.
  unsigned int iters = pickIters(static_cast<double>(probeUs), targetTimeUsLocal, forcedIters);
  float timed = runBatch(iters);
  if (timed <= 0.0f)
    return -1.0f;
  return timed / static_cast<float>(iters);
}

int WebgpuPeak::runAll()
{
  const WebgpuRuntime *rt = webgpuRuntime();
  if (!rt)
  {
    log->note("WebGPU: " + webgpuLoadDiagnostic() + "\n");
    return 0;
  }

  InstanceHolder holder;
  if (!holder.init(rt))
  {
    log->note("WebGPU: failed to create instance\n");
    return -1;
  }
  if (holder.adapters.empty())
  {
    log->note("WebGPU: no adapters found\n");
    return 0;
  }

  auto backendScope = log->beginBackend("WebGPU");

  for (size_t d = 0; d < holder.adapters.size(); d++)
  {
    if (clpeak::cancelRequested())
      break;
    if (!isDeviceSelected(static_cast<int>(d)))
      continue;

    WebgpuDevice dev;
    if (!dev.init(rt, holder.instance, holder.adapters[d], holder.timedWaitAny,
                  holder.callbackStates))
    {
      log->note("WebGPU: failed to init device " + std::to_string(d) + "\n");
      continue;
    }

    // WebGPU does not expose cache size or dedicated VRAM size, so pass 0/0
    // and let benchmark_config_t::forDevice pick the standard working-set floor.
    benchmark_config_t cfg = benchmark_config_t::forDevice(dev.info.deviceType, 0, 0);
    cfg.targetTimeUs = targetTimeUs;
    if (forceIters)
      cfg.kernelLatencyIters = specifiedIters;

    std::vector<logger::Prop> deviceProps;
    deviceProps.push_back({"WebGPU backend", dev.info.backendName});
    deviceProps.push_back({"Adapter type", dev.info.adapterTypeStr});
    if (!dev.info.architecture.empty())
      deviceProps.push_back({"Architecture", dev.info.architecture});
    deviceProps.push_back({"Max storage buf",
                           std::to_string(dev.info.maxStorageBufferBindingSize >> 20) + " MB"});
    if (!rt->path.empty())
      deviceProps.push_back({"Runtime", rt->path});

    auto deviceScope = backendScope.beginDevice({
      dev.info.deviceName,
      "",
      dev.info.driverDescription,
      deviceProps,
      -1,
      static_cast<int>(d),
      dev.info.deviceType
    });
    currentDeviceScope = &deviceScope;

    // ---- Compute (FLOPS / OPS) -------------------------------------------
    if (auto run = admit(Benchmark::ComputeSP))     runComputeSP(dev, cfg);
    if (auto run = admit(Benchmark::ComputeHP))     runComputeHP(dev, cfg);
    if (auto run = admit(Benchmark::ComputeMP))     runComputeMP(dev, cfg);
    if (auto run = admit(Benchmark::ComputeInt))    runComputeInt32(dev, cfg);
    if (auto run = admit(Benchmark::ComputeInt8DP)) runComputeInt8DP(dev, cfg);

    // ---- Bandwidth (B/s) -------------------------------------------------
    if (auto run = admit(Benchmark::GlobalBW))      runGlobalBandwidth(dev, cfg);
    if (auto run = admit(Benchmark::LocalBW))       runLocalBandwidth(dev, cfg);
    if (auto run = admit(Benchmark::ImageBW))       runImageBandwidth(dev, cfg);
    if (auto run = admit(Benchmark::TransferBW))    runTransferBandwidth(dev, cfg);

    // ---- Latency (s) -----------------------------------------------------
    if (auto run = admit(Benchmark::KernelLatency)) runKernelLatency(dev, cfg);

    currentDeviceScope = nullptr;
  }

  return 0;
}

BackendInventory WebgpuPeak::enumerate()
{
  BackendInventory inv;
  inv.id = kBackend;

  const WebgpuRuntime *rt = webgpuRuntime();
  if (!rt)
  {
    inv.unavailableReason = webgpuLoadDiagnostic();
    return inv;
  }

  InstanceHolder holder;
  if (!holder.init(rt))
  {
    inv.unavailableReason = "failed to create a WebGPU instance";
    return inv;
  }

  inv.available = !holder.adapters.empty();
  if (!inv.available)
    inv.unavailableReason = "no adapters found";
  if (!rt->path.empty())
    inv.info = rt->path;

  InventoryPlatform plat;
  plat.index = 0;
  plat.name = "WebGPU";

  for (size_t i = 0; i < holder.adapters.size(); i++)
  {
    WGPUAdapterInfo ai = WGPU_ADAPTER_INFO_INIT;
    InventoryDevice dev;
    dev.index = static_cast<int>(i);
    if (rt->api.wgpuAdapterGetInfo(holder.adapters[i], &ai) == WGPUStatus_Success)
    {
      dev.name = wgpuAdapterDisplayName(ai);
      dev.typeStr = wgpuAdapterTypeName(ai.adapterType);
      dev.arch = wgpuToStdString(ai.architecture);
      dev.driverVersion = wgpuToStdString(ai.description);
      dev.apiVersion = wgpuBackendTypeName(ai.backendType);
      rt->api.wgpuAdapterInfoFreeMembers(ai);
    }
    WGPULimits lim = WGPU_LIMITS_INIT;
    if (rt->api.wgpuAdapterGetLimits(holder.adapters[i], &lim) == WGPUStatus_Success)
      dev.maxAllocBytes = lim.maxStorageBufferBindingSize;
    dev.hasFp16 = rt->api.wgpuAdapterHasFeature(holder.adapters[i], WGPUFeatureName_ShaderF16) != 0;
    plat.devices.push_back(std::move(dev));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

#endif // ENABLE_WEBGPU
