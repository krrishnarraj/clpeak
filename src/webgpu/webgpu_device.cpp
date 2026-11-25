#ifdef ENABLE_WEBGPU

#include "webgpu_runtime.h"
#include <webgpu/webgpu_peak.h>
#include <common/common.h>
#include <algorithm>
#include <vector>

namespace
{

DeviceType toNeutralDeviceType(WGPUAdapterType at)
{
  switch (at)
  {
  case WGPUAdapterType_DiscreteGPU:
  case WGPUAdapterType_IntegratedGPU:
    return DeviceType::Gpu;
  case WGPUAdapterType_CPU:
    return DeviceType::Cpu;
  default:
    return DeviceType::Unknown;
  }
}

void onUncapturedError(WGPUDevice const *, WGPUErrorType type,
                       WGPUStringView message, void *userdata1, void *)
{
  auto *state = static_cast<WebgpuDevice::CallbackState *>(userdata1);
  if (state)
    state->hadError.store(true, std::memory_order_release);
  clpeak::logMessage(clpeak::LogLevel::Warning, "webgpu",
                     "uncaptured error (" + std::to_string(static_cast<int>(type)) +
                     "): " + wgpuToStdString(message));
}

void onDeviceLost(WGPUDevice const *, WGPUDeviceLostReason reason,
                  WGPUStringView message, void *userdata1, void *)
{
  // Destroyed and CallbackCancelled are expected during normal teardown.
  if (reason == WGPUDeviceLostReason_Destroyed ||
      reason == WGPUDeviceLostReason_CallbackCancelled)
    return;
  auto *state = static_cast<WebgpuDevice::CallbackState *>(userdata1);
  if (state)
  {
    state->deviceLost.store(true, std::memory_order_release);
    state->hadError.store(true, std::memory_order_release);
  }
  clpeak::logMessage(clpeak::LogLevel::Error, "webgpu",
                     "device lost: " + wgpuToStdString(message));
}

} // namespace

WebgpuDevice::~WebgpuDevice()
{
  cleanup();
}

bool WebgpuDevice::init(const WebgpuRuntime *runtime, WGPUInstanceImpl *inst,
                        WGPUAdapterImpl *adp, bool hasTimedWaitAny,
                        std::vector<std::shared_ptr<CallbackState>> &statePool)
{
  rt = runtime;
  instance = inst;
  adapter = adp;
  timedWaitAny = hasTimedWaitAny;
  cbState = std::make_shared<CallbackState>();
  statePool.push_back(cbState);

  WGPUAdapterInfo ai = WGPU_ADAPTER_INFO_INIT;
  if (rt->api.wgpuAdapterGetInfo(adapter, &ai) == WGPUStatus_Success)
  {
    info.vendor = wgpuToStdString(ai.vendor);
    info.architecture = wgpuToStdString(ai.architecture);
    info.deviceName = wgpuAdapterDisplayName(ai);
    info.driverDescription = wgpuToStdString(ai.description);
    info.vendorID = ai.vendorID;
    info.deviceID = ai.deviceID;
    info.backendName = wgpuBackendTypeName(ai.backendType);
    info.adapterTypeStr = wgpuAdapterTypeName(ai.adapterType);
    info.deviceType = toNeutralDeviceType(ai.adapterType);
    info.isIntegratedOrCpu =
        (ai.adapterType == WGPUAdapterType_IntegratedGPU ||
         ai.adapterType == WGPUAdapterType_CPU);
    rt->api.wgpuAdapterInfoFreeMembers(ai);
  }

  // Request optional features the adapter advertises that our tests use.
  std::vector<WGPUFeatureName> reqFeatures;
  if (rt->api.wgpuAdapterHasFeature(adapter, WGPUFeatureName_ShaderF16))
    reqFeatures.push_back(WGPUFeatureName_ShaderF16);

  // Request the adapter's real limits (especially maxStorageBufferBindingSize
  // and maxBufferSize so global bandwidth can outgrow last-level cache).
  WGPULimits adapterLimits = WGPU_LIMITS_INIT;
  bool haveLimits =
      (rt->api.wgpuAdapterGetLimits(adapter, &adapterLimits) == WGPUStatus_Success);
  if (haveLimits)
    adapterLimits.nextInChain = nullptr;

  WGPUDeviceDescriptor devDesc = WGPU_DEVICE_DESCRIPTOR_INIT;
  devDesc.label = wgpuStrView("clpeak-webgpu-device");
  devDesc.requiredFeatureCount = reqFeatures.size();
  devDesc.requiredFeatures = reqFeatures.empty() ? nullptr : reqFeatures.data();
  devDesc.requiredLimits = haveLimits ? &adapterLimits : nullptr;
  devDesc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
  devDesc.deviceLostCallbackInfo.callback = onDeviceLost;
  devDesc.deviceLostCallbackInfo.userdata1 = cbState.get();
  devDesc.uncapturedErrorCallbackInfo.callback = onUncapturedError;
  devDesc.uncapturedErrorCallbackInfo.userdata1 = cbState.get();

  struct ReqState
  {
    std::atomic<bool> done{false};
    WGPURequestDeviceStatus status = WGPURequestDeviceStatus_Error;
    WGPUDevice dev = nullptr;
    std::string msg;
  };
  auto [req, ud] = wgpuMakeCallbackState<ReqState>();

  WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
  cb.mode = timedWaitAny ? WGPUCallbackMode_WaitAnyOnly
                         : WGPUCallbackMode_AllowProcessEvents;
  cb.callback = [](WGPURequestDeviceStatus status, WGPUDevice d,
                   WGPUStringView msg, void *ud1, void *)
  {
    auto s = wgpuTakeCallbackState<ReqState>(ud1);
    s->status = status;
    s->dev = d;
    s->msg = wgpuToStdString(msg);
    s->done.store(true, std::memory_order_release);
  };
  cb.userdata1 = ud;

  WGPUFuture f = rt->api.wgpuAdapterRequestDevice(adapter, &devDesc, cb);
  bool waited = wgpuWaitOne(*rt, instance, timedWaitAny, f, &req->done);
  if (!waited || !req->done.load(std::memory_order_acquire))
  {
    CLPEAK_LOG(Error, "WebGPU: requestDevice wait failed");
    return false;
  }
  if (req->status != WGPURequestDeviceStatus_Success || !req->dev)
  {
    CLPEAK_LOG(Error, "WebGPU: requestDevice failed: %s", req->msg.c_str());
    return false;
  }

  device = req->dev;
  queue = rt->api.wgpuDeviceGetQueue(device);

  WGPULimits devLimits = WGPU_LIMITS_INIT;
  if (rt->api.wgpuDeviceGetLimits(device, &devLimits) == WGPUStatus_Success)
  {
    info.maxWGSize = std::min(devLimits.maxComputeWorkgroupSizeX,
                              std::min(devLimits.maxComputeInvocationsPerWorkgroup,
                                       static_cast<uint32_t>(MAX_WG_SIZE)));
    info.maxWGCount = devLimits.maxComputeWorkgroupsPerDimension;
    info.maxTextureDimension2D = devLimits.maxTextureDimension2D;
    info.maxWorkgroupStorageSize = devLimits.maxComputeWorkgroupStorageSize;
    info.maxStorageBufferBindingSize = devLimits.maxStorageBufferBindingSize;
    info.maxBufferSize = devLimits.maxBufferSize;
  }

  info.float16Supported = rt->api.wgpuDeviceHasFeature(device, WGPUFeatureName_ShaderF16) != 0;
  if (rt->api.wgpuInstanceHasWGSLLanguageFeature)
  {
    info.packed4x8DotProductSupported =
        rt->api.wgpuInstanceHasWGSLLanguageFeature(
            instance, WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct) != 0;
  }

  CLPEAK_VLOG("WebGPU device: %s (backend=%s, type=%s, f16=%d, dp4a=%d, "
              "timedWaitAny=%d, maxBuf=%llu MB, maxStorageBind=%llu MB, "
              "maxWGCount=%u, maxTex2D=%u)\n",
              info.deviceName.c_str(), info.backendName.c_str(),
              info.adapterTypeStr.c_str(),
              info.float16Supported ? 1 : 0,
              info.packed4x8DotProductSupported ? 1 : 0,
              timedWaitAny ? 1 : 0,
              static_cast<unsigned long long>(info.maxBufferSize >> 20),
              static_cast<unsigned long long>(info.maxStorageBufferBindingSize >> 20),
              info.maxWGCount, info.maxTextureDimension2D);

  return true;
}

void WebgpuDevice::cleanup()
{
  if (!rt)
    return;
  if (queue)
  {
    rt->api.wgpuQueueRelease(queue);
    queue = nullptr;
  }
  if (device)
  {
    rt->api.wgpuDeviceDestroy(device);
    rt->api.wgpuDeviceRelease(device);
    device = nullptr;
  }
  if (instance)
    rt->api.wgpuInstanceProcessEvents(instance);
  // Adapter and instance are owned by the caller (WebgpuPeak).
  adapter = nullptr;
  instance = nullptr;
}

void WebgpuDevice::pushErrorScopes()
{
  if (!device || !rt->api.wgpuDevicePushErrorScope || !rt->api.wgpuDevicePopErrorScope)
    return;
  rt->api.wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
  rt->api.wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);
  rt->api.wgpuDevicePushErrorScope(device, WGPUErrorFilter_Internal);
}

bool WebgpuDevice::popErrorScope(std::string *msg)
{
  struct ScopeRes
  {
    std::atomic<bool> done{false};
    WGPUPopErrorScopeStatus status = WGPUPopErrorScopeStatus_Error;
    WGPUErrorType type = WGPUErrorType_Unknown;
    std::string msg;
  };
  auto [sr, ud] = wgpuMakeCallbackState<ScopeRes>();

  WGPUPopErrorScopeCallbackInfo cb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
  cb.mode = timedWaitAny ? WGPUCallbackMode_WaitAnyOnly
                         : WGPUCallbackMode_AllowProcessEvents;
  cb.callback = [](WGPUPopErrorScopeStatus status, WGPUErrorType type,
                   WGPUStringView m, void *ud1, void *)
  {
    auto s = wgpuTakeCallbackState<ScopeRes>(ud1);
    s->status = status;
    s->type = type;
    s->msg = wgpuToStdString(m);
    s->done.store(true, std::memory_order_release);
  };
  cb.userdata1 = ud;

  WGPUFuture f = rt->api.wgpuDevicePopErrorScope(device, cb);
  bool waited = wgpuWaitOne(*rt, instance, timedWaitAny, f, &sr->done);
  if (!waited || !sr->done.load(std::memory_order_acquire))
  {
    if (msg)
      *msg = "popErrorScope wait failed";
    return false;
  }
  if (sr->status != WGPUPopErrorScopeStatus_Success || sr->type != WGPUErrorType_NoError)
  {
    if (msg)
      *msg = sr->msg.empty() ? "WebGPU error scope failed" : sr->msg;
    return false;
  }
  return true;
}

bool WebgpuDevice::popErrorScopes(std::string *msg)
{
  if (!device || !rt->api.wgpuDevicePushErrorScope || !rt->api.wgpuDevicePopErrorScope)
    return true;
  bool ok = true;
  for (int i = 0; i < 3; i++)
  {
    std::string scopeMsg;
    if (!popErrorScope(&scopeMsg))
    {
      if (ok && msg)
        *msg = std::move(scopeMsg);
      ok = false;
    }
  }
  return ok;
}

WGPUBufferImpl *WebgpuDevice::createBuffer(uint64_t size, uint64_t usage,
                                           bool mappedAtCreation)
{
  if (!device || isLost())
    return nullptr;

  pushErrorScopes();

  WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
  desc.size = size;
  desc.usage = static_cast<WGPUBufferUsage>(usage);
  desc.mappedAtCreation = mappedAtCreation ? WGPU_TRUE : WGPU_FALSE;
  WGPUBuffer buf = rt->api.wgpuDeviceCreateBuffer(device, &desc);

  std::string errMsg;
  if (!popErrorScopes(&errMsg) || !buf)
  {
    CLPEAK_VLOG("WebGPU buffer alloc (%llu bytes) failed: %s\n",
                static_cast<unsigned long long>(size), errMsg.c_str());
    if (buf)
    {
      rt->api.wgpuBufferDestroy(buf);
      rt->api.wgpuBufferRelease(buf);
    }
    return nullptr;
  }
  return buf;
}

WGPUTextureImpl *WebgpuDevice::createTexture(const WGPUTextureDescriptor *desc)
{
  if (!device || !desc || isLost())
    return nullptr;

  pushErrorScopes();
  WGPUTexture tex = rt->api.wgpuDeviceCreateTexture(device, desc);

  std::string errMsg;
  if (!popErrorScopes(&errMsg))
  {
    CLPEAK_VLOG("WebGPU texture creation failed: %s\n", errMsg.c_str());
    rt->api.wgpuTextureDestroy(tex);
    rt->api.wgpuTextureRelease(tex);
    return nullptr;
  }
  return tex;
}

WGPUComputePipelineImpl *WebgpuDevice::createComputePipeline(
    const std::string &wgsl,
    WGPUPipelineLayoutImpl *pipeLayout,
    const WGPUConstantEntry *constants,
    size_t constantCount)
{
  if (!device || isLost())
    return nullptr;

  pushErrorScopes();

  WGPUShaderSourceWGSL wgslSource = WGPU_SHADER_SOURCE_WGSL_INIT;
  wgslSource.code = {wgsl.data(), wgsl.size()};

  WGPUShaderModuleDescriptor smDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
  smDesc.nextInChain = &wgslSource.chain;
  WGPUShaderModule mod = rt->api.wgpuDeviceCreateShaderModule(device, &smDesc);

  WGPUComputePipelineDescriptor cpDesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
  cpDesc.layout = pipeLayout;
  cpDesc.compute.module = mod;
  cpDesc.compute.entryPoint = wgpuStrView("main");
  cpDesc.compute.constantCount = constantCount;
  cpDesc.compute.constants = constants;
  WGPUComputePipeline pipe = rt->api.wgpuDeviceCreateComputePipeline(device, &cpDesc);
  rt->api.wgpuShaderModuleRelease(mod);

  std::string errMsg;
  if (!popErrorScopes(&errMsg))
  {
    CLPEAK_VLOG("WebGPU pipeline creation error: %s\n", errMsg.c_str());
    rt->api.wgpuComputePipelineRelease(pipe);
    return nullptr;
  }
  return pipe;
}

bool WebgpuDevice::submitAndWait(WGPUCommandBufferImpl *cmdBuf)
{
  if (!queue || !cmdBuf || isLost())
    return false;

  clearError();
  rt->api.wgpuQueueSubmit(queue, 1, &cmdBuf);

  struct DoneState
  {
    std::atomic<bool> done{false};
    WGPUQueueWorkDoneStatus status = WGPUQueueWorkDoneStatus_Error;
  };
  auto [ds, ud] = wgpuMakeCallbackState<DoneState>();

  WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
  cb.mode = timedWaitAny ? WGPUCallbackMode_WaitAnyOnly
                         : WGPUCallbackMode_AllowProcessEvents;
  cb.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView,
                   void *ud1, void *)
  {
    auto s = wgpuTakeCallbackState<DoneState>(ud1);
    s->status = status;
    s->done.store(true, std::memory_order_release);
  };
  cb.userdata1 = ud;

  WGPUFuture f = rt->api.wgpuQueueOnSubmittedWorkDone(queue, cb);
  bool waited = wgpuWaitOne(*rt, instance, timedWaitAny, f, &ds->done);
  if (!waited || !ds->done.load(std::memory_order_acquire))
    return false;
  return ds->status == WGPUQueueWorkDoneStatus_Success && !isLost() && !hasError();
}

#endif // ENABLE_WEBGPU
