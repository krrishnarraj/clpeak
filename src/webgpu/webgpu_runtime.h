#ifndef CLPEAK_WEBGPU_RUNTIME_H
#define CLPEAK_WEBGPU_RUNTIME_H

// Load-on-demand WebGPU C API (standard webgpu.h).
//
// The WebGPU implementation library is dlopen'd at run time so clpeak ships
// with no hard build or runtime dependency on any specific WebGPU library,
// matching the LiteRT and ONNX backends: a system without one gets a single
// diagnostic line and no rows.

#include <atomic>
#include <memory>
#include <string>
#include "webgpu.h"

#define CLPEAK_WEBGPU_REQUIRED(X)                  \
  X(wgpuCreateInstance)                            \
  X(wgpuInstanceRelease)                           \
  X(wgpuInstanceProcessEvents)                     \
  X(wgpuInstanceRequestAdapter)                    \
  X(wgpuAdapterGetInfo)                            \
  X(wgpuAdapterInfoFreeMembers)                    \
  X(wgpuAdapterGetLimits)                          \
  X(wgpuAdapterHasFeature)                         \
  X(wgpuAdapterRequestDevice)                      \
  X(wgpuAdapterRelease)                            \
  X(wgpuDeviceGetQueue)                            \
  X(wgpuDeviceGetLimits)                           \
  X(wgpuDeviceHasFeature)                          \
  X(wgpuDeviceCreateBuffer)                        \
  X(wgpuDeviceCreateTexture)                       \
  X(wgpuDeviceCreateShaderModule)                  \
  X(wgpuDeviceCreateBindGroupLayout)               \
  X(wgpuDeviceCreatePipelineLayout)                \
  X(wgpuDeviceCreateBindGroup)                     \
  X(wgpuDeviceCreateComputePipeline)               \
  X(wgpuDeviceCreateCommandEncoder)                \
  X(wgpuDeviceDestroy)                             \
  X(wgpuDeviceRelease)                             \
  X(wgpuQueueSubmit)                               \
  X(wgpuQueueWriteBuffer)                          \
  X(wgpuQueueOnSubmittedWorkDone)                  \
  X(wgpuQueueRelease)                              \
  X(wgpuBufferGetMappedRange)                      \
  X(wgpuBufferUnmap)                               \
  X(wgpuBufferDestroy)                             \
  X(wgpuBufferRelease)                             \
  X(wgpuTextureCreateView)                         \
  X(wgpuTextureDestroy)                            \
  X(wgpuTextureRelease)                            \
  X(wgpuTextureViewRelease)                        \
  X(wgpuShaderModuleRelease)                       \
  X(wgpuBindGroupLayoutRelease)                    \
  X(wgpuPipelineLayoutRelease)                     \
  X(wgpuBindGroupRelease)                          \
  X(wgpuComputePipelineRelease)                    \
  X(wgpuCommandEncoderBeginComputePass)            \
  X(wgpuCommandEncoderCopyBufferToBuffer)          \
  X(wgpuCommandEncoderCopyBufferToTexture)         \
  X(wgpuCommandEncoderFinish)                      \
  X(wgpuCommandEncoderRelease)                     \
  X(wgpuComputePassEncoderSetPipeline)             \
  X(wgpuComputePassEncoderSetBindGroup)            \
  X(wgpuComputePassEncoderDispatchWorkgroups)      \
  X(wgpuComputePassEncoderEnd)                     \
  X(wgpuComputePassEncoderRelease)                 \
  X(wgpuCommandBufferRelease)

#define CLPEAK_WEBGPU_OPTIONAL(X)                  \
  X(wgpuHasInstanceFeature)                        \
  X(wgpuInstanceWaitAny)                           \
  X(wgpuInstanceHasWGSLLanguageFeature)            \
  X(wgpuDevicePushErrorScope)                      \
  X(wgpuDevicePopErrorScope)

struct WebgpuApi
{
#define CLPEAK_WEBGPU_DECL(name) decltype(&::name) name = nullptr;
  CLPEAK_WEBGPU_REQUIRED(CLPEAK_WEBGPU_DECL)
  CLPEAK_WEBGPU_OPTIONAL(CLPEAK_WEBGPU_DECL)
#undef CLPEAK_WEBGPU_DECL
};

struct WebgpuRuntime
{
  void *lib = nullptr;
  std::string path;
  WebgpuApi api;
};

// Returns the loaded runtime, or nullptr if loading failed (see webgpuLoadDiagnostic()).
const WebgpuRuntime *webgpuRuntime();

// Helpers for WGPUStringView conversions.
static inline WGPUStringView wgpuStrView(const char *s)
{
  WGPUStringView v = WGPU_STRING_VIEW_INIT;
  v.data = s;
  v.length = WGPU_STRLEN;
  return v;
}

static inline std::string wgpuToStdString(WGPUStringView v)
{
  if (!v.data || v.length == 0)
    return {};
  if (v.length == WGPU_STRLEN)
    return std::string(v.data);
  return std::string(v.data, v.length);
}

static inline const char *wgpuBackendTypeName(WGPUBackendType bt)
{
  switch (bt)
  {
  case WGPUBackendType_Vulkan:   return "Vulkan";
  case WGPUBackendType_Metal:    return "Metal";
  case WGPUBackendType_D3D12:    return "D3D12";
  case WGPUBackendType_D3D11:    return "D3D11";
  case WGPUBackendType_OpenGL:   return "OpenGL";
  case WGPUBackendType_OpenGLES: return "OpenGLES";
  case WGPUBackendType_WebGPU:   return "WebGPU";
  case WGPUBackendType_Null:     return "Null";
  default:                       return "Unknown";
  }
}

static inline const char *wgpuAdapterTypeName(WGPUAdapterType at)
{
  switch (at)
  {
  case WGPUAdapterType_DiscreteGPU:   return "Discrete GPU";
  case WGPUAdapterType_IntegratedGPU: return "Integrated GPU";
  case WGPUAdapterType_CPU:           return "CPU";
  default:                            return "Other";
  }
}

static inline std::string wgpuAdapterDisplayName(const WGPUAdapterInfo &ai)
{
  std::string name = wgpuToStdString(ai.device);
  if (name.empty())
    name = wgpuToStdString(ai.description);
  if (name.empty())
    name = "WebGPU Adapter";
  return name;
}

// Heap-owned state for one-shot WGPUFuture callbacks.  A future's callback is
// guaranteed to fire once (including with *_CallbackCancelled on instance
// release), so if a wait fails and the caller returns early, storing a
// shared_ptr copy in userdata1 keeps the state alive until that callback runs.
template <typename T>
struct WgpuCallbackHandle
{
  std::shared_ptr<T> state;
  void *userdata;
};

template <typename T>
WgpuCallbackHandle<T> wgpuMakeCallbackState()
{
  auto state = std::make_shared<T>();
  return {state, new std::shared_ptr<T>(state)};
}

template <typename T>
std::shared_ptr<T> wgpuTakeCallbackState(void *userdata)
{
  std::unique_ptr<std::shared_ptr<T>> holder(
      static_cast<std::shared_ptr<T> *>(userdata));
  return *holder;
}

// Wait for a single WGPUFuture using wgpuInstanceWaitAny (when TimedWaitAny is
// enabled) or polling wgpuInstanceProcessEvents until *done becomes true.
bool wgpuWaitOne(const WebgpuRuntime &rt, WGPUInstance inst, bool timedWaitAny,
                 WGPUFuture future, const std::atomic<bool> *done);

#endif // CLPEAK_WEBGPU_RUNTIME_H
