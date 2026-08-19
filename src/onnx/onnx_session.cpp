#ifdef ENABLE_ONNX

#include "onnx_session.h"

#include <common/common.h>

#include <mutex>
#include <vector>

std::string onnxStatusText(const OrtRuntime &rt, OrtStatus *st)
{
  if (!st)
    return "";
  const char *msg = rt.api->GetErrorMessage(st);
  std::string out = msg ? msg : "unknown error";
  rt.api->ReleaseStatus(st);
  // Session-creation errors from EP compilers can run to many lines; the
  // result rows are line-oriented, so keep the first line only.
  size_t nl = out.find('\n');
  if (nl != std::string::npos)
    out.resize(nl);
  return out;
}

OrtEnv *onnxEnv(const OrtRuntime &rt)
{
  static OrtEnv *env = nullptr;
  static std::once_flag once;
  std::call_once(once, [&rt] {
    // A provider declining a graph is an expected outcome here -- it becomes
    // an Unsupported row -- but ORT reports it at ERROR level and writes it
    // straight to stderr.  Normal runs stay silent (the skip row carries the
    // message); --verbose opens the runtime's own log up.
    OrtLoggingLevel level = clpeak::verboseEnabled() ? ORT_LOGGING_LEVEL_WARNING
                                                     : ORT_LOGGING_LEVEL_FATAL;
    OrtStatus *st = rt.api->CreateEnv(level, "clpeak", &env);
    if (st)
    {
      CLPEAK_VLOG("onnx: CreateEnv failed: %s\n", onnxStatusText(rt, st).c_str());
      env = nullptr;
    }
  });
  return env;
}

// ---------------------------------------------------------------------------
// Per-EP registration.  These options are the ones a well-behaved app would
// pass to reach the vendor's accelerator, and they are part of what the
// backend measures, so every provider's wiring lives in this one place.
//
// Two registration shapes exist in the ORT C API and both are needed: most
// providers take a name plus string key/values, while CUDA, TensorRT, ROCm
// and MIGraphX have typed options structs of their own.
// ---------------------------------------------------------------------------

namespace
{

struct EpOptions
{
  const char *registrationName;     // name the generic append API expects
  std::vector<std::pair<const char *, const char *>> kv;
};

// Providers registered through the generic string-keyed API.  Returns false
// for anything not handled here; the caller then tries the typed paths.
bool genericEpOptions(const std::string &providerKey, EpOptions &out)
{
  if (providerKey == "CoreMLExecutionProvider")
  {
    // MLProgram + Neural Engine: the fp16-native ANE path.  CoreML has no
    // NPU-only mode; CPUAndNeuralEngine is the strictest available request.
    out = {"CoreML",
           {{"ModelFormat", "MLProgram"},
            {"MLComputeUnits", "CPUAndNeuralEngine"}}};
    return true;
  }
  if (providerKey == "QNNExecutionProvider")
  {
    // HTP is the NPU proper; without this the EP would settle for the DSP or
    // the CPU reference backend and the row would not mean what it says.
#if defined(_WIN32)
    out = {"QNN", {{"backend_path", "QnnHtp.dll"}}};
#else
    out = {"QNN", {{"backend_path", "libQnnHtp.so"}}};
#endif
    return true;
  }
  if (providerKey == "OpenVINOExecutionProvider")
  {
    out = {"OpenVINO", {{"device_type", "NPU"}}};
    return true;
  }
  if (providerKey == "VitisAIExecutionProvider")
  {
    out = {"VitisAI", {}};
    return true;
  }
  if (providerKey == "NvTensorRTRTXExecutionProvider")
  {
    out = {"NvTensorRtRtx", {}};
    return true;
  }
  if (providerKey == "XnnpackExecutionProvider")
  {
    out = {"XNNPACK", {}};
    return true;
  }
  if (providerKey == "DmlExecutionProvider")
  {
    out = {"DML", {}};
    return true;
  }
  if (providerKey == "WebGpuExecutionProvider")
  {
    out = {"WebGPU", {}};
    return true;
  }
  return false;
}

// Register `providerKey` on `so`.  Returns an empty string on success, or a
// one-line reason -- including "no wiring", so an unknown provider is
// reported as unsupported rather than silently run with defaults.
std::string appendProvider(const OrtRuntime &rt, OrtSessionOptions *so,
                           const std::string &providerKey)
{
  const OrtApi *api = rt.api;

  EpOptions opts;
  if (genericEpOptions(providerKey, opts))
  {
    std::vector<const char *> keys, vals;
    for (auto &kvp : opts.kv)
    {
      keys.push_back(kvp.first);
      vals.push_back(kvp.second);
    }
    return onnxStatusText(rt, api->SessionOptionsAppendExecutionProvider(
        so, opts.registrationName, keys.data(), vals.data(), keys.size()));
  }

  // ---- Typed-options providers -----------------------------------------
  // Device 0 throughout: this backend enumerates providers, not the physical
  // GPUs behind them, so a multi-GPU box benchmarks its first device.
  if (providerKey == "CUDAExecutionProvider")
  {
    OrtCUDAProviderOptionsV2 *cuda = nullptr;
    if (OrtStatus *st = api->CreateCUDAProviderOptions(&cuda))
      return onnxStatusText(rt, st);
    std::string err = onnxStatusText(
        rt, api->SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda));
    api->ReleaseCUDAProviderOptions(cuda);
    return err;
  }
  if (providerKey == "TensorrtExecutionProvider")
  {
    OrtTensorRTProviderOptionsV2 *trt = nullptr;
    if (OrtStatus *st = api->CreateTensorRTProviderOptions(&trt))
      return onnxStatusText(rt, st);
    std::string err = onnxStatusText(
        rt, api->SessionOptionsAppendExecutionProvider_TensorRT_V2(so, trt));
    api->ReleaseTensorRTProviderOptions(trt);
    return err;
  }
  if (providerKey == "ROCMExecutionProvider")
  {
    OrtROCMProviderOptions rocm{};
    rocm.device_id = 0;
    return onnxStatusText(
        rt, api->SessionOptionsAppendExecutionProvider_ROCM(so, &rocm));
  }
  if (providerKey == "MIGraphXExecutionProvider")
  {
    OrtMIGraphXProviderOptions mgx{};
    mgx.device_id = 0;
    return onnxStatusText(
        rt, api->SessionOptionsAppendExecutionProvider_MIGraphX(so, &mgx));
  }

  return "clpeak has no session wiring for " + providerKey + " yet";
}

} // namespace

OnnxSessionResult onnxCreateSession(const OrtRuntime &rt,
                                    const onnx_ep_info_t &ep,
                                    const std::string &modelBytes)
{
  OnnxSessionResult res;
  const OrtApi *api = rt.api;

  OrtEnv *env = onnxEnv(rt);
  if (!env)
  {
    res.error = "onnxruntime environment creation failed";
    return res;
  }

  OrtSessionOptions *so = nullptr;
  OrtStatus *st = api->CreateSessionOptions(&so);
  if (st)
  {
    res.error = onnxStatusText(rt, st);
    return res;
  }

  // The CPU EP is implicit -- every session already has it -- so it is the
  // one provider that needs no registration and keeps its fallback.
  if (ep.providerKey != "CPUExecutionProvider")
  {
    res.error = appendProvider(rt, so, ep.providerKey);
    if (!res.error.empty())
    {
      api->ReleaseSessionOptions(so);
      return res;
    }

    // The honesty guard: a graph this EP cannot take entirely must fail
    // loudly, not fall back to the bundled CPU EP and report a CPU number
    // under an NPU heading.
    st = api->AddSessionConfigEntry(so, "session.disable_cpu_ep_fallback", "1");
    if (st)
      CLPEAK_VLOG("onnx: disable_cpu_ep_fallback rejected: %s\n",
                  onnxStatusText(rt, st).c_str());

    // ORT's MatMulAddFusion rewrites `MatMul` + `Add` into a single `Gemm`,
    // and several NPU providers implement MatMul but not Gemm -- the CoreML
    // EP accepts 20 of the transformer block's 22 nodes and refuses exactly
    // the two fused ones, which fails the whole session under the guard
    // above.  Turning the fusion off keeps every provider running the graph
    // as authored, which is both what makes the numbers comparable and what
    // an app targeting that NPU would have to do anyway.
    st = api->AddSessionConfigEntry(
        so, "optimization.disable_specified_optimizers", "MatMulAddFusion");
    if (st)
      CLPEAK_VLOG("onnx: disable_specified_optimizers rejected: %s\n",
                  onnxStatusText(rt, st).c_str());
  }

  OrtSession *session = nullptr;
  st = api->CreateSessionFromArray(env, modelBytes.data(), modelBytes.size(),
                                   so, &session);
  api->ReleaseSessionOptions(so);
  if (st)
  {
    res.error = onnxStatusText(rt, st);
    return res;
  }

  res.session = session;
  return res;
}

#endif // ENABLE_ONNX
