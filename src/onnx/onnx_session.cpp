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
    OrtLoggingLevel level = clpeak::verboseEnabled() ? ORT_LOGGING_LEVEL_WARNING
                                                     : ORT_LOGGING_LEVEL_ERROR;
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
// Per-EP registration options.  These are the options a well-behaved app
// would pass to reach the vendor's accelerator; they are part of what the
// backend measures, so they live in one visible place.
// ---------------------------------------------------------------------------

namespace
{

struct EpOptions
{
  const char *registrationName;     // name the generic append API expects
  std::vector<std::pair<const char *, const char *>> kv;
};

// Returns false when clpeak has no wiring for this provider yet (the caller
// reports it as such, rather than measuring something unintended).
bool epOptionsFor(const std::string &providerKey, EpOptions &out)
{
  if (providerKey == "CPUExecutionProvider")
  {
    out = {nullptr, {}};            // implicit: every session has the CPU EP
    return true;
  }
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
  return false;   // CUDA / TensorRT / ROCm / MIGraphX: wiring comes later
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

  EpOptions opts;
  if (!epOptionsFor(ep.providerKey, opts))
  {
    res.error = "clpeak has no session wiring for " + ep.providerKey + " yet";
    return res;
  }

  OrtSessionOptions *so = nullptr;
  OrtStatus *st = api->CreateSessionOptions(&so);
  if (st)
  {
    res.error = onnxStatusText(rt, st);
    return res;
  }

  if (opts.registrationName)
  {
    std::vector<const char *> keys, vals;
    for (auto &kvp : opts.kv)
    {
      keys.push_back(kvp.first);
      vals.push_back(kvp.second);
    }
    st = api->SessionOptionsAppendExecutionProvider(
        so, opts.registrationName, keys.data(), vals.data(), keys.size());
    if (st)
    {
      res.error = onnxStatusText(rt, st);
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
