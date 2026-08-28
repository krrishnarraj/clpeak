#ifdef ENABLE_ONNX

#include "onnx_session.h"

#include <common/common.h>
#include <common/console_mute.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
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

    // Stop the partitioner bisecting a graph it cannot take.
    //
    // When TensorRT fails to parse a graph it does not give up: it splits it
    // and retries, up to trt_max_partition_iterations (1000 by default),
    // hunting for the largest subgraph it can claim.  Every attempt logs the
    // same import failure, which is where the flood of "cannot be imported
    // into TensorRT" lines comes from -- four seconds and dozens of identical
    // errors for one float8 E5M2 graph it was never going to accept.
    //
    // None of that search can help here.  Every session in this backend runs
    // with CPU fallback disabled, so a partition TensorRT only partly claims
    // fails the session exactly as a rejected one does.  One attempt answers
    // the only question clpeak asks: all of it, or none.
    {
      const char *keys[] = {"trt_max_partition_iterations"};
      const char *vals[] = {"1"};
      if (OrtStatus *st = api->UpdateTensorRTProviderOptions(trt, keys, vals, 1))
        CLPEAK_VLOG("onnx: trt_max_partition_iterations rejected: %s\n",
                    onnxStatusText(rt, st).c_str());
    }

    std::string err = onnxStatusText(
        rt, api->SessionOptionsAppendExecutionProvider_TensorRT_V2(so, trt));
    api->ReleaseTensorRTProviderOptions(trt);
    return err;
  }
  if (providerKey == "DnnlExecutionProvider")
  {
    OrtDnnlProviderOptions *dnnl = nullptr;
    if (OrtStatus *st = api->CreateDnnlProviderOptions(&dnnl))
      return onnxStatusText(rt, st);
    std::string err = onnxStatusText(
        rt, api->SessionOptionsAppendExecutionProvider_Dnnl(so, dnnl));
    api->ReleaseDnnlProviderOptions(dnnl);
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

// Where a profile file may be written.  Never the working directory: this is
// a benchmark, not something that should leave files where it was run from.
static std::string profilePrefixPath()
{
#ifdef _WIN32
  const char *tmp = std::getenv("TEMP");
  return std::string(tmp ? tmp : ".") + "\\clpeak_onnx_prof";
#else
  const char *tmp = std::getenv("TMPDIR");
  std::string dir = tmp ? tmp : "/tmp";
  if (!dir.empty() && dir.back() == '/')
    dir.pop_back();
  return dir + "/clpeak_onnx_prof";
#endif
}

std::vector<std::string> onnxCollectExecutedOps(const OrtRuntime &rt,
                                                OrtSession *session)
{
  std::vector<std::string> ops;
  if (!session)
    return ops;

  OrtAllocator *alloc = nullptr;
  if (OrtStatus *st = rt.api->GetAllocatorWithDefaultOptions(&alloc))
  {
    rt.api->ReleaseStatus(st);
    return ops;
  }

  char *path = nullptr;
  if (OrtStatus *st = rt.api->SessionEndProfiling(session, alloc, &path))
  {
    CLPEAK_VLOG("onnx: SessionEndProfiling failed: %s\n",
                onnxStatusText(rt, st).c_str());
    return ops;
  }
  if (!path)
    return ops;

  std::string file(path);
  if (OrtStatus *st = rt.api->AllocatorFree(alloc, path))
    rt.api->ReleaseStatus(st);

  // The profile is JSON, and every executed kernel carries an "op_name".
  // Scanning for that key beats parsing: no dependency, and the format has
  // been stable for years.
  std::ifstream in(file, std::ios::binary);
  std::string json((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  in.close();
  CLPEAK_VLOG("onnx: profile %s (%zu bytes)\n", file.c_str(), json.size());
  std::remove(file.c_str());

  // ONNX Runtime writes `"op_name" : "QLinearMatMul"`, spaces around the
  // colon included, so the separator is skipped rather than matched
  // literally -- a fixed `"op_name":"` finds nothing.
  const std::string key = "\"op_name\"";
  size_t pos = 0;
  while ((pos = json.find(key, pos)) != std::string::npos)
  {
    pos += key.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
      pos++;
    if (pos >= json.size() || json[pos] != ':')
      continue;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
      pos++;
    if (pos >= json.size() || json[pos] != '"')
      continue;
    pos++;
    const size_t end = json.find('"', pos);
    if (end == std::string::npos)
      break;
    std::string name = json.substr(pos, end - pos);
    pos = end;
    if (!name.empty() &&
        std::find(ops.begin(), ops.end(), name) == ops.end())
      ops.push_back(std::move(name));
  }
  return ops;
}

// The kernels that do the multiply in integer arithmetic, across providers.
// Matched loosely because providers prefix and suffix their own fusions.
static const char *kQuantMarkers[] = {
  "QLinearMatMul", "MatMulInteger", "QGemm", "QLinearGemm",
  "MatMulIntegerToFloat", "QuantizeLinearMatMul", "QOrderedMatMul",
  // Weight-only: ORT's own kernel for narrow blocked weights against
  // floating-point activations, and what a quantized language model runs on.
  "MatMulNBits",
};

std::string onnxQuantizedKernelName(const std::vector<std::string> &ops)
{
  for (const auto &op : ops)
    for (const char *m : kQuantMarkers)
      if (op.find(m) != std::string::npos)
        return op;
  return std::string();
}

bool onnxOpsRanQuantizedMatMul(const std::vector<std::string> &ops)
{
  if (ops.empty())
    return true;          // no profile to judge by; do not reject on silence

  bool sawQuantizeNode = false, sawPlainMatMul = false;
  for (const auto &op : ops)
  {
    // Exact names: "MatMul" is a substring of QLinearMatMul and
    // MatMulInteger, so a loose match here would reject every success.
    if (op == "MatMul" || op == "Gemm" || op == "FusedMatMul")
      sawPlainMatMul = true;
    if (op == "DequantizeLinear" || op == "QuantizeLinear")
      sawQuantizeNode = true;
  }
  return !(sawPlainMatMul && sawQuantizeNode);
}

OnnxSessionResult onnxCreateSession(const OrtRuntime &rt,
                                    const onnx_ep_info_t &ep,
                                    const std::string &modelBytes,
                                    bool keepConstantsUnfolded,
                                    bool profile,
                                    bool keepQdqUnfused)
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

  if (profile)
  {
    const std::string prefix = profilePrefixPath();
#ifdef _WIN32
    std::wstring wide(prefix.begin(), prefix.end());
    st = api->EnableProfiling(so, wide.c_str());
#else
    st = api->EnableProfiling(so, prefix.c_str());
#endif
    if (st)
      CLPEAK_VLOG("onnx: EnableProfiling rejected: %s\n",
                  onnxStatusText(rt, st).c_str());
  }

  // Optimizers this backend turns off, set once: ORT keeps one value per
  // config key, so a second AddSessionConfigEntry for the same key silently
  // replaces the first and warns about it.
  //
  // MatMulAddFusion rewrites `MatMul` + `Add` into a single `Gemm`, and
  // several NPU providers implement MatMul but not Gemm -- the CoreML EP
  // accepts 20 of the transformer block's 22 nodes and refuses exactly the
  // two fused ones, failing the whole session under the fallback guard
  // below.  It is disabled for every provider, the CPU one included: the
  // point is that each runs the graph as authored, and a provider running a
  // differently-optimised graph is not being compared with the others.
  //
  // ConstantFolding is disabled only for the throughput models, whose two
  // operands are both constants and would otherwise be multiplied once at
  // load time.
  {
    std::string disabled = "MatMulAddFusion";
    if (keepConstantsUnfolded)
      disabled += ";ConstantFolding";
    // See the header: QLinearMatMul cannot carry float8, so letting the QDQ
    // selector fire on a float8 graph turns a valid model into an invalid one.
    if (keepQdqUnfused)
      disabled += ";QDQSelectorActionTransformer";
    st = api->AddSessionConfigEntry(
        so, "optimization.disable_specified_optimizers", disabled.c_str());
    if (st)
      CLPEAK_VLOG("onnx: disable_specified_optimizers rejected: %s\n",
                  onnxStatusText(rt, st).c_str());
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
  }

  // Registering a provider can drag in a second copy of the ONNX schema
  // registry -- the XNNPACK EP emits hundreds of "Schema error: ... already
  // registered" lines from the bundled ONNX library, straight to the console
  // and below any ORT log level.  Mute the build; --verbose keeps it visible.
  OrtSession *session = nullptr;
  {
    clpeak::ScopedConsoleMute mute;
    st = api->CreateSessionFromArray(env, modelBytes.data(), modelBytes.size(),
                                     so, &session);
  }
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
