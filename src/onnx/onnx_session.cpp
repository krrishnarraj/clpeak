#ifdef ENABLE_ONNX

#include "onnx_session.h"
#include "onnx_model.h"

#include <common/common.h>
#include <common/console_mute.h>
#include <common/dynlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

std::string onnxDtypeUnsupportedReason(const OrtRuntime &rt, int dtype)
{
  const int      opset  = onnxOpsetForDtype(dtype);
  const uint32_t needApi = onnxMinOrtApiForOpset(opset);
  if (needApi && rt.apiVersion < needApi)
    return "needs opset " + std::to_string(opset) +
           ", which arrived in ONNX Runtime 1." + std::to_string(needApi) +
           "; this runtime is " + rt.versionString;

  // 1.18 is the first release that honours the request not to fuse.
  if (onnxIsQuantElem(dtype) && !onnxQdqFusionIsLegal(dtype) &&
      rt.apiVersion < 18)
    return "needs ONNX Runtime 1.18 or newer, the first that honours the "
           "request not to fuse a quantized matmul into QLinearMatMul -- which "
           "is an 8-bit integer operator and cannot carry this type; this "
           "runtime is " + rt.versionString;

  return std::string();
}

std::string onnxStatusText(const OrtRuntime &rt, OrtStatus *st)
{
  if (!st)
    return "";
  const char *msg = rt.api->GetErrorMessage(st);
  std::string out = msg ? msg : "";
  rt.api->ReleaseStatus(st);
  // A status with an empty message is as useless as no status: the row would
  // report a refusal with nothing in it, which is what a stock ONNX Runtime
  // 1.23 does for the float4 graphs.  Never hand back an empty reason.
  if (out.empty())
    return "the runtime refused it without saying why";
  // Session-creation errors from EP compilers can run to many lines; the
  // result rows are line-oriented, so keep the first line only.
  size_t nl = out.find('\n');
  if (nl != std::string::npos)
    out.resize(nl);
  return out;
}

OrtEnv *onnxEnv(const OrtRuntime &rt)
{
  // One OrtEnv per loaded runtime.  The GUI can hot-swap the ONNX Runtime
  // library in-process (Settings → Choose library…); enumeration is what
  // loads the runtime and needs no Env, so it succeeds, but the old Env
  // belongs to the old shared object's LoggingManager singleton.
  // Reusing it with the new OrtApi triggers
  //   "Attempt to use DefaultLogger but none has been registered."
  // Track which runtime the cached Env was created from and recreate it
  // when the runtime changes.  Works on Windows/Linux/macOS/Android;
  // on iOS (CLPEAK_ONNX_STATIC) the runtime never changes so the Env
  // is still created once.
  static std::mutex mutex;
  static OrtEnv *env = nullptr;
  static const OrtApi *envApi = nullptr;
  static const OrtApiBase *envBase = nullptr;
  std::lock_guard<std::mutex> lock(mutex);
  if (env && envApi == rt.api && envBase == rt.base)
    return env;
  if (env)
  {
    // Release with the API that created it; the old shared object is still
    // mapped (g_rt.lib is leaked on purpose) so its ReleaseEnv remains valid.
    if (envApi && envApi->ReleaseEnv)
      envApi->ReleaseEnv(env);
    else if (rt.api && rt.api->ReleaseEnv)
      rt.api->ReleaseEnv(env);
    env = nullptr;
    envApi = nullptr;
    envBase = nullptr;
  }
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
    return nullptr;
  }
  if (env)
  {
    envApi = rt.api;
    envBase = rt.base;
  }
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
bool genericEpOptions(const onnx_ep_info_t &ep, EpOptions &out)
{
  const std::string &providerKey = ep.providerKey;
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
    // One registration per enumerated target (see onnxAvailableEps):
    // NPU, GPU and CPU are different silicon behind one provider name,
    // and the target is part of what the row measures.  The pointer
    // borrows ep.epDevice, which outlives the append call below.
    const char *dev = ep.epDevice.empty() ? "NPU" : ep.epDevice.c_str();
    out = {"OpenVINO", {{"device_type", dev}}};
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

// Register `ep` on `so`.  Returns an empty string on success, or a
// one-line reason -- including "no wiring", so an unknown provider is
// reported as unsupported rather than silently run with defaults.
std::string appendProvider(const OrtRuntime &rt, OrtSessionOptions *so,
                           const onnx_ep_info_t &ep)
{
  const OrtApi *api = rt.api;
  const std::string &providerKey = ep.providerKey;

  EpOptions opts;
  if (genericEpOptions(ep, opts))
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

  // ---- NNAPI: an exported function, not an OrtApi entry -----------------
  // Android's EP predates the generic string-keyed API and never moved onto
  // it, so registration goes through a plain exported symbol taking a flag
  // word.  That symbol lives in the runtime we dlopen'd, which is also why
  // this path cannot exist on a statically linked build -- and does not need
  // to, NNAPI being Android-only.
  if (providerKey == "NnapiExecutionProvider")
  {
    if (!rt.lib)
      return "NNAPI needs a dynamically loaded onnxruntime";

    using AppendNnapiFn = OrtStatus *(ORT_API_CALL *)(OrtSessionOptions *,
                                                      uint32_t);
    auto append = reinterpret_cast<AppendNnapiFn>(clpeak::dynSym(
        rt.lib, "OrtSessionOptionsAppendExecutionProvider_Nnapi"));
    if (!append)
      return "this onnxruntime was built without the NNAPI provider";

    // NNAPI_FLAG_CPU_DISABLED (0x004).  NNAPI falls back to its own
    // nnapi-reference CPU implementation for anything the accelerator will
    // not take, and a row that quietly measured that would be an NPU number
    // in name only -- the same reason QNN is pinned to HTP and CoreML to
    // CPUAndNeuralEngine above.  A model the NPU cannot take then fails
    // loudly here instead, which is the honest outcome.  Ignored below
    // Android API 29; the app's minSdk is 33.
    return onnxStatusText(rt, append(so, 0x004));
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
#ifdef _MSC_VER
  char *tmpBuf = nullptr;
  size_t tmpLen = 0;
  if (_dupenv_s(&tmpBuf, &tmpLen, "TEMP") != 0) tmpBuf = nullptr;
  std::string out = std::string(tmpBuf ? tmpBuf : ".") + "\\clpeak_onnx_prof";
  free(tmpBuf);
  return out;
#else
  const char *tmp = std::getenv("TEMP");
  return std::string(tmp ? tmp : ".") + "\\clpeak_onnx_prof";
#endif
#else
#ifdef _MSC_VER
  char *tmpBuf = nullptr;
  size_t tmpLen = 0;
  if (_dupenv_s(&tmpBuf, &tmpLen, "TMPDIR") != 0) tmpBuf = nullptr;
  std::string dir = tmpBuf ? tmpBuf : "/tmp";
  free(tmpBuf);
#else
  const char *tmp = std::getenv("TMPDIR");
  std::string dir = tmp ? tmp : "/tmp";
#endif
  if (!dir.empty() && dir.back() == '/')
    dir.pop_back();
  return dir + "/clpeak_onnx_prof";
#endif
}

// ORT appends a timestamp to the prefix when it writes the profile. Make the
// prefix itself unique so a failed session can remove only its own file.
static std::string uniqueProfilePrefixPath()
{
  static std::atomic<uint64_t> serial{0};
  const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
#ifdef _WIN32
  const uint64_t pid = static_cast<uint64_t>(_getpid());
#else
  const uint64_t pid = static_cast<uint64_t>(getpid());
#endif
  return profilePrefixPath() + "_" + std::to_string(pid) + "_" +
         std::to_string(now) + "_" +
         std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
}

// SessionEndProfiling gives us the exact path on success. A provider can
// reject the model while creating the session, though, after ORT has already
// opened a profile file and before there is a session to end. The unique
// prefix above gives that failure path a safe cleanup target.
static void removeProfileArtifacts(const std::string &prefix)
{
  if (prefix.empty())
    return;

  const std::filesystem::path prefixPath(prefix);
  const std::filesystem::path dir = prefixPath.parent_path().empty()
      ? std::filesystem::path(".") : prefixPath.parent_path();
  const std::string namePrefix = prefixPath.filename().string();

  std::error_code ec;
  std::filesystem::directory_iterator it(dir, ec), end;
  for (; !ec && it != end; it.increment(ec))
  {
    const std::string name = it->path().filename().string();
    if (name.compare(0, namePrefix.size(), namePrefix) != 0)
      continue;

    std::error_code removeEc;
    std::filesystem::remove(it->path(), removeEc);
    if (removeEc)
      CLPEAK_VLOG("onnx: could not remove profile %s: %s\n",
                  it->path().string().c_str(), removeEc.message().c_str());
  }
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

  std::string profilePrefix;
  if (profile)
  {
    profilePrefix = uniqueProfilePrefixPath();
#ifdef _WIN32
    std::wstring wide(profilePrefix.begin(), profilePrefix.end());
    st = api->EnableProfiling(so, wide.c_str());
#else
    st = api->EnableProfiling(so, profilePrefix.c_str());
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
    res.error = appendProvider(rt, so, ep);
    if (!res.error.empty())
    {
      api->ReleaseSessionOptions(so);
      removeProfileArtifacts(profilePrefix);
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
    removeProfileArtifacts(profilePrefix);
    return res;
  }
  // Success and no session is a combination the API does not promise but has
  // been seen: the caller then reports a failure with an empty reason.
  if (!session)
  {
    res.error = "the runtime returned no session and no error";
    removeProfileArtifacts(profilePrefix);
    return res;
  }

  res.session = session;
  return res;
}

#endif // ENABLE_ONNX
