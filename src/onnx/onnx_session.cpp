#ifdef ENABLE_ONNX

#include "onnx_session.h"
#include "onnx_coreml_plan.h"
#include "onnx_model.h"
#include "onnx_plugin.h"

#include <common/common.h>
#include <common/console_mute.h>
#include <common/dynlib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::string onnxProviderFenceReason(const onnx_ep_info_t &ep, int dtype,
                                    bool qdq)
{
  // TensorRT for RTX on a per-tensor float4 QDQ matmul: GetCapability takes
  // the whole graph and the engine build that follows dies with an access
  // violation (exit 0xC0000005 -- NvTensorRTRTX EP 0.3.0 from the Windows ML
  // package 2.30.43, ONNX Runtime 1.27.1, RTX 5060), after the fp32, fp16,
  // bf16 and fp8_e4m3 rows had built and run on the same provider.  It is
  // the graph classic TensorRT *declines*, with "CHECK(output_quantize_axis_
  // .has_value()) failed": a float4 path that wants a quantization axis,
  // which only block scaling has.  The RTX library does not survive its own
  // check.  The block-scaled float4 rows (nvfp4, fp4_weight) are a different
  // graph -- the one that check asks for -- and are still put to it.
  if (ep.providerKey == "NvTensorRTRTXExecutionProvider" &&
      dtype == ONNX_DT_FLOAT4E2M1 && qdq)
    return "TensorRT for RTX takes the process down (an access violation in "
           "its engine build) on a per-tensor float4 QDQ matmul instead of "
           "declining it as TensorRT does -- its float4 path wants a block "
           "scale, which the nvfp4 row has -- so this graph is not sent to it";

  return std::string();
}

// Device loss, latched.  Written from ORT's logger thread and from whatever
// thread a status came back on, read by runAll between tests.
static std::atomic<bool> g_deviceLost{false};

bool onnxReasonIsDeviceLoss(const std::string &reason)
{
  // What the WebGPU/Dawn EP says on a driver reset, in the three places it
  // surfaces: the device-lost callback, the failed readback that follows, and
  // the Vulkan error underneath.  "device lost" also covers D3D12's
  // DXGI_ERROR_DEVICE_REMOVED path, which ORT reports with the same words.
  static const char *kMarkers[] = {
      "Device] is lost",
      "device lost",
      "Device lost",
      "DEVICE_LOST",
      "DEVICE_REMOVED",
      "Failed to download data from buffer",
  };
  for (const char *m : kMarkers)
    if (reason.find(m) != std::string::npos)
      return true;
  return false;
}

// Latch when `text` carries the loss; harmless for anything else.
static void noteDeviceLost(const std::string &text)
{
  if (!g_deviceLost.load(std::memory_order_relaxed) && onnxReasonIsDeviceLoss(text))
    g_deviceLost.store(true, std::memory_order_relaxed);
}

bool onnxDeviceLost() { return g_deviceLost.load(std::memory_order_relaxed); }
void onnxClearDeviceLost() { g_deviceLost.store(false, std::memory_order_relaxed); }

ResultStatus onnxFailureStatus(const std::string &reason, ResultStatus current)
{
  if (current == ResultStatus::Error)
    return current;
  return onnxReasonIsDeviceLoss(reason) ? ResultStatus::Error : ResultStatus::Unsupported;
}

std::string onnxStatusText(const OrtRuntime &rt, OrtStatus *st)
{
  if (!st)
    return "";
  const char *msg = rt.api->GetErrorMessage(st);
  std::string out = msg ? msg : "";
  rt.api->ReleaseStatus(st);
  // Before the first line is kept below: the loss is often named further down.
  noteDeviceLost(out);
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

namespace
{

// While the viability probe runs, the ORT logger's relay is demoted to
// debug: a target with no hardware behind it (OpenVINO NPU with no NPU)
// reports the refusal at ERROR severity, and with no LogSink installed
// (--list-devices, the GUI catalog) every Error reaches the terminal --
// past ScopedConsoleMute, which deliberately lets clpeak's own diagnostics
// through.  The probe already keeps the refusal reason from the returned
// status, so nothing is lost.  Nesting-safe; the latch in ortLogMessage
// stays first so a loss announced mid-probe is still caught.
std::atomic<int> g_relaySuppressDepth{0};

// OrtLoggingFunction: ORT's severities onto the run log's levels.  INFO is
// the session-creation narration -- providers registered, transformers
// applied -- which is debug material and dropped unless --verbose; ERROR
// includes a provider declining a graph, which is exactly why a reading is
// Unsupported, so it is kept.  May be called from ORT's own threads.
void ORT_API_CALL ortLogMessage(void *, OrtLoggingLevel severity,
                                const char *category, const char *,
                                const char *codeLocation, const char *message)
{
  // Before any filtering: the WebGPU EP announces the loss at INFO
  // ("WebGPU device lost (1): vkWaitForFences failed with VK_ERROR_DEVICE_LOST"),
  // which is debug material here and dropped without --verbose -- so latching
  // after the filter would see it only in verbose runs, the ones least in need
  // of the guard.
  if (message)
    noteDeviceLost(message);

  clpeak::LogLevel level;
  switch (severity)
  {
  case ORT_LOGGING_LEVEL_FATAL:
  case ORT_LOGGING_LEVEL_ERROR:   level = clpeak::LogLevel::Error;   break;
  case ORT_LOGGING_LEVEL_WARNING: level = clpeak::LogLevel::Warning; break;
  default:                        level = clpeak::LogLevel::Debug;   break;
  }
  if (level != clpeak::LogLevel::Debug &&
      g_relaySuppressDepth.load(std::memory_order_relaxed) > 0)
    level = clpeak::LogLevel::Debug;
  if (level == clpeak::LogLevel::Debug && !clpeak::verboseEnabled())
    return;
  // The optimizer narrates every pass over every session -- forty
  // "GraphTransformer X modified: 0" lines per session, none of them about
  // the device -- and clpeak creates sessions by the hundred.  The TensorRT
  // for RTX plugin's pool allocator does the same per *run*: a
  // "CudaMempoolAllocator::DoAlloc" and a "::DoFree" at INFO for every
  // Run(), which over a ladder is tens of thousands of lines saying the
  // output buffer came and went.  Everything else INFO says (provider
  // registration, partitioning, the session options, the pool's creation
  // and the arena's growth) is kept.
  if (level == clpeak::LogLevel::Debug && message &&
      (std::strncmp(message, "GraphTransformer ", 17) == 0 ||
       std::strncmp(message, "Running graph optimizations", 27) == 0 ||
       std::strncmp(message, "CudaMempoolAllocator::DoAlloc", 29) == 0 ||
       std::strncmp(message, "CudaMempoolAllocator::DoFree", 28) == 0))
    return;
  std::string text;
  if (category && *category && std::strcmp(category, "onnxruntime") != 0)
    text = std::string("[") + category + "] ";
  text += message ? message : "";
  if (codeLocation && *codeLocation)
    text += std::string(" (") + codeLocation + ")";
  clpeak::logMessage(level, "onnxruntime", text);
}

} // namespace

void onnxSuppressOrtRelay(bool on)
{
  int d = g_relaySuppressDepth.load(std::memory_order_relaxed) + (on ? 1 : -1);
  if (d < 0)
    d = 0;
  g_relaySuppressDepth.store(d, std::memory_order_relaxed);
}

// Why the last onnxEnv() refused, for the rows that report it.  Written
// and read under onnxEnv()'s lock or on the same thread after it returned
// null, so a plain string will do.
static std::string g_envError;

// The process's one OrtEnv, created on first use and never released.  One
// runtime per process (onnx_runtime.h) means nothing ever needs a second
// environment, and releasing one is where runtimes have crashed on their own
// teardown: before 1.25 the WebGPU provider's device-lost callback logs
// through the logger the environment has just destroyed (ReleaseEnv ->
// WebGpuContextFactory::Cleanup -> the Dawn device's DeviceLostEvent ->
// LOGS_DEFAULT, "Attempt to use DefaultLogger but none has been
// registered", thrown out of a destructor, so std::terminate; ORT 1.24.4
// for macOS, reproduced; microsoft/onnxruntime#27569), and before 1.29 a
// plugin provider's allocators held references teardown could outlive (an
// access violation AppVerifier caught; microsoft/onnxruntime#29770) -- the
// likeliest reading of the GUI dying as it left DirectML's 1.24.4 with
// Windows ML providers registered.  A change of plugin libraries is synced
// onto the live environment instead (onnxSyncEpLibraries).
static std::mutex g_envLock;
static OrtEnv *g_env = nullptr;
static const OrtApiBase *g_envBase = nullptr;   // the runtime it belongs to
// The plugin configuration the environment was last synced to (onnx_plugin.h).
static uint64_t g_envGeneration = 0;
// A runtime that refused once refuses again: remembered, so enumeration
// does not pay for the same failed attempt at every probe.
static const OrtApiBase *g_failedBase = nullptr;

OrtEnv *onnxEnv(const OrtRuntime &rt)
{
  std::lock_guard<std::mutex> lock(g_envLock);
  const uint64_t generation = onnxEpConfigGeneration();
  if (g_env && g_envBase == rt.base)
  {
    if (g_envGeneration != generation)
    {
      g_envGeneration = generation;
      onnxSyncEpLibraries(rt, g_env);
    }
    return g_env;
  }
  if (!g_env && g_failedBase == rt.base)
    return nullptr;
  // An environment of another runtime cannot be here -- the runtime is fixed
  // for the process -- and would not be released if it were: its runtime is
  // still mapped, so leaving it costs nothing a crash would not.
  g_env = nullptr;
  // The runtime's own log goes to the run log (ortLogMessage) instead of
  // to stderr: a provider explaining why it declined a graph, or which
  // nodes fell back to the CPU, is the line that explains an Unsupported
  // row or a slow one, and on a phone there is no stderr to read it from.
  // The Env is created once and its level is fixed, while --verbose can
  // differ from run to run in the GUI, so it is opened at INFO and the
  // callback applies the current run's verbosity.
  OrtStatus *st = rt.api->CreateEnvWithCustomLogger(
      ortLogMessage, nullptr, ORT_LOGGING_LEVEL_INFO, "clpeak", &g_env);
  if (st || !g_env)
  {
    // Two package-built runtimes (Debian's, Homebrew's) share one system
    // libonnx, whose schema registry is process-wide: the second to create
    // an environment finds every schema "already registered" and refuses.
    // The reason is the runtime's own words, kept for the skip rows.
    g_envError = st ? onnxStatusText(rt, st)
                    : "the runtime returned no environment and no error";
    CLPEAK_VLOG("onnx: CreateEnv failed: %s\n", g_envError.c_str());
    g_env = nullptr;
    g_failedBase = rt.base;
    return nullptr;
  }
  g_envBase = rt.base;
  g_envGeneration = generation;
  g_envError.clear();
  // Plugin providers live on the Env, so this is where they are registered
  // -- before anything enumerates or attaches them.
  onnxSyncEpLibraries(rt, g_env);
  return g_env;
}

std::string onnxEnvError()
{
  return g_envError;
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
  std::vector<std::pair<std::string, std::string>> kv;
};

// The QNN backend library for a device: `backend_path` naming the file
// beside the plugin library when it is there to be named, else
// `backend_type` and the provider's own search.  A built-in QNN provider
// resolves a bare file name against its runtime's directory (and the
// plugin against its own), but naming the file outright when it is in
// sight leaves nothing to a loader search that depends on which
// directory the process started in.
void qnnBackend(const onnx_ep_info_t &ep, const char *type, const char *file,
                EpOptions &out)
{
  if (ep.epDevicePtr)
  {
    const std::string lib = onnxEpLibraryPath(ep.library);
    const size_t slash = lib.find_last_of("/\\");
    if (slash != std::string::npos)
    {
      const std::string beside = lib.substr(0, slash + 1) + file;
      std::error_code ec;
      if (std::filesystem::exists(beside, ec))
      {
        out.kv.emplace_back("backend_path", beside);
        return;
      }
    }
    out.kv.emplace_back("backend_type", type);
    return;
  }
  out.kv.emplace_back("backend_path", file);
}

// The compute units the CoreML provider is asked for.  CoreML has no
// NPU-only mode; CPUAndNeuralEngine is the strictest available request,
// and what Core ML then does with each operation is read back from its
// compute plan (onnx_coreml_plan.h).
constexpr const char *kCoremlComputeUnits = "CPUAndNeuralEngine";

// Providers registered through the generic string-keyed API -- and the
// plugin providers, which take the same key/value options through the
// OrtEpDevice append (onnx_plugin.h).  Returns false for anything not
// handled here; the caller then tries the typed paths.
// `wantPlan` asks the CoreML provider to log its compute plan while it
// loads the model (ProfileComputePlan, ORT 1.20+): the provider then loads
// the compiled model a second time to read MLComputePlan, which is what
// the native backend pays too and the price of knowing where the work ran.
// Off for the probes and the attach-only check, which never time a run.
bool genericEpOptions(const onnx_ep_info_t &ep, EpOptions &out, bool wantPlan,
                      const std::string &nativeProfile)
{
  const std::string &providerKey = ep.providerKey;
  if (providerKey == "CoreMLExecutionProvider")
  {
    // MLProgram + Neural Engine: the fp16-native ANE path.
    out = {"CoreML",
           {{"ModelFormat", "MLProgram"},
            {"MLComputeUnits", kCoremlComputeUnits}}};
    if (wantPlan)
      out.kv.emplace_back("ProfileComputePlan", "1");
    return true;
  }
  if (providerKey == "QNNExecutionProvider")
  {
#if defined(_WIN32)
    const char *htpFile = "QnnHtp.dll";
    const char *gpuFile = "QnnGpu.dll";
    const char *cpuFile = "QnnCpu.dll";
#else
    const char *htpFile = "libQnnHtp.so";
    const char *gpuFile = "libQnnGpu.so";
    const char *cpuFile = "libQnnCpu.so";
#endif
    out = {"QNN", {}};
    // The plugin enumerates each backend's device on its own (the HTP as
    // an NPU, the Adreno as a GPU, the reference backend as a CPU), and
    // the row is the device it says.  The built-in provider enumerates
    // once, and there HTP is the NPU proper: without naming it the EP
    // would settle for the DSP or the CPU reference backend and the row
    // would not mean what it says.
    if (ep.epDevicePtr && ep.deviceType == DeviceType::Gpu)
      qnnBackend(ep, "gpu", gpuFile, out);
    else if (ep.epDevicePtr && ep.deviceType == DeviceType::Cpu)
      qnnBackend(ep, "cpu", cpuFile, out);
    else
    {
      qnnBackend(ep, "htp", htpFile, out);
      // Peak clocks.  The HTP's default performance mode is a power
      // saver; "burst" is what a benchmark -- and Qualcomm's own
      // profiling tools -- ask for, and the difference is a large
      // multiple on sustained work.
      out.kv.emplace_back("htp_performance_mode", "burst");
      out.kv.emplace_back("qnn_context_priority", "high");
      // The most optimised graph the finalizer will produce.  Costs
      // preparation time, which the session-creation budgets bound.
      out.kv.emplace_back("htp_graph_finalization_optimization_mode", "3");
      // Keep the graph-boundary QuantizeLinear/DequantizeLinear on the
      // HTP.  The default hands them to the CPU EP, which the fallback
      // guard then refuses -- and a quantized input arriving through
      // a dequantize is exactly what the numeric-error and live QDQ
      // graphs have at their boundary.
      out.kv.emplace_back("offload_graph_io_quantization", "0");
    }
    // Per-operation device timings, written as CSV.  "detailed" is the
    // level that breaks a graph down by node; every QNN backend has it.
    if (!nativeProfile.empty())
    {
      out.kv.emplace_back("profiling_level", "detailed");
      out.kv.emplace_back("profiling_file_path", nativeProfile);
    }
    return true;
  }
  if (providerKey == "OpenVINOExecutionProvider")
  {
    // One registration per enumerated target (see onnxAvailableEps):
    // NPU, GPU and CPU are different silicon behind one provider name,
    // and the target is part of what the row measures.  A plugin OpenVINO
    // is appended for its OrtEpDevice, which already names the silicon.
    if (ep.epDevicePtr)
    {
      out = {"OpenVINO", {}};
      return true;
    }
    out = {"OpenVINO",
           {{"device_type", ep.epDevice.empty() ? "NPU" : ep.epDevice}}};
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
                           const onnx_ep_info_t &ep, bool wantPlan,
                           const std::string &nativeProfile = std::string())
{
  const OrtApi *api = rt.api;
  const std::string &providerKey = ep.providerKey;

  // ---- Plugin providers: appended for their OrtEpDevice -----------------
  // The string-keyed append below knows the built-in names only and
  // answers "not supported in this build" for a plugin, however it was
  // registered.  The options are the same table: a plugin QNN takes the
  // HTP options a built-in one does.  A plugin clpeak has no wiring for
  // runs with the provider's own defaults -- the person naming the library
  // asked for exactly that, and the fallback guard still fails a session
  // the provider does not take whole.
  if (ep.epDevicePtr)
  {
    EpOptions opts;
    if (!genericEpOptions(ep, opts, wantPlan, nativeProfile))
      opts = {"", {}};
    return onnxAppendPluginDevice(rt, so, ep, opts.kv);
  }

  EpOptions opts;
  if (genericEpOptions(ep, opts, wantPlan, nativeProfile))
  {
    std::vector<const char *> keys, vals;
    for (auto &kvp : opts.kv)
    {
      keys.push_back(kvp.first.c_str());
      vals.push_back(kvp.second.c_str());
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

// Attach-only check for the viability probe: provider registration with no
// model behind it.  A target the provider cannot serve (OpenVINO NPU with
// no NPU, a QNN backend_path pointing at a missing library) fails here,
// before any graph is built or compiled.
std::string onnxProviderAttach(const OrtRuntime &rt, const onnx_ep_info_t &ep)
{
  // The CPU EP is implicit and needs no registration; it always attaches.
  if (ep.providerKey == "CPUExecutionProvider")
    return std::string();

  // Provider appends log through ORT's DefaultLogger, which only exists
  // once an Env has registered it.  Session creation always builds the
  // Env first; this attach-only path must do the same, or EPs that log at
  // registration (TensorRT, CUDA) fail with "Attempt to use DefaultLogger
  // but none has been registered" instead of attaching.
  if (!onnxEnv(rt))
    return "onnxruntime environment creation failed: " + onnxEnvError();

  OrtSessionOptions *so = nullptr;
  if (OrtStatus *st = rt.api->CreateSessionOptions(&so))
    return onnxStatusText(rt, st);
  // Appending a target with no hardware behind it (OpenVINO NPU with no
  // NPU) makes the provider log straight to the console, below any ORT
  // log level -- the same class of spam session creation already mutes.
  // Mute the append; the returned status still carries the reason.
  std::string err;
  {
    clpeak::ScopedConsoleMute mute;
    err = appendProvider(rt, so, ep, /*wantPlan=*/false);
  }
  rt.api->ReleaseSessionOptions(so);
  return err;
}

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

std::string onnxNativeProfilePath(const onnx_ep_info_t &ep)
{
  // The providers whose own profiler genericEpOptions knows how to ask for.
  if (ep.providerKey == "QNNExecutionProvider")
    return uniqueProfilePrefixPath() + "_qnn.csv";
  return std::string();
}

void onnxLogNativeProfile(const std::string &path, const std::string &tag)
{
  if (path.empty())
    return;
  std::ifstream in(path, std::ios::binary);
  if (!in)
  {
    CLPEAK_VLOG("onnx-native-profile[%s]: the provider wrote nothing to %s\n",
                tag.c_str(), path.c_str());
    return;
  }
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);)
  {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (!line.empty())
      lines.push_back(std::move(line));
  }
  in.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);

  // Bounded: the head holds the column names and the one-off setup events,
  // the tail the last few runs, which is what the timings are read from.
  constexpr size_t kHead = 40, kTail = 360;
  CLPEAK_VLOG("onnx-native-profile[%s]: %zu lines\n", tag.c_str(), lines.size());
  for (size_t i = 0; i < lines.size(); i++)
  {
    if (i == kHead && lines.size() > kHead + kTail)
    {
      CLPEAK_VLOG("onnx-native-profile[%s]: ... %zu lines skipped ...\n",
                  tag.c_str(), lines.size() - kHead - kTail);
      i = lines.size() - kTail;
    }
    CLPEAK_VLOG("onnx-native-profile[%s]: %s\n", tag.c_str(), lines[i].c_str());
  }
}

const char *onnxProfileTypeName(int dtype)
{
  switch (dtype)
  {
  case ONNX_DT_FLOAT:    return "float";
  case ONNX_DT_FLOAT16:  return "float16";
  case ONNX_DT_BFLOAT16: return "bfloat16";
  default:               return ""; // quantized: no plain kernel to check
  }
}

std::vector<std::string> onnxCollectExecutedOps(const OrtRuntime &rt,
                                                OrtSession *session,
                                                std::string *opInType,
                                                const char *ofOp)
{
  std::vector<std::string> ops;
  if (opInType)
    opInType->clear();
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

    // Capture the element type of the kernel's first input.  ORT writes
    // `"input_type_shape" : [ { "float16" : [32,32] }, ...` in the same
    // event's args; find it near this op_name and read the first quoted key
    // inside the first brace.
    const bool wanted = ofOp ? (name == ofOp)
                             : (name == "MatMul" || name == "FusedMatMul" ||
                                name == "Gemm");
    if (opInType && opInType->empty() && wanted)
    {
      const std::string itsKey = "\"input_type_shape\"";
      // The args object holds op_name and input_type_shape together; search a
      // bounded window on either side rather than counting braces.
      size_t lo = (pos > 800) ? pos - 800 : 0;
      size_t its = json.find(itsKey, lo);
      if (its != std::string::npos && its < pos + 800)
      {
        size_t br = json.find('{', its); // first tensor's { "<type>": ... }
        size_t q1 = (br == std::string::npos) ? std::string::npos
                                              : json.find('"', br);
        if (q1 != std::string::npos)
        {
          size_t q2 = json.find('"', q1 + 1);
          if (q2 != std::string::npos)
            *opInType = json.substr(q1 + 1, q2 - q1 - 1);
        }
      }
    }

    if (!name.empty())
      ops.push_back(std::move(name));
  }
  return ops;
}

size_t onnxCountOp(const std::vector<std::string> &ops, const char *name)
{
  size_t n = 0;
  for (const auto &op : ops)
    if (op == name)
      n++;
  return n;
}

std::string onnxJoinOps(const std::vector<std::string> &ops)
{
  std::string out;
  std::vector<std::string> seen;
  for (const auto &op : ops)
  {
    if (std::find(seen.begin(), seen.end(), op) != seen.end())
      continue;
    seen.push_back(op);
    out += (out.empty() ? "" : ", ") + op;
  }
  return out;
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
                                    bool keepQdqUnfused,
                                    bool verifyPlacement,
                                    const std::string &nativeProfile)
{
  OnnxSessionResult res;
  const OrtApi *api = rt.api;

  // The one provider whose runtime can be asked where the work went.  The
  // unit is the one its MLComputeUnits request names; a request no single
  // unit answers for is not verified.
  const OnnxCoremlUnit unit =
      (verifyPlacement && ep.providerKey == "CoreMLExecutionProvider")
          ? onnxCoremlUnitFor(kCoremlComputeUnits)
          : OnnxCoremlUnit{};
  const bool wantPlan = !unit.cls.empty();

  OrtEnv *env = onnxEnv(rt);
  if (!env)
  {
    res.error = "onnxruntime environment creation failed: " + onnxEnvError();
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
    res.error = appendProvider(rt, so, ep, wantPlan, nativeProfile);
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
  // When the CoreML provider was asked for its compute plan, the console is
  // where it arrives (NSLog, one line per operation, before the session is
  // initialised), so that build is captured rather than discarded.
  OrtSession *session = nullptr;
  std::string console;
  {
    clpeak::ScopedConsoleMute mute(wantPlan ? clpeak::ScopedConsoleMute::Capture::Always
                                            : clpeak::ScopedConsoleMute::Capture::Verbose);
    st = api->CreateSessionFromArray(env, modelBytes.data(), modelBytes.size(),
                                     so, &session);
    if (wantPlan)
    {
      mute.finish();
      console = mute.text();
    }
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

  // The placement guard.  ORT's fallback guard above proved the provider
  // took every node; this proves its runtime ran them on the unit the row is
  // named for, by the same 5%-of-cost rule the native Core ML backend
  // applies to its own plans.  A plan that never arrived -- an OS before
  // macOS 14.4 / iOS 17.4, where MLComputePlan does not exist and the
  // provider says so through its logger -- leaves the session unverified
  // rather than refused: it is what every other provider gets, and the
  // verbose log says which it was.
  if (wantPlan)
  {
    const OnnxCoremlPlan plan = onnxParseCoremlPlan(console);
    if (!plan.known)
      CLPEAK_VLOG("onnx: no compute plan from the CoreML provider (%s); placement unverified\n",
                  plan.failure.empty() ? "no placement lines on the console" : plan.failure.c_str());
    else
    {
      size_t onUnit = 0;
      for (const auto &op : plan.ops)
        onUnit += (op.device == unit.cls) ? 1 : 0;
      CLPEAK_VLOG("onnx: compute plan: %zu of %zu operations on %s\n", onUnit, plan.ops.size(),
                  unit.name.c_str());
      const std::string why = onnxCoremlPlanOffDeviceReason(plan, unit.cls, unit.name);
      if (!why.empty())
      {
        api->ReleaseSession(session);
        removeProfileArtifacts(profilePrefix);
        res.error = why;
        res.offDevice = true;
        return res;
      }
    }
  }

  res.session = session;
  return res;
}

#endif // ENABLE_ONNX
