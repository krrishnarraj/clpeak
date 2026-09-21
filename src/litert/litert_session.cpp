#ifdef ENABLE_LITERT

#include "litert_session.h"

#include <common/common.h>
#include <common/console_mute.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

#include "litert/c/litert_profiler_event.h"

namespace
{

using Clock = std::chrono::steady_clock;

double elapsedUs(Clock::time_point t0)
{
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
// LiteRT logs through a process-wide default logger, to stderr (or logcat)
// unless replaced.  It is replaced once per runtime with a sink logger: the
// console stays clean, every line reaches the run log at LiteRT's own
// severity under --verbose, and a session that fails to compile can quote
// what the runtime said instead of a bare status code -- which for an NPU
// compiler declining a graph is the whole answer.
std::mutex g_logMutex;
LiteRtLogger g_sink = nullptr;
const LitertRuntime *g_sinkRuntime = nullptr;

void installSink(const LitertRuntime &rt)
{
  std::lock_guard<std::mutex> lock(g_logMutex);
  if (g_sinkRuntime == &rt || !rt.api.LiteRtCreateSinkLogger || !rt.api.LiteRtSetDefaultLogger)
    return;
  LiteRtLogger sink = nullptr;
  if (rt.api.LiteRtCreateSinkLogger(&sink) != kLiteRtStatusOk || !sink)
    return;
  if (rt.api.LiteRtSetMinLoggerSeverity)
    rt.api.LiteRtSetMinLoggerSeverity(sink, kLiteRtLogSeverityInfo);
  if (rt.api.LiteRtSetDefaultLogger(sink) != kLiteRtStatusOk)
    return;
  g_sink = sink;
  g_sinkRuntime = &rt;
}

// LiteRT prefixes each sink line with its severity name and "[file:line]".
clpeak::LogLevel levelOf(const std::string &line)
{
  if (line.rfind("ERROR", 0) == 0)
    return clpeak::LogLevel::Error;
  if (line.rfind("WARNING", 0) == 0)
    return clpeak::LogLevel::Warning;
  return clpeak::LogLevel::Debug;
}

} // namespace

std::string litertDrainLog(const LitertRuntime &rt)
{
  std::lock_guard<std::mutex> lock(g_logMutex);
  std::string text;
  if (!g_sink || g_sinkRuntime != &rt || !rt.api.LiteRtGetSinkLoggerSize)
    return text;
  size_t n = 0;
  if (rt.api.LiteRtGetSinkLoggerSize(g_sink, &n) != kLiteRtStatusOk)
    return text;
  for (size_t i = 0; i < n; i++)
  {
    const char *msg = nullptr;
    if (rt.api.LiteRtGetSinkLoggerMessage(g_sink, i, &msg) != kLiteRtStatusOk || !msg)
      continue;
    std::string line = msg;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
      line.pop_back();
    if (line.empty())
      continue;
    const clpeak::LogLevel lvl = levelOf(line);
    if (lvl != clpeak::LogLevel::Debug || clpeak::verboseEnabled())
      clpeak::logMessage(lvl, "litert", line);
    text += line;
    text += '\n';
  }
  if (rt.api.LiteRtClearSinkLogger)
    rt.api.LiteRtClearSinkLogger(g_sink);
  return text;
}

const char *litertAccelName(LitertAccel a)
{
  switch (a)
  {
  case LitertAccel::Cpu: return "CPU";
  case LitertAccel::Gpu: return "GPU";
  case LitertAccel::Npu: return "NPU";
  }
  return "?";
}

namespace
{

// The last error or warning lines of what the runtime said -- its sink log
// and whatever it printed to the console -- for a row reason.  LiteRT's own
// lines look like "ERROR: [file.cc:12] text"; TFLite's kernel errors reach
// stderr as "ERROR: tflite/kernels/x.cc:302 text"; the accelerator
// libraries' glog lines ("E0000 ...") carry their severity in the first
// letter.  Only the sentence survives.
std::string lastLines(const std::string &log, int keep = 3)
{
  std::vector<std::string> lines;
  size_t pos = 0;
  while (pos < log.size())
  {
    size_t nl = log.find('\n', pos);
    if (nl == std::string::npos)
      nl = log.size();
    std::string l = log.substr(pos, nl - pos);
    pos = nl + 1;
    const bool glogBad = (l.size() > 5 && (l[0] == 'E' || l[0] == 'W') && l[1] == '0');
    if (levelOf(l) == clpeak::LogLevel::Debug && !glogBad)
      continue;
    if (glogBad)
    {
      // "E0000 00:00:123.456 7 file.cc:9] text"
      const size_t br = l.find("] ");
      if (br != std::string::npos)
        l = l.substr(br + 2);
    }
    else
    {
      const size_t colon = l.find(": ");
      if (colon != std::string::npos)
        l = l.substr(colon + 2);            // past "ERROR: "
      if (!l.empty() && l[0] == '[')
      {
        const size_t br = l.find("] ");     // "[file.cc:12] "
        if (br != std::string::npos)
          l = l.substr(br + 2);
      }
      else
      {
        // "tflite/kernels/x.cc:302 text": a path token then the sentence.
        const size_t sp = l.find(' ');
        if (sp != std::string::npos && l.find(".cc:") < sp)
          l = l.substr(sp + 1);
      }
    }
    if (!l.empty() && (lines.empty() || lines.back() != l))
      lines.push_back(l);
  }
  std::string out;
  const size_t from = lines.size() > (size_t)keep ? lines.size() - keep : 0;
  for (size_t i = from; i < lines.size(); i++)
  {
    if (!out.empty())
      out += "; ";
    out += lines[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Environments
// ---------------------------------------------------------------------------
// One LiteRtEnvironment per accelerator for the life of the process.  An
// environment is where LiteRT loads and initialises accelerators (the GPU
// accelerator's OpenCL/Metal context, an NPU's dispatch library), which is
// expensive and, for the GPU, keeps a context alive that later sessions
// share.  Each is created with only its own accelerator registered, so the
// CPU device's runs never touch the GPU library and the GPU device's
// fallback path is the CPU built into the runtime rather than a delegate.
//
// Each belongs to the runtime that created it.  The GUI can point the
// backend at a different libLiteRt between runs (Settings), and an
// environment is an object of the library that made it -- handing it to
// another library's LiteRtCreateCompiledModel would run one build's code
// over another build's object, and would measure the old library's
// accelerator under the new one's name even when the builds happen to
// match.  So a record whose owner is not the current runtime is torn down
// with its owner's own entry points and rebuilt, and a remembered creation
// failure goes with it: the library that had no GPU accelerator beside it
// was the old one.  The owner is the library handle, unique per mapped file
// and never unmapped (common/dynlib.h), so the same path picked again keeps
// its environment.  Sessions are never alive across a switch -- the FFI
// refuses a library change during a run -- but the record checks anyway.
struct EnvRecord
{
  const void *owner = nullptr;   // LitertRuntime::lib of the runtime that created it
  LitertApi api;                 // that runtime's entry points, for the matching destroy
  LiteRtEnvironment env = nullptr;
  std::string error;             // why creation failed, when env is null
  std::string log;               // what LiteRT said while bringing it up
  int models = 0;                // compiled models created on it, ever
  int live = 0;                  // sessions currently alive on it
};
std::mutex g_envMutex;
std::map<int, EnvRecord> g_envs;   // by accelerator

// The Metal accelerator adds a residency set to its command queue for every
// compiled model and never removes one, and IOGPUMetalCommandQueue asserts
// -- an abort -- on the 33rd ("command queue residency set limit of 32
// exceeded").  Disabling the sets through the GPU options does not stop it,
// and neither does draining an autorelease pool; only a new command queue
// does, which means a new environment.  So on Apple the GPU environment is
// torn down and rebuilt after this many models, between sessions, at the
// cost of ~300 ms of Metal initialisation each time.  A doubling ladder
// over nine formats compiles that many models before its second test.
#ifdef __APPLE__
constexpr int kGpuModelsPerEnvironment = 24;
#else
constexpr int kGpuModelsPerEnvironment = 0;   // no such limit known elsewhere
#endif

LiteRtAny anyString(const char *s)
{
  LiteRtAny a;
  a.type = kLiteRtAnyTypeString;
  a.str_value = s;
  return a;
}
LiteRtAny anyInt(int64_t v)
{
  LiteRtAny a;
  a.type = kLiteRtAnyTypeInt;
  a.int_value = v;
  return a;
}

LiteRtEnvironment environmentFor(const LitertRuntime &rt, LitertAccel accel, std::string &error)
{
  std::lock_guard<std::mutex> lock(g_envMutex);
  const int key = (int)accel;
  auto it = g_envs.find(key);
  if (it != g_envs.end() && it->second.owner != rt.lib)
  {
    // Another runtime's environment, or its refusal: see the note above.
    if (it->second.live > 0)
    {
      error = "the " + std::string(litertAccelName(accel)) +
              " environment of the previously loaded LiteRT still has sessions alive";
      return nullptr;
    }
    CLPEAK_VLOG("litert: dropping the %s environment of the previously loaded runtime\n",
                litertAccelName(accel));
    if (it->second.env)
      it->second.api.LiteRtDestroyEnvironment(it->second.env);
    g_envs.erase(it);
    it = g_envs.end();
  }
  if (it != g_envs.end() && it->second.env && accel == LitertAccel::Gpu &&
      kGpuModelsPerEnvironment > 0 && it->second.models >= kGpuModelsPerEnvironment &&
      it->second.live == 0)
  {
    CLPEAK_VLOG("litert: recreating the GPU environment after %d compiled models\n",
                it->second.models);
    it->second.api.LiteRtDestroyEnvironment(it->second.env);
    g_envs.erase(it);
    it = g_envs.end();
  }
  if (it != g_envs.end())
  {
    if (!it->second.env)
    {
      error = it->second.error;
      return nullptr;
    }
    it->second.models++;
    it->second.live++;
    return it->second.env;
  }

  // Strings must outlive the call only; LiteRT copies option values.
  const std::string libDir = rt.libraryDir;
  const std::string npuDir = litertNpuDir();
  std::vector<LiteRtEnvOption> opts;
  int mask = 0;
  switch (accel)
  {
  case LitertAccel::Cpu: mask = kLiteRtHwAcceleratorCpu; break;
  case LitertAccel::Gpu: mask = kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu; break;
  case LitertAccel::Npu: mask = kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorNpu; break;
  }
  opts.push_back({kLiteRtEnvOptionTagAutoRegisterAccelerators, anyInt(mask)});
  if (!libDir.empty())
    opts.push_back({kLiteRtEnvOptionTagRuntimeLibraryDir, anyString(libDir.c_str())});
  if (accel == LitertAccel::Npu && !npuDir.empty())
  {
    opts.push_back({kLiteRtEnvOptionTagDispatchLibraryDir, anyString(npuDir.c_str())});
    opts.push_back({kLiteRtEnvOptionTagCompilerPluginLibraryDir, anyString(npuDir.c_str())});
  }

  LiteRtEnvironment env = nullptr;
  LiteRtStatus st;
  std::string logged;
  {
    // Accelerator libraries announce themselves on the console (the Metal
    // one through glog) below any logger LiteRT lets us set.
    clpeak::ScopedConsoleMute mute(clpeak::ScopedConsoleMute::Capture::Always);
    st = rt.api.LiteRtCreateEnvironment((int)opts.size(), opts.data(), &env);
    mute.finish();
    logged = mute.text();
  }
  logged += litertDrainLog(rt);
  EnvRecord rec;
  rec.owner = rt.lib;
  rec.api = rt.api;
  rec.log = logged;
  if (st != kLiteRtStatusOk || !env)
  {
    error = "LiteRT could not create an environment for the " + std::string(litertAccelName(accel)) +
            ": " + litertStatusText(rt, st);
    const std::string why = lastLines(logged);
    if (!why.empty())
      error += " (" + why + ")";
    rec.error = error;
    g_envs[key] = rec;
    return nullptr;
  }
  rec.env = env;
  rec.models = 1;
  rec.live = 1;
  g_envs[key] = rec;
  return env;
}

void releaseEnvironment(LitertAccel accel)
{
  std::lock_guard<std::mutex> lock(g_envMutex);
  auto it = g_envs.find((int)accel);
  if (it != g_envs.end() && it->second.live > 0)
    it->second.live--;
}

// An opaque option: a TOML string under the identifier the accelerator
// reads it by (litert/c/options/*.cc upstream serialize exactly this).
bool addOpaque(const LitertRuntime &rt, LiteRtOptions options, const char *identifier,
               const std::string &toml, std::string &error)
{
  char *payload = static_cast<char *>(std::malloc(toml.size() + 1));
  if (!payload)
  {
    error = "out of memory";
    return false;
  }
  std::memcpy(payload, toml.data(), toml.size());
  payload[toml.size()] = 0;
  LiteRtOpaqueOptions opaque = nullptr;
  LiteRtStatus st = rt.api.LiteRtCreateOpaqueOptions(
      identifier, payload, [](void *p) { std::free(p); }, &opaque);
  if (st != kLiteRtStatusOk || !opaque)
  {
    std::free(payload);
    error = std::string("LiteRT rejected ") + identifier + " options: " + litertStatusText(rt, st);
    return false;
  }
  st = rt.api.LiteRtAddOpaqueOptions(options, opaque);
  if (st != kLiteRtStatusOk)
  {
    rt.api.LiteRtDestroyOpaqueOptions(opaque);
    error = std::string("LiteRT rejected ") + identifier + " options: " + litertStatusText(rt, st);
    return false;
  }
  return true;
}

} // namespace

bool litertPrepareEnvironment(const LitertRuntime &rt, LitertAccel accel, std::string &error)
{
  installSink(rt);
  LiteRtEnvironment env = environmentFor(rt, accel, error);
  if (env)
    releaseEnvironment(accel);   // environmentFor counted a session that is not coming
  return env != nullptr;
}

void litertResetEnvironment(const LitertRuntime &rt, LitertAccel accel)
{
  std::lock_guard<std::mutex> lock(g_envMutex);
  auto it = g_envs.find((int)accel);
  if (it == g_envs.end())
    return;
  if (it->second.env)
    it->second.api.LiteRtDestroyEnvironment(it->second.env);
  g_envs.erase(it);
  litertDrainLog(rt);
}

std::string litertEnvironmentLog(LitertAccel accel)
{
  std::lock_guard<std::mutex> lock(g_envMutex);
  auto it = g_envs.find((int)accel);
  return it == g_envs.end() ? std::string() : it->second.log;
}

// ---------------------------------------------------------------------------
// LitertSession
// ---------------------------------------------------------------------------

std::unique_ptr<LitertSession> LitertSession::create(const LitertRuntime &rt,
                                                     const litert_device_info_t &dev,
                                                     clpeak_tflite::TfliteBytes &&model,
                                                     const LitertSessionConfig &cfg,
                                                     std::string &error)
{
  installSink(rt);
  litertDrainLog(rt);   // whatever an earlier failure left behind is not ours

  std::unique_ptr<LitertSession> s(new LitertSession());
  s->rt_ = &rt;
  s->bytes_ = std::move(model);
  s->profile_ = cfg.profile;
  s->accel_ = dev.accel;
  s->accelName_ = litertAccelName(dev.accel);

  s->env_ = environmentFor(rt, dev.accel, error);
  if (!s->env_)
    return nullptr;
  s->holdsEnv_ = true;

  std::string console;   // what the runtime printed while creating
  auto fail = [&](const std::string &what, LiteRtStatus st) -> std::unique_ptr<LitertSession> {
    s->creationLog = console + litertDrainLog(rt);
    error = what + ": " + litertStatusText(rt, st);
    const std::string why = lastLines(s->creationLog);
    if (!why.empty())
      error += " (" + why + ")";
    return nullptr;
  };

  // ---- options -------------------------------------------------------------
  LiteRtStatus st = rt.api.LiteRtCreateOptions(&s->options_);
  if (st != kLiteRtStatusOk)
    return fail("LiteRT could not create options", st);

  LiteRtHwAcceleratorSet accel = kLiteRtHwAcceleratorCpu;
  switch (dev.accel)
  {
  case LitertAccel::Cpu: accel = kLiteRtHwAcceleratorCpu; break;
  case LitertAccel::Gpu: accel = kLiteRtHwAcceleratorGpu; break;
  case LitertAccel::Npu: accel = kLiteRtHwAcceleratorNpu; break;
  }
  st = rt.api.LiteRtSetOptionsHardwareAccelerators(s->options_, accel);
  if (st != kLiteRtStatusOk)
    return fail("LiteRT rejected the accelerator selection", st);

  std::string optErr;
  if (dev.accel == LitertAccel::Cpu)
  {
    int threads = cfg.cpuThreads;
    if (threads <= 0)
      threads = (int)std::thread::hardware_concurrency();
    if (threads <= 0)
      threads = 1;
    if (!addOpaque(rt, s->options_, "xnnpack", "num_threads = " + std::to_string(threads) + "\n", optErr))
      return fail(optErr, kLiteRtStatusErrorInvalidArgument);
  }
  if (dev.accel == LitertAccel::Gpu)
  {
    std::string toml;
    if (cfg.gpuPrecision != kLiteRtDelegatePrecisionDefault)
      toml += "precision = " + std::to_string((int)cfg.gpuPrecision) + "\n";
    if (cfg.gpuAllowQuantized)
      toml += "allow_src_quantized_fc_conv_ops = true\n";
    if (!toml.empty() && !addOpaque(rt, s->options_, "gpu_options", toml, optErr))
      return fail(optErr, kLiteRtStatusErrorInvalidArgument);
  }
  if (dev.accel == LitertAccel::Npu)
  {
    // Peak clocks, the way the ONNX backend asks QNN for "burst": every
    // vendor's default performance mode is a power policy for an app, and
    // the difference on sustained work is a large multiple.  The payload is
    // the vendor's own TOML (the identifiers and keys of the
    // litert/c/options/*.cc parsers; the values are the enums in the
    // vendored headers), and only the vendor whose dispatch library came up
    // gets one -- an identifier no plugin consumes would be ignored, but
    // there is no reason to send it.
    const char *identifier = nullptr;
    std::string toml;
    if (dev.vendor == "Qualcomm")
    {
      identifier = "qualcomm";
      toml = "htp_performance_mode = " +
             std::to_string((int)kLiteRtQualcommHtpPerformanceModeBurst) + "\n" +
             "dsp_performance_mode = " +
             std::to_string((int)kLiteRtQualcommDspPerformanceModeBurst) + "\n";
    }
    else if (dev.vendor == "Google")
    {
      identifier = "google_tensor";
      toml = "performance_mode = " +
             std::to_string((int)kLiteRtGoogleTensorOptionsPerformanceModeBurst) + "\n";
    }
    else if (dev.vendor == "MediaTek")
    {
      identifier = "mediatek";
      toml = "performance_mode = " +
             std::to_string((int)kLiteRtMediatekNeuronAdapterPerformanceModeNeuronPreferTurboBoost) +
             "\n";
    }
    if (identifier && !addOpaque(rt, s->options_, identifier, toml, optErr))
      return fail(optErr, kLiteRtStatusErrorInvalidArgument);
  }
  if (cfg.profile &&
      !addOpaque(rt, s->options_, "runtime_options_string", "enable_profiling = true\n", optErr))
    return fail(optErr, kLiteRtStatusErrorInvalidArgument);

  // ---- model + compilation -------------------------------------------------
  // Muted and captured: a kernel that refuses a type says so on stderr
  // through TFLite's error reporter, and that sentence is the row's reason.
  const auto t0 = Clock::now();
  LiteRtStatus loadSt, compileSt = kLiteRtStatusOk;
  {
    clpeak::ScopedConsoleMute mute(clpeak::ScopedConsoleMute::Capture::Always);
    loadSt = rt.api.LiteRtCreateModelFromBuffer(s->env_, s->bytes_.data(), s->bytes_.size(), &s->model_);
    if (loadSt == kLiteRtStatusOk && s->model_)
      compileSt = rt.api.LiteRtCreateCompiledModel(s->env_, s->model_, s->options_, &s->compiled_);
    mute.finish();
    console = mute.text();
  }
  if (loadSt != kLiteRtStatusOk || !s->model_)
    return fail("LiteRT could not load the model", loadSt);
  if (compileSt != kLiteRtStatusOk || !s->compiled_)
    return fail(std::string("LiteRT could not compile the model for the ") + s->accelName_, compileSt);
  s->createUs = elapsedUs(t0);
  s->creationLog = console + litertDrainLog(rt);
  if (!console.empty() && clpeak::verboseEnabled())
    CLPEAK_VLOG("litert: console during creation:\n%s", console.c_str());
  // Where a session's time went, for the run log: the host generating the
  // model against the runtime loading and compiling it.  On a phone both
  // can dwarf the measurement that follows.
  CLPEAK_VLOG("litert: %s: %.1f MB built in %.0f ms, loaded and compiled for the %s in %.0f ms\n",
              s->bytes_.description.c_str(), (double)s->bytes_.size() / 1048576.0, s->bytes_.buildUs / 1000.0,
              s->accelName_.c_str(), s->createUs / 1000.0);

  if (dev.accel != LitertAccel::Cpu)
  {
    bool full = false;
    st = rt.api.LiteRtCompiledModelIsFullyAccelerated(s->compiled_, &full);
    s->fullyAccelerated_ = (st == kLiteRtStatusOk) && full;
  }

  // ---- buffers -------------------------------------------------------------
  // Sizes and types come from the model LiteRT parsed; the requirements from
  // the compiled model say which memory the accelerator wants them in, and
  // a managed buffer of that kind is what a real application would use.
  LiteRtParamIndex mainIdx = 0;
  LiteRtSubgraph sg = nullptr;
  if (rt.api.LiteRtGetMainModelSubgraphIndex(s->model_, &mainIdx) != kLiteRtStatusOk ||
      rt.api.LiteRtGetModelSubgraph(s->model_, mainIdx, &sg) != kLiteRtStatusOk || !sg)
    return fail("LiteRT could not find the model's main subgraph", kLiteRtStatusErrorNotFound);

  LiteRtParamIndex nIn = 0, nOut = 0;
  rt.api.LiteRtGetNumSubgraphInputs(sg, &nIn);
  rt.api.LiteRtGetNumSubgraphOutputs(sg, &nOut);

  for (LiteRtParamIndex i = 0; i < nIn + nOut; i++)
  {
    const bool isIn = i < nIn;
    const LiteRtParamIndex idx = isIn ? i : i - nIn;
    LiteRtTensor tensor = nullptr;
    st = isIn ? rt.api.LiteRtGetSubgraphInput(sg, idx, &tensor)
              : rt.api.LiteRtGetSubgraphOutput(sg, idx, &tensor);
    if (st != kLiteRtStatusOk || !tensor)
      return fail("LiteRT could not read a model tensor", st);
    LiteRtRankedTensorType type;
    std::memset(&type, 0, sizeof type);
    st = rt.api.LiteRtGetRankedTensorType(tensor, &type);
    if (st != kLiteRtStatusOk)
      return fail("LiteRT could not read a tensor's type", st);

    LiteRtTensorBufferRequirements req = nullptr;
    st = isIn ? rt.api.LiteRtGetCompiledModelInputBufferRequirements(s->compiled_, 0, idx, &req)
              : rt.api.LiteRtGetCompiledModelOutputBufferRequirements(s->compiled_, 0, idx, &req);
    if (st != kLiteRtStatusOk || !req)
      return fail("LiteRT could not report a buffer requirement", st);

    LiteRtTensorBuffer buf = nullptr;
    st = rt.api.LiteRtCreateManagedTensorBufferFromRequirements(s->env_, &type, req, &buf);
    if (st != kLiteRtStatusOk || !buf)
      return fail("LiteRT could not allocate a tensor buffer", st);
    size_t packed = 0;
    rt.api.LiteRtGetTensorBufferPackedSize(buf, &packed);
    if (isIn)
    {
      s->inputs_.push_back(buf);
      s->inputBytes_.push_back(packed);
    }
    else
    {
      s->outputs_.push_back(buf);
      s->outputBytes_.push_back(packed);
    }
  }
  return s;
}

LitertSession::~LitertSession()
{
  if (!rt_)
    return;
  for (auto b : inputs_)
    rt_->api.LiteRtDestroyTensorBuffer(b);
  for (auto b : outputs_)
    rt_->api.LiteRtDestroyTensorBuffer(b);
  if (compiled_)
    rt_->api.LiteRtDestroyCompiledModel(compiled_);
  if (model_)
    rt_->api.LiteRtDestroyModel(model_);
  if (options_)
    rt_->api.LiteRtDestroyOptions(options_);
  if (holdsEnv_)
    releaseEnvironment(accel_);
  litertDrainLog(*rt_);
}

std::string LitertSession::offDevice() const
{
  if (fullyAccelerated_)
    return std::string();
  return "LiteRT handed part of this graph back to the CPU: the " + accelName_ +
         " accelerator did not take every operation, so this would not be a " + accelName_ + " number";
}

bool LitertSession::writeInput(size_t i, const void *data, size_t bytes, std::string &error)
{
  if (i >= inputs_.size())
  {
    error = "no input " + std::to_string(i);
    return false;
  }
  if (bytes != inputBytes_[i])
  {
    error = "input " + std::to_string(i) + " is " + std::to_string(inputBytes_[i]) +
            " bytes, not " + std::to_string(bytes);
    return false;
  }
  void *p = nullptr;
  LiteRtStatus st = rt_->api.LiteRtLockTensorBuffer(inputs_[i], &p, kLiteRtTensorBufferLockModeWrite);
  if (st != kLiteRtStatusOk || !p)
  {
    error = "LiteRT could not map an input buffer: " + litertStatusText(*rt_, st);
    return false;
  }
  std::memcpy(p, data, bytes);
  st = rt_->api.LiteRtUnlockTensorBuffer(inputs_[i]);
  if (st != kLiteRtStatusOk)
  {
    error = "LiteRT could not unmap an input buffer: " + litertStatusText(*rt_, st);
    return false;
  }
  return true;
}

bool LitertSession::run(std::string &error)
{
  // The CPU path prepares its kernels on the first inference rather than at
  // compile time, and a kernel that refuses a type says so on stderr then
  // ("input->type != kTfLiteFloat32 (BFLOAT16 != FLOAT32)").  The first run
  // is therefore muted and captured like creation is; every later one runs
  // bare, since a mute is two dup2 calls that a timed loop must not pay.
  std::string console;
  LiteRtStatus st;
  if (!ran_)
  {
    ran_ = true;
    const auto t0 = Clock::now();
    clpeak::ScopedConsoleMute mute(clpeak::ScopedConsoleMute::Capture::Always);
    st = rt_->api.LiteRtRunCompiledModel(compiled_, 0, inputs_.size(), inputs_.data(),
                                         outputs_.size(), outputs_.data());
    mute.finish();
    console = mute.text();
    if (st == kLiteRtStatusOk)
    {
      std::string ignored;
      sync(ignored);   // the first run's time is to completion, like the rest
    }
    firstRunUs = elapsedUs(t0);
    if (!console.empty() && clpeak::verboseEnabled())
      CLPEAK_VLOG("litert: console during the first inference:\n%s", console.c_str());
    CLPEAK_VLOG("litert: %s: first inference %.0f ms\n", bytes_.description.c_str(), firstRunUs / 1000.0);
  }
  else
    st = rt_->api.LiteRtRunCompiledModel(compiled_, 0, inputs_.size(), inputs_.data(),
                                         outputs_.size(), outputs_.data());
  if (st != kLiteRtStatusOk)
  {
    const std::string logged = console + litertDrainLog(*rt_);
    error = "inference failed: " + litertStatusText(*rt_, st);
    const std::string why = lastLines(logged);
    if (!why.empty())
      error += " (" + why + ")";
    return false;
  }
  return true;
}

bool LitertSession::sync(std::string &error)
{
  if (outputs_.empty())
    return true;
  void *p = nullptr;
  LiteRtStatus st = rt_->api.LiteRtLockTensorBuffer(outputs_[0], &p, kLiteRtTensorBufferLockModeRead);
  if (st != kLiteRtStatusOk)
  {
    error = "LiteRT could not wait for an inference: " + litertStatusText(*rt_, st);
    return false;
  }
  st = rt_->api.LiteRtUnlockTensorBuffer(outputs_[0]);
  if (st != kLiteRtStatusOk)
  {
    error = "LiteRT could not unmap an output buffer: " + litertStatusText(*rt_, st);
    return false;
  }
  return true;
}

double LitertSession::timeRuns(unsigned n, std::string &error, bool syncEach)
{
  if (n == 0)
    return 0.0;
  const auto t0 = Clock::now();
  for (unsigned i = 0; i < n; i++)
  {
    if (!run(error))
      return -1.0;
    if (syncEach && !sync(error))
      return -1.0;
  }
  if (!syncEach && !sync(error))
    return -1.0;
  return elapsedUs(t0) / n;
}

bool LitertSession::outputBytes(size_t i, std::vector<uint8_t> &out, std::string &error)
{
  if (i >= outputs_.size())
  {
    error = "no output " + std::to_string(i);
    return false;
  }
  void *p = nullptr;
  LiteRtStatus st = rt_->api.LiteRtLockTensorBuffer(outputs_[i], &p, kLiteRtTensorBufferLockModeRead);
  if (st != kLiteRtStatusOk || !p)
  {
    error = "LiteRT could not map an output buffer: " + litertStatusText(*rt_, st);
    return false;
  }
  out.assign(static_cast<const uint8_t *>(p), static_cast<const uint8_t *>(p) + outputBytes_[i]);
  st = rt_->api.LiteRtUnlockTensorBuffer(outputs_[i]);
  if (st != kLiteRtStatusOk)
  {
    error = "LiteRT could not unmap an output buffer: " + litertStatusText(*rt_, st);
    return false;
  }
  return true;
}

std::vector<std::string> LitertSession::profileOps(std::string &error)
{
  std::vector<std::string> out;
  if (!profile_ || !rt_->api.LiteRtCompiledModelGetProfiler || !rt_->api.LiteRtStartProfiler ||
      !rt_->api.LiteRtStopProfiler || !rt_->api.LiteRtGetNumProfilerEvents ||
      !rt_->api.LiteRtGetProfilerEvents)
  {
    error = "profiling is not available";
    return out;
  }
  LiteRtProfiler prof = nullptr;
  if (rt_->api.LiteRtCompiledModelGetProfiler(compiled_, &prof) != kLiteRtStatusOk || !prof)
  {
    error = "LiteRT exposes no profiler for this model";
    return out;
  }
  if (rt_->api.LiteRtResetProfiler)
    rt_->api.LiteRtResetProfiler(prof);
  rt_->api.LiteRtStartProfiler(prof);
  const bool ok = run(error);
  rt_->api.LiteRtStopProfiler(prof);
  if (!ok)
    return out;
  int n = 0;
  if (rt_->api.LiteRtGetNumProfilerEvents(prof, &n) != kLiteRtStatusOk || n <= 0)
    return out;
  std::vector<ProfiledEventData> events((size_t)n);
  if (rt_->api.LiteRtGetProfilerEvents(prof, n, events.data()) != kLiteRtStatusOk)
    return out;
  for (const auto &e : events)
  {
    if (!(e.event_type & (OPERATOR_INVOKE_EVENT | DELEGATE_OPERATOR_INVOKE_EVENT |
                          DELEGATE_PROFILED_OPERATOR_INVOKE_EVENT)))
      continue;
    std::string line = e.tag ? e.tag : "?";
    switch (e.event_source)
    {
    case TFLITE_INTERPRETER: line += " [interpreter]"; break;
    case TFLITE_DELEGATE: line += " [delegate]"; break;
    case LITERT: line += " [litert]"; break;
    }
    out.push_back(line);
  }
  return out;
}

#endif // ENABLE_LITERT
