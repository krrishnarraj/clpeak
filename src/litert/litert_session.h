#ifndef CLPEAK_LITERT_SESSION_H
#define CLPEAK_LITERT_SESSION_H

// One compiled LiteRT model on one accelerator, with its input and output
// buffers, plus what the runtime says about where its operations ran.  Pure
// C++ surface over the dlopen'd C API; every test is a .cpp against this.

#include <litert/litert_peak.h>
#include "litert_runtime.h"
#include "tflite_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Per-session knobs.  The accelerator comes from the device; these are the
// policies a test varies on top of it.
struct LitertSessionConfig
{
  // GPU: the precision the accelerator computes in.  Default lets ML Drift
  // choose (fp16 on mobile GPUs); Fp32 forces full precision; Fp16 forces
  // half storage and arithmetic; Fp16WithFp32Accum keeps fp32 accumulation
  // in the matmul-class operators.  Ignored elsewhere.
  LiteRtDelegatePrecision gpuPrecision = kLiteRtDelegatePrecisionDefault;
  // GPU: let quantized FULLY_CONNECTED / CONV_2D run on the GPU as such
  // rather than being handed back (allow_src_quantized_fc_conv_ops).
  bool gpuAllowQuantized = false;
  // CPU: XNNPACK threads; 0 = every hardware thread.
  int cpuThreads = 0;
  // Record per-operation events so profileOps() can name what ran where.
  // Costs a little per run; used for placement checks, never for a timing.
  bool profile = false;
};

class LitertSession
{
public:
  // Loads the model bytes (kept alive by the session, LiteRT reads them in
  // place), compiles it for the device's accelerator, allocates its buffers.
  // Null with `error` set when any step fails: the runtime's own message,
  // with whatever LiteRT logged while failing, is the answer the row reports.
  static std::unique_ptr<LitertSession> create(const LitertRuntime &rt,
                                               const litert_device_info_t &dev,
                                               clpeak_tflite::TfliteBytes &&model,
                                               const LitertSessionConfig &cfg,
                                               std::string &error);
  ~LitertSession();

  // Creation cost in microseconds: model load plus compilation (on an NPU,
  // the vendor compiler; on the GPU, shader compilation and weight upload).
  double createUs = 0.0;

  // Did every operation land on the device this session was created for?
  // Always true on the CPU.  For the GPU and NPU this is LiteRT's own
  // answer (LiteRtCompiledModelIsFullyAccelerated): the runtime keeps the
  // CPU as a fallback for anything an accelerator declines, and a matmul
  // that fell back would otherwise be timed on the CPU under the
  // accelerator's name.  `offDevice()` names what fell back when the
  // profiler can say, and explains the refusal otherwise.
  bool onDevice() const { return fullyAccelerated_; }
  std::string offDevice() const;

  size_t numInputs() const { return inputs_.size(); }
  size_t numOutputs() const { return outputs_.size(); }

  // Copy `bytes` into input `i`; fails when the size disagrees with the
  // buffer LiteRT allocated.
  bool writeInput(size_t i, const void *data, size_t bytes, std::string &error);

  // One inference.
  bool run(std::string &error);

  // Mean microseconds per inference over `n` of them; negative with `error`
  // set on failure.
  double timeRuns(unsigned n, std::string &error);

  // Bytes of output `i` after the last run().
  bool outputBytes(size_t i, std::vector<uint8_t> &out, std::string &error);

  // One profiled run: the operator events LiteRT recorded, each as
  // "TAG [source]" where the source says whether the interpreter (CPU
  // reference kernels), a delegate (XNNPACK, the GPU accelerator) or the
  // LiteRT runtime itself executed it.  Empty when profiling is off.
  std::vector<std::string> profileOps(std::string &error);

  // What LiteRT logged while this session was created (already relayed to
  // the run log); the last few lines are what a refusal reason quotes.
  std::string creationLog;

private:
  LitertSession() = default;
  const LitertRuntime *rt_ = nullptr;
  LiteRtEnvironment env_ = nullptr;      // shared per accelerator, not owned
  LiteRtOptions options_ = nullptr;
  LiteRtModel model_ = nullptr;
  LiteRtCompiledModel compiled_ = nullptr;
  clpeak_tflite::TfliteBytes bytes_;
  std::vector<LiteRtTensorBuffer> inputs_, outputs_;
  std::vector<size_t> inputBytes_, outputBytes_;
  bool fullyAccelerated_ = true;
  bool profile_ = false;
  bool holdsEnv_ = false;
  bool ran_ = false;
  LitertAccel accel_ = LitertAccel::Cpu;
  std::string accelName_;
};

// Drain LiteRT's sink logger into the run log (debug under --verbose, errors
// and warnings always) and return the drained text.  Called after every
// creation and failure so a refusal carries the runtime's own words.
std::string litertDrainLog(const LitertRuntime &rt);

// What LiteRT logged while creating an accelerator's environment -- which
// GPU accelerator library it loaded, what an NPU dispatch library said.
std::string litertEnvironmentLog(LitertAccel accel);

// Tear down an accelerator's environment; the next session rebuilds it.
// No session may be alive on it.
void litertResetEnvironment(const LitertRuntime &rt, LitertAccel accel);

#endif // CLPEAK_LITERT_SESSION_H
