#ifndef LITERT_PEAK_H
#define LITERT_PEAK_H

#ifdef ENABLE_LITERT

#include <common/answer_check.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>

#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct CliOptions;
struct LitertRuntime;

// Ceiling on the iteration count of a timed batch, for every throughput test
// in this backend; same reasoning and value as the ONNX and Core ML backends
// (include/onnx/onnx_peak.h).
constexpr unsigned int kLitertMaxIters = 500;

// Ceilings on model creation.  On the CPU and the GPU a compiled model is
// milliseconds of packing and shader compilation; on an NPU it is a vendor
// compiler run, and the doubling ladders need the same cliff detection the
// ONNX backend has for QNN and TensorRT.
constexpr double kLitertMaxCreateUs = 30.0e6;
constexpr double kLitertMaxBlockCreateUs = 60.0e6;
constexpr double kLitertCreateGrowthFactor = 6.0;
constexpr double kLitertCreateGrowthFloor = 2.0e6;

// The gemm ladder's compile cap and the next doubling's predicted compile:
// the ONNX backend's (kOnnxMaxCreateUs, onnxPredictCreateUs in
// include/onnx/onnx_peak.h, which has the evidence from QNN's HTP and
// TensorRT), because the ladder is its sixteen-layer chain and an NPU
// compiles it as slowly.  The growth already seen is carried forward, 4x to
// 16x, and squared on the size after one that failed to gain.
constexpr double kLitertMaxChainCreateUs = 240.0e6;
inline double litertPredictCreateUs(double prevUs, double prevPrevUs, bool confirming)
{
  double growth = 4.0;
  if (prevPrevUs > 0.0 && prevUs > kLitertCreateGrowthFloor)
  {
    growth = prevUs / prevPrevUs;
    if (confirming)
      growth *= growth;
    growth = growth < 4.0 ? 4.0 : (growth > 16.0 ? 16.0 : growth);
  }
  return prevUs * growth;
}

// Which hardware accelerator a device row drives.  LiteRT's model of the
// machine is a bitmask of three -- CPU (XNNPACK), GPU (its ML Drift
// accelerator over OpenCL, Metal or WebGPU) and NPU (a vendor dispatch
// library) -- and, as in the Core ML backend, each is presented as one
// device so the same micro-graphs run side by side on all of them.
enum class LitertFormat;   // src/litert/litert_model.h

enum class LitertAccel { Cpu, Gpu, Npu };
const char *litertAccelName(LitertAccel a);   // "CPU" / "GPU" / "NPU"

// How a raced format's graph is written and run (gemm.cpp, conv.cpp,
// block.cpp, each with its numbers): the operator its layers are, whether a
// GPU may use its 8-bit kernels, and whether an integer graph's inputs and
// outputs are float.  The first two are raced on rate; the last is settled
// by the answer (LitertPeak::resolveIo), since it moves no rate.
struct LitertForm
{
  // The layers as 1x1 CONV_2Ds over a grid of positions, not FULLY_CONNECTED.
  bool conv1x1 = false;
  // GPU: allow_src_quantized_fc_conv_ops -- the accelerator's 8-bit FC and
  // convolution kernels, which it otherwise disallows -- with the
  // enable_constant_tensors_sharing its documentation requires where the
  // graph's constants allow it (LitertPlan::gpuShareConstants).
  bool gpuInt8Kernels = false;
  // An integer format's graph takes and returns float32, quantized inside it
  // (QUANTIZE in, DEQUANTIZE out), the converter's default form.
  bool floatIo = false;
};

struct litert_device_info_t
{
  LitertAccel accel = LitertAccel::Cpu;
  std::string displayName;   // "Qualcomm Hexagon NPU via LiteRT", "GPU via LiteRT (Apple M1 Pro)"
  std::string typeStr;       // "NPU" / "GPU" / "CPU"
  DeviceType deviceType = DeviceType::Unknown;
  std::string vendor;        // NPU: which dispatch library ("Qualcomm", "MediaTek", ...); GPU: the backend name
  std::string detail;        // GPU device name / NPU SoC, when known
};

class LitertPeak : public Peak
{
public:
  LitertPeak();
  ~LitertPeak() override;

  static constexpr Backend kBackend = Backend::Litert;
  Backend backend() const override { return kBackend; }
  int runAll() override;

  static BackendInventory enumerate();

  // Per-benchmark entry points (one .cpp each, like the other backends).
  int runGemm(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);
  int runConv(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);
  int runNumericError(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);
  int runBlock(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);
  int runActivation(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);
  int runTensorBandwidth(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);
  int runDispatchLatency(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg);

  logger::DeviceScope *currentDeviceScope = nullptr;

  // litert_numeric_error's measurement for one format on one device,
  // memoised: the relative RMS error of the accelerator's 1024-cubed matmul
  // against the host's double-precision reference, in ppm, or the status
  // and reason when it could not be measured.  The rate tests ask before
  // publishing a format's rate -- a kernel that returns a wrong answer
  // (Mali's int8 path on a Pixel 7a was 250% off) is not a capability, and
  // its speed is not a number anyone should divide by.  The line and the
  // words are the three ML backends' (include/common/answer_check.h).
  struct AnswerCheck : clpeak::AnswerCheck
  {
    // The output is filled with a marker before the run.  The share of its
    // elements still holding it once the run returned, and -- read only for
    // a wrong answer -- after a second run, with that run's figure: a marker
    // that survives both is an answer never written, one the second run
    // replaces is an answer that lands after the run returns.
    double unwritten = -1.0, unwrittenRerun = -1.0, rerunPpm = -1.0;
  };
  // The format's answer in one form (LitertForm): `conv1x1` is the same
  // product written as a 1x1 CONV_2D, `gpuInt8Kernels` with the GPU's 8-bit
  // kernels allowed, `floatIo` an integer graph with float inputs and outputs.
  const AnswerCheck &answerCheck(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f,
                                 const LitertForm &form = LitertForm());
  // Empty when the format's answer in that form is right or could not be
  // checked; otherwise the reason a rate row is refused with.
  std::string wrongAnswer(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f,
                          const LitertForm &form = LitertForm());
  // The inputs and outputs an integer format's graph takes in `form` on this
  // device: its own int8 or int16 ones while its answer is right with them,
  // float ones quantized inside the graph when only those answer right --
  // LiteRT 2.2.0's OpenCL accelerator never converts an integer graph's
  // inputs and outputs (src/litert/AGENTS.md has the source).  Any other
  // format's are float already, and `form` comes back as it went in.
  LitertForm resolveIo(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f,
                       LitertForm form);
  // What a rate row says of a form resolveIo gave float inputs and outputs:
  // "the graph's input and output float, since with int8 ones this
  // accelerator never wrote its answer".  Empty for any other form.
  std::string floatIoClause(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f,
                            const LitertForm &form);

private:
  // (accelerator, format, conv1x1, gpuInt8Kernels, floatIo)
  std::map<std::tuple<int, int, bool, bool, bool>, AnswerCheck> answerChecks_;
};

// The accelerators this machine can actually run, accelerators first and the
// CPU last: each has compiled and run one tiny model with nothing handed
// back to the CPU.  Shared by enumerate() and runAll() so listing and runs
// agree.  `skipped` receives the accelerators that were tried and declined,
// with the reason, for the verbose-only notes.
std::vector<litert_device_info_t> litertUsableDevices(
    const LitertRuntime &rt,
    std::vector<std::pair<litert_device_info_t, std::string>> *skipped = nullptr);

// Choose which LiteRT library to load, ahead of the platform's conventional
// names; empty clears the choice.  Backs `--litert-lib` and the FFI's
// clpeak_set_litert_library().  Re-declared here so the CLI and the FFI can
// set it without the backend's private loader header.  Between runs only.
void litertSetLibraryOverride(const std::string &path);

// Where the NPU dispatch / compiler-plugin libraries live; backs
// `--litert-npu-dir`.  Empty means beside the runtime library.
void litertSetNpuDirOverride(const std::string &dir);

// A writable directory the backend may stage one vendor's NPU shims in
// (Android; the FFI's clpeak_set_litert_npu_stage_dir).  LiteRT loads the
// first libLiteRtDispatch_* it lists in one directory, so an app that
// carries every vendor's shims gives the one for this SoC a directory of
// its own here -- links to the packaged files, remade at every launch.
// Empty (the default) leaves the runtime's own directory as it is.
void litertSetNpuStageDir(const std::string &dir);

// Why the runtime failed to load, ready to show a user; empty when it loaded.
std::string litertLoadDiagnostic();

// What a settings screen needs to say about the runtime in one place.
struct LitertRuntimeStatus
{
  bool available = false;
  std::string version;   // the ABI version this build speaks; LiteRT has no runtime version string
  std::string path;      // what was loaded
  std::string error;     // populated only when !available
  // A library chosen after the runtime loaded, which loads at the next start
  // (src/litert/litert_runtime.h): `pendingPath` empty for the default search.
  bool pending = false;
  std::string pendingPath;
};
LitertRuntimeStatus litertRuntimeStatus();

#endif // ENABLE_LITERT
#endif // LITERT_PEAK_H
