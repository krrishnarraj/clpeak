#ifndef LITERT_PEAK_H
#define LITERT_PEAK_H

#ifdef ENABLE_LITERT

#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>

#include <map>
#include <string>
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

// Which hardware accelerator a device row drives.  LiteRT's model of the
// machine is a bitmask of three -- CPU (XNNPACK), GPU (its ML Drift
// accelerator over OpenCL, Metal or WebGPU) and NPU (a vendor dispatch
// library) -- and, as in the Core ML backend, each is presented as one
// device so the same micro-graphs run side by side on all of them.
enum class LitertFormat;   // src/litert/litert_model.h

enum class LitertAccel { Cpu, Gpu, Npu };
const char *litertAccelName(LitertAccel a);   // "CPU" / "GPU" / "NPU"

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
  // its speed is not a number anyone should divide by.
  struct AnswerCheck
  {
    double ppm = -1.0;
    ResultStatus status = ResultStatus::Ok;
    std::string error;
  };
  const AnswerCheck &answerCheck(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f);
  // Empty when the format's answer is right or could not be checked;
  // otherwise the reason a rate row is refused with.
  std::string wrongAnswer(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f);

private:
  std::map<std::pair<int, int>, AnswerCheck> answerChecks_;   // (accelerator, format)
};

// A relative RMS error past this is a wrong answer, not a loss of precision:
// int8 with the result itself quantized costs ~1%, nothing legitimate here
// reaches 10%.
constexpr double kLitertWrongAnswerPpm = 100000.0;

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
};
LitertRuntimeStatus litertRuntimeStatus();

#endif // ENABLE_LITERT
#endif // LITERT_PEAK_H
