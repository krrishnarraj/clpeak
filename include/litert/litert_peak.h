#ifndef LITERT_PEAK_H
#define LITERT_PEAK_H

#ifdef ENABLE_LITERT

#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>

#include <string>
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
