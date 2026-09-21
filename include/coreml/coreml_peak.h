#ifndef COREML_PEAK_H
#define COREML_PEAK_H

#ifdef ENABLE_COREML

#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>

#include <string>
#include <vector>

struct CliOptions;

// Ceiling on the iteration count of a timed batch, for every throughput test
// in this backend.  The time budget alone sizes a batch from the device's
// speed, and at the small end of a ladder that asks for thousands of
// repetitions of something that took microseconds; the mean stops moving
// after a few hundred.  Same reasoning, same value as the ONNX backend's
// kOnnxMaxIters (include/onnx/onnx_peak.h).
constexpr unsigned int kCoremlMaxIters = 500;

// Ceilings on model creation -- writing, compiling and loading a model, which
// on the Neural Engine is a compiler run and not a bookkeeping call.  30 s for
// the doubling ladders (gemm, conv), 60 s for the fixed-geometry block whose
// longest context legitimately needs the better part of a minute on the ANE
// compiler.  A create that grew more than kCoremlCreateGrowthFactor since the
// previous rung, once past kCoremlCreateGrowthFloor, is a compile cliff and
// ends a ladder before the next model is even built.
constexpr double kCoremlMaxCreateUs = 30.0e6;
constexpr double kCoremlMaxBlockCreateUs = 60.0e6;
constexpr double kCoremlCreateGrowthFactor = 6.0;
constexpr double kCoremlCreateGrowthFloor = 2.0e6;

// Which piece of silicon a Core ML compute device is.  Core ML has no
// exclusive mode -- every configuration keeps the CPU as a fallback -- so a
// "device" here is the strictest MLComputeUnits request that prefers it, and
// the compute plan (coreml_session.h) proves per operation that the request
// was honoured.
enum class CoremlDeviceKind { NeuralEngine, Gpu, Cpu };

// One benchmarkable device of this backend: an entry of MLAllComputeDevices(),
// presented the way the ONNX backend presents an execution provider.  Running
// the same micro-graphs on the Neural Engine, the GPU and the CPU through one
// framework is what makes the three comparable on one machine -- and the ANE
// is reachable through no other public API at all.
struct coreml_device_info_t
{
  CoremlDeviceKind kind = CoremlDeviceKind::Cpu;
  std::string displayName;   // "Apple Neural Engine", "GPU via Core ML (Apple M1 Pro)"
  std::string typeStr;       // "NPU" / "GPU" / "CPU"
  DeviceType deviceType = DeviceType::Unknown;
  int coreCount = 0;         // ANE only
  int gpuIndex = -1;         // index into MTLCopyAllDevices order, GPU only
  std::string gpuName;       // Metal device name, GPU only
};

class CoreMLPeak : public Peak
{
public:
  CoreMLPeak();
  ~CoreMLPeak() override;

  // Which backend this is -- the one place that says so; the registry,
  // the inventory and the device selector all read it from here.
  static constexpr Backend kBackend = Backend::Coreml;
  Backend backend() const override { return kBackend; }
  int runAll() override;

  static BackendInventory enumerate();

  // Per-benchmark entry points (one .cpp each, like the other backends).
  int runGemm(const coreml_device_info_t &dev, benchmark_config_t &cfg);
  int runNumericError(const coreml_device_info_t &dev, benchmark_config_t &cfg);
  int runConv(const coreml_device_info_t &dev, benchmark_config_t &cfg);
  int runBlock(const coreml_device_info_t &dev, benchmark_config_t &cfg);
  int runActivation(const coreml_device_info_t &dev, benchmark_config_t &cfg);
  int runTensorBandwidth(const coreml_device_info_t &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(const coreml_device_info_t &dev, benchmark_config_t &cfg);
  int runDispatchLatency(const coreml_device_info_t &dev, benchmark_config_t &cfg);

  logger::DeviceScope *currentDeviceScope = nullptr;
};

// The compute devices Core ML reports on this machine, accelerators first.
// Empty, with `why` set, on an OS too old to enumerate them.  Shared by
// enumerate() and runAll() so listing and runs agree.
std::vector<coreml_device_info_t> coremlDevices(std::string *why = nullptr);

// Core ML specification version this OS accepts (8 on macOS 14 / iOS 17, 9
// on macOS 15 / iOS 18, 10 on macOS 26 / iOS 26), which is what decides
// which MIL operations -- and therefore which datatypes -- a model may use.
int coremlSpecVersion();

// "macOS 26.1" / "iOS 18.4" -- the runtime that compiled and ran the models.
std::string coremlOsVersionString();
// Its major version (26, 27, ...), for fences on an OS release's faults.
int coremlOsMajorVersion();

#endif // ENABLE_COREML
#endif // COREML_PEAK_H
