#ifndef ONNX_PEAK_H
#define ONNX_PEAK_H

#ifdef ENABLE_ONNX

#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>

#include <string>
#include <vector>

struct CliOptions;
struct OrtRuntime;

// Ceiling on the iteration count of a timed batch, for every throughput test
// in this backend.  It is not a time budget -- each test keeps its own of
// those -- it is the point past which more samples stop buying anything.
//
// Without it the budget alone sizes the batch, and at the small end of a
// sweep that runs away: a 1x1 convolution over a 32x32 feature map takes
// 166 us on an M1 Pro's Neural Engine, so a 2 s budget asks for twelve
// thousand repetitions of it.  The mean stopped moving after a few hundred,
// and the extra eleven thousand are the difference between measuring the
// device and waiting for it.  The effect grows with the hardware: the faster
// the provider, the more of the ladder lands in this regime, so an RTX 5060
// under TensorRT spends more of its run here than a phone does.
//
// 500 leaves the search's own tolerance (3%) far above the sampling noise at
// every rung measured, and it binds only where a rung is already fast enough
// for the count to be meaningless.  Slow rungs never reach it and are
// governed by the time budget exactly as before.
//
// The dispatch-latency test is the deliberate exception: there the
// per-submission overhead *is* the measurement, one submission is
// microseconds, and it passes its own far larger cap.
constexpr unsigned int kOnnxMaxIters = 500;

// Ceiling on session creation (graph compilation) time.  Separate from
// kMaxIterUs which gates per-iteration execution time: on QNN HTP the
// compilation dominates (1024^3: 33s, 2048^3: 313s, ~9x per 2x dim) and the
// execution gate never fires because per-iter stays in ms.  Two guards:
//
//  * absolute: one create > kOnnxMaxCreateUs -> stop ladder after this rung
//    (keep its result, skip larger).  30s for the doubling ladders (gemm,
//    conv), 60s for the fixed-geometry block whose 8192 context legitimately
//    needs ~1 min on CoreML/TensorRT AOT toolchains.
//
//  * factor: create grew > kOnnxCreateGrowthFactor since previous rung
//    (memory 4x, flops 8x per 2x dim; 6x tolerates jitter but catches QNN's
//    9.3x).  Applies only to ladders where D doubles.
//
// The first rung (kMinDim) is allowed to exceed the absolute once - its time
// is the seed for the factor gate; truncating it would discard a valid peak.
constexpr double kOnnxMaxCreateUs = 30.0e6;
constexpr double kOnnxMaxBlockCreateUs = 60.0e6;
constexpr double kOnnxCreateGrowthFactor = 6.0;

// One benchmarkable "device" of this backend: an ONNX Runtime execution
// provider (EP).  NPUs are reachable only through such vendor runtimes --
// the EP is the closest thing to an ISA they expose -- so each EP the
// loaded runtime offers is enumerated as a device, including CPU/GPU EPs:
// running the same micro-graphs on them makes NPU-vs-GPU-vs-CPU numbers
// comparable on one machine.
struct onnx_ep_info_t
{
  std::string providerKey;   // ORT registration name, e.g. "CoreMLExecutionProvider"
  std::string displayName;   // e.g. "CoreML (Apple Neural Engine)"
  std::string typeStr;       // "NPU" / "GPU" / "CPU"
  DeviceType  deviceType = DeviceType::Unknown;
};

class OnnxPeak : public Peak
{
public:
  std::vector<int> deviceIndices;  // empty = run all enumerated EPs

  OnnxPeak();
  ~OnnxPeak() override;

  void applyOptions(const CliOptions &opts) override;
  int  runAll() override;

  static BackendInventory enumerate();
  static void printInventory(const BackendInventory &inv, std::ostream &os);

  // Per-benchmark entry points (one .cpp each, like the other backends).
  int runGemm(const OrtRuntime &rt, const onnx_ep_info_t &ep,
              benchmark_config_t &cfg);
  int runConv(const OrtRuntime &rt, const onnx_ep_info_t &ep,
              benchmark_config_t &cfg);
  int runNumericError(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                      benchmark_config_t &cfg);
  int runBlock(const OrtRuntime &rt, const onnx_ep_info_t &ep,
               benchmark_config_t &cfg);
  int runActivation(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                    benchmark_config_t &cfg);
  int runTensorBandwidth(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                         benchmark_config_t &cfg);
  int runTransferBandwidth(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                           benchmark_config_t &cfg);
  int runDispatchLatency(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                         benchmark_config_t &cfg);

  logger::DeviceScope *currentDeviceScope = nullptr;
};

// List the loaded runtime's execution providers in benchmark order
// (accelerators first, CPU last).  Shared by enumerate() and runAll().
std::vector<onnx_ep_info_t> onnxAvailableEps(const OrtRuntime &rt);

// Choose which onnxruntime library to load, ahead of the platform's
// conventional names; empty clears the choice.  Backs
// `--onnx-lib` and the FFI's clpeak_set_onnx_library().  Re-declared here so
// the CLI and the FFI can set it without reaching into the backend's private
// loader header (and its ONNX Runtime include).  Details, including the
// between-runs-only contract: src/onnx/onnx_runtime.h.
void onnxSetLibraryOverride(const std::string &path);

// Why the runtime failed to load, ready to show a user; empty when it loaded.
// Details: src/onnx/onnx_runtime.h.
std::string onnxLoadDiagnostic();

// What a settings screen needs to say about the runtime in one place: which
// one is loaded and from where, or why none is.  The device catalog carries
// the version and the provider list already, but not the reason a chosen
// library was refused -- and that reason is the whole feedback loop for
// picking one.
struct OnnxRuntimeStatus
{
  bool        available = false;
  bool        linkedIn  = false;  // built in rather than loaded (iOS)
  std::string version;            // "1.29.0"
  std::string path;               // what was loaded; empty = found by name
  std::string error;              // populated only when !available
};
OnnxRuntimeStatus onnxRuntimeStatus();

#endif // ENABLE_ONNX
#endif // ONNX_PEAK_H
