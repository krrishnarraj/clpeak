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
struct OrtEpDevice; // ONNX Runtime's (EP, hardware device) pair; opaque here

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
//    (keep its result, skip larger).  60s for the doubling ladders (gemm,
//    conv), 60s for the fixed-geometry block whose 8192 context legitimately
//    needs ~1 min on CoreML/TensorRT AOT toolchains.
//
//  * factor: create grew > kOnnxCreateGrowthFactor since previous rung, and
//    is itself past kOnnxCreateGrowthFloor (memory 4x, flops 8x per 2x dim;
//    6x tolerates jitter but catches QNN's 9.3x).  Applies only to ladders
//    where D doubles.
//
// The first rung (kMinDim) is allowed to exceed the absolute once - its time
// is the seed for the factor gate; truncating it would discard a valid peak.
constexpr double kOnnxMaxCreateUs = 60.0e6;
constexpr double kOnnxMaxBlockCreateUs = 60.0e6;
constexpr double kOnnxCreateGrowthFactor = 6.0;
// ...and only once creation is expensive enough for its growth to mean
// anything.  The factor exists to catch an ahead-of-time compiler's cliff
// before the next model is built, but a ratio between two trivial numbers is
// not a cliff: ONNX Runtime's CPU EP went from 0.1 s to 0.7 s simply
// serializing a larger model, tripped 7.8x, and lost the rung that gave the
// same provider on another OS 11% more.  Below this the absolute gate and
// the predicted-create gate are the ones that matter, and both still apply.
constexpr double kOnnxCreateGrowthFloor = 2.0e6;
// Tiny probe budget: 64^3 model should compile in <10s even on AOT.
// If it exceeds this, the dtype is emulated/slow and the full 1024
// ladder will be minutes - skip the variant early.
constexpr double kOnnxTinyMaxCreateUs = 10.0e6;
// How many sizes a doubling ladder climbs past a provider's runtime sending
// the work to another compute unit (onnx_session.h, offDevice) before
// concluding the unit will never take the shape.  Core ML's planner is the
// case: it keeps a 1024-cube matmul and a 32-square convolution on the CPU
// under the Neural Engine configuration and sends the next sizes up, and
// its resident-tensor ladder was refused twice (8 and 32 MB) before the
// 128 MB rung landed.  Four covers that with one to spare, and bounds what
// an fp32 graph -- which the Neural Engine never takes -- costs in compiles
// before its row says so.
constexpr int kOnnxOffDevicePatience = 4;

// One benchmarkable "device" of this backend: an ONNX Runtime execution
// provider (EP).  NPUs are reachable only through such vendor runtimes --
// the EP is the closest thing to an ISA they expose -- so each EP the
// loaded runtime offers is enumerated as a device, including CPU/GPU EPs:
// running the same micro-graphs on them makes NPU-vs-GPU-vs-CPU numbers
// comparable on one machine.
//
// OpenVINO is the exception: one EP fronts three different pieces of
// silicon (NPU, GPU, CPU) selected by its `device_type` option, so it
// enumerates as three devices sharing one providerKey and differing only
// in epDevice.  A target with no hardware behind it (an Arc dGPU box has
// no NPU) is filtered by the viability probe before it ever lists.
//
// Providers can also arrive as *plugin libraries* (ONNX Runtime 1.22+):
// a separately shipped shared library registered on the environment --
// Qualcomm's QNN EP since its 2.0, the ones Windows ML installs from the
// Store.  Those the runtime enumerates per hardware device (`GetEpDevices`),
// so each (provider, device) pair is one entry here, with `epDevicePtr`
// naming the OrtEpDevice a session is appended for, `library` the
// registration it came from and `vendor` / `hardware` what the runtime
// says about the silicon.  The pointer belongs to the runtime's
// environment and is valid until that environment is recreated -- a
// runtime switch or a change of plugin libraries between runs -- which is
// also when enumeration runs again.
struct onnx_ep_info_t
{
  std::string providerKey; // ORT registration name, e.g. "CoreMLExecutionProvider"
  std::string displayName; // e.g. "CoreML (Apple Neural Engine)"
  std::string typeStr;     // "NPU" / "GPU" / "CPU"
  DeviceType deviceType = DeviceType::Unknown;
  std::string epDevice; // OpenVINO `device_type` ("NPU"/"GPU"/"CPU"), a
                        // plugin's device label ("NPU", "GPU#1"); empty otherwise

  // Plugin-provider devices only.
  const OrtEpDevice *epDevicePtr = nullptr;
  std::string library;                                       // registration name of the plugin library
  std::string vendor;                                        // the hardware vendor the runtime reports
  std::vector<std::pair<std::string, std::string>> hardware; // device metadata, as reported
};

// One plugin execution-provider library to register on the runtime's
// environment: `--onnx-ep NAME=PATH`, the FFI setter, or what the Windows ML
// catalog resolved.  `name` is the registration name the provider expects
// (Qualcomm's QNN insists on "QNNExecutionProvider"); `path` is what ORT
// loads, absolute or -- on Android, where a bare soname resolves out of the
// APK -- just the file name.  `named` says a person asked for it: a named
// library that will not register is reported in normal output, like a named
// runtime that will not load; an implicit one (a plugin the app bundles
// speculatively) fails into the verbose log only.
struct OnnxEpLibrary
{
  std::string name;
  std::string path;
  bool named = true;
};

// Replace the set of plugin libraries to register.  Between runs only, like
// onnxSetLibraryOverride: the environment they are registered on is rebuilt
// on the next use, and every session on it must be gone by then.
void onnxSetEpLibraries(std::vector<OnnxEpLibrary> libs);
const std::vector<OnnxEpLibrary> &onnxEpLibraries();

// Windows ML's execution-provider catalog (`--onnx-winml [PATH]`): the
// vendor providers Windows 11 installs from the Microsoft Store (Qualcomm
// QNN, Intel OpenVINO, AMD Vitis AI, NVIDIA TensorRT for RTX) resolved
// through Microsoft.Windows.AI.MachineLearning.dll and registered as
// plugin libraries.  `path` names that DLL or its directory, or is empty
// to search beside the loaded runtime and the executable.  Installing a
// provider is a download, so this is opt-in.  A no-op off Windows, where
// the status simply says so.
void onnxSetWinml(bool enabled, const std::string &path);

// How each requested plugin library fared on the environment: registered,
// or the runtime's reason.  Filled when the environment is created; empty
// until then.  Both the run's notes and a settings screen read it.
struct OnnxEpLibraryStatus
{
  OnnxEpLibrary lib;
  bool registered = false;
  std::string error; // when !registered
};
std::vector<OnnxEpLibraryStatus> onnxEpLibraryStatus();

class OnnxPeak : public Peak
{
public:
  OnnxPeak();
  ~OnnxPeak() override;

  // Which backend this is -- the one place that says so; the registry,
  // the inventory and the device selector all read it from here.
  static constexpr Backend kBackend = Backend::Onnx;
  Backend backend() const override { return kBackend; }
  int runAll() override;

  static BackendInventory enumerate();

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

// The providers from onnxAvailableEps that can actually run here: each
// has created at least one tiny session (see onnxEpViable).  A provider
// the build contains but the hardware cannot serve -- OpenVINO NPU with
// no NPU, NNAPI declining every graph -- is dropped instead of listing
// as a device that only ever reports Unsupported.  When `skipped` is
// given it also receives each dropped entry with its refusal reason, for
// the verbose-only notes in runAll() and --list-devices (a missing
// accelerator is the normal case, not a warning).
std::vector<onnx_ep_info_t> onnxUsableEps(
    const OrtRuntime &rt,
    std::vector<std::pair<onnx_ep_info_t, std::string>> *skipped = nullptr);

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
  bool available = false;
  bool linkedIn = false; // built in rather than loaded (iOS)
  std::string version;   // "1.29.0"
  std::string path;      // what was loaded; the resolved file even when found
                           // by name, empty only when statically linked
  std::string error;     // populated only when !available
  // The plugin libraries and how they registered (see onnxEpLibraryStatus);
  // the Windows ML catalog's own state when it is enabled.
  std::vector<OnnxEpLibraryStatus> epLibraries;
  bool winmlEnabled = false;
  std::string winmlPath;  // the catalog DLL that answered; empty when none
  std::string winmlError; // why the catalog gave nothing, when enabled;
                          // both empty while nothing has resolved it yet
};
OnnxRuntimeStatus onnxRuntimeStatus();

#endif // ENABLE_ONNX
#endif // ONNX_PEAK_H
