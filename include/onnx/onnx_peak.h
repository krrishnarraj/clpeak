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
              benchmark_config_t &cfg, Category category);
  int runConv(const OrtRuntime &rt, const onnx_ep_info_t &ep,
              benchmark_config_t &cfg);
  int runNumericError(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                      benchmark_config_t &cfg);
  int runBlock(const OrtRuntime &rt, const onnx_ep_info_t &ep,
               benchmark_config_t &cfg);
  int runTensorBandwidth(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                         benchmark_config_t &cfg);
  int runDispatchLatency(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                         benchmark_config_t &cfg);

  logger::DeviceScope *currentDeviceScope = nullptr;
};

// List the loaded runtime's execution providers in benchmark order
// (accelerators first, CPU last).  Shared by enumerate() and runAll().
std::vector<onnx_ep_info_t> onnxAvailableEps(const OrtRuntime &rt);

#endif // ENABLE_ONNX
#endif // ONNX_PEAK_H
