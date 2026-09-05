#ifndef CLPEAK_ONNX_PROBE_H
#define CLPEAK_ONNX_PROBE_H

#ifdef ENABLE_ONNX

#include <string>
#include <unordered_map>

struct OrtRuntime;
struct onnx_ep_info_t;

struct OnnxProbeResult
{
  bool ok = false;
  std::string reason; // when !ok
  double createUs = 0.0;
  // for quantized / weight-only
  std::string ranAs;
  int actDtype = 0;
  int wgtDtype = 0;
  const char *schemeName = "";
  bool castedActs = false;
  bool reduceInFloat = false;
  // Graph shape the ladder must reproduce: true = Add-0 operand trick
  // (unfoldable), false = old result-scaled-only shape.  Decided per
  // (provider, dtype) by profiling the 32^3 session for an inserted Cast --
  // a trick whose elementwise promotes (CPU fp16) measures the wrong
  // arithmetic, so the probe falls back to the old shape there.
  bool unfoldActs = false;
};

using OnnxProbeCache = std::unordered_map<std::string, OnnxProbeResult>;

// Probe every gemm variant once at 32^3 (tiny) and cache result.
// The cache is keyed by variant label (e.g. "fp16", "int8_qdq").
// Returned reference is cached per EP (providerKey plus OpenVINO target)
// for the lifetime of the process - subsequent calls for same EP return
// the same map without rebuilding.
const OnnxProbeCache &onnxProbeGemmCache(const OrtRuntime &rt,
                                         const onnx_ep_info_t &ep);

// Build fresh cache (no memoization) - used by the global probe once
// per EP before all tests.  Exposed for testing.
OnnxProbeCache onnxProbeGemmVariants(const OrtRuntime &rt,
                                     const onnx_ep_info_t &ep);

// Can this EP run anything at all?  Tries one tiny session each for fp32,
// fp16 and int8 QDQ (both spellings) and answers on the first success.
// GetAvailableProviders reports what the build contains, not what is in
// the box -- an OpenVINO NPU target with no NPU, or NNAPI on a phone whose
// accelerator declines every graph -- so listing and runAll() filter
// through this first instead of printing a device that only ever reports
// Unsupported.  Creation only, no runs; memoized per runtime and target.
// `reason` carries the refusal when it answers false.
bool onnxEpViable(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                  std::string &reason);

#endif // ENABLE_ONNX
#endif // CLPEAK_ONNX_PROBE_H
