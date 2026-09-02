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
};

using OnnxProbeCache = std::unordered_map<std::string, OnnxProbeResult>;

// Probe every gemm variant once at 64^3 (tiny) and cache result.
// The cache is keyed by variant label (e.g. "fp16", "int8_qdq").
// Returned reference is cached per EP (providerKey) for the lifetime
// of the process - subsequent calls for same EP return the same map
// without rebuilding.
const OnnxProbeCache &onnxProbeGemmCache(const OrtRuntime &rt,
                                         const onnx_ep_info_t &ep);

// Build fresh cache (no memoization) - used by the global probe once
// per EP before all tests.  Exposed for testing.
OnnxProbeCache onnxProbeGemmVariants(const OrtRuntime &rt,
                                     const onnx_ep_info_t &ep);

#endif // ENABLE_ONNX
#endif // CLPEAK_ONNX_PROBE_H
