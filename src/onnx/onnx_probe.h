#ifndef CLPEAK_ONNX_PROBE_H
#define CLPEAK_ONNX_PROBE_H

#ifdef ENABLE_ONNX

#include "onnx_model.h"

#include <string>
#include <unordered_map>
#include <vector>

struct OrtRuntime;
struct onnx_ep_info_t;

// What one tiny (32^3) build of a gemm variant found out about a provider,
// and the graph shape the ladder must therefore reproduce.
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
  // Graph shapes the ladder may use, in preference order, each proven at
  // 32^3 to build, fuse where the row needs it, and run the multiply at the
  // row's own width (see OnnxLiveShape).  The first is the fastest safe
  // choice -- result-scaled where it is viable, since a live operand costs a
  // pass and, on the ANE, compiles to a ~20% slower program -- and the rest
  // are fallbacks the ladder drops to only when the one before it is caught
  // folding.  Result-scaled is the only foldable shape; the live ones cannot
  // be evaluated at build time, so a provider whose compiler folds ends up on
  // one of them, and a provider that does not keeps the fast result-scaled.
  std::vector<OnnxLiveShape> shapes;
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

// Can this EP run anything at all?  Tries, cheapest first: provider attach
// with no model, one trivial-Mul session (the dispatch-latency graph), then
// one tiny session each for fp32, fp16 and int8 QDQ (both spellings),
// stopping at the first success.  GetAvailableProviders reports what the
// build contains, not what is in the box -- an OpenVINO NPU target with no
// NPU, or NNAPI on a phone whose accelerator declines every graph -- so
// listing and runAll() filter through this first instead of printing a
// device that only ever reports Unsupported.  Creation only, no runs;
// memoized per runtime and target.  `reason` carries the refusal when it
// answers false.
bool onnxEpViable(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                  std::string &reason);

// The floating-point width this provider streams a resident tensor through
// fastest: ONNX_DT_FLOAT16 or ONNX_DT_FLOAT.  One [1, K] x [K, N] matrix-
// vector product over eight megabytes of weights, timed in each, and the
// faster kept (fp16 on a tie).  The bandwidth-shaped tests take their
// element type from this rather than assuming fp16: a provider without a
// native fp16 kernel for the operation -- ONNX Runtime's CPU EP converts
// every fp16 tensor on the way in -- would otherwise report its conversion
// rate under a bandwidth heading (2 GB/s on a Threadripper whose fp32 rows
// stream 38).  Memoized per runtime and target.
int onnxStreamDtype(const OrtRuntime &rt, const onnx_ep_info_t &ep);

// The bandwidth that probe measured in the width it chose, in bytes/second,
// or 0 when neither width could be timed.  It is a read of eight megabytes
// of resident weights through the operation every provider tunes hardest, so
// nothing that also has to *write* a tensor can honestly exceed it -- which
// makes it the ceiling a differential measurement is checked against.  The
// activation rows subtract a reference graph from a measurement, and where
// the operation costs little the remainder is mostly the noise of two large
// numbers: Core ML's softmax swung between a refusal and 139 GB/s that way,
// on a device that streams 85.
double onnxStreamBps(const OrtRuntime &rt, const onnx_ep_info_t &ep);

// onnx-gemm and onnx-numeric-error are a rate/accuracy pair over their
// overlapping labels (the plain-float dtypes plus int8_qdq; the weight-only
// and nvfp4 rows have no accuracy counterpart by design).  When gemm's ladder
// proves the provider folded the resident operands at compile time, its rate
// is suppressed as an error -- and the accuracy row for the same label is
// suppressed with it, even though its non-resident graph did run.  That graph
// cannot fold (its activations are a runtime input), so without this its
// number would stand alone beside a rate that was refused as meaningless.
// runGemm records each folded label here; runNumericError consults it.
// Keyed by EP exactly like the probe cache above.  Cleared per EP in runAll
// (and again at runGemm entry for direct callers) so a stale run cannot
// suppress a later one.
void onnxNoteGemmFolded(const onnx_ep_info_t &ep, const std::string &label);
bool onnxGemmFolded(const onnx_ep_info_t &ep, const std::string &label);
void onnxClearGemmFolded(const onnx_ep_info_t &ep);

#endif // ENABLE_ONNX
#endif // CLPEAK_ONNX_PROBE_H
