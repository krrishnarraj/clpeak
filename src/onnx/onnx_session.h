#ifndef CLPEAK_ONNX_SESSION_H
#define CLPEAK_ONNX_SESSION_H

// Session construction shared by every ONNX benchmark: one OrtEnv per
// process, per-EP registration options, and the honesty guard -- sessions
// on a non-CPU EP are created with ORT's CPU fallback disabled, so a graph
// the EP cannot run entirely fails session creation (and the row reports
// Unsupported) instead of silently measuring the CPU.

#include <string>
#include <vector>

#include "onnx_runtime.h"
#include <onnx/onnx_peak.h>

// Process-wide OrtEnv (created on first use; log level follows --verbose).
OrtEnv *onnxEnv(const OrtRuntime &rt);

struct OnnxSessionResult
{
  OrtSession *session = nullptr;
  std::string error;               // set when session == nullptr
};

// Build a session for `ep` from in-memory model bytes.  For non-CPU EPs the
// EP is appended with clpeak's default options for that provider and CPU
// fallback is disabled.  On failure `error` carries a one-line reason
// suitable for a skip row.
// `keepConstantsUnfolded` stops ORT evaluating a subgraph of constants at
// load time.  Needed only by the throughput models whose operands are both
// initializers -- without it the whole matmul is computed once during session
// creation and every timed run measures nothing.
// `profile` records which kernels the provider actually runs; collect them
// afterwards with onnxCollectExecutedOps().
// `keepQdqUnfused` additionally holds off the QDQ selector, which rewrites
// DequantizeLinear/MatMul/QuantizeLinear into QLinearMatMul.  That fusion is
// what makes an int8 row an int8 row, but QLinearMatMul is an integer operator
// and has no float8 type constraint, so on a float8 graph the rewrite produces
// a model that fails its own type check -- "Type 'tensor(float8e4m3fn)' of
// input parameter (A_q) of operator (QLinearMatMul) is invalid".  A provider
// with real float8 matmul hardware consumes the QDQ nodes itself and never
// wanted the rewrite.
OnnxSessionResult onnxCreateSession(const OrtRuntime &rt,
                                    const onnx_ep_info_t &ep,
                                    const std::string &modelBytes,
                                    bool keepConstantsUnfolded = false,
                                    bool profile = false,
                                    bool keepQdqUnfused = false);

// Names of the kernels a profiled session executed, distinct, in first-seen
// order.  Empty when profiling was off or unavailable.
//
// This answers a question no timing can: whether a row measured the operation
// its name claims.  ONNX Runtime rewrites graphs before running them, and a
// provider that declines to fuse a quantized matmul will dequantize the
// operands and multiply them in floating point instead -- producing a
// perfectly good number that is not an int8 number at all.
std::vector<std::string> onnxCollectExecutedOps(const OrtRuntime &rt,
                                                OrtSession *session);

// Did the provider actually multiply in integers?
//
// Evidence comes in two shapes.  A provider that executes ONNX operators one
// at a time names each kernel, and a fused quantized matmul shows up as
// something like QLinearMatMul.  A provider that compiles whole subgraphs --
// TensorRT, Core ML, QNN -- reports one opaque kernel of its own instead,
// whose name says nothing at all: TensorRT builds a perfectly good int8
// engine and calls it `TRTKernel_graph_clpeak_7216741020808563463_0`.
//
// So the test looks for the *failure* rather than the success: a bare
// floating-point MatMul sitting beside the dequantize nodes, which is exactly
// what a provider that declined to fuse leaves behind.  Anything else ran as
// the provider's own quantized kernel, and it cannot have quietly run on the
// CPU instead, because the fallback guard would have failed the session.
bool onnxOpsRanQuantizedMatMul(const std::vector<std::string> &ops);

// The recognisable quantized kernel among `ops`, or an empty string when the
// provider fused everything into a kernel of its own naming.
std::string onnxQuantizedKernelName(const std::vector<std::string> &ops);
// One-line human-readable form of an OrtStatus (releases the status).
std::string onnxStatusText(const OrtRuntime &rt, OrtStatus *st);

#endif // CLPEAK_ONNX_SESSION_H
