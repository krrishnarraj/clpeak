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
// Null when the runtime refuses to create one, with the reason in
// onnxEnvError(); a refusal is remembered per runtime rather than retried.
OrtEnv *onnxEnv(const OrtRuntime &rt);
std::string onnxEnvError();

struct OnnxSessionResult
{
  OrtSession *session = nullptr;
  std::string error;               // set when session == nullptr
  // The provider took the graph but its runtime placed the work on another
  // unit (onnx_coreml_plan.h): a refusal about size or shape, not about the
  // format, so a ladder may climb past it where it would stop at any other.
  bool offDevice = false;
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
// `verifyPlacement` asks the provider's runtime where it put the work and
// refuses the session when that is not the unit the row is named for.
// Today that is the CoreML provider's compute plan (onnx_coreml_plan.h);
// every other provider is judged by the fallback guard alone.  It is off
// for the probes -- the viability check and the 32-cube fusion probe --
// which ask whether a graph builds and fuses, not where a real size runs:
// Core ML sends anything that small to the CPU, and verifying it would
// declare the provider dead on a machine whose Neural Engine takes every
// 2048-cube it is offered.
OnnxSessionResult onnxCreateSession(const OrtRuntime &rt,
                                    const onnx_ep_info_t &ep,
                                    const std::string &modelBytes,
                                    bool keepConstantsUnfolded = false,
                                    bool profile = false,
                                    bool keepQdqUnfused = false,
                                    bool verifyPlacement = true);

// Names of the kernels a profiled session executed, one entry per kernel
// launch in execution order -- so a kernel that ran twice appears twice.
// Empty when profiling was off or unavailable.
//
// This answers a question no timing can: whether a row measured the operation
// its name claims.  ONNX Runtime rewrites graphs before running them, and a
// provider that declines to fuse a quantized matmul will dequantize the
// operands and multiply them in floating point instead -- producing a
// perfectly good number that is not an int8 number at all.  Duplicates are
// kept because counts matter too: a graph shape that adds one more Cast to
// a provider that already casts once is running an extra pass, and the
// shape probe compares the counts.
// When `opInType` is non-null it also receives the element type the compute
// kernel actually consumed, as ORT names it in the profile ("float",
// "float16", "bfloat16", ...), or empty when no such kernel ran.  `ofOp`
// picks which kernel to read it from; null means the MatMul family.
// SessionEndProfiling is one-shot, so this is the only chance to read it.
//
// It is the one signal that catches a provider quietly widening the
// arithmetic: ORT inserts a "precision-free" Cast in front of an fp16 kernel
// when the one it picks wants fp32, and the row then measures fp32 under the
// fp16 label -- the CPU EP's fp16 convolution rows land on its fp32 rows to
// three figures for exactly this reason.  A cast *count* cannot see it, since
// the widened graph can carry fewer Cast nodes than the narrow one.
std::vector<std::string> onnxCollectExecutedOps(const OrtRuntime &rt,
                                                OrtSession *session,
                                                std::string *opInType = nullptr,
                                                const char *ofOp = nullptr);

// The element type ORT names in a profile for `dtype` ("float", "float16",
// "bfloat16"), or "" for a type with no plain kernel to read.
const char *onnxProfileTypeName(int dtype);

// How many times `name` appears among `ops`.
size_t onnxCountOp(const std::vector<std::string> &ops, const char *name);

// `ops` distinct, in first-seen order, comma-joined -- for skip reasons and
// the verbose log.
std::string onnxJoinOps(const std::vector<std::string> &ops);

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
// Empty when this runtime can be asked to run `dtype` at all; otherwise the
// reason it cannot, phrased for a skip row.
//
// Two constraints, and the binding one is named.  A datatype cannot be spelled
// below the opset that introduced it, and a model declaring a newer opset fails
// to *load*, with a message about IR versions that says nothing about the
// datatype.  And a quantized type QLinearMatMul cannot carry additionally needs
// a runtime that honours `disable_specified_optimizers`, since the QDQ selector
// would otherwise rewrite the graph into one that fails its own type check --
// ONNX Runtime before 1.18 accepts that request and ignores it, the same defect
// the constant-folding guard in gemm.cpp reports.
std::string onnxDtypeUnsupportedReason(const OrtRuntime &rt, int dtype);

// Empty when `ep` can be handed a `dtype` graph -- quantized in and out with
// a per-tensor scale when `qdq` -- otherwise the reason it must not be,
// phrased for a skip row.
//
// Every other refusal in this backend is learned by asking: the provider
// builds the graph or says why not, and the row reports its words.  This is
// for the graphs a provider does not decline but takes the process down with,
// where asking is the fault -- the answer arrives as an access violation, and
// every row after it, on every provider, is never run.  Checked before the
// gemm probe builds a variant, which is what every test consults, and again
// by the accuracy row, which builds the same graph in its own shape.
std::string onnxProviderFenceReason(const onnx_ep_info_t &ep, int dtype,
                                    bool qdq);

// Attach `ep` to throwaway session options: the provider-registration half
// of session creation, with no model and no session.  Empty when the
// provider accepts clpeak's options for this target (an OpenVINO target
// with no hardware behind it fails here, fast, with nothing compiled);
// otherwise the one-line refusal.  The viability probe tries this before
// building any graph.
std::string onnxProviderAttach(const OrtRuntime &rt, const onnx_ep_info_t &ep);

// One-line human-readable form of an OrtStatus (releases the status).
std::string onnxStatusText(const OrtRuntime &rt, OrtStatus *st);

// Has the runtime reported its accelerator device lost since the flag was
// last cleared?
//
// A GPU that is reset out from under the provider -- a Mali driver timing out
// a hung shader and answering vkWaitForFences with VK_ERROR_DEVICE_LOST, which
// is what a Pixel 7a does to the WebGPU EP on the first non-trivial graph --
// does not come back.  Every later session still builds, every later inference
// still fails, and the run spends its whole budget proving it: 52 of 83 rows
// and 53 of 58 seconds in the report that prompted this.  So the loss is
// latched the moment the runtime mentions it, whether that is in a status
// message handed back to us or a line it only logged, and runAll() abandons
// the provider rather than asking it 52 more questions.
//
// Set from the ORT logger (above any verbosity filter -- the WebGPU EP reports
// the loss at INFO, so a non-verbose run would otherwise never see it) and
// from onnxStatusText().  Cleared per provider in runAll().
bool onnxDeviceLost();
void onnxClearDeviceLost();

// True when `reason` is a runtime or device failure rather than the provider
// saying it has no kernel for this format.
bool onnxReasonIsDeviceLoss(const std::string &reason);

// The status a refusal deserves.  ONNX's ordinary refusals *are* capability
// facts -- a provider declining nodes under the CPU-fallback guard, a missing
// bf16 kernel, the empty status ORT returns for a float4 graph -- so
// Unsupported stays the default and only a device failure is promoted to
// Error.  Reporting a dead GPU as "unsupported" would be a claim about the
// format, and a reader has no way to tell it from a real one: this run said
// "unsupported" against fp32 matmul on a GPU that does fp32 matmul perfectly
// well when it is alive.
// `current` is never downgraded: a caller that has already concluded the
// ladder failed for a reason of its own keeps its Error.
ResultStatus onnxFailureStatus(const std::string &reason,
                               ResultStatus current = ResultStatus::Unsupported);

#endif // CLPEAK_ONNX_SESSION_H
