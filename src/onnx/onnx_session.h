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
OnnxSessionResult onnxCreateSession(const OrtRuntime &rt,
                                    const onnx_ep_info_t &ep,
                                    const std::string &modelBytes,
                                    bool keepConstantsUnfolded = false,
                                    bool profile = false);

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

// True if any of `ops` is a kernel that does integer matrix arithmetic.
bool onnxOpsIncludeQuantizedMatMul(const std::vector<std::string> &ops);

// A wall-clock budget for one test on one provider.
//
// Every sweep in this backend decides how far to climb from measured
// throughput, which is the right rule when running a graph is the cost.  On
// the providers that compile ahead of time it is not: Core ML turns a 16 MB
// elementwise graph into machine code in seconds and a 64 MB one in more
// than ten minutes, and no amount of throughput data predicts that.  A sweep
// that consults the clock between steps stops climbing when the provider
// turns out to be expensive to compile for, and reports what it has instead
// of appearing to hang.
class OnnxDeadline
{
public:
  explicit OnnxDeadline(double seconds);
  bool expired() const;

private:
  double m_startUs;
  double m_budgetUs;
};

// Budget per test, per provider.  Generous next to the ~60 s the slowest
// test needs on a warm cache, so it only bites when something is genuinely
// pathological.
constexpr double kOnnxTestBudgetSec = 180.0;

// One-line human-readable form of an OrtStatus (releases the status).
std::string onnxStatusText(const OrtRuntime &rt, OrtStatus *st);

#endif // CLPEAK_ONNX_SESSION_H
