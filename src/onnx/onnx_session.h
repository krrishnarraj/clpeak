#ifndef CLPEAK_ONNX_SESSION_H
#define CLPEAK_ONNX_SESSION_H

// Session construction shared by every ONNX benchmark: one OrtEnv per
// process, per-EP registration options, and the honesty guard -- sessions
// on a non-CPU EP are created with ORT's CPU fallback disabled, so a graph
// the EP cannot run entirely fails session creation (and the row reports
// Unsupported) instead of silently measuring the CPU.

#include <string>

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
OnnxSessionResult onnxCreateSession(const OrtRuntime &rt,
                                    const onnx_ep_info_t &ep,
                                    const std::string &modelBytes);

// One-line human-readable form of an OrtStatus (releases the status).
std::string onnxStatusText(const OrtRuntime &rt, OrtStatus *st);

#endif // CLPEAK_ONNX_SESSION_H
