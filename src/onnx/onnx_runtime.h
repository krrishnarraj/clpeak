#ifndef CLPEAK_ONNX_RUNTIME_H
#define CLPEAK_ONNX_RUNTIME_H

// Load-on-demand ONNX Runtime.  The library is dlopen'd so clpeak ships with
// no hard dependency: only one symbol (OrtGetApiBase) is ever resolved by
// name -- every other entry point comes through the OrtApi function-pointer
// table that call returns.  When no usable runtime is found the backend
// reports itself unavailable, matching how the GPU backends behave on a
// machine without their driver.

#include <string>
#include <onnxruntime_c_api.h>  // vendored: third_party/onnxruntime/

struct OrtRuntime
{
  void             *lib  = nullptr;
  const OrtApiBase *base = nullptr;
  const OrtApi     *api  = nullptr;   // table for `apiVersion`
  uint32_t          apiVersion = 0;   // highest version the runtime granted
  std::string       versionString;   // e.g. "1.29.0" (base->GetVersionString)
};

// Load once per process; returns nullptr when no runtime library is found or
// it exposes no API version we can use.  Never unloaded: ONNX Runtime keeps
// worker threads alive, so dlclosing it is not safe.
const OrtRuntime *ortRuntime();

#endif // CLPEAK_ONNX_RUNTIME_H
