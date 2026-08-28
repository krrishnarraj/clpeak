#ifndef CLPEAK_ONNX_RUNTIME_H
#define CLPEAK_ONNX_RUNTIME_H

// Load-on-demand ONNX Runtime.  The library is dlopen'd so clpeak ships with
// no hard dependency: only one symbol (OrtGetApiBase) is ever resolved by
// name -- every other entry point comes through the OrtApi function-pointer
// table that call returns.  When no usable runtime is found the backend
// reports itself unavailable, matching how the GPU backends behave on a
// machine without their driver.
//
// CLPEAK_ONNX_STATIC builds are the exception, and iOS is why: Apple's
// official pod ships onnxruntime.xcframework as *static* libraries, and iOS
// will not dlopen anything that was not linked into the app bundle to begin
// with.  There the runtime is linked in and OrtGetApiBase is called directly;
// there is no library to search for and no path to override.

#include <string>
#include <onnxruntime_c_api.h>  // vendored: third_party/onnxruntime/

struct OrtRuntime
{
  void             *lib  = nullptr;   // null on a statically linked build
  const OrtApiBase *base = nullptr;
  const OrtApi     *api  = nullptr;   // table for `apiVersion`
  uint32_t          apiVersion = 0;   // highest version the runtime granted
  std::string       versionString;   // e.g. "1.29.0" (base->GetVersionString)
  std::string       path;            // what was loaded; empty when static
};

// Point the loader at a specific library, ahead of $CLPEAK_ONNXRUNTIME_LIB and
// the platform's conventional names.  Backs `--onnx-lib` and the FFI's
// clpeak_set_onnx_library(), which is how the GUI's settings screen chooses
// between installed runtimes.  An empty path clears the override.
//
// Naming a different library after one is already loaded takes effect: the
// next ortRuntime() call loads the new one.  The old handle is deliberately
// leaked rather than dlclosed -- ONNX Runtime keeps worker threads alive past
// the last session, so unloading it is not safe.  No-op when statically
// linked.
//
// Call it between runs only: ortRuntime() hands out a pointer to the loader's
// own record, and changing the library repoints that record.
void onnxSetLibraryOverride(const std::string &path);

// Load on first use; returns nullptr when no runtime library is found or it
// exposes no API version we can use.  A failed load is remembered, so a
// missing runtime costs one search rather than one per call.
const OrtRuntime *ortRuntime();

// Why the last load attempt failed, ready to show a user; empty when the
// runtime loaded or has not been asked for yet.  A refusal is only as useful
// as the sentence it comes with, and "not found" is the wrong sentence when
// the real answer is that the named file is not an ONNX Runtime.
std::string onnxLoadDiagnostic();

#endif // CLPEAK_ONNX_RUNTIME_H
