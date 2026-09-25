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
  std::string       path;            // what was loaded; the resolved file even
                                     // when found by name, empty only when static
};

// The runtime setup -- which library, and whether (and from where) the
// Windows ML catalog adds its providers -- is fixed for the process by the
// first runtime that loads.  Until one has, a change applies at the next
// ortRuntime() call, however many attempts that takes (a library that fails
// to load sets nothing up).  From then on a change is kept, reported by
// onnxPendingSetup(), and applies the next time the process starts: one
// process never holds a second runtime.  Two did not coexist safely --
// provider libraries stay bound to the runtime that loaded them, and
// releasing a runtime's environment is where runtimes crashed
// (src/onnx/AGENTS.md, "One runtime setup per process").  The CLI sets its
// options once, before anything loads, and is unaffected.
//
// Point the loader at a specific library, ahead of the platform's
// conventional names.  Backs `--onnx-lib` and the FFI's
// clpeak_set_onnx_library(), which is how the GUI's settings screen chooses
// between installed runtimes.  An empty path clears the override.  A handle
// is never dlclosed -- ONNX Runtime keeps worker threads alive past the last
// session, and unloading a file that turned out not to be an ONNX Runtime
// is no safer (common/dynlib.h has the crash that proved it).  No-op when
// statically linked.
void onnxSetLibraryOverride(const std::string &path);

// Windows ML's execution-provider catalog (`--onnx-winml [PATH]`), part of
// the runtime setup above: a catalog directory with no library named makes
// the onnxruntime.dll beside it the runtime, and the catalog decides which
// providers join it.  `path` names Microsoft.Windows.AI.MachineLearning.dll
// or its directory, or is empty to search beside the loaded runtime and the
// executable.  Accepted everywhere; off Windows the resolution says why it
// did nothing.
void onnxSetWinml(bool enabled, const std::string &path);

// The Windows ML setup in effect: the one the loaded runtime was set up
// with, or the latest before one has loaded.
bool onnxWinmlEnabled();
std::string onnxWinmlPathHint();

// Bumped whenever the Windows ML setup in effect changes, which can only
// happen before a runtime has loaded; the catalog's resolution is memoized
// against it (onnx_winml.cpp).
uint64_t onnxWinmlGeneration();

// Load on first use; returns nullptr when no runtime library is found or it
// exposes no API version we can use.  A failed load is remembered, so a
// missing runtime costs one search rather than one per call; the record a
// successful one returns is never rewritten.
const OrtRuntime *ortRuntime();

// A setup chosen after the runtime loaded, which takes effect at the next
// start: `library` empty for the default search.  False when nothing waits.
struct OnnxPendingSetup
{
  std::string library;
  bool winml = false;
  std::string winmlPath;
};
bool onnxPendingSetup(OnnxPendingSetup &out);

// Why the last load attempt failed, ready to show a user; empty when the
// runtime loaded or has not been asked for yet.  A refusal is only as useful
// as the sentence it comes with, and "not found" is the wrong sentence when
// the real answer is that the named file is not an ONNX Runtime.
std::string onnxLoadDiagnostic();

#endif // CLPEAK_ONNX_RUNTIME_H
