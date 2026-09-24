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

// Point the loader at a specific library, ahead of the platform's conventional
// names.  Backs `--onnx-lib` and the FFI's
// clpeak_set_onnx_library(), which is how the GUI's settings screen chooses
// between installed runtimes.  An empty path clears the override.
//
// Naming a different library after one is already loaded takes effect: the
// next ortRuntime() call loads the new one -- unless a plugin library has
// been loaded into the current runtime (onnxPinRuntime), in which case the
// choice waits for the next start.  The old handle is deliberately
// leaked rather than dlclosed -- ONNX Runtime keeps worker threads alive past
// the last session, so unloading it is not safe (nor is unloading a file
// that turned out not to be an ONNX Runtime at all; common/dynlib.h has the
// crash that proved it).  No-op when statically linked.
//
// Call it between runs only.  The record ortRuntime() handed out stays valid
// and unchanged (every runtime loaded is kept for the life of the process),
// but a run's environment and devices belong to the runtime it started on.
void onnxSetLibraryOverride(const std::string &path);

// Load on first use; returns nullptr when no runtime library is found or it
// exposes no API version we can use.  A failed load is remembered, so a
// missing runtime costs one search rather than one per call.
const OrtRuntime *ortRuntime();

// Ask the loader to run its default search again on the next ortRuntime():
// the plugin configuration steers that search (an --onnx-winml directory's
// own runtime comes first), so a change to it is a reason to look again.
// A named library is unaffected, and so is a pinned runtime.  Between runs
// only, like the override.
void onnxRuntimeRecheck();

// `rt` stays the runtime until the process exits, for `why` -- a clause
// naming the runtime, filed under `key` so a later pin for the same cause
// replaces its sentence rather than adding one.  Two causes pin: a plugin
// library mapped into the process on `rt`'s behalf (onnxRegisterEpLibraries;
// handed to a second runtime, one left the GUI's enumeration running
// forever on Windows), and an environment `rt` cannot release without
// crashing (onnx_session.cpp, g_envUnreleasable), since a switch releases
// it.  src/onnx/AGENTS.md, "Some runtimes stay until the process exits", has
// the account.  A later choice of runtime is kept and reported by
// onnxPendingRuntime() instead of loaded.  Sticky: a library the runtime has
// since unregistered can still be mapped.
void onnxPinRuntime(const OrtRuntime &rt, const std::string &key,
                    const std::string &why);

// A runtime chosen after the loaded one was pinned: `path` is the library
// that loads at the next start (empty for the default search) and `reason`
// the sentence saying why it is not loaded now.  False when nothing waits.
bool onnxPendingRuntime(std::string &path, std::string &reason);

// Why the last load attempt failed, ready to show a user; empty when the
// runtime loaded or has not been asked for yet.  A refusal is only as useful
// as the sentence it comes with, and "not found" is the wrong sentence when
// the real answer is that the named file is not an ONNX Runtime.
std::string onnxLoadDiagnostic();

#endif // CLPEAK_ONNX_RUNTIME_H
