#ifndef CLPEAK_LITERT_RUNTIME_H
#define CLPEAK_LITERT_RUNTIME_H

// Load-on-demand LiteRT.  libLiteRt is dlopen'd so clpeak ships with no hard
// dependency, exactly as the ONNX backend treats onnxruntime: a machine
// without it gets a one-line "library not found" and no rows, the shape a
// missing GPU driver produces.  Unlike ONNX Runtime there is no single entry
// point returning a function table, so every C function the backend calls
// is resolved by name into `LitertApi`; the list is one X-macro so a missing
// symbol is reported with its name rather than as a crash.
//
// Where the runtime is found: `--litert-lib PATH` / the FFI setter, then the
// platform's conventional names (the bare soname on Android, where the app
// packages the AAR's .so; the app bundle's Frameworks directory on iOS, where
// the Runner embeds Google's dylib; the pip wheel's dylib/so/dll paths on
// desktops).
// The directory the library loaded from is remembered: LiteRT looks for its
// GPU accelerator (libLiteRtOpenClAccelerator / libLiteRtMetalAccelerator /
// libLiteRtWebGpuAccelerator) and the NPU dispatch libraries next to it
// unless told otherwise, and `--litert-npu-dir` is that telling.

#include <string>

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_metrics.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_profiler.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/options/litert_cpu_options.h"
#include "litert/c/options/litert_gpu_options.h"
#include "litert/c/options/litert_google_tensor_options.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/c/options/litert_mediatek_options.h"
#include "litert/c/options/litert_qualcomm_options.h"
#include "litert/c/options/litert_runtime_options.h"

// Every entry point the backend uses.  REQUIRED ones fail the load when
// absent -- the library is then not a LiteRT this backend can drive;
// OPTIONAL ones are null when absent and callers check (the profiler and
// the sink logger are diagnostics, not measurements).
//
// Accelerator and vendor options (litert/c/options/*.h) are deliberately
// not here: those `Lrt*` builders are client-side source in LiteRT's C++
// SDK, not exports of libLiteRt.  What the runtime actually receives is an
// opaque payload -- a TOML string under an identifier such as "gpu_options"
// or "xnnpack" -- and litert_session.cpp writes those strings directly.
#define CLPEAK_LITERT_REQUIRED(X)                                     \
  X(LiteRtCreateEnvironment)                                          \
  X(LiteRtDestroyEnvironment)                                         \
  X(LiteRtCreateOptions)                                              \
  X(LiteRtDestroyOptions)                                             \
  X(LiteRtSetOptionsHardwareAccelerators)                             \
  X(LiteRtAddOpaqueOptions)                                           \
  X(LiteRtCreateOpaqueOptions)                                        \
  X(LiteRtDestroyOpaqueOptions)                                       \
  X(LiteRtCreateModelFromBuffer)                                      \
  X(LiteRtDestroyModel)                                               \
  X(LiteRtGetMainModelSubgraphIndex)                                  \
  X(LiteRtGetModelSubgraph)                                           \
  X(LiteRtGetNumSubgraphInputs)                                       \
  X(LiteRtGetSubgraphInput)                                           \
  X(LiteRtGetNumSubgraphOutputs)                                      \
  X(LiteRtGetSubgraphOutput)                                          \
  X(LiteRtGetRankedTensorType)                                        \
  X(LiteRtCreateCompiledModel)                                        \
  X(LiteRtDestroyCompiledModel)                                       \
  X(LiteRtGetCompiledModelInputBufferRequirements)                    \
  X(LiteRtGetCompiledModelOutputBufferRequirements)                   \
  X(LiteRtRunCompiledModel)                                           \
  X(LiteRtCompiledModelIsFullyAccelerated)                            \
  X(LiteRtCreateManagedTensorBufferFromRequirements)                  \
  X(LiteRtDestroyTensorBuffer)                                        \
  X(LiteRtLockTensorBuffer)                                           \
  X(LiteRtUnlockTensorBuffer)                                         \
  X(LiteRtGetTensorBufferPackedSize)                                  \
  X(LiteRtGetTensorBufferRequirementsBufferSize)                      \
  X(LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes)         \
  X(LiteRtGetTensorBufferRequirementsSupportedTensorBufferType)

// LiteRtGetStatusString is optional because the Linux x86_64 wheel of
// ai-edge-litert 2.2.0 does not export it (the macOS and Android libraries
// do); litertStatusText() falls back to the numeric code.
#define CLPEAK_LITERT_OPTIONAL(X)                                     \
  X(LiteRtGetStatusString)                                            \
  X(LiteRtCreateSinkLogger)                                           \
  X(LiteRtDestroyLogger)                                              \
  X(LiteRtSetDefaultLogger)                                           \
  X(LiteRtGetDefaultLogger)                                           \
  X(LiteRtSetMinLoggerSeverity)                                       \
  X(LiteRtGetSinkLoggerSize)                                          \
  X(LiteRtGetSinkLoggerMessage)                                       \
  X(LiteRtClearSinkLogger)                                            \
  X(LiteRtCompiledModelGetProfiler)                                   \
  X(LiteRtStartProfiler)                                              \
  X(LiteRtStopProfiler)                                               \
  X(LiteRtResetProfiler)                                              \
  X(LiteRtGetNumProfilerEvents)                                       \
  X(LiteRtGetProfilerEvents)                                          \
  X(LiteRtCompiledModelGetErrorMessages)                              \
  X(LiteRtCompiledModelClearErrors)                                   \
  X(LiteRtEnvironmentSupportsFP16)

struct LitertApi
{
#define CLPEAK_LITERT_DECL(name) decltype(&name) name = nullptr;
  CLPEAK_LITERT_REQUIRED(CLPEAK_LITERT_DECL)
  CLPEAK_LITERT_OPTIONAL(CLPEAK_LITERT_DECL)
#undef CLPEAK_LITERT_DECL
};

struct LitertRuntime
{
  void *lib = nullptr;
  LitertApi api;
  std::string path;        // the file that loaded (resolved when known)
  std::string libraryDir;  // its directory; where accelerator libraries live
  std::string abiVersion;  // LITERT_RUNTIME_ABI_VERSION this build was compiled against
};

// Point the loader at a specific library, ahead of the platform's conventional
// names; empty clears the choice.  Backs `--litert-lib` and the FFI's
// clpeak_set_litert_library().  Takes effect on the next litertRuntime()
// call; the previously loaded library stays mapped for the process's life
// (LiteRT starts worker threads and accelerator contexts that are not safe
// to unload).  Call it between runs only.
void litertSetLibraryOverride(const std::string &path);

// Where the NPU dispatch and compiler-plugin libraries are
// (libLiteRtDispatch_<Vendor>.so, libLiteRtCompilerPlugin_<Vendor>.so, and
// the vendor runtime beside them).  Empty means the runtime library's own
// directory.  Backs `--litert-npu-dir`.
void litertSetNpuDirOverride(const std::string &dir);
std::string litertNpuDir();

// Load on first use; nullptr when no runtime is found or it lacks a required
// entry point.  A failed search is remembered, so a missing runtime costs one
// search rather than one per call.
const LitertRuntime *litertRuntime();

// Why the last load attempt failed, ready to show a user; empty when it
// loaded or has not been asked for yet.
std::string litertLoadDiagnostic();

// LiteRT's status codes as text, for row reasons.
std::string litertStatusText(const LitertRuntime &rt, LiteRtStatus st);

#endif // CLPEAK_LITERT_RUNTIME_H
