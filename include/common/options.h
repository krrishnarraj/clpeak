#ifndef CLPEAK_OPTIONS_H
#define CLPEAK_OPTIONS_H

#include <bitset>
#include <string>
#include <utility>
#include <vector>
#include <common/benchmark_enums.h>  // Backend, Benchmark, Category
#include <common/common.h>           // DEFAULT_TARGET_TIME_US
#include <common/run_document.h>     // Invocation

// One backend as the command line and the result document know it.  The
// flag is the name in lower case; `builtIn` says whether this binary
// carries the backend at all.  Every flag parses in every build -- a script
// may say --no-cuda on a Mac -- and a backend that is asked for by name but
// not built in is reported, not rejected (see requestedButNotBuilt).
struct BackendInfo {
  Backend     id;
  const char *name;     // "OpenCL", "CUDA", "CoreML" -- as printed and as the document's `backend`
  const char *flag;     // "opencl", "cuda", "coreml" -- --<flag> / --no-<flag>
  bool        builtIn;
};

const BackendInfo &backendInfo(Backend b);

// One item of --devices, `backend:index`, exactly as --list-devices prints
// it.  The list is an allow-list: when it is given, only the devices on it
// run, and a backend with none of its devices listed does not run at all.
struct DeviceSelector {
  Backend backend = Backend::COUNT;
  int     index   = 0;
};

// Shared CLI options populated once by parseCliOptions and consumed by every
// backend.  Peak::applyOptions copies the relevant fields into the backend
// so the rest of its code can stay backend-flavored.
struct CliOptions {
  // Which backends run.  Default: every one that is built in.  The first
  // positive --<backend> flag flips this to allow-list mode ("only the
  // listed"); --no-<backend> always subtracts.
  std::bitset<static_cast<size_t>(Backend::COUNT)> enabledBackends;
  // The backends named positively on the command line, kept apart from
  // enabledBackends so the caller can tell "asked for CUDA and it is not in
  // this build" from "CUDA is simply not in this build".
  std::bitset<static_cast<size_t>(Backend::COUNT)> requestedBackends;

  // --devices: the devices that run.  Empty = every device of every enabled
  // backend.
  std::vector<DeviceSelector> devices;

  // --onnx-lib: absolute path to the onnxruntime shared library to load,
  // overriding the platform's conventional names.  Empty = search the
  // default names (see src/onnx/onnx_runtime.cpp).  Ignored on a build that
  // links ONNX Runtime statically, where there is nothing to load.
  std::string onnxLibPath;

  // --onnx-ep NAME=PATH (repeatable): plugin execution-provider libraries
  // to register on the runtime (ONNX Runtime 1.22+), each under the
  // registration name the provider expects -- Qualcomm's QNN plugin is
  // `QNNExecutionProvider=<dir>/onnxruntime_providers_qnn.dll`.  See
  // src/onnx/onnx_plugin.h.
  std::vector<std::pair<std::string, std::string>> onnxEpLibraries;

  // --onnx-winml [PATH]: register the execution providers Windows ML's
  // catalog installs from the Microsoft Store (Windows 11 24H2+), through
  // Microsoft.Windows.AI.MachineLearning.dll -- the file, or its directory,
  // named by PATH, else searched beside the loaded runtime and the
  // executable.  Opt-in because a missing provider is downloaded.
  bool        onnxWinml = false;
  std::string onnxWinmlPath;

  // --litert-lib: the LiteRT shared library (libLiteRt) to load, ahead of
  // the platform's conventional names; --litert-npu-dir: where the NPU
  // dispatch and compiler-plugin libraries and the vendor runtime live,
  // when not beside the runtime library (see src/litert/litert_runtime.cpp).
  std::string litertLibPath;
  std::string litertNpuDir;

  // Iters / warmup.  When forceIters is false, each backend's runKernel
  // calibrates iters from a one-shot timed warmup so the timed phase lands
  // at ~targetTimeUs regardless of device speed.
  bool         forceIters    = false;
  unsigned int iters         = 0;
  unsigned int warmupCount   = 2;
  unsigned int targetTimeUs  = DEFAULT_TARGET_TIME_US; // --max-time, in us
  // CPU backend uses its own (longer) budget; --max-time does not affect it.
  unsigned int targetTimeUsCpu = DEFAULT_CPU_TARGET_TIME_US; // --max-time-cpu, in us

  // Test selection.  Default: every category and every test enabled.  The
  // first positive --<test> flag flips enabledTests to allow-list mode
  // ("deny by default; enable picked"); --no-<test> always subtracts.
  // The first positive --<category> flag flips enabledCategories the same
  // way; --no-<category> always subtracts.  A test runs iff its primary
  // category is enabled AND its own bit is set (see isAllowed).
  std::bitset<static_cast<size_t>(Benchmark::COUNT)>  enabledTests;
  std::bitset<static_cast<size_t>(Category::Unknown)> enabledCategories;

  // Output / compare.  One format, one flag: `-o file` writes the v3 JSON
  // document (run_document.h).
  bool        enableOutput = false;
  std::string outputFile;
  std::string compareFile;

  // Listing mode (no benchmarks run; just print devices).
  bool listDevices = false;

  // Verbose diagnostics: print backend debug logs (kernel build logs, API /
  // launch errors, library exceptions) that are suppressed by default.
  bool verbose = false;

  // Print, alongside the readings, what each test and each reading measures
  // (the descriptions authored at the beginTest()/emit() call sites).  Off by
  // default: the plain output is a table for people who already know it.
  bool describe = false;

  CliOptions()
  {
    enabledBackends.set();
    enabledTests.set();
    enabledCategories.set();
  }

  // Whether a backend has anything to run: its flag is on, and if --devices
  // was given, at least one of its devices is on the list.  The run loop
  // skips the rest without constructing them, so `--devices cuda:0` never
  // loads the ONNX runtime.
  bool backendEnabled(Backend b) const
  {
    if (!enabledBackends.test(static_cast<size_t>(b)))
      return false;
    if (devices.empty())
      return true;
    for (const DeviceSelector &sel : devices)
      if (sel.backend == b)
        return true;
    return false;
  }

  // Backends named with a positive flag that this binary does not carry.
  // The caller tells the user; running nothing silently would read as "no
  // devices".
  std::vector<Backend> requestedButNotBuilt() const;
};

// Describe how clpeak was asked to run, for the result document's `invocation`
// block.  Every number in a run is sensitive to this -- a shorter --max-time
// measures a different thing, and a selective run is not a full one even though
// the file looks the same shape -- so it is recorded rather than inferred.
// Lives here because the category and test flag-name tables do.
Invocation invocationFrom(const CliOptions &opts, int argc, char **argv);

// Parse argv into out.  On --help / --version / parse error this calls
// exit() directly.  Returns 0 on success.
int parseCliOptions(int argc, char **argv, CliOptions &out);

// Embedding-safe variant: never calls exit().  Returns true on success;
// on failure (parse error, or --help/--version which have no meaning when
// embedded) returns false with a human-readable message in errorMsg.
// Used by clpeak_ffi so a bad argv can't kill the host GUI process.
bool parseCliOptionsNoExit(int argc, char **argv, CliOptions &out,
                           std::string &errorMsg);

#endif // CLPEAK_OPTIONS_H
