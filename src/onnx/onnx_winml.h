#ifndef CLPEAK_ONNX_WINML_H
#define CLPEAK_ONNX_WINML_H

// Windows ML's execution-provider catalog, reached through the flat C API
// of Microsoft.Windows.AI.MachineLearning.dll (the `WinMLEp*` entry points
// of WinMLEpCatalog.h in the Microsoft.Windows.AI.MachineLearning NuGet).
//
// On Windows 11 24H2 the vendor providers -- Qualcomm's QNN, Intel's
// OpenVINO, AMD's Vitis AI, NVIDIA's TensorRT for RTX -- are not part of
// any runtime: the catalog knows which of them fit the machine, installs
// them from the Microsoft Store on request, and hands back the path of
// each one's plugin library, which then registers like any other
// (onnx_plugin.h).  That is the one route to those NPUs that needs no
// vendor download and no matching of package versions by hand.
//
// The DLL is Microsoft's and is not shipped with clpeak: it is loaded from
// the path `--onnx-winml` names, or found beside the loaded onnxruntime
// (the NuGet package carries both in one directory) or the executable.
// Everything here is dlopen'd, like the runtime itself; a build needs no
// Windows ML SDK.  Off Windows the resolution simply reports that the
// catalog is a Windows feature.

#ifdef ENABLE_ONNX

#include <string>
#include <vector>

struct OrtRuntime;

struct OnnxWinmlProvider
{
  std::string name;           // the catalog's provider name ("QNNExecutionProvider", ...)
  std::string version;
  std::string packageFamily;  // the Store package that carries it
  std::string libraryPath;    // the plugin library, once the provider is ready
  bool certified = false;     // only certified providers are registered
  bool ready = false;         // installed and added to this process
  bool installed = false;     // on the machine, whether or not added yet
  std::string error;          // why it could not be made ready, if it could not
};

struct OnnxWinmlResolution
{
  std::string dllPath;                       // the catalog DLL that answered
  std::string error;                         // when the catalog could not be used at all
  std::vector<OnnxWinmlProvider> providers;  // what it listed, ready or not
};

// Resolve the catalog for the current configuration: load the DLL,
// enumerate the providers that fit this machine, install and prepare the
// certified ones, and read their library paths.  Memoized per configuration
// (and per runtime, since the search for the DLL starts beside it): the
// install step is a download, and it runs once.  `pathHint` is what
// --onnx-winml named -- the DLL or its directory -- or empty to search.
const OnnxWinmlResolution &onnxWinmlResolve(const OrtRuntime *rt,
                                            const std::string &pathHint);

// The memoized answer for the current configuration and `rt`, or null when
// nothing has resolved it yet.  For status queries, which must never be the
// thing that installs a provider.
const OnnxWinmlResolution *onnxWinmlResolved(const OrtRuntime *rt);

#endif // ENABLE_ONNX
#endif // CLPEAK_ONNX_WINML_H
