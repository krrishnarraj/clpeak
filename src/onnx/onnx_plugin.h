#ifndef CLPEAK_ONNX_PLUGIN_H
#define CLPEAK_ONNX_PLUGIN_H

// Plugin execution providers (ONNX Runtime 1.22+).
//
// A provider no longer has to be compiled into the runtime: a separately
// shipped shared library exporting CreateEpFactories is registered on the
// environment by path, and the runtime then enumerates every hardware
// device it can serve through GetEpDevices.  Qualcomm's QNN EP moved to
// this shape with its 2.0 (Microsoft's built-in QNN packages stop at ORT
// 1.24), and it is how Windows ML hands out the vendor providers it
// installs from the Store -- so without it, a stock runtime on a Snapdragon
// laptop reaches the CPU and DirectML and nothing else.
//
// Two things differ from a built-in provider, and both are kept in this
// file rather than spread through the session code: the registration
// happens once per environment (onnxEnv() calls onnxRegisterEpLibraries
// right after creating one, and rebuilds the environment when the set of
// libraries changes), and a session is attached through the
// OrtEpDevice-based append (SessionOptionsAppendExecutionProvider_V2) --
// the string-keyed append knows only the built-in names and answers "not
// supported in this build" for a plugin, whatever was registered.

#ifdef ENABLE_ONNX

#include "onnx_runtime.h"
#include <onnx/onnx_peak.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Bumped by every change to the plugin configuration (onnxSetEpLibraries,
// onnxSetWinml).  onnxEnv() compares it with the generation its environment
// was registered under and recreates the environment when they differ.
uint64_t onnxEpConfigGeneration();

// The libraries a fresh environment registers: the configured set plus,
// when the Windows ML catalog is enabled, the providers it resolved (which
// may install them first -- see onnx_winml.h).  Memoized per configuration,
// so asking twice costs nothing and downloads nothing.
std::vector<OnnxEpLibrary> onnxEffectiveEpLibraries(const OrtRuntime &rt);

// Register the effective libraries on `env`, just created for `rt` (or
// emptied by onnxUnregisterEpLibraries), and record how each fared
// (onnxEpLibraryStatus()).  Also remembers which provider names the
// environment listed *before* registering, so that onnxPluginDevices can
// tell a plugin's devices from the built-in ones.  True when a plugin
// library is mapped into the process afterwards -- registered, or refused
// after loading -- which pins `rt` for the rest of it (onnxPinRuntime).
bool onnxRegisterEpLibraries(const OrtRuntime &rt, OrtEnv *env);

// Unregister from `env` every library onnxRegisterEpLibraries registered
// on it: how a new plugin set reaches an environment that must not be
// released (onnx_session.cpp, g_envUnreleasable).  Between runs only.
void onnxUnregisterEpLibraries(const OrtRuntime &rt, OrtEnv *env);

// One entry per (plugin provider, hardware device) the environment
// enumerates, accelerators first, CPU-class devices last, named from what
// the runtime reports.  Empty when no plugin registered, or the runtime
// predates the API.
std::vector<onnx_ep_info_t> onnxPluginDevices(const OrtRuntime &rt);

// Attach the plugin provider behind `ep` to `so` with `kv` as its options.
// Empty on success, otherwise the runtime's one-line reason.
std::string onnxAppendPluginDevice(
    const OrtRuntime &rt, OrtSessionOptions *so, const onnx_ep_info_t &ep,
    const std::vector<std::pair<std::string, std::string>> &kv);

// The path the plugin library `registrationName` was registered from, or
// empty when it is unknown or was not registered.  The QNN wiring uses it to
// name the backend library beside the plugin outright, instead of leaving
// the plugin to search for it.
std::string onnxEpLibraryPath(const std::string &registrationName);

// kEpTable lookup (onnx_peak.cpp): the display name, type label and device
// class clpeak gives a provider it knows.  False for a provider it does not.
bool onnxEpTableEntry(const std::string &providerKey, std::string &display,
                      std::string &typeStr, DeviceType &deviceType);

// The Windows ML configuration as set by onnxSetWinml.
bool onnxWinmlEnabled();
std::string onnxWinmlPathHint();

#endif // ENABLE_ONNX
#endif // CLPEAK_ONNX_PLUGIN_H
