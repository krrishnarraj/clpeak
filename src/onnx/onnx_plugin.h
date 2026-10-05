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
// file rather than spread through the session code: the libraries are
// registered on the environment (onnxEnv() syncs them when it creates the
// launch's one environment, and again whenever the configured set changes:
// what went is unregistered, what came is registered), and a session is
// attached through the OrtEpDevice-based append
// (SessionOptionsAppendExecutionProvider_V2) -- the string-keyed append
// knows only the built-in names and answers "not supported in this build"
// for a plugin, whatever was registered.

#ifdef ENABLE_ONNX

#include "onnx_runtime.h"
#include <onnx/onnx_peak.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Bumped by every change to the configured plugin libraries
// (onnxSetEpLibraries).  onnxEnv() compares it with the generation its
// environment was synced to and syncs it again when they differ.
uint64_t onnxEpConfigGeneration();

// Bring `env`'s plugin libraries in line with the configured set plus, when
// the Windows ML catalog is on, the providers it resolved (which may install
// them first -- see onnx_winml.h; memoized per runtime and fixed for the
// process with the rest of the runtime setup, so a later sync downloads
// nothing).  Records how each fared (onnxEpLibraryStatus()) and the devices
// registering it added, which is how onnxPluginDevices knows each device's
// library.  A library no longer wanted is unregistered (and unloaded), a new
// one registered, and one that stays keeps its answer.  The configured
// libraries go in first, and a catalog provider one of them stands for --
// one registered under its name, or one serving a provider of that name --
// is left out (OnnxEpLibraryStatus::replaces).  The environment is never
// rebuilt for this -- it lives for the process (onnx_session.cpp).  Between
// runs only: a library in use by a session cannot be unregistered.
void onnxSyncEpLibraries(const OrtRuntime &rt, OrtEnv *env);

// What identifies a plugin device's origin: the provider's own name plus
// the version it reports, when it reports one.  Not the registration name:
// that is whatever the person registering the library typed (the GUI
// suggests the provider's name, the CLI takes any), so one library would
// read differently in the two.  The listing wraps it as
// "EP plugin (<here>)" (InventoryDevice::origin); the run reports it as the
// value of its "EP plugin" prop.  One helper so the two never drift apart.
inline std::string onnxPluginOrigin(const onnx_ep_info_t &ep)
{
  if (!ep.epDevicePtr)
    return std::string();
  if (ep.pluginVersion.empty())
    return ep.providerKey;
  return ep.providerKey + " " + ep.pluginVersion;
}

// One entry per (plugin provider, hardware device) a registered library
// added to the environment, with `library` the registration it came from;
// accelerators first, CPU-class devices last, named from what the runtime
// reports.  Empty when no plugin registered, or the runtime predates the
// API.
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

#endif // ENABLE_ONNX
#endif // CLPEAK_ONNX_PLUGIN_H
