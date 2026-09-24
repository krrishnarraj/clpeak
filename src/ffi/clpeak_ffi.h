#ifndef CLPEAK_FFI_H
#define CLPEAK_FFI_H

// ── clpeak C ABI ────────────────────────────────────────────────────────────
//
// The single native bridge for every clpeak GUI (Flutter desktop, Android,
// iOS — all consume this via Dart FFI).  Exposes device enumeration, a
// blocking benchmark launch with a streaming event callback, cooperative
// cancellation, and saved-result loading.
//
// String ownership: functions returning `char *` return a malloc'd UTF-8
// string the caller must release with clpeak_free_string().  `const char *`
// returns are static and must not be freed.

#if defined(_WIN32)
#define CLPEAK_FFI_EXPORT __declspec(dllexport)
#else
#define CLPEAK_FFI_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---- Version / catalog ------------------------------------------------------

// clpeak version string (e.g. "2.0.18-3-gabc1234").  Static; do not free.
CLPEAK_FFI_EXPORT const char *clpeak_version(void);

// Device catalog for every backend compiled into this library, as the
// inventoryToJson() document (include/common/inventory.h has the keys):
//   {"backends":[{"name","flag","available",info?,reason?,notes?,
//     "platforms":[{"index","name",
//       "devices":[{"index","name","type",arch?,driver?,api?,...}]}]}]}
// `flag` is the backend's command-line name; `--devices <flag>:<index>`
// names a device, and a run's argv narrows to exactly the devices listed.
// `reason` says why an unavailable backend is unavailable; `info` is a
// backend-level fact (the ONNX Runtime version, the OS release).
CLPEAK_FFI_EXPORT char *clpeak_copy_backend_catalog_json(void);

CLPEAK_FFI_EXPORT void clpeak_free_string(char *s);

// Choose which onnxruntime shared library the ONNX backend loads, ahead of
// the platform's conventional names.  Pass an
// absolute path, or NULL/"" to go back to searching.  This is the settings-
// screen entry point: the CLI grammar's `--onnx-lib` cannot serve it, because
// clpeak_copy_backend_catalog_json() takes no arguments and enumeration is
// what loads the runtime -- so the choice has to be in place before the
// catalog is asked for.
//
// Naming a different library than the one already loaded takes effect on the
// next enumeration or run -- unless the current runtime is pinned: a plugin
// execution-provider library (one Windows ML resolved, or one
// clpeak_set_onnx_ep_libraries() named) has been loaded into it, or it
// cannot release its environment without crashing.  That runtime then stays
// for the life of the process, and the choice waits for the next start
// (clpeak_copy_onnx_status_json's `pendingRuntime`).  Call it between runs
// only.  A no-op on builds that link ONNX Runtime statically (iOS) or omit
// the backend entirely.
CLPEAK_FFI_EXPORT void clpeak_set_onnx_library(const char *path);

// Plugin execution-provider libraries to register on the ONNX Runtime
// environment (ONNX Runtime 1.22+; src/onnx/onnx_plugin.h): `spec` is one
// `NAME=PATH` per line ('\n'-separated), NAME being the registration name
// the provider expects ("QNNExecutionProvider" for Qualcomm's plugin) and
// PATH the library -- absolute, or on Android a bare soname the APK's lib
// dir resolves.  Lines starting with '!' name an implicit library: one the
// app bundles on the off-chance the hardware is there, whose failure to
// register goes to the verbose log rather than the run's notes.  NULL/""
// clears the set.  Between runs only, like clpeak_set_onnx_library(): the
// environment is rebuilt on the next enumeration or run.
CLPEAK_FFI_EXPORT void clpeak_set_onnx_ep_libraries(const char *spec);

// Windows ML's execution-provider catalog (Windows 11 24H2+): when
// `enabled`, the vendor providers the catalog installs from the Microsoft
// Store are registered as plugin libraries on the next enumeration or run.
// `path` names Microsoft.Windows.AI.MachineLearning.dll or its directory,
// or is NULL/"" to search beside the loaded runtime and the executable; with
// a path and no library chosen through clpeak_set_onnx_library(), the
// onnxruntime.dll beside the catalog becomes the runtime.  Installing a
// provider is a download, which is why this is a switch and not a default.
// Accepted everywhere; off Windows the status reports the catalog as
// unavailable.
CLPEAK_FFI_EXPORT void clpeak_set_onnx_winml(int enabled, const char *path);

// State of the ONNX Runtime, for a settings screen to report back with:
//   {"available":bool,"linkedIn":bool,"version":str,"path":str,"error":str,
//    "epLibraries":[{"name":str,"path":str,"named":bool,"registered":bool,
//                    "error":str}],
//    "winml":{"enabled":bool,"path":str,"error":str},
//    "pendingRuntime"?:{"path":str,"reason":str}}
// `linkedIn` means the runtime is built into this binary (iOS) and
// clpeak_set_onnx_library() has nothing to do.  `path` is what was loaded,
// the resolved file even when it was found by name (empty only when
// statically linked).  `error` says why nothing loaded --
// naming a library that cannot be opened is the ordinary way to get here.
// `epLibraries` is what the last environment registered, so a library set
// since the last enumeration is absent until the next one; `winml.path` is
// the catalog DLL that answered and `winml.error` why it did not.  When
// enabled but nothing has resolved the catalog yet both are empty --
// pending until the next enumeration or run, like `epLibraries`.
// `pendingRuntime`, present only then, is a library chosen after the loaded
// runtime was pinned (see clpeak_set_onnx_library): `path` loads at the next
// start (empty = the default search), and `reason` is the clause saying why
// not now.
// {"available":false,"error":"ONNX backend not built in"} without one.
CLPEAK_FFI_EXPORT char *clpeak_copy_onnx_status_json(void);

// The same two entry points for LiteRT: which libLiteRt to load (absolute
// path, or NULL/"" to search the conventional names) and, separately, the
// directory holding the NPU dispatch / compiler-plugin libraries and the
// vendor runtime (NULL/"" = beside the LiteRT library).  Between runs only;
// no-ops on a build without the backend.  On iOS the runtime is dlopen'd
// from the app bundle's own Frameworks directory, the one place the platform
// loads a library from, so a path named here could never be loaded there
// and the settings screen does not offer one.
CLPEAK_FFI_EXPORT void clpeak_set_litert_library(const char *path);
CLPEAK_FFI_EXPORT void clpeak_set_litert_npu_dir(const char *dir);

// Android: a writable directory (the app's support directory) where the
// backend stages one vendor's NPU shims when the APK carries several.
// LiteRT loads the first libLiteRtDispatch_* it lists in a directory, so
// the shims for this SoC (ro.soc.manufacturer) get `<dir>/<vendor>/` of
// links to the packaged files, remade at every launch.  Before enumeration;
// a no-op elsewhere and with one vendor or none packaged.
CLPEAK_FFI_EXPORT void clpeak_set_litert_npu_stage_dir(const char *dir);

// State of the LiteRT runtime, for a settings screen:
//   {"available":bool,"version":str,"path":str,"error":str}
// `version` is the ABI version this build was compiled against -- LiteRT
// exposes no runtime version string.  {"available":false,"error":"LiteRT
// backend not built in"} without the backend.
CLPEAK_FFI_EXPORT char *clpeak_copy_litert_status_json(void);

// ---- Event stream -------------------------------------------------------------

// Every run event arrives as one malloc'd UTF-8 JSON document.  OWNERSHIP
// TRANSFERS to the callee: it must release the string with
// clpeak_free_string() once consumed.  This makes the callback safe for
// asynchronous consumers (Dart NativeCallable.listener) that decode the
// payload after the native call has already returned.
//
// The documents mirror LogEvent (include/common/logger.h).  Kinds ("t"):
//   backend_begin {backend}
//   device        {backend, platform, device, driver, platform_index,
//                  type, props:[{k,v}...]}
//
// Every event carries `backend`, `platform`, `device`, `driver` and
// `device_index`.  The index is part of the device's identity, not a detail of
// the device event: a name does not identify a device on its own (MoltenVK
// exposes one GPU twice, and a multi-GPU box has N identical cards), and two
// devices whose readings merge produce a test with two of everything.
//   test_begin    {..., test, title, variant, axis, category, shape,
//                  direction, quantity, unit, desc, reopened}
//   metric        {..., test, variant, metric, label,
//                  value | (status, reason),
//                  unit, quantity, direction,   // only when overriding
//                  minfo}
//   test_skipped  {... same header as test_begin ...,
//                  metrics:[...], status, reason}
//
// The test header arrives once, on test_begin, and a `metric` carries only
// what identifies its test plus the reading itself -- so a consumer builds
// the test node up front and appends readings to it.  `shape` says whether
// the readings may be collapsed to one number (homogeneous) or each stands
// alone (heterogeneous); `direction` which way is better; `unit` is an
// SI base unit (flops, ops, bps, texels, s) and the presenter picks G/T/P/µ/n
// prefixes from it.  `reopened` marks a test_begin that resumes an already-
// announced test to append readings.
//
// A reading omits `status` when it succeeded (it has a `value` instead), and
// carries unit fields only when it overrides its test's -- the case that lets
// one test hold both flops and ops readings.
//
// `desc` explains what the test measures and `minfo` what one reading means;
// both are empty for tests and readings that carry no documentation.  A
// reading's note travels with the reading, never up-front, because that is
// where it is authored (logger::EmitOptions::description).
//   test_end      {}          device_end {}          backend_end {}
//   log           {..., test, variant, level, source, elapsed_s, message}
//   done          {status, cancelled}   // ALWAYS the last event of a launch
//
// `log` is one line of the run's diagnostic stream -- the same entry the
// saved document holds in its `log` array (docs/format-v3.md): `level` is
// error | warning | info | debug, `source` names the library a message was
// relayed from ("onnxruntime", "vulkan", "opencl", "console") or is empty
// for clpeak's own, `elapsed_s` is seconds since the run started, and the
// scope fields say where it fired.  Debug entries arrive only when argv has
// --verbose.  The scope fields are empty for a message outside any backend
// (a rejected argument, a backend not in this build).
//
// Callbacks fire on the thread that called clpeak_launch(), except `log`
// events relayed from a vendor runtime's own logging callback or a console
// capture, which fire on whatever thread produced them.  The Dart listener
// (NativeCallable.listener) is built for exactly that.
typedef void (*ClpeakEventCallback)(void *user_data, char *event_json);

// ---- Run -----------------------------------------------------------------------

#define CLPEAK_RUN_OK         0   /* all backends completed                  */
/* > 0: OR'd backend error statuses (driver init / runtime failures)         */
#define CLPEAK_RUN_BAD_ARGS  (-1) /* argv rejected; nothing ran              */
#define CLPEAK_RUN_CANCELLED (-2) /* clpeak_request_cancel() honored         */
#define CLPEAK_RUN_BUSY      (-3) /* another launch is already in progress   */

// Run benchmarks.  Blocking — call from a worker thread; events stream via
// on_event as they happen and a final `done` event is emitted before this
// returns (including on bad args).  argv follows the CLI flag grammar
// (src/common/options.cpp); --help/--version/--list-devices are not
// meaningful here and are rejected.  `-o <file>` is honored at the end of the
// run exactly like the CLI, so partial results of a cancelled run still get
// saved -- with `"cancelled": true` in the document to say they are partial.
// While the run is in flight, `<file>` with `.json` swapped for `.log` holds
// the diagnostic stream so far, one JSON object per line (run_log.h); it is
// removed once the document is written, so one left behind means the process
// died mid-run and the sidecar is that run's only record.
// Never calls exit().
CLPEAK_FFI_EXPORT int clpeak_launch(int argc, const char **argv,
                                    ClpeakEventCallback on_event,
                                    void *user_data);

// ---- Cancellation ----------------------------------------------------------------

// Request cooperative cancellation of the in-flight launch.  Observed at
// test boundaries (the currently-running test finishes first); remaining
// tests and devices are skipped silently.  Safe to call from any thread.
// The flag auto-resets at the start of the next clpeak_launch().
CLPEAK_FFI_EXPORT void clpeak_request_cancel(void);

// ---- Saved results ------------------------------------------------------------------
//
// There is no loader here.  Result files are JSON in exactly the shape a
// consumer wants (docs/format-v3.md), so the GUI reads them directly rather
// than round-tripping a file through this library -- which also means run
// history stays readable when the native library cannot be loaded at all.

#ifdef __cplusplus
}
#endif

#endif // CLPEAK_FFI_H
