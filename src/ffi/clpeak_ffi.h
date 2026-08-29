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
// inventoryToJson() document:
//   {"backends":[{"name","available","platforms":[{"index","name",
//     "devices":[{"index","name","type",...}]}]}]}
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
// next enumeration or run.  Call it between runs only.  A no-op on builds
// that link ONNX Runtime statically (iOS) or omit the backend entirely.
CLPEAK_FFI_EXPORT void clpeak_set_onnx_library(const char *path);

// State of the ONNX Runtime, for a settings screen to report back with:
//   {"available":bool,"linkedIn":bool,"version":str,"path":str,"error":str}
// `linkedIn` means the runtime is built into this binary (iOS) and
// clpeak_set_onnx_library() has nothing to do.  `path` is what was loaded,
// empty when it was found by name.  `error` says why nothing loaded --
// naming a library that cannot be opened is the ordinary way to get here.
// {"available":false,"error":"ONNX backend not built in"} without one.
CLPEAK_FFI_EXPORT char *clpeak_copy_onnx_status_json(void);

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
//                  device_index, props:[{k,v}...]}
//   test_begin    {backend,..., test, display, unit, category, desc}
//   metric        {backend, platform, device, driver, category, test,
//                  display, metric, unit, value, status, reason, sub, desc,
//                  minfo}
//   test_skipped  {..., metrics:[...], status, reason, desc}
//
// `desc` explains what the test measures and `minfo` what one reading means;
// both are empty for tests and readings that carry no documentation.  A
// reading's note travels with the reading, never up-front, because that is
// where it is authored (logger::EmitOptions::description).
//   test_end      {}          device_end {}          backend_end {}
//   note          {message}
//   done          {status, cancelled}   // ALWAYS the last event of a launch
//
// Callbacks fire on the thread that called clpeak_launch().
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
// meaningful here and are rejected.  --xml-file/--json-file/--csv-file are
// honored at the end of the run exactly like the CLI, so partial results of
// a cancelled run still get saved.  Never calls exit().
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

// Load a previously-saved result file (XML / JSON / CSV, format_version 2)
// and return it re-serialized as the saveJson document:
//   {"format_version":2,"clpeak_version":...,"os":...,
//    "devices":[{backend, platform, device, driver, props:[{k,v}...]}],
//    "entries":[...]}
// "devices" carries the same per-device detail as the live `device` event, so
// a reloaded run shows what a live one does.  It is absent for CSV (which
// cannot store it) and for files written before it existed.
// Returns NULL when the file is unreadable, rejected, or empty.
CLPEAK_FFI_EXPORT char *clpeak_load_result_file_json(const char *path);

#ifdef __cplusplus
}
#endif

#endif // CLPEAK_FFI_H
