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
//                  type, props:[{k,v}...]}
//
// Every event carries `backend`, `platform`, `device`, `driver` and
// `device_index`.  The index is part of the device's identity, not a detail of
// the device event: a name does not identify a device on its own (MoltenVK
// exposes one GPU twice, and a multi-GPU box has N identical cards), and two
// devices whose readings merge produce a test with two of everything.
//   test_begin    {..., test, title, variant, axis, category, shape,
//                  direction, quantity, unit, scale, desc, reopened}
//   metric        {..., test, variant, metric, label,
//                  value | (status, reason),
//                  unit, quantity, scale, direction,   // only when overriding
//                  minfo}
//   test_skipped  {... same header as test_begin ...,
//                  metrics:[...], status, reason}
//
// The test header arrives once, on test_begin, and a `metric` carries only
// what identifies its test plus the reading itself -- so a consumer builds
// the test node up front and appends readings to it.  `shape` says whether
// the readings may be collapsed to one number (homogeneous) or each stands
// alone (heterogeneous); `direction` which way is better; `scale` multiplies
// a value into its SI base unit, which is how a presenter picks an SI prefix.
// `reopened` marks a test_begin that resumes an already-announced test to
// append readings.
//
// A reading omits `status` when it succeeded (it has a `value` instead), and
// carries unit fields only when it overrides its test's -- the case that lets
// one test hold both TFLOPS and TOPS readings.
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
// meaningful here and are rejected.  `-o <file>` is honored at the end of the
// run exactly like the CLI, so partial results of a cancelled run still get
// saved -- with `"cancelled": true` in the document to say they are partial.
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
