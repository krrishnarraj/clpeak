# clpeak GUI

One Flutter app for Android, iOS, macOS, Linux, and Windows, driving the
clpeak benchmark backends through the `src/ffi` C-ABI bridge (Dart FFI).

Features: system/device dashboard, Quick/Full/Custom runs (device, category,
and time-budget selection), live streaming results with cancellation, and an
auto-saved run history (canonical clpeak XML) with rename/export/delete.

See [AGENTS.md](AGENTS.md) for per-platform build instructions, the
dev-loop `CLPEAK_FFI_PATH` override, and the list of hand-edited generated
files.
