# src/common — Shared Backend-Neutral Code

Base classes, utilities, result store, and inventory shared by every backend.
The `logger` base (`logger.cpp` + `include/common/logger.h`) turns the RAII
scope API backends use into a single typed event stream (`LogEvent`); output
channels implement one hook — `onEvent()`. Two channels exist: `LoggerText`
(`logger_text.cpp`, indented text to an injectable `std::ostream`, used by the
desktop CLI) and `LoggerFfi` (`src/ffi/logger_ffi.cpp`, JSON over a C callback,
used by the Flutter GUI on every platform).

## Quick Lookups

- Looking for the Peak base class? → `peak.cpp` + `include/common/peak.h`
- Understanding result recording API? → `include/common/logger.h` (header only)
- Understanding result output format? → `result_store.cpp` + `include/common/result_store.h`
- Understanding calibration? → `common.cpp` (`pickIters()`) + `include/common/common.h`
- Understanding gating? → `peak.cpp` + `include/common/peak.h` (gating lives in Peak)
- Adding a new backend? → The `Peak` interface is in `include/common/peak.h`
- Understanding device inventory structs / JSON? → `inventory.cpp` + `include/common/inventory.h`

## Key Files

| File | Purpose |
|------|---------|
| `peak.cpp` | `Peak` base class: `applyOptions()` copies CLI state (including gating) |
| `common.cpp` | `benchmark_config_t::forDevice()`, `pickIters()` calibration |
| `result_store.cpp` | `ResultEntry`/`ResultStore` serialization: JSON, CSV, XML |
| `logger.cpp` | Base `logger` class: result-scope API (`emit()`/`skip()`/`skipAll()`) dispatching `LogEvent`s to the single `onEvent()` hook |
| `logger_text.cpp` | `LoggerText` — renders the event stream as indented/aligned text + baseline deltas (desktop CLI) |
| `inventory.cpp` | `inventoryToJson()` — device inventory JSON serializer (no backend includes) |
| `options.cpp` | `parseCliOptions()` (CLI, exits on error) + `parseCliOptionsNoExit()` (embedded, used by `src/ffi`) |
| `common.cpp` (also) | `clpeak::requestCancel()/cancelRequested()` — cooperative run cancellation observed in `Peak::isAllowed()` + backend device loops |

## When You Change This Directory

- If you change the `Peak` interface → update `include/common/peak.h` + all backend `AGENTS.md` files.
- If you add a utility function → update this file's Key Files table.
- If you change the result format → update `include/common/result_store.h` and bump `RESULT_FORMAT_VERSION`.
- If you change `LogEvent` or the event kinds → update `src/ffi/logger_ffi.cpp` (JSON mirror) and the Dart decoder `app/lib/src/ffi/clpeak_events.dart`.
- If you add/remove a file → update `src/common/CMakeLists.txt` and this file.
