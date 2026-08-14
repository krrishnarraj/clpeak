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
| `result_store.cpp` | `ResultEntry`/`ResultStore` + `DeviceInfo` serialization: JSON, CSV, XML |
| — | Test tags are *not* reversible to display names: `emitVariants` (`src/cpu/compute_common.h`) slugs a runtime ISA onto the base tag, so `ResultEntry::display` must be persisted, never re-derived |
| `logger.cpp` | Base `logger` class: result-scope API (`emit()`/`skip()`/`skipAll()`) dispatching `LogEvent`s to the single `onEvent()` hook |
| — | Documentation (`TestSpec::description`, `EmitOptions::description`) is whitespace-collapsed to one line here and stamped onto the rows, since all three dump formats are line-oriented — see `include/common/AGENTS.md` |
| `logger_text.cpp` | `LoggerText` — renders the event stream as indented/aligned text + baseline deltas, and under `--describe` the wrapped test/reading documentation (desktop CLI) |
| `inventory.cpp` | `inventoryToJson()` — device inventory JSON serializer (no backend includes) |
| `options.cpp` | `parseCliOptions()` (CLI, exits on error) + `parseCliOptionsNoExit()` (embedded, used by `src/ffi`) |
| `common.cpp` (also) | `clpeak::requestCancel()/cancelRequested()` — cooperative run cancellation observed in `Peak::isAllowed()` + backend device loops |

## Scope invariant: one open test at a time

A `TestScope` must be closed before a sibling one opens. `LoggerText` buffers a
test's metric rows until `TestEnd` (it needs them all to align the column), so
an overlapping `beginTest` used to *discard* the pending rows — silently, since
the `ResultStore` kept them and only the text output lost rows (JSON/CSV looked
complete). Two guards now make that impossible:

- `DeviceScope::beginTest()` detects an already-open test, calls
  `logger::closeOpenTest()` so the previous test ends cleanly, and emits a
  `Note` naming both tags. The pre-existing `assert` still fails fast in debug
  builds; the note + implicit close are what keep release builds correct.
- `TestScope::end()` only emits `TestEnd` when it is still the open scope
  (matched by `curTestSeq`), so the implicitly-closed scope's destructor cannot
  emit a second `TestEnd` — every `TestBegin` has exactly one `TestEnd` in the
  stream, which the FFI/GUI channel depends on.
- `LoggerText::renderTestBegin()` flushes rather than clears, as defence in
  depth: no measured row can be dropped even if the scope layer is bypassed.

Prefer still calling `end()` explicitly (see `runCacheBandwidth`) — it is the
intent-revealing form and avoids the note.

## When You Change This Directory

- If you change the `Peak` interface → update `include/common/peak.h` + all backend `AGENTS.md` files.
- If you add a utility function → update this file's Key Files table.
- If you change the result format → update `include/common/result_store.h`, and
  bump `RESULT_FORMAT_VERSION` only for a *breaking* change. Both loaders skip
  elements/keys they don't recognise, so purely additive fields (device `<prop>`
  was one) round-trip without a bump — and a bump would make every file a user
  has already saved unreadable, including the GUI's whole run history.
- If you change `LogEvent` or the event kinds → update `src/ffi/logger_ffi.cpp` (JSON mirror) and the Dart decoder `app/lib/src/ffi/clpeak_events.dart`.
- If you add/remove a file → update `src/common/CMakeLists.txt` and this file.
