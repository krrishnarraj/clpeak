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
| `common.cpp` | `benchmark_config_t::forDevice()`, `pickIters()` calibration, and `clpeak::requestCancel()/cancelRequested()` — cooperative cancellation observed in `Peak::isAllowed()` and the backend device loops |
| `result_store.cpp` | `ResultEntry`/`ResultStore` + `DeviceInfo` serialization: JSON, CSV, XML. Test tags are **not** reversible to display names (`emitVariants` slugs a runtime ISA onto the base tag), so `ResultEntry::display` must be persisted, never re-derived |
| `logger.cpp` | Base `logger` class: result-scope API (`emit()`/`skip()`/`skipAll()`) dispatching `LogEvent`s to the single `onEvent()` hook. Also whitespace-collapses the documentation strings to one line, since all three dump formats are line-oriented |
| `logger_text.cpp` | `LoggerText` — renders the event stream as indented/aligned text + baseline deltas, and under `--describe` the wrapped test/reading documentation (desktop CLI) |
| `inventory.cpp` | `inventoryToJson()` — device inventory JSON serializer (no backend includes) |
| `options.cpp` | `parseCliOptions()` (CLI, exits on error) + `parseCliOptionsNoExit()` (embedded, used by `src/ffi`) |
| `console_mute.cpp` | `clpeak::ScopedConsoleMute` — silences stdout+stderr at the fd level for a scope, so vendor runtimes that print below any log level (hipBLASLt's Tensile internals, the ONNX schema registry) cannot wreck the results table. A no-op under `--verbose` |
| `dynlib.cpp` | `dynOpen()`/`dynSym()`/`dynClose()` — load-on-demand vendor libraries (cuBLASLt / hipBLASLt / rocBLAS) so the shipped binary needs only the driver |

## Scope invariant: one open test at a time

A `TestScope` must be closed before a sibling one opens. `LoggerText` buffers a
test's metric rows until `TestEnd` (it needs them all to align the column), so
an overlapping `beginTest` silently drops the pending rows from the text output
while `ResultStore` keeps them — JSON/CSV look complete, the table doesn't.

Three guards make that impossible now: `beginTest()` implicitly closes an open
test and emits a `Note` naming both tags; `TestScope::end()` only emits
`TestEnd` while it is still the open scope, so every `TestBegin` has exactly one
`TestEnd` (which the FFI/GUI channel depends on); and `renderTestBegin()`
flushes rather than clears. Still prefer calling `end()` explicitly (see
`runCacheBandwidth`) — it is the intent-revealing form and avoids the note.

## The global-bandwidth working set must outgrow the cache

Every backend warms one buffer and then re-reads that same buffer for the whole
timed phase, so anything that stays cache-resident is counted as memory traffic
it never was. `benchmark_config_t::forDevice()` owns the sizing for all of them;
the reasoning for each constant is at the constant.

A backend passes what its API knows, and nothing more:

| passes | why |
|---|---|
| CUDA, oneAPI, OpenCL | cache size only — their query returns the true last level |
| ROCm | cache size **and** board memory — HIP's `l2CacheSize` excludes the MALL / Infinity Cache |
| Vulkan | device-local heap only, and only for a discrete GPU — Vulkan has no cache query at all |
| Metal, ONNX | neither — unified memory, so the memory proxy would be system RAM |

Never pass memory that is not the device's own (an iGPU heap, a unified-memory
part): the proxy assumes the last level is a fixed fraction of board memory, and
that ratio only holds for discrete parts.

Every backend prints the working set it settled on under `--verbose`. That line
is the first thing to look at when a bandwidth number comes in above what the
memory can physically do — see `src/cpu/AGENTS.md` for the same rule applied to
the CPU's own STREAM arrays, which are sized separately.

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
