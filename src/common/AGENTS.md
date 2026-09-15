# src/common — Shared Backend-Neutral Code

Base classes, utilities, result store, and inventory shared by every backend.
The `logger` base (`logger.cpp` + `include/common/logger.h`) turns the RAII
scope API backends use into a single typed event stream (`LogEvent`) *and*
accumulates the run tree (`RunDocument`) as it goes; output channels implement
one hook — `onEvent()`. Two channels exist: `LoggerText`
(`logger_text.cpp`, indented text to an injectable `std::ostream`, used by the
desktop CLI) and `LoggerFfi` (`src/ffi/logger_ffi.cpp`, JSON over a C callback,
used by the Flutter GUI on every platform).

## Quick Lookups

- Looking for the Peak base class? → `peak.cpp` + `include/common/peak.h`
- Understanding result recording API? → `include/common/logger.h` (header only)
- Understanding result output format? → `run_document.cpp` + `include/common/run_document.h`, schema in `docs/format-v3.md`
- Understanding calibration? → `common.cpp` (`pickIters()`) + `include/common/common.h`
- Understanding gating? → `peak.cpp` + `include/common/peak.h` (gating lives in Peak)
- Adding a new backend? → The `Peak` interface is in `include/common/peak.h` (`kBackend` + `backend()`, `runAll()`, `isAllowed()`, `isDeviceSelected()`); a `Backend` enum value in `include/common/benchmark_enums.h` (its position is the run order) and a row in the backend table in `options.cpp` give it its flags; `src/registry/backend_registry.cpp` and `cmake/backends.cmake` put it in both binaries
- Understanding device inventory structs / JSON? → `inventory.cpp` + `include/common/inventory.h`

## Key Files

| File | Purpose |
|------|---------|
| `peak.cpp` | `Peak` base class: `applyOptions()` copies CLI state (gating, and the `--device` items that name `backend()`) |
| `common.cpp` | `benchmark_config_t::forDevice()`, `pickIters()` calibration, and `clpeak::requestCancel()/cancelRequested()` — cooperative cancellation observed in `Peak::isAllowed()` and the backend device loops |
| `run_document.cpp` | The result tree + its single JSON serialization. Also `RunDocument::append`, which folds each backend's logger into the document a host saves |
| `units.cpp` | The unit table: token → symbol, quantity, default direction (+ `formatScaledValue()` — SI prefix for display). Keyed by the tokens backends already pass, so adding these fields cost no backend churn |
| `json.cpp` | Recursive-descent JSON parser, classic-locale numbers. Hand-rolled: it is the only parser clpeak needs, and it is smaller than the XML/CSV line scanners it replaced |
| `host_info.cpp` | `probeHost()`. Deliberately records no hostname, username or serial — result files get shared |
| `logger.cpp` | Base `logger` class: result-scope API (`emit()`/`skip()`/`skipAll()`) dispatching `LogEvent`s to the single `onEvent()` hook. Also whitespace-collapses the documentation strings to one line, since all three dump formats are line-oriented |
| `logger_text.cpp` | `LoggerText` — renders the event stream as indented/aligned text with per-row units + baseline deltas, and under `--describe` the wrapped test/reading documentation (desktop CLI) |
| `inventory.cpp` | `inventoryToJson()` — device inventory JSON serializer for the GUI catalog; `printInventory()` — the one `--list-devices` format for every backend, each device line starting with its `backend:index` token |
| `options.cpp` | `parseCliOptions()` (CLI, exits on error) + `parseCliOptionsNoExit()` (embedded, used by `src/ffi`); the backend table (`backendInfo()`: name, flag, built-in), the category and test flag tables, and `--help` generated from them |
| `console_mute.cpp` | `clpeak::ScopedConsoleMute` — silences stdout+stderr at the fd level for a scope, so vendor runtimes that print below any log level (hipBLASLt's Tensile internals, the ONNX schema registry) cannot wreck the results table. A no-op under `--verbose` |
| `coreml_cache.mm` / `coreml_cache_stub.cpp` | `clpeak::purgeCoreMLCompileCache()` — removes this process's Core ML runtime cache (`~/Library/Caches/<process>/com.apple.e5rt.e5bundlecache`), which keeps every compiled model with its weights and never evicts; called by the Core ML backend per session and by the ONNX backend per provider. Apple-only, a stub elsewhere |
| `dynlib.cpp` | `dynOpen()`/`dynSym()`/`dynClose()` — load-on-demand vendor libraries (cuBLASLt / hipBLASLt / rocBLAS) so the shipped binary needs only the driver |

## Scope invariant: one open test at a time

A `TestScope` must be closed before a sibling one opens. `LoggerText` streams
each metric as it arrives (with `mergedPad` for incremental alignment), so an
overlapping `beginTest` would interleave rows from two tests — the saved file
would still be complete in the `RunDocument`, the table wouldn't.

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
| Metal, ONNX, Core ML | neither — unified memory, so the memory proxy would be system RAM |

Never pass memory that is not the device's own (an iGPU heap, a unified-memory
part): the proxy assumes the last level is a fixed fraction of board memory, and
that ratio only holds for discrete parts.

Every backend prints the working set it settled on under `--verbose`. That line
is the first thing to look at when a bandwidth number comes in above what the
memory can physically do — see `src/cpu/AGENTS.md` for the same rule applied to
the CPU's own STREAM arrays, which are sized separately.

## Runtimes that compile models leave things on disk; a backend cleans up its own

Every backend that hands work to a vendor runtime has to know what that
runtime persists, because a benchmark compiles hundreds of one-off models
per run and nothing about "a cache" assumes that.  What was checked:

| runtime | persists | bounded? | cleaned by |
|---|---|---|---|
| Core ML (E5RT), reached by the Core ML backend, the ONNX CoreML provider, and any future LiteRT Core ML delegate | every compiled model **with its weights, twice**, under `~/Library/Caches/<process or bundle id>/com.apple.e5rt.e5bundlecache` (inside the app container on iOS) | **no** — 292 GB / 12k entries accumulated; a 16384-cube session adds 4.5 GB; macOS reports it "available" as purgeable while writes get ENOSPC | `clpeak::purgeCoreMLCompileCache()` (`coreml_cache.mm`): per Core ML session, per ONNX provider, and as a backstop after every backend in the CLI and FFI run loops |
| ONNX Runtime CoreML EP's own `.mlpackage`, and its profiling JSON | temp directory, per session | removed by ORT on release / `removeProfileArtifacts` | the ONNX backend |
| ONNX TensorRT / OpenVINO / QNN / NNAPI engine or context caches | only when the provider option enabling them is set | off — clpeak sets none | n/a |
| CUDA / OpenCL PTX JIT (`~/.nv/ComputeCache`), Mesa shader cache, Metal shader cache, DirectML, driver pipeline caches | small compiled kernels | yes — driver-capped or OS-managed, kilobytes per kernel here | the driver |
| SYCL persistent JIT cache | only with `SYCL_CACHE_PERSISTENT=1` | off by default | n/a |

The rule for a new backend: find out where its runtime writes, and if the
answer is "per model, unbounded", clean it in the session teardown and add
a row here.  Temporary files a backend writes itself carry the process id
in their name and are swept at the next start (`src/coreml/coreml_session.mm`
is the pattern), because a process that dies mid-session cleans nothing.

## When You Change This Directory

- If you change the `Peak` interface → update `include/common/peak.h` + all backend `AGENTS.md` files.
- If you add a utility function → update this file's Key Files table.
- If you change the result format → update `docs/format-v3.md` and
  `include/common/run_document.h`, and bump `RESULT_FORMAT_VERSION` only for a
  *breaking* change. The loader skips keys it doesn't recognise and
  absent-means-default is the format's own convention, so an optional field
  round-trips without a bump — and a bump makes every file a user has already
  saved unreadable, the GUI's whole run history included.
- If you change `LogEvent` or the event kinds → update `src/ffi/logger_ffi.cpp` (JSON mirror) and the Dart decoder `app/lib/src/ffi/clpeak_events.dart`.
- If you add/remove a file → update `src/common/CMakeLists.txt` and this file.
