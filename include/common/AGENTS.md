# include/common — Shared Neutral Headers

Public headers for types, enums, and interfaces shared by every backend.
No backend-specific includes live here.

## Quick Lookups

- Looking for the Peak base class? → `peak.h`
- Looking for benchmark enums? → `benchmark_enums.h`
- Looking for benchmark constants + calibration? → `common.h`
- Looking for CLI options struct? → `options.h`
- Looking for result output format? → `run_document.h` (model) / `docs/format-v3.md` (schema)
- Looking for logger interface? → `logger.h` (base) / `logger_text.h` (shared text formatter)
- Looking for device inventory structs? → `inventory.h`
- Looking for gating? → `peak.h` (gating is part of Peak)
- Classifying or documenting a test? → see *What a backend authors at beginTest()* below

## What a backend authors at `beginTest()`

Six things, all authored where the measurement lives — never collected into a
table somewhere else, because none of them can be recovered from anywhere else.

| field | says |
|---|---|
| `shape` | whether the readings are comparable to one another (below) |
| `axis` | what varies across them: `"vector width"`, `"data type"`, `"cache level"`, `"direction"`, `"threads"` |
| `direction` | which way is better, when the unit table's default is wrong |
| `variant` | a runtime qualifier that is *not* identity — a CPU ISA, a GPU arch |
| `description` | what the test measures (below) |
| `unit` | the token the unit table resolves (`src/common/units.cpp`) |

### `shape` — homogeneous or heterogeneous

`TestShape::Homogeneous` means the readings are interchangeable variants of one
measurement — `float` / `float2` / `float4`, `int8_dp` chain depths — so the
best of them is the test's answer and the GUI shows that one number.
`TestShape::Heterogeneous` means each reading is its own measurement — nine
GEMM datatypes, L1/L2/L3/DRAM, h2d vs d2h, ST vs MT — so the GUI shows them all
and no headline.

It cannot be inferred from anything. `wmma_fp16` has one reading and is
homogeneous; `mps-gemm-fp` has three and is not; both are tflops. The same tag
even differs by backend: a GPU's `global_memory_bandwidth` is a vector-width
sweep, the CPU's is read/copy/triad.

Heterogeneous is the default, so an unclassified test is verbose, never wrong.
Classifying one is what turns the collapsed row on.

### `variant`, not a slugged tag

A runtime qualifier belongs in `TestSpec::variant`, never appended to `tag`.
Slugging it in is lossy — `"NEON + SME2 (SVL=512b)"` and `"AMX/SME"` both
collapse to underscores — and it makes the tag differ between machines, which
breaks `--compare` across them. `variant` participates in the test's key
(`id@variant`) so two ISAs of one test stay two rows, and the GUI shows it
beside the title.

### A reading in another unit

`EmitOptions::unit` overrides the test's for one reading. That is what lets a
single heterogeneous GEMM test hold both TFLOPS and TOPS readings, instead of
the `-fp` / `-int` twins that existed only because the unit string had to
differ. `DeviceScope::beginTest()` on a tag already recorded for the device
*reopens* it and appends, so the two halves can be measured in different
category phases and still land in one test.

Two things are easy to get wrong here:

- **Skips need the unit as much as measurements do.** `skip()` has an
  `EmitOptions` overload for exactly this. An unsupported int8 row with no unit
  inherits its test's, and then claims to be flops.
- **The reopening `TestSpec` still carries the unit of the readings it is about
  to produce.** The test keeps the unit of its *first* open — that is what the
  document records — and on a category-filtered run the integer phase's open
  *is* the first, so a `tops` reopen that declared no unit would record the
  whole test in flops. Both presenters print the unit per row, so no row is
  mislabelled by the test header.

**A test's `CLPEAK_VLOG` lines belong after its `beginTest()`.** The table
streams per metric; a diagnostic emitted during the *setup* that precedes
`beginTest()` therefore appears under the previous test's readings, where it
reads as belonging to them. Open the scope first — it needs nothing from the
setup — and let the diagnostic land under its own header. The
global-bandwidth working-set line is the worked example, in all five GPU
backends. Device-scope diagnostics (a failed program build, an enumeration
dump) are the exception and correctly precede every test.

**Reopen across phases, not within one.** A family whose data types are each
measured in their own block should open ONE scope and pass it down, not open
the test per block — see the tensor-core and cooperative-matrix runners, whose
descriptors take a `logger::TestScope *`. Reopening per reading works, but
each metric flushes immediately so the label column widens down the page
(via `mergedPad`). The genuine use for a reopen is the integer phase arriving
after the rest of the run, where a new header is what you want anyway.

## Test documentation

Two levels, each authored where its own code lives — never collected into a
table somewhere else:

- **The test**: `logger::TestSpec::description`, one or two sentences on what
  the test measures, at the `beginTest()` call site.
- **One reading**: `logger::EmitOptions::description`, at the `emit()` /
  `skip()` call that produces it. The three-argument sugar is the usual form:
  `test.emit("DRAM x8", ns, "Eight reads in flight at once.")`, and
  `skip(metric, status, reason, description)` documents a reading that didn't
  happen. For a loop over variants, hang the note on the table the loop walks
  (see the `Level` struct in `src/cpu/latency.cpp`) so name and note stay
  adjacent.

Authoring natively, next to the measurement, is what makes this work at all:
CPU tags are ISA-slugged at runtime, so no static table keyed by tag could
cover them, and the same tag means different things across backends.

The strings travel on the document (`TestResult::description`,
`MetricResult::description`), so the dump round-trips them and a reopened file
explains itself.  Whitespace is collapsed to single spaces on the way in, so a
literal wrapped across source lines stays one line in the output.

Two consumers render them: the GUI behind an info glyph, and the CLI under
`--describe` (off by default — the plain output stays a table).  Undocumented
tests and readings carry empty strings and neither shows anything for them, so
documenting is purely additive.

Where a backend runs its tests through a shared runner, the strings ride
alongside the rest of the per-test data — a field on the descriptor struct
(`mtl_compute_desc_t::description`), on the entry table (ROCm's `WmmaEntry`),
or a parameter (`clPeak::runComputeTest`) — so the runner forwards them on
every path, skips included.  A phrase shared by several readings gets one
helper per backend (`mtlWidthNote()`, `vkWidthNote()`, …), never a
cross-backend table: the same label means different things in different
backends.  Vulkan's `int8_dp2` is a second *chain*, not a wider vector.

Where a test reports a value by a non-obvious convention, the description is
the place to say so — `transfer_bandwidth`'s zero-copy rows read `0.00` on
unified-memory devices, and OpenCL's test description explains that.

`src/cpu/latency.cpp` (`memory_latency`) is the worked example of both levels.

Style: plain language for someone who doesn't know the term in the metric
name; no "lower/higher is better" (the GUI already orders and scales the
readings).  Say "the device", not "the GPU": OpenCL, Vulkan and oneAPI
enumerate CPU devices too and show the same string for them -- and "the host"
for the side submitting the work, since "the CPU" is ambiguous once the device
is one.  A claim that only holds for graphics hardware names it as such
("consumer graphics parts run these many times slower than 32-bit").

Trap: the sugar overload takes `const char *`, not `std::string` — a
std::string parameter makes existing braced calls (`emit(m, v, {true})`)
ambiguous, since a braced bool also matches std::string's `initializer_list`
constructor. Pass `.c_str()` for a composed description, or use `EmitOptions`.

See also: `app/AGENTS.md` (the GUI affordance), `src/ffi/AGENTS.md` (the
`desc` / `minfo` event fields) and `logger_text.h` (`--describe` rendering).

## Key Files

| File | Purpose |
|------|---------|
| `peak.h` | `Peak` abstract base class + gating — every backend implements this |
| `benchmark_enums.h` | `Benchmark`, `Category`, `DeviceType` enums, `categoryOf()` |
| `common.h` | OS macros, tuning constants, `benchmark_config_t`, `pickIters()` calibration |
| `options.h` | `CliOptions` struct + `parseCliOptions()` / `parseCliOptionsNoExit()` declarations |
| `run_document.h` | `RunDocument`/`DeviceResult`/`TestResult`/`MetricResult` + `TestShape` + JSON save/load. The one dump format |
| `units.h` | `Quantity`, `Direction`, `UnitInfo` — resolves a unit token into symbol, quantity, SI `scale` and which way is better; `formatScaledValue()` picks the display SI prefix |
| `json.h` | Minimal JSON DOM parser (reading side only; the writers stream text) |
| `host_info.h` | `probeHost()` — the machine a run happened on, never its owner |
| `logger.h` | `LogEvent` + `logger` abstract base — result-scope API, single `onEvent()` hook, accumulated `results` |
| `logger_text.h` | `LoggerText` — indented/aligned text rendering to an injectable `std::ostream` + baseline deltas (CLI) |
| `inventory.h` | `InventoryDevice`, `BackendInventory`, `inventoryToJson()` |
| `dynlib.h` | `dynOpen()`/`dynSym()`/`dynClose()` — load-on-demand vendor libraries, so the shipped binary needs only the GPU driver |

## When You Change This Directory

- If you change the `Peak` interface → update all backend `AGENTS.md` files.
- If you add/remove a header → update this file's Key Files table.
- If you change result format → update `docs/format-v3.md`, and bump
  `RESULT_FORMAT_VERSION` in `run_document.h` only for a *breaking* change.
  Adding an optional field is not one: the loader ignores keys it doesn't know,
  and absent-means-default is the format's own convention. A bump makes every
  file already saved unreadable, the GUI's whole run history included, so it
  also means bumping `formatVersion` in
  `app/lib/src/services/run_history_store.dart`.
- If you change benchmark enums → make sure all backends handle the new enum values.
