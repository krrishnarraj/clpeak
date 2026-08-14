# include/common — Shared Neutral Headers

Public headers for types, enums, and interfaces shared by every backend.
No backend-specific includes live here.

## Quick Lookups

- Looking for the Peak base class? → `peak.h`
- Looking for benchmark enums? → `benchmark_enums.h`
- Looking for benchmark constants + calibration? → `common.h`
- Looking for CLI options struct? → `options.h`
- Looking for result output format? → `result_store.h`
- Looking for logger interface? → `logger.h` (base) / `logger_text.h` (shared text formatter)
- Looking for device inventory structs? → `inventory.h`
- Looking for gating? → `peak.h` (gating is part of Peak)
- Documenting a test for non-expert readers? → see *Test documentation* below

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

The strings travel on the result rows (`ResultEntry::description`,
`::metricDescription`), so all three dump formats round-trip them and a
reopened file explains itself.  Whitespace is collapsed to single spaces on
the way in, so the line-oriented CSV/XML/JSON writers stay safe no matter how
the literal is wrapped in source.

Two consumers render them: the GUI behind an info glyph, and the CLI under
`--describe` (off by default — the plain output stays a table).  Undocumented
tests and readings carry empty strings and neither shows anything for them, so
documenting is purely additive.

Coverage is per backend, one backend at a time.  **`src/cpu/` is fully
documented and is the reference** — every `TestSpec` there carries a
`description`, and it shows all three authoring shapes: the plain literal
(`microarch.cpp`), a note table walked by a loop (`latency.cpp`,
`bandwidth.cpp`), and one note shared by a family of tests written once at the
runner (`compute_common.h`'s `ST`/`MT` notes, which serve ~25 tests).
**`src/metal/` is done too**, and shows the fourth shape: a backend whose tests
run through a descriptor struct carries the strings on that struct
(`mtl_compute_desc_t::description`, `mtl_compute_variant_t::description`) so
the shared runner forwards them — the same move the OpenCL / Vulkan / CUDA /
ROCm / oneAPI backends will need, since they are built the same way and are
still undocumented.

Style: plain language for someone who doesn't know the term in the metric
name; no "lower/higher is better" (the GUI already orders and scales the
readings).

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
| `result_store.h` | `ResultEntry`/`ResultStore` + JSON/CSV/XML serialization + `resultsToJson()` |
| `logger.h` | `LogEvent` + `logger` abstract base — result-scope API, single `onEvent()` hook, accumulated `results` |
| `logger_text.h` | `LoggerText` — indented/aligned text rendering to an injectable `std::ostream` + baseline deltas (CLI) |
| `inventory.h` | `InventoryDevice`, `BackendInventory`, `inventoryToJson()` |

## When You Change This Directory

- If you change the `Peak` interface → update all backend `AGENTS.md` files.
- If you add/remove a header → update this file's Key Files table.
- If you change result format → bump `RESULT_FORMAT_VERSION` in `result_store.h`.
  Purely additive fields don't need it: every loader ignores keys/attributes/
  trailing CSV columns it doesn't know, which is how `display`, the `devices`
  block and the documentation fields all landed on v2.
- If you change benchmark enums → make sure all backends handle the new enum values.
