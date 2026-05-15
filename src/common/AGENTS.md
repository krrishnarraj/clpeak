# src/common — Shared Backend-Neutral Code

Base classes, utilities, result store, calibration, and inventory shared
by every backend.

## Quick Lookups

- Looking for the Peak base class? → `peak.cpp` + `include/peak.h`
- Understanding result recording? → `logger.cpp` + `include/logger.h`
- Understanding result output format? → `result_store.cpp` + `include/result_store.h`
- Understanding CLI option gating? → `backend_gating.cpp` + `include/backend_gating.h`
- Understanding calibration? → `calibrate.cpp` + `include/calibrate.h`
- Adding a new backend? → The `Peak` interface is in `include/peak.h`
- Understanding device enumeration? → `inventory.cpp` + `include/inventory.h`

## Key Files

| File | Purpose |
|------|---------|
| `peak.cpp` | `Peak` base class: constructor, `applyOptions()` common logic |
| `common.cpp` | `benchmark_config_t::forDevice()`, `Timer`, `populate()`, utilities |
| `logger.cpp` | `logger` class: result-scope API, `recordSkip`, stdout printing |
| `result_store.cpp` | `ResultEntry`/`ResultStore` serialization: JSON, CSV, XML |
| `backend_gating.cpp` | `BackendGating::copyFrom()` — copies CLI gating state |
| `calibrate.cpp` | `pickIters()` — runtime iteration calibration |
| `inventory.cpp` | `enumerateAllBackends()`, `inventoryToJson()` — device listing |
| `options.cpp` | `parseCliOptions()` — CLI argument parsing |

## When You Change This Directory

- If you change the `Peak` interface → update `include/peak.h` + all backend `AGENTS.md` files.
- If you add a utility function → update this file's Key Files table.
- If you change the result format → update `include/result_store.h` and bump `RESULT_FORMAT_VERSION`.
- If you add/remove a file → update the root `cmake/sources.cmake` and this file.
