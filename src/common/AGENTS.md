# src/common — Shared Backend-Neutral Code

Base classes, utilities, result store, calibration, and inventory shared
by every backend. Does NOT contain a logger implementation — each consumer
(desktop CLI, Android) provides its own (`src/cli/logger.cpp` or
`android/.../logger_android.cpp`).

## Quick Lookups

- Looking for the Peak base class? → `peak.cpp` + `include/common/peak.h`
- Understanding result recording API? → `include/common/logger.h` (header only)
- Understanding result output format? → `result_store.cpp` + `include/common/result_store.h`
- Understanding CLI option gating? → `backend_gating.cpp` + `include/common/backend_gating.h`
- Understanding calibration? → `calibrate.cpp` + `include/common/calibrate.h`
- Adding a new backend? → The `Peak` interface is in `include/common/peak.h`
- Understanding device inventory structs / JSON? → `inventory.cpp` + `include/common/inventory.h`

## Key Files

| File | Purpose |
|------|---------|
| `peak.cpp` | `Peak` base class: constructor, `applyOptions()` common logic |
| `common.cpp` | `benchmark_config_t::forDevice()`, `Timer`, `populate()`, utilities |
| `result_store.cpp` | `ResultEntry`/`ResultStore` serialization: JSON, CSV, XML |
| `logger.cpp` | Base `logger` class: result-scope API, `emit()`, `recordSkip` |
| `backend_gating.cpp` | `BackendGating::copyFrom()` — copies CLI gating state |
| `calibrate.cpp` | `pickIters()` — runtime iteration calibration |
| `inventory.cpp` | `inventoryToJson()` — device inventory JSON serializer (no backend includes) |
| `options.cpp` | `parseCliOptions()` — CLI argument parsing |

## When You Change This Directory

- If you change the `Peak` interface → update `include/common/peak.h` + all backend `AGENTS.md` files.
- If you add a utility function → update this file's Key Files table.
- If you change the result format → update `include/common/result_store.h` and bump `RESULT_FORMAT_VERSION`.
- If you add/remove a file → update `src/common/CMakeLists.txt` and this file.
