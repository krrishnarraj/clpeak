# include/common — Shared Neutral Headers

Public headers for types, enums, and interfaces shared by every backend.
No backend-specific includes live here.

## Quick Lookups

- Looking for the Peak base class? → `peak.h`
- Looking for benchmark enums? → `benchmark_enums.h`
- Looking for benchmark constants (loop counts, sizes)? → `benchmark_constants.h`
- Looking for CLI options struct? → `options.h`
- Looking for result output format? → `result_store.h`
- Looking for logger interface? → `logger.h`
- Looking for calibration? → `calibrate.h`
- Looking for device inventory structs? → `inventory.h`
- Looking for backend gating? → `backend_gating.h`
- Looking for OS macros / Timer / util functions? → `common.h`

## Key Files

| File | Purpose |
|------|---------|
| `peak.h` | `Peak` abstract base class — every backend implements this |
| `benchmark_enums.h` | `Benchmark`, `Category`, `DeviceType` enums, `categoryOf()` |
| `benchmark_constants.h` | `benchmark_config_t` defaults and tuning constants |
| `options.h` | `CliOptions` struct + `parseCliOptions()` declaration |
| `result_store.h` | `ResultEntry`/`ResultStore` + JSON/CSV/XML serialization |
| `logger.h` | `logger` class — result-scope API, stdout, baseline compare |
| `calibrate.h` | `pickIters()` / `DEFAULT_TARGET_TIME_US` |
| `inventory.h` | `InventoryDevice`, `BackendInventory`, `inventoryToJson()` |
| `backend_gating.h` | `BackendGating` — per-test/category enable/disable |
| `common.h` | OS macros, `Timer`, `benchmark_config_t`, utility functions |

## When You Change This Directory

- If you change the `Peak` interface → update all backend `AGENTS.md` files.
- If you add/remove a header → update this file's Key Files table.
- If you change result format → bump `RESULT_FORMAT_VERSION` in `result_store.h`.
- If you change benchmark enums → make sure all backends handle the new enum values.
