# include/common — Shared Neutral Headers

Public headers for types, enums, and interfaces shared by every backend.
No backend-specific includes live here.

## Quick Lookups

- Looking for the Peak base class? → `peak.h`
- Looking for benchmark enums? → `benchmark_enums.h`
- Looking for benchmark constants + calibration? → `common.h`
- Looking for CLI options struct? → `options.h`
- Looking for result output format? → `result_store.h`
- Looking for logger interface? → `logger.h`
- Looking for device inventory structs? → `inventory.h`
- Looking for gating? → `peak.h` (gating is part of Peak)

## Key Files

| File | Purpose |
|------|---------|
| `peak.h` | `Peak` abstract base class + gating — every backend implements this |
| `benchmark_enums.h` | `Benchmark`, `Category`, `DeviceType` enums, `categoryOf()` |
| `common.h` | OS macros, tuning constants, `benchmark_config_t`, `pickIters()` calibration |
| `options.h` | `CliOptions` struct + `parseCliOptions()` declaration |
| `result_store.h` | `ResultEntry`/`ResultStore` + JSON/CSV/XML serialization |
| `logger.h` | `logger` class — result-scope API, stdout, baseline compare |
| `inventory.h` | `InventoryDevice`, `BackendInventory`, `inventoryToJson()` |

## When You Change This Directory

- If you change the `Peak` interface → update all backend `AGENTS.md` files.
- If you add/remove a header → update this file's Key Files table.
- If you change result format → bump `RESULT_FORMAT_VERSION` in `result_store.h`.
- If you change benchmark enums → make sure all backends handle the new enum values.
