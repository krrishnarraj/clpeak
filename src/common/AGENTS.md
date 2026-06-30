# src/common — Shared Backend-Neutral Code

Base classes, utilities, result store, and inventory shared
by every backend. Also owns the shared text logger (`logger_text.cpp` —
`LoggerText`, writes to an injectable `std::ostream`). Consumers that need a
non-text channel subclass it: the desktop CLI uses `LoggerText` directly over
`std::cout`; Android (`android/.../logger_android.cpp`) and iOS
(`ios/.../logger_ios.mm`) subclass to also forward structured data over JNI /
callbacks.

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
| `logger.cpp` | Base `logger` class: result-scope API, `emit()`, `recordSkip` |
| `logger_text.cpp` | `LoggerText` — indented/aligned text formatting to an injectable `std::ostream` + baseline deltas (derives from `logger`); shared by CLI + Android |
| `inventory.cpp` | `inventoryToJson()` — device inventory JSON serializer (no backend includes) |
| `options.cpp` | `parseCliOptions()` — CLI argument parsing |

## When You Change This Directory

- If you change the `Peak` interface → update `include/common/peak.h` + all backend `AGENTS.md` files.
- If you add a utility function → update this file's Key Files table.
- If you change the result format → update `include/common/result_store.h` and bump `RESULT_FORMAT_VERSION`.
- If you add/remove a file → update `src/common/CMakeLists.txt` and this file.
