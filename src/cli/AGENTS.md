# src/cli — Desktop CLI Entry Point

The desktop command-line application. Orchestrates backend selection,
runs benchmarks, and handles centralized file output.

## Quick Lookups

- CLI argument parsing? → `../common/options.cpp` + `include/common/options.h`
- How backends are wired? → `main.cpp` — creates backend objects, merges results
- Logger implementation? → shared `LoggerText` in `../common/logger_text.cpp` + `include/common/logger_text.h` (CLI constructs it over `std::cout`)
- Device listing? → `main.cpp` — `--list-devices` prints each enabled backend's `enumerate()` through its own `printInventory`

## Key Files

| File | Purpose |
|------|---------|
| `main.cpp` | `main()` — backend dispatch, `--list-devices`, centralized file output; constructs `LoggerText(std::cout, …)` |

## When You Change This Directory

- If you add a CLI flag → update `include/common/options.h` + `../common/options.cpp`.
  A test flag names what is measured and applies to every backend that runs
  (`--gemm`, not `--cuda-gemm`); a backend is a `Backend` enum value + a row in
  the backend table, and `--device backend:index` then works for it unasked.
- If you change how backends are invoked → update `main.cpp` (and mirror it in `../ffi/clpeak_ffi.cpp`, which ports this loop for the GUI).
- If you change the text output format → update `../common/logger_text.cpp`.
