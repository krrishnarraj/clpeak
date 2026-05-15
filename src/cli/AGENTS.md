# src/cli — Desktop CLI Entry Point

`main.cpp` — parses CLI options, enumerates backends (for `--list-devices`),
instantiates each enabled backend, runs benchmarks, and writes output files.

## Quick Lookups

- Looking for the CLI entry point? → `main.cpp` (`main()`)
- Looking for option parsing? → `src/common/options.cpp` + `include/common/options.h`
- Looking for the result merge logic? → `main.cpp` (`mergeResults()`)
- Looking for output format writers? → `src/common/result_store.cpp`

## Key Files

| File | Purpose |
|------|---------|
| `main.cpp` | `main()`, `mergeResults()`, `enumerateAllBackends()` — CLI dispatch |
| `CMakeLists.txt` | Builds `clpeak` executable, links `peak_common` + enabled backends |

## When You Change This Directory

- If you add a new backend → add its `#ifdef` block in `main.cpp` + update `CMakeLists.txt`.
- If you change CLI argument handling → update `src/common/options.cpp`.
- If you change output dispatch → update `src/common/result_store.cpp`.
