# src/cli — Desktop CLI Entry Point

The desktop command-line application. Orchestrates backend selection,
runs benchmarks, and handles centralized file output.

## Quick Lookups

- CLI argument parsing? → `../common/options.cpp` + `include/common/options.h`
- How backends are wired? → `main.cpp` — creates backend objects, merges results
- Logger implementation? → `logger_cli.cpp` + `include/cli/logger_cli.h`
- Device listing? → `main.cpp` — `enumerateAllBackends()` with per-backend printing

## Key Files

| File | Purpose |
|------|---------|
| `main.cpp` | `main()` — backend dispatch, `enumerateAllBackends()`, centralized file output |
| `logger_cli.cpp` | `LoggerCli` — stdout printing, baseline comparison (derives from `logger`) |

## When You Change This Directory

- If you add a CLI flag → update `include/common/options.h` + `../common/options.cpp`.
- If you change how backends are invoked → update `main.cpp`.
- If you change output format → update `logger.cpp` + `include/common/logger.h`.
