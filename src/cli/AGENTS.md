# src/cli — Desktop CLI Entry Point

The desktop command-line application. Orchestrates backend selection,
runs benchmarks, and handles centralized file output.

## Quick Lookups

- CLI argument parsing? → `../common/options.cpp` + `include/common/options.h`
- How backends are wired? → `main.cpp` iterates `backendRegistry()` (`../registry/backend_registry.cpp`, shared with the GUI bridge) — creates each enabled backend, runs it, merges its results
- Logger implementation? → shared `LoggerText` in `../common/logger_text.cpp` + `include/common/logger_text.h` (CLI constructs it over `std::cout`)
- Device listing? → `main.cpp` collects each enabled backend's `enumerate()`; `printInventory()` in `../common/inventory.cpp` prints them all in one format, each device line starting with the `backend:index` token `--devices` takes

## Key Files

| File | Purpose |
|------|---------|
| `main.cpp` | `main()` — `setProgramName(argv[0])` first, then backend dispatch, `--list-devices`, `--resume` (`include/common/run_resume.h`), centralized file output; constructs `LoggerText(std::cout, …)` |
| `CMakeLists.txt` | the `xpupeak` target at the build root, and `clpeak` beside it (a link; a copy of the exe on Windows); the root `CMakeLists.txt` installs and tests both |

## Two names, one program

`clpeak`, the 3.x command, is `xpupeak` under its old name: the same flags,
output and exit codes. The only difference is the name in help, usage and
error messages, which is the one it was started as (`programName()` in
`include/common/options.h`). A message that names the program takes it from
there, never a literal `xpupeak:`.

## When You Change This Directory

- If you add a CLI flag → update `include/common/options.h` + `../common/options.cpp`.
  A test flag names what is measured and applies to every backend that runs
  (`--gemm`, not `--cuda-gemm`); a backend is a `Backend` enum value + a row in
  the backend table, and `--devices backend:index` and the listing then work for it unasked.
- If you change how backends are invoked → update `main.cpp` (and mirror it in `../ffi/xpupeak_ffi.cpp`, which ports this loop for the GUI). Which backends exist and in what order is not this file's to decide: both loops walk `backendRegistry()`.
- If you change the text output format → update `../common/logger_text.cpp`.
