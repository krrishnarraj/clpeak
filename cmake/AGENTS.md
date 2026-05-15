# cmake — Shared CMake Modules

Build-system helpers: version derivation from git, and the generated
`version.h.in` template.

## Quick Lookups

- Looking for version derivation? → `version.cmake` (configure-time) + `GenVersion.cmake` (build-time)
- Looking for the version header template? → `version.h.in`
- Looking for how backends are enabled/disabled? → root `CMakeLists.txt` (`CLPEAK_ENABLE_*` options)

## Key Files

| File | Purpose |
|------|---------|
| `version.cmake` | `clpeak_setup_version()` — git-describe → `CLPEAK_VERSION_STR` |
| `GenVersion.cmake` | Build-time version regeneration (write-if-different) |
| `version.h.in` | Template for `generated/version.h` |

## When You Change This Directory

- If you change the version scheme → update `version.cmake`, `GenVersion.cmake`, and `version.h.in`.
- If you bump the fallback version → update `CLPEAK_VERSION_FALLBACK` in `version.cmake`.
- If you add a new cmake module → update this file's Key Files table.
