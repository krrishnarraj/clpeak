# src/common/cmake — Version Handling and Embedded Data

Git-describe version derivation and the generated `version.h.in` template;
the headline catalog embedded in the build.

## Quick Lookups

- Looking for version derivation? → `version.cmake`, configure time **only**
- Looking for the version header template? → `version.h.in`
- Looking for how backends are enabled/disabled? → root `CMakeLists.txt` (`XPUPEAK_ENABLE_*` options)

## Key Files

| File | Purpose |
|------|---------|
| `version.cmake` | git-describe → `XPUPEAK_VERSION_STR` (`"unknown"` when git-describe cannot run) → `generated/version.h`; `xpupeak_setup_version(<target>)` puts that header on the target's include path |
| `version.h.in` | Template for `generated/version.h` |
| `embed_catalog.cmake` | `cmake -P` script: `app/assets/measures.json` → `measures_catalog.cpp` (a byte array, `HeadlineCatalog::bundled()`'s, `include/common/measures.h`).  Run by `src/common/CMakeLists.txt` as a build step that depends on the file, so a catalog edit rebuilds without a reconfigure -- unlike the version, the catalog is data the build should follow |

## Rules

- **Derive the version once, at configure time — never during the build.** The
  GUI build rewrites tracked files (the Flutter SDK owns `pubspec.lock`,
  `analysis_options.yaml`, the generated plugin registrants), so anything
  re-deriving mid-build stamps release binaries `-dirty` from a clean checkout.
  Excluding those files is not a fix — the SDK decides that set, not us.
- The trade-off is that the string is fixed until the next configure: after
  committing, re-run `cmake -B build` to refresh it. The configure summary
  prints the version so it is visible.
- **Never hardcode a version number as the fallback.** A build with no git
  metadata (source tarball, no git binary) reports `"unknown"`, which shows up
  in `--version`, the JSON/XML result headers and the CPack archive name. A
  stale hardcoded number there would be indistinguishable from a real release.

## When You Change This Directory

- If you change the version scheme → update `version.cmake` and `version.h.in`.
- If you add a new cmake module → update this file's Key Files table.
