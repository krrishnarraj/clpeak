# src/common/cmake — Version Handling

Git-describe version derivation and the generated `version.h.in` template.

## Quick Lookups

- Looking for version derivation? → `version.cmake`, configure time **only**
- Looking for the version header template? → `version.h.in`
- Looking for how backends are enabled/disabled? → root `CMakeLists.txt` (`CLPEAK_ENABLE_*` options)

## Key Files

| File | Purpose |
|------|---------|
| `version.cmake` | git-describe → `CLPEAK_VERSION_STR` → `generated/version.h`; `clpeak_setup_version(<target>)` puts that header on the target's include path |
| `version.h.in` | Template for `generated/version.h` |

## Rules

- **Derive the version once, at configure time.** There used to be a build-time
  regeneration step (`GenVersion.cmake` + a `clpeak_version_gen` target) so the
  string tracked new commits without reconfiguring. It tied the version to the
  tree's state *during* the build, and the GUI build dirties the tree — the
  Flutter SDK rewrites `pubspec.lock`, `analysis_options.yaml` and the generated
  plugin registrants whenever its version differs from the one that produced the
  committed copies. CPack's `preinstall` pass then rebuilt against that and
  relinked release binaries as `-dirty` from a pristine tag checkout. An
  exclude-list of those files was tried first and was already incomplete on its
  first CI run; deriving once, before anything has run, is what actually closes
  it. Do not reintroduce build-time derivation.
- The trade-off is that the string is fixed until the next configure: after
  committing, re-run `cmake -B build` to refresh it. The configure summary
  prints the version so it is visible.

## When You Change This Directory

- If you change the version scheme → update `version.cmake` and `version.h.in`.
- If you bump the fallback version → update `CLPEAK_VERSION_FALLBACK` in `version.cmake`.
- If you add a new cmake module → update this file's Key Files table.
