# packaging/flatpak
Flathub packaging for clpeak (Linux Flatpak; CLI / console-application).

## Quick Lookups
- Build chain / sandbox permissions? → `io.github.krrishnarraj.clpeak.yaml`
- Store listing (name, summary, description, releases)? → `io.github.krrishnarraj.clpeak.metainfo.xml`

## Key Files
| File | Purpose |
|------|---------|
| `io.github.krrishnarraj.clpeak.yaml` | Flatpak manifest. Enables Vulkan+OpenCL+CPU only; builds `shaderc` (for `glslc`) and the OpenCL ICD loader, then clpeak pinned to a git tag+commit. |
| `io.github.krrishnarraj.clpeak.metainfo.xml` | AppStream MetaInfo. `console-application` component — no `.desktop`/icon needed. |

## Local Build

```console
flatpak run org.flatpak.Builder --force-clean --user --install build-dir packaging/flatpak/io.github.krrishnarraj.clpeak.yaml
flatpak run io.github.krrishnarraj.clpeak --list-devices
flatpak uninstall -y clpeak
```

## Notes
- App ID `io.github.krrishnarraj.clpeak` (Flathub form for GitHub-hosted projects).
- The manifest pins clpeak to a release tag+commit; bump both on each release and
  add a matching `<release>` entry to the MetaInfo.
- shaderc's `third_party/*` deps are vendored at the revisions from shaderc's
  `DEPS` file because Flathub builds offline — re-sync them when bumping shaderc.
- CUDA/ROCm/oneAPI/Metal are intentionally disabled: they need vendor toolkits or
  proprietary drivers Flathub cannot ship. See root `AGENTS.md` for backends.

## When You Change This Directory
- If you bump the clpeak version → update both the manifest `tag`/`commit` and the
  MetaInfo `<releases>`.
- If you add packaging for another store → add it under `packaging/` and update the
  root `AGENTS.md` Directory Map.
