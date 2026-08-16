# packaging/flatpak
Flathub packaging for clpeak (Linux Flatpak). Ships both front-ends from one
build: `clpeak` (CLI, the Flatpak's default `command`) and `clpeak-gui` (the
Flutter desktop app, what the exported `.desktop` entry launches).

## Quick Lookups
- Build chain / sandbox permissions? → `io.github.krrishnarraj.clpeak.yaml`
- Store listing (name, summary, description, screenshots, releases)? → `io.github.krrishnarraj.clpeak.metainfo.xml`
- What the launcher runs? → the `.desktop` files below

## Key Files
| File | Purpose |
|------|---------|
| `io.github.krrishnarraj.clpeak.yaml` | Flatpak manifest. Enables Vulkan+OpenCL+CPU only; builds `shaderc` (for `glslc`) and the OpenCL ICD loader, then clpeak pinned to a git tag+commit. Stages the Flutter SDK tarball as a module source so `clpeak-gui` builds too. |
| `io.github.krrishnarraj.clpeak.metainfo.xml` | AppStream MetaInfo. `desktop-application` component. |
| `io.github.krrishnarraj.clpeak.desktop` | GUI launcher (`Exec=clpeak-gui`). Installed as the app-id desktop file where the GUI was built. |
| `clpeak-cli.desktop` | Fallback launcher (`Exec=clpeak`, `Terminal=true`), installed under the *same* app-id name on arches with no GUI. Never exported under its own name. |

## Local Build

```console
flatpak run org.flatpak.Builder --force-clean --user --install build-dir packaging/flatpak/io.github.krrishnarraj.clpeak.yaml
flatpak run io.github.krrishnarraj.clpeak --list-devices
flatpak run --command=clpeak-gui io.github.krrishnarraj.clpeak
flatpak uninstall -y clpeak
```

## Notes
- App ID `io.github.krrishnarraj.clpeak` (Flathub form for GitHub-hosted projects).
- Runtime is `org.gnome.Platform//49`, not freedesktop: a Flutter Linux app links
  GTK3, which the freedesktop runtime does not carry. GNOME 49 sits on
  freedesktop 25.08, so the `llvm22` SDK extension branch is unchanged.
- The GUI is x86_64-only — Flutter publishes no linux-arm64 desktop SDK. The SDK
  source carries `only-arches: [x86_64]`; on aarch64 CMake finds no Flutter and
  reports `GUI: disabled` instead of failing, matching the arm64 CI jobs. The
  post-install picks `clpeak-cli.desktop` there so the MetaInfo's `desktop-id`
  launchable still resolves and AppStream validates on every arch.
- `--share=network` on the clpeak module is for `flutter pub get` and the engine
  artifact download. `HOME`/`PUB_CACHE` are redirected into the module build dir
  so nothing the Flutter tool writes escapes it.
- Install layout comes from the root CMake install rules: `/app/bin/clpeak`,
  `/app/bin/clpeak-gui` (a wrapper) and the Flutter bundle whole under
  `/app/gui/`. The bundle's runner only starts with its sibling `data/` and
  `lib/` — don't flatten it.
- shaderc's `third_party/*` deps are vendored at the revisions from shaderc's
  `DEPS` file because Flathub builds offline — re-sync them when bumping shaderc.
- CUDA/ROCm/oneAPI/Metal are intentionally disabled: they need vendor toolkits or
  proprietary drivers Flathub cannot ship. See root `AGENTS.md` for backends.
- MetaInfo screenshots point at `raw.githubusercontent.com/.../master/docs/assets/img/`,
  so they only resolve once `docs/` is on the default branch.

## When You Change This Directory
- If you bump the clpeak version → update both the manifest `tag`/`commit` and the
  MetaInfo `<releases>`.
- If you bump the Flutter SDK → update `url` **and** `sha256` from
  `https://storage.googleapis.com/flutter_infra_release/releases/releases_linux.json`.
- If you add packaging for another store → add it under `packaging/` and update the
  root `AGENTS.md` Directory Map.
