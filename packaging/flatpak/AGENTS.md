# packaging/flatpak
Flathub packaging for XpuPeak (Linux Flatpak), never published yet. Ships both
front-ends from one build: `xpupeak` (CLI, the Flatpak's default `command`)
and `xpupeak-gui` (the Flutter desktop app, what the exported `.desktop` entry
launches).

## Quick Lookups
- Build chain / sandbox permissions? → `io.github.krrishnarraj.XpuPeak.yaml`
- Store listing (name, summary, description, screenshots, releases)? → `io.github.krrishnarraj.XpuPeak.metainfo.xml`
- What the launcher runs? → the `.desktop` files below

## Key Files
| File | Purpose |
|------|---------|
| `io.github.krrishnarraj.XpuPeak.yaml` | Flatpak manifest. Enables Vulkan+OpenCL+CPU only; builds `shaderc` (for `glslc`) and the OpenCL ICD loader, then XpuPeak pinned to a git tag+commit. Stages the Flutter SDK tarball as a module source so `xpupeak-gui` builds too. |
| `io.github.krrishnarraj.XpuPeak.metainfo.xml` | AppStream MetaInfo. `desktop-application` component. |
| `io.github.krrishnarraj.XpuPeak.desktop` | GUI launcher (`Exec=xpupeak-gui`). Installed as the app-id desktop file where the GUI was built. |
| `xpupeak-cli.desktop` | Fallback launcher (`Exec=xpupeak`, `Terminal=true`), installed under the *same* app-id name on arches with no GUI. Never exported under its own name. |

## Local Build

```console
flatpak run org.flatpak.Builder --force-clean --user --install build-dir packaging/flatpak/io.github.krrishnarraj.XpuPeak.yaml
flatpak run io.github.krrishnarraj.XpuPeak --list-devices
flatpak run --command=xpupeak-gui io.github.krrishnarraj.XpuPeak
flatpak uninstall -y io.github.krrishnarraj.XpuPeak
```

## Notes
- App ID `io.github.krrishnarraj.XpuPeak` (Flathub form for GitHub-hosted
  projects). The name ends in `XpuPeak` as people read it; the commands stay
  lowercase.
- Runtime is `org.gnome.Platform//49`, not freedesktop: a Flutter Linux app links
  GTK3, which the freedesktop runtime does not carry. GNOME 49 sits on
  freedesktop 25.08, so the `llvm22` SDK extension branch is unchanged.
- The GUI is x86_64-only — Flutter publishes no linux-arm64 desktop SDK. The SDK
  source carries `only-arches: [x86_64]`; on aarch64 CMake finds no Flutter and
  reports `GUI: disabled` instead of failing, matching the arm64 CI jobs. The
  post-install picks `xpupeak-cli.desktop` there so the MetaInfo's `desktop-id`
  launchable still resolves and AppStream validates on every arch.
- The GUI's window reports `kr.XpuPeak` (the Linux runner's application ID,
  set as its program name) as its Wayland app_id and X11 WM_CLASS, not the
  Flatpak ID, so the GUI entry carries `StartupWMClass=kr.XpuPeak`.
- The icon is the app's own Linux icon (`app/linux/runner/resources/xpupeak_icon.png`,
  256 px, generated from the XpuPeak master by `tools/icons/generate_icons.py`),
  installed at that one size; AppStream compose renders the catalog sizes.
- `--share=network` on the `xpupeak` module is for `flutter pub get` and the engine
  artifact download. `HOME`/`PUB_CACHE` are redirected into the module build dir
  so nothing the Flutter tool writes escapes it.
- Install layout comes from the root CMake install rules: `/app/bin/xpupeak`,
  `/app/bin/xpupeak-gui` (a wrapper) and the Flutter bundle whole under
  `/app/gui/`. The bundle's runner only starts with its sibling `data/` and
  `lib/` — don't flatten it.
- shaderc's `third_party/*` deps are vendored at the revisions from shaderc's
  `DEPS` file because Flathub builds offline — re-sync them when bumping shaderc.
- CUDA/ROCm/oneAPI/Metal are intentionally disabled: they need vendor toolkits or
  proprietary drivers Flathub cannot ship. See root `AGENTS.md` for backends.
- The source, homepage and screenshot URLs name `krrishnarraj/xpupeak`, the repo
  as renamed just before 4.0.0. The screenshots point at
  `raw.githubusercontent.com/.../master/docs/assets/img/`, so they only resolve
  once that rename is done and `docs/` is on the default branch.

## When You Change This Directory
- If you bump the version → update both the manifest `tag`/`commit` and the
  MetaInfo `<releases>` (newest first, with the tag's date). For 4.0.0 the
  `commit:` and the release date are still to be filled in at release.
- If you bump the Flutter SDK → update `url` **and** `sha256` from
  `https://storage.googleapis.com/flutter_infra_release/releases/releases_linux.json`.
- If you add packaging for another store → add it under `packaging/` and update the
  root `AGENTS.md` Directory Map.
