#!/bin/sh
# make_dmg.sh — wrap clpeak-gui.app in a drag-to-Applications disk image.
#
#   make_dmg.sh <app-bundle> <output.dmg> [volume-name]
#
# Driven by the clpeak-gui-dmg CMake target.  Uses only hdiutil/ditto from the
# base system: no create-dmg, no Homebrew, nothing for CI to install.
#
# The app is ad-hoc signed (see src/ffi/CMakeLists.txt), so a downloaded image
# is quarantined by Gatekeeper — first launch needs right-click → Open, or
# `xattr -dr com.apple.quarantine /Applications/clpeak-gui.app`.  Shipping a
# clean first-launch means a Developer ID identity + notarytool in CI.
set -eu

app=${1:?usage: make_dmg.sh <app-bundle> <output.dmg> [volume-name]}
dmg=${2:?usage: make_dmg.sh <app-bundle> <output.dmg> [volume-name]}
vol=${3:-clpeak}

if [ ! -d "$app" ]; then
    echo "make_dmg.sh: no app bundle at $app (build the clpeak-gui target first)" >&2
    exit 1
fi

staging=$(mktemp -d "${TMPDIR:-/tmp}/clpeak-dmg.XXXXXX")
trap 'rm -rf "$staging"' EXIT INT TERM

# ditto preserves the framework symlinks and the code signature.
ditto "$app" "$staging/$(basename "$app")"
ln -s /Applications "$staging/Applications"

rm -f "$dmg"
hdiutil create -volname "$vol" -srcfolder "$staging" -ov -quiet \
    -format UDZO "$dmg"

echo "make_dmg.sh: wrote $dmg"
