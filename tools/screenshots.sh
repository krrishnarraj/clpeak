#!/usr/bin/env bash
# Regenerate the screenshots the README and the docs site show, from the saved
# runs in results/ -- no benchmark runs and no screen capture.
#
#   tools/screenshots.sh [DEVICE]
#
# app/integration_test/screenshots_test.dart renders each screen of the real
# app on the desktop engine (DEVICE: macos / linux / windows, default this
# host's) into raw 2x captures, and tools/screenshots/frame.py frames them
# into docs/assets/img/.  Needs the Flutter SDK and Pillow.  To change what is
# shown, edit the test: which run, which device chip, where it scrolls.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -s)" in
  Darwin) host=macos ;;
  Linux) host=linux ;;
  *) host=windows ;;
esac
device="${1:-$host}"

raw="$(mktemp -d)"
trap 'rm -rf "$raw"' EXIT

(cd "$repo/app" &&
  CLPEAK_SHOTS_OUT="$raw" CLPEAK_SHOTS_REPO="$repo" \
    flutter test integration_test/screenshots_test.dart -d "$device")
python3 "$repo/tools/screenshots/frame.py" "$raw" "$repo/docs/assets/img"
