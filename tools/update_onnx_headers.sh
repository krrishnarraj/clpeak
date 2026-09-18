#!/bin/sh
# update_onnx_headers.sh — refresh the vendored ONNX Runtime C API headers.
#
#   update_onnx_headers.sh <tag>        e.g. update_onnx_headers.sh v1.30.0
#   update_onnx_headers.sh --check      report the pinned tag and the latest
#                                       upstream release, then exit
#
# The ONNX backend dlopens the runtime, so clpeak needs only these headers to
# build — see src/onnx/AGENTS.md.  They are checked in rather than submoduled
# because a --depth 1 clone of microsoft/onnxruntime costs ~814 MB (10,866
# files) to deliver three files.  This script is the "easy update" half of
# that trade: it re-fetches all three from one release tag and rewrites the
# tag recorded in the README, so the pin never drifts from the contents.
#
# Uses only curl and the base system: nothing for CI to install.
set -eu

repo=https://github.com/microsoft/onnxruntime
raw=https://raw.githubusercontent.com/microsoft/onnxruntime
srcdir=include/onnxruntime/core/session

# Every header the vendored onnxruntime_c_api.h needs, transitively.
files="onnxruntime_c_api.h onnxruntime_error_code.h onnxruntime_ep_c_api.h"

here=$(cd "$(dirname "$0")/.." && pwd)
dest=$here/third_party/onnxruntime
readme=$dest/README.md

pinned_tag() {
    sed -n 's/^- \*\*Tag:\*\* `\([^`]*\)`.*/\1/p' "$readme"
}

latest_tag() {
    # Releases are chronological; the newest non-prerelease tag is the head of
    # the ls-remote list once sorted by version.
    git ls-remote --tags --refs "$repo" 'v*' 2>/dev/null |
        sed 's:.*refs/tags/::' |
        grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' |
        sort -V | tail -1
}

if [ "${1:-}" = "--check" ]; then
    echo "pinned:   $(pinned_tag)"
    echo "upstream: $(latest_tag)"
    exit 0
fi

tag=${1:?usage: update_onnx_headers.sh <tag>|--check   (e.g. v1.30.0)}

if [ ! -d "$dest" ]; then
    echo "update_onnx_headers.sh: no vendored dir at $dest" >&2
    exit 1
fi

staging=$(mktemp -d "${TMPDIR:-/tmp}/clpeak-ort.XXXXXX")
trap 'rm -rf "$staging"' EXIT INT TERM

# Fetch everything before touching the tree, so a 404 on the last file cannot
# leave a half-updated mix of two releases behind.
for f in $files; do
    if ! curl -sfL --max-time 120 "$raw/$tag/$srcdir/$f" -o "$staging/$f"; then
        echo "update_onnx_headers.sh: cannot fetch $f at $tag" >&2
        echo "  check the tag exists: $repo/releases" >&2
        exit 1
    fi
done

api=$(sed -n 's/^#define ORT_API_VERSION *\([0-9]*\).*/\1/p' "$staging/onnxruntime_c_api.h")
if [ -z "$api" ]; then
    echo "update_onnx_headers.sh: no ORT_API_VERSION in the fetched header" >&2
    exit 1
fi

for f in $files; do
    cp "$staging/$f" "$dest/$f"
done

# Keep the recorded pin in step with what was just written.
tmp=$staging/README.md
sed -e "s|^- \*\*Tag:\*\* .*|- **Tag:** \`$tag\`  (\`ORT_API_VERSION\` $api)|" \
    "$readme" > "$tmp"
cp "$tmp" "$readme"

echo "Updated third_party/onnxruntime to $tag (ORT_API_VERSION $api)."
echo "Next: rebuild, and check kMinApiVersion in src/onnx/onnx_runtime.cpp"
echo "      still names the oldest runtime worth supporting."
