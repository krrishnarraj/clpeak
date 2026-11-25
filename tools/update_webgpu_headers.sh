#!/bin/sh
# update_webgpu_headers.sh — refresh the vendored WebGPU C API header.
#
#   update_webgpu_headers.sh <commit|main>  refetch webgpu.h and rewrite pin
#   update_webgpu_headers.sh --check        report pinned commit vs upstream HEAD
#
# The WebGPU backend dlopens a conforming webgpu.h runtime at run time, so
# clpeak needs only this single header to build — see src/webgpu/AGENTS.md.
# It is checked in rather than submoduled so plain clones, desktop CI, and
# source tarballs build WebGPU without initializing submodules; this script
# fetches webgpu.h from a specific upstream commit and rewrites the SHA
# recorded in third_party/webgpu/README.md so the pin never drifts.
#
# Uses only curl, git ls-remote and the base system: nothing for CI to install.
set -eu

repo=https://github.com/webgpu-native/webgpu-headers
raw=https://raw.githubusercontent.com/webgpu-native/webgpu-headers

here=$(cd "$(dirname "$0")/.." && pwd)
dest=$here/third_party/webgpu
readme=$dest/README.md

pinned_commit() {
    sed -n 's/^- \*\*Commit:\*\* `\([^`]*\)`.*/\1/p' "$readme"
}

resolve_ref() {
    git ls-remote "$repo" "$1" 2>/dev/null | awk 'NR == 1 { print $1 }'
}

if [ "${1:-}" = "--check" ]; then
    echo "pinned:   $(pinned_commit)"
    echo "upstream: $(resolve_ref HEAD)"
    exit 0
fi

rev=${1:?usage: update_webgpu_headers.sh <commit|main>|--check}

if [ ! -d "$dest" ]; then
    echo "update_webgpu_headers.sh: no vendored dir at $dest" >&2
    exit 1
fi

case "$rev" in
    HEAD|main)
        commit=$(resolve_ref "$rev")
        if [ -z "$commit" ]; then
            echo "update_webgpu_headers.sh: cannot resolve $rev from $repo" >&2
            exit 1
        fi
        ;;
    *[!0-9a-fA-F]*)
        commit=$(resolve_ref "$rev")
        if [ -z "$commit" ]; then
            echo "update_webgpu_headers.sh: cannot resolve ref '$rev' from $repo" >&2
            exit 1
        fi
        ;;
    *)
        if [ "${#rev}" -ne 40 ]; then
            echo "update_webgpu_headers.sh: pass a full 40-character commit SHA or 'main'" >&2
            exit 1
        fi
        commit=$rev
        ;;
esac

staging=$(mktemp -d "${TMPDIR:-/tmp}/clpeak-webgpu.XXXXXX")
trap 'rm -rf "$staging"' EXIT INT TERM

if ! curl -sfL --max-time 120 "$raw/$commit/webgpu.h" -o "$staging/webgpu.h"; then
    echo "update_webgpu_headers.sh: cannot fetch webgpu.h at $commit" >&2
    echo "  check the commit exists: $repo/commits" >&2
    exit 1
fi

if ! grep -q '^#ifndef WEBGPU_H_' "$staging/webgpu.h"; then
    echo "update_webgpu_headers.sh: fetched file does not look like webgpu.h" >&2
    exit 1
fi

cp "$staging/webgpu.h" "$dest/webgpu.h"

# Keep the recorded pin in step with what was just written.
tmp=$staging/README.md
sed -e "s|^- \*\*Commit:\*\* .*|- **Commit:** \`$commit\`|" \
    "$readme" > "$tmp"
cp "$tmp" "$readme"

echo "Updated third_party/webgpu/webgpu.h to $commit."
echo "Next: rebuild and verify ABI/symbol compatibility against a conforming WebGPU runtime."
