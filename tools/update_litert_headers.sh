#!/bin/sh
# update_litert_headers.sh — refresh the vendored LiteRT C API headers.
#
#   update_litert_headers.sh <tag>        e.g. update_litert_headers.sh v2.3.0
#   update_litert_headers.sh --check      report the pinned tag and the latest
#                                         upstream release, then exit
#
# The LiteRT backend dlopens the runtime, so clpeak needs only the C headers
# to build — see src/litert/AGENTS.md.  They come from the `litert_cc_sdk.zip`
# asset of a GitHub release (280 KB) rather than a clone of the repository
# (the whole runtime and converter, over a gigabyte).  Only the transitive
# include closure of the headers the backend uses is kept: the same root set
# every time, so an update cannot silently drop a header the code needs or
# add the C++ wrapper's Abseil-dependent ones.
#
# Uses curl, unzip and awk from the base system: nothing for CI to install.
set -eu

repo=https://github.com/google-ai-edge/LiteRT

# The headers src/litert includes; everything they include comes along.
roots="litert/c/litert_common.h
litert/c/litert_environment.h
litert/c/litert_environment_options.h
litert/c/litert_options.h
litert/c/litert_model.h
litert/c/litert_compiled_model.h
litert/c/litert_tensor_buffer.h
litert/c/litert_tensor_buffer_requirements.h
litert/c/litert_profiler.h
litert/c/litert_metrics.h
litert/c/litert_opaque_options.h
litert/c/litert_op_code.h
litert/c/internal/litert_logging.h
litert/c/internal/litert_runtime_c_api.h
litert/c/options/litert_gpu_options.h
litert/c/options/litert_cpu_options.h
litert/c/options/litert_qualcomm_options.h
litert/c/options/litert_mediatek_options.h
litert/c/options/litert_google_tensor_options.h
litert/c/options/litert_intel_openvino_options.h
litert/c/options/litert_samsung_options.h
litert/c/options/litert_runtime_options.h"

here=$(cd "$(dirname "$0")/.." && pwd)
dest=$here/third_party/litert
readme=$dest/README.md

pinned_tag() {
    sed -n 's/^- \*\*Tag:\*\* `\([^`]*\)`.*/\1/p' "$readme"
}

latest_tag() {
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

tag=${1:?usage: update_litert_headers.sh <tag>|--check   (e.g. v2.3.0)}

if [ ! -d "$dest" ]; then
    echo "update_litert_headers.sh: no vendored dir at $dest" >&2
    exit 1
fi

staging=$(mktemp -d "${TMPDIR:-/tmp}/clpeak-litert.XXXXXX")
trap 'rm -rf "$staging"' EXIT INT TERM

if ! curl -sfL --max-time 300 "$repo/releases/download/$tag/litert_cc_sdk.zip" \
        -o "$staging/sdk.zip"; then
    echo "update_litert_headers.sh: cannot fetch litert_cc_sdk.zip at $tag" >&2
    echo "  check the tag exists: $repo/releases" >&2
    exit 1
fi
(cd "$staging" && unzip -q sdk.zip)
src=$staging/litert_cc_sdk

# Walk the include closure: a worklist of relative paths, each opened once.
closure=$staging/closure
: > "$closure"
printf '%s\n' $roots > "$staging/todo"
while [ -s "$staging/todo" ]; do
    f=$(head -1 "$staging/todo")
    sed -i.bak '1d' "$staging/todo"
    grep -qx "$f" "$closure" && continue
    [ -f "$src/$f" ] || continue
    echo "$f" >> "$closure"
    sed -n 's/^#include[[:space:]]*"\([^"]*\)".*/\1/p' "$src/$f" >> "$staging/todo"
done

abi=$(sed -n 's/^#define LITERT_RUNTIME_ABI_VERSION *"\([^"]*\)".*/\1/p' \
      "$src/litert/c/internal/litert_runtime_c_api.h" 2>/dev/null || true)

# Replace the vendored set in one step; the generated build_config.h stand-in
# and this README are ours and stay.
find "$dest/litert/c" -name '*.h' -delete
while read -r f; do
    mkdir -p "$dest/$(dirname "$f")"
    cp "$src/$f" "$dest/$f"
done < "$closure"

n=$(wc -l < "$closure" | tr -d ' ')
tmp=$staging/README.md
sed -e "s|^- \*\*Tag:\*\* .*|- **Tag:** \`$tag\`  (\`LITERT_RUNTIME_ABI_VERSION\` ${abi:-unknown})|" \
    -e "s|(\`litert/c/\`, [0-9]* headers)|(\`litert/c/\`, $n headers)|" \
    "$readme" > "$tmp"
cp "$tmp" "$readme"

echo "Updated third_party/litert to $tag ($n headers, ABI ${abi:-unknown})."
echo "Next: rebuild, and check that every symbol src/litert/litert_runtime.cpp"
echo "      resolves is still exported by libLiteRt."
