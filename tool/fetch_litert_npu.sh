#!/bin/sh
# fetch_litert_npu.sh — stage LiteRT's NPU dispatch libraries for the Android app.
#
#   tool/fetch_litert_npu.sh [tag]      default: the tag pinned in
#                                       third_party/litert/README.md
#
# LiteRT reaches an NPU through a vendor "dispatch" library
# (libLiteRtDispatch_<Vendor>.so) and, for on-device compilation, a compiler
# plugin (libLiteRtCompilerPlugin_<Vendor>.so).  Google publishes them as
# `litert_npu_runtime_libraries_jit.zip` on each GitHub release, not on
# Maven, laid out as Play feature modules -- one per Hexagon generation for
# Qualcomm, one for Google Tensor.  This script unpacks that zip and copies
# the arm64 shims into app/android/app/src/main/jniLibs/arm64-v8a/ (ignored
# by git), where Gradle packages them beside libLiteRt.so so the backend's
# default NPU directory finds them.
#
# What it does NOT fetch is the vendor runtime itself.  MediaTek's and
# Google Tensor's live on the device as system libraries; Qualcomm's (the
# QAIRT/QNN HTP libraries, tens of MB per Hexagon generation) must be
# bundled by the app, and the zip's own fetch_qualcomm_library.sh downloads
# them from Qualcomm's software center into the per-generation module
# directories.  Run that script inside the unpacked zip and copy the
# libraries for the generations you ship next to the shims, or -- for a
# Play release -- wire the module directories in as dynamic feature modules
# gated by device group, which is the layout they come in.
#
# Uses curl and unzip from the base system.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
readme=$here/third_party/litert/README.md
tag=${1:-$(sed -n 's/^- \*\*Tag:\*\* `\([^`]*\)`.*/\1/p' "$readme")}
repo=https://github.com/google-ai-edge/LiteRT
dest=$here/app/android/app/src/main/jniLibs/arm64-v8a

staging=$(mktemp -d "${TMPDIR:-/tmp}/clpeak-litert-npu.XXXXXX")
trap 'rm -rf "$staging"' EXIT INT TERM

if ! curl -sfL --max-time 300 "$repo/releases/download/$tag/litert_npu_runtime_libraries_jit.zip" \
        -o "$staging/npu.zip"; then
    echo "fetch_litert_npu.sh: cannot fetch litert_npu_runtime_libraries_jit.zip at $tag" >&2
    exit 1
fi
(cd "$staging" && unzip -q npu.zip)

mkdir -p "$dest"
n=0
# One copy of each vendor's shims: the per-generation Qualcomm modules carry
# identical files, differing only in the QNN libraries they are meant to hold.
for so in $(find "$staging" -path '*/jni/arm64-v8a/libLiteRt*.so' | sort); do
    base=$(basename "$so")
    if [ ! -f "$dest/$base" ]; then
        cp "$so" "$dest/$base"
        n=$((n + 1))
        echo "staged $base"
    fi
done
echo "Staged $n NPU shim libraries from $tag into ${dest#$here/}."
echo "The unpacked release (with fetch_qualcomm_library.sh) is in $staging until this script exits;"
echo "re-run with QAIRT libraries copied beside the shims to reach a Qualcomm NPU."
