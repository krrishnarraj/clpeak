#!/bin/sh
# fetch_litert_npu.sh — stage LiteRT's NPU dispatch libraries for the Android app.
#
#   tool/fetch_litert_npu.sh qualcomm|google_tensor [tag]
#                                       tag defaults to the one pinned in
#                                       third_party/litert/README.md
#
# LiteRT reaches an NPU through a vendor "dispatch" library
# (libLiteRtDispatch_<Vendor>.so) and, for on-device compilation, a compiler
# plugin (libLiteRtCompilerPlugin_<Vendor>.so).  Google publishes them as
# `litert_npu_runtime_libraries_jit.zip` on each GitHub release, not on
# Maven, laid out as Play feature modules -- one per Hexagon generation for
# Qualcomm, one for Google Tensor.  This script unpacks that zip and copies
# one vendor's arm64 shims into app/android/app/src/main/jniLibs/arm64-v8a/
# (ignored by git), where Gradle packages them beside libLiteRt.so so the
# backend's default NPU directory finds them.
#
# One vendor at a time, because LiteRT loads the first libLiteRtDispatch_*
# it lists in that directory and warns about the rest (litert_dispatch.cc):
# two vendors' shims side by side would leave which NPU is tried to the
# filesystem's listing order.  A Play release avoids the question with one
# feature module per vendor, delivered by device group.  Staging any shim
# also switches the app to extracting its native libraries at install
# (build.gradle.kts), since LiteRT finds the shim by listing a directory,
# which an APK's internal lib/ path is not.
#
# What it does NOT fetch is the vendor runtime itself.  MediaTek's and
# Google Tensor's live on the device as system libraries; Qualcomm's (the
# QAIRT/QNN HTP libraries, tens of MB per Hexagon generation) must be
# bundled by the app.  The simplest way is the Gradle opt-in in
# app/android/app/build.gradle.kts (`clpeakQnn=true`), which pulls
# com.qualcomm.qti:qnn-runtime from Maven Central -- every generation's
# Skel/Stub plus libQnnHtp, libQnnSystem and libQnnHtpPrepare, 67 MB
# compressed.  The zip's own fetch_qualcomm_library.sh is the other route:
# it downloads the full QAIRT SDK from Qualcomm's software center into the
# per-generation module directories (adding libQnnIr/libQnnSaver, which the
# Maven package lacks); run it inside the unpacked zip and copy the
# libraries for the generations you ship next to the shims, or -- for a
# Play release -- wire the module directories in as dynamic feature modules
# gated by device group, which is the layout they come in.
#
# Uses curl and unzip from the base system.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
readme=$here/third_party/litert/README.md
vendor=${1:-}
case "$vendor" in
    qualcomm)      module=qualcomm_runtime_v75 ;;   # every generation carries the same shims
    google_tensor) module=google_tensor_runtime ;;
    *)
        echo "usage: tool/fetch_litert_npu.sh qualcomm|google_tensor [tag]" >&2
        exit 2 ;;
esac
tag=${2:-$(sed -n 's/^- \*\*Tag:\*\* `\([^`]*\)`.*/\1/p' "$readme")}
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

# Whatever was staged before goes: the directory holds nothing else.
rm -rf "$dest"
mkdir -p "$dest"
n=0
for so in $(find "$staging/$module" -path '*/jni/arm64-v8a/libLiteRt*.so' | sort); do
    base=$(basename "$so")
    cp "$so" "$dest/$base"
    n=$((n + 1))
    echo "staged $base"
done
echo "Staged $n $vendor shim libraries from $tag into ${dest#$here/}."
if [ "$vendor" = qualcomm ]; then
    echo "Qualcomm's own runtime is a separate step: set clpeakQnn=true in app/android/gradle.properties"
    echo "(Maven Central, 67 MB) or run fetch_qualcomm_library.sh from the unpacked release, which is in"
    echo "$staging until this script exits."
fi
