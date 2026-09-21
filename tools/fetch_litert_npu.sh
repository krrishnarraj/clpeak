#!/bin/sh
# fetch_litert_npu.sh — stage LiteRT's NPU dispatch libraries for the Android app.
#
#   tools/fetch_litert_npu.sh qualcomm|google_tensor|all [tag]
#                                       tag defaults to the one pinned in
#                                       third_party/litert/README.md
#
# LiteRT reaches an NPU through a vendor "dispatch" library
# (libLiteRtDispatch_<Vendor>.so) and, for on-device compilation, a compiler
# plugin (libLiteRtCompilerPlugin_<Vendor>.so).  Google publishes them as
# `litert_npu_runtime_libraries_jit.zip` on each GitHub release, not on
# Maven, laid out as Play feature modules -- one per Hexagon generation for
# Qualcomm, one for Google Tensor (G3 and later).  This script unpacks that
# zip and copies the named vendors' arm64 shims into
# app/android/app/src/main/jniLibs/arm64-v8a/ (ignored by git), where Gradle
# packages them beside libLiteRt.so.
#
# `all` stages every vendor, for the one APK that serves a Pixel and a
# Snapdragon alike.  LiteRT loads the first libLiteRtDispatch_* it lists in
# its dispatch directory and warns about the rest (litert_dispatch.cc), so
# the app does not hand it the lib dir then: at launch the LiteRT backend
# picks the vendor of this SoC (ro.soc.manufacturer) and links that
# vendor's shims into a directory of their own under the app's support
# directory (litertStageNpuVendor in src/litert/litert_peak.cpp).  A Play
# release could instead ship one feature module per vendor, delivered by
# device group, which is the layout the zip comes in.  Staging any shim
# switches the app to extracting its native libraries at install
# (build.gradle.kts), since both LiteRT and the backend find shims by
# listing a directory, which an APK's internal lib/ path is not.
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
    qualcomm)      vendors=qualcomm ;;
    google_tensor) vendors=google_tensor ;;
    all)           vendors="qualcomm google_tensor" ;;
    *)
        echo "usage: tools/fetch_litert_npu.sh qualcomm|google_tensor|all [tag]" >&2
        exit 2 ;;
esac
module_of() {
    case "$1" in
        qualcomm)      echo qualcomm_runtime_v75 ;;   # every generation carries the same shims
        google_tensor) echo google_tensor_runtime ;;
    esac
}
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
for v in $vendors; do
    module=$(module_of "$v")
    for so in $(find "$staging/$module" -path '*/jni/arm64-v8a/libLiteRt*.so' | sort); do
        base=$(basename "$so")
        cp "$so" "$dest/$base"
        n=$((n + 1))
        echo "staged $base"
    done
done
echo "Staged $n shim libraries ($vendors) from $tag into ${dest#$here/}."
case " $vendors " in *" qualcomm "*)
    echo "Qualcomm's own runtime is a separate step: set clpeakQnn=true in app/android/gradle.properties"
    echo "(Maven Central, 67 MB; also packages Qualcomm's ONNX Runtime QNN plugin) or run"
    echo "fetch_qualcomm_library.sh from the unpacked release, which is in $staging until this script exits."
    ;;
esac
