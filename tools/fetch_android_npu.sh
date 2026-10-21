#!/bin/sh
# fetch_android_npu.sh — stage the NPU libraries the Android app packages.
#
#   tools/fetch_android_npu.sh [tag]
#                                       tag: the LiteRT release whose shims
#                                       are staged, by default the one
#                                       pinned in third_party/litert/README.md
#
# Two backends reach an Android NPU -- LiteRT through vendor shims, ONNX
# Runtime through Qualcomm's plugin execution provider -- and neither's
# vendor libraries come with the runtimes the Gradle build packages (the
# onnxruntime-android and LiteRT AARs).  This script stages every vendor's
# into the NPU runtime modules, app/android/npu/<module>/jniLibs/arm64-v8a/ (ignored
# by git), replacing whatever was staged before, and the build packages
# each module it finds staged as a Play feature module that installs with
# the app only on the chipsets of its device groups
# (app/android/app/device_targeting_configuration.xml): nothing staged, no
# NPU libraries, which is the CI APK's case.
#
# LiteRT reaches an NPU through a vendor "dispatch" library
# (libLiteRtDispatch_<Vendor>.so) and, for on-device compilation, a compiler
# plugin (libLiteRtCompilerPlugin_<Vendor>.so).  Google publishes the pair
# for each vendor as one AAR on its Maven repository,
# com.google.ai.edge.litert:litert-npu-runtime-<vendor> (since 2.3.0; 2.2.0
# had them in a zip on the GitHub release): Qualcomm, Google Tensor and
# MediaTek, each into its own module.  Samsung's are built from LiteRT's
# source against Samsung's SDK and published nowhere.
#
# Every vendor's, for the one bundle that serves a Pixel and a Snapdragon
# alike: Play gives each phone its own vendor's modules, and a universal
# APK built from the bundle carries them all.  For that APK: LiteRT loads the first libLiteRtDispatch_* it lists in
# its dispatch directory and warns about the rest (litert_dispatch.cc), so
# the app does not hand it the lib dir then: at launch the LiteRT backend
# picks the vendor of this SoC (ro.soc.manufacturer) and links that
# vendor's shims into a directory of their own under the app's support
# directory (litertStageNpuVendor in src/litert/litert_peak.cpp).  Staging
# anything switches the app to extracting its native libraries at install
# (build.gradle.kts), since both LiteRT and the backend find shims by
# listing a directory, which an APK's internal lib/ path is not.
#
# The vendor runtime the shims drive is not in those AARs.  MediaTek's and
# Google Tensor's live on the device as system libraries.  Qualcomm's -- the
# QNN libraries of its QAIRT SDK -- must be bundled by the app, and serves
# both backends, so the script stages two things beside its shims, of one
# QNN build:
#  - com.qualcomm.qti:qnn-runtime from Maven Central: libQnnHtp,
#    libQnnSystem, libQnnHtpPrepare and every Hexagon generation's
#    Skel/Stub, 67 MB compressed;
#  - com.qualcomm.qti:onnxruntime-android-qnn from Maven Central: one
#    libonnxruntime_providers_qnn.so, which the app registers on the
#    onnxruntime-android it packages (SettingsService.effectiveOnnxEpLibraries).
# LiteRT opens the QNN libraries at run time, and only those of the backend
# it is asked for: libQnnIr.so for its IR backend and libQnnSaver.so for
# saver_output_dir, neither of which clpeak sets, so the Maven package's
# lack of both costs nothing.  (2.2.0's Android compiler plugin was linked
# against them, and Android's linker refuses a library one of whose
# DT_NEEDED entries it cannot find; 2.3.0's links no QNN library.)
# Each Hexagon generation's Skel/Stub pair goes into that generation's
# module (qualcomm_runtime_vNN) and the rest into qualcomm_runtime, which
# every Snapdragon on the whitelist gets.
# A generation the runtime brings with no module for it stops the script
# before anything is replaced.  The build refuses a Qualcomm set with any
# of the QNN libraries missing: it would install, run and list no Hexagon
# NPU.
#
# Uses curl and unzip from the base system.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
readme=$here/third_party/litert/README.md
case "${1:-}" in
    ''|v*) ;;
    *)
        echo "usage: tools/fetch_android_npu.sh [tag]   (a LiteRT release tag, e.g. v2.3.0)" >&2
        exit 2 ;;
esac
# Each vendor's shims: the AAR on Google's Maven, and the module they go in.
vendors="qualcomm google_tensor mediatek"
artifact_of() {
    case "$1" in
        qualcomm)      echo litert-npu-runtime-qualcomm ;;
        google_tensor) echo litert-npu-runtime-google-tensor ;;
        mediatek)      echo litert-npu-runtime-mediatek ;;
    esac
}
module_of() {
    case "$1" in
        qualcomm)      echo qualcomm_runtime ;;
        google_tensor) echo google_tensor_runtime ;;
        mediatek)      echo mediatek_runtime ;;
    esac
}
tag=${1:-$(sed -n 's/^- \*\*Tag:\*\* `\([^`]*\)`.*/\1/p' "$readme")}
litert=${tag#v}
google_maven=https://dl.google.com/android/maven2/com/google/ai/edge/litert
npu=$here/app/android/npu

# Qualcomm's set, one QNN build for every library in the APK: the Maven
# runtime (2.50.0 carries v2.50.0.260828221209, QAIRT 2.50.0.260828) and
# the ONNX Runtime plugin Qualcomm validated against it.  LiteRT 2.3.0's
# shims were built against the same QAIRT build
# (third_party/qairt/workspace.bzl at the tag); its QnnManager accepts a
# runtime whose API minor version is newer, with a warning, and refuses
# only a major mismatch.  Bump the two together, and with LiteRT.
qnn=2.50.0
ort_qnn=2.6.0
maven=https://repo1.maven.org/maven2/com/qualcomm/qti

staging=$(mktemp -d "${TMPDIR:-/tmp}/clpeak-android-npu.XXXXXX")
trap 'rm -rf "$staging"' EXIT INT TERM

# fetch_aar <artifact> <version> <dir>: the arm64 libraries of one of
# Qualcomm's AARs on Maven Central into <dir>.
fetch_aar() {
    curl -sfL --max-time 900 "$maven/$1/$2/$1-$2.aar" -o "$staging/$1.aar" &&
        unzip -q -j -o "$staging/$1.aar" 'jni/arm64-v8a/*.so' -d "$3"
}

# Every vendor's shims first, each AAR's arm64 libraries into a directory
# named after its module.
for v in $vendors; do
    a=$(artifact_of "$v")
    mkdir -p "$staging/$(module_of "$v")"
    if ! curl -sfL --max-time 300 "$google_maven/$a/$litert/$a-$litert.aar" -o "$staging/$a.aar" ||
        ! unzip -q -j -o "$staging/$a.aar" 'jni/arm64-v8a/libLiteRt*.so' -d "$staging/$(module_of "$v")"; then
        echo "fetch_android_npu.sh: cannot fetch com.google.ai.edge.litert:$a:$litert" >&2
        exit 1
    fi
done

# Qualcomm's runtime before anything is replaced: a staging that would
# leave the shims without it is no staging at all.
mkdir "$staging/qnn"
if ! fetch_aar qnn-runtime "$qnn" "$staging/qnn"; then
    echo "fetch_android_npu.sh: cannot fetch com.qualcomm.qti:qnn-runtime:$qnn" >&2
    exit 1
fi
if ! fetch_aar onnxruntime-android-qnn "$ort_qnn" "$staging/qnn"; then
    echo "fetch_android_npu.sh: cannot fetch com.qualcomm.qti:onnxruntime-android-qnn:$ort_qnn" >&2
    exit 1
fi

# place <library> <module> <origin>: one library into its module's set,
# gathered under $staging/out before anything staged is replaced.
out=$staging/out
n=0
place() {
    if [ ! -f "$npu/$2/AndroidManifest.xml" ]; then
        echo "fetch_android_npu.sh: $(basename "$1") belongs in app/android/npu/$2, which does" \
            "not exist: add the module and its device group (app/AGENTS.md, NPU delivery)" >&2
        exit 1
    fi
    mkdir -p "$out/$2"
    cp "$1" "$out/$2/"
    n=$((n + 1))
    echo "staged $(basename "$1") into $2 ($3)"
}
for v in $vendors; do
    module=$(module_of "$v")
    for so in "$staging/$module"/libLiteRt*.so; do
        place "$so" "$module" "LiteRT $litert"
    done
    if [ "$v" = qualcomm ]; then
        for so in "$staging"/qnn/*.so; do
            base=$(basename "$so")
            case "$base" in
                libonnxruntime_providers_qnn.so) from="onnxruntime-android-qnn $ort_qnn" ;;
                *)                               from="qnn-runtime $qnn" ;;
            esac
            case "$base" in
                libQnnHtpV[0-9]*Skel.so|libQnnHtpV[0-9]*Stub.so)
                    gen=$(echo "$base" | sed 's/^libQnnHtpV\([0-9]*\).*/\1/')
                    place "$so" "qualcomm_runtime_v$gen" "$from" ;;
                *)
                    place "$so" qualcomm_runtime "$from" ;;
            esac
        done
    fi
done

# Whatever was staged before goes: in the modules, and in the base, where
# an older version of this script staged everything for every phone.
rm -rf "$here/app/android/app/src/main/jniLibs"
for d in "$npu"/*/; do
    rm -rf "${d}jniLibs"
done
for d in "$out"/*/; do
    m=$(basename "$d")
    mkdir -p "$npu/$m/jniLibs/arm64-v8a"
    cp "$d"*.so "$npu/$m/jniLibs/arm64-v8a/"
done
echo "Staged $n libraries ($vendors) into the NPU modules in app/android/npu;"
echo "the next Android build packages them."
