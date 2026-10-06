#!/bin/sh
# fetch_android_npu.sh — stage the NPU libraries the Android app packages.
#
#   tools/fetch_android_npu.sh qualcomm|google_tensor|all [tag]
#                                       tag: the LiteRT release whose shims
#                                       are staged, by default the one
#                                       pinned in third_party/litert/README.md
#
# Two backends reach an Android NPU -- LiteRT through vendor shims, ONNX
# Runtime through Qualcomm's plugin execution provider -- and neither's
# vendor libraries come with the runtimes the Gradle build packages (the
# onnxruntime-android and LiteRT AARs).  This script stages them in
# app/android/app/src/main/jniLibs/arm64-v8a/ (ignored by git), replacing
# whatever was staged before, and the build packages what it finds there:
# nothing staged, no NPU libraries, which is the CI APK's case.
#
# LiteRT reaches an NPU through a vendor "dispatch" library
# (libLiteRtDispatch_<Vendor>.so) and, for on-device compilation, a compiler
# plugin (libLiteRtCompilerPlugin_<Vendor>.so).  Google publishes them as
# `litert_npu_runtime_libraries_jit.zip` on each GitHub release, not on
# Maven, laid out as Play feature modules -- one per Hexagon generation for
# Qualcomm, one for Google Tensor (G3 and later).
#
# `all` stages every vendor, for the one APK that serves a Pixel and a
# Snapdragon alike.  LiteRT loads the first libLiteRtDispatch_* it lists in
# its dispatch directory and warns about the rest (litert_dispatch.cc), so
# the app does not hand it the lib dir then: at launch the LiteRT backend
# picks the vendor of this SoC (ro.soc.manufacturer) and links that
# vendor's shims into a directory of their own under the app's support
# directory (litertStageNpuVendor in src/litert/litert_peak.cpp).  Staging
# anything switches the app to extracting its native libraries at install
# (build.gradle.kts), since both LiteRT and the backend find shims by
# listing a directory, which an APK's internal lib/ path is not.
#
# The vendor runtime the shims drive is not in that zip.  MediaTek's and
# Google Tensor's live on the device as system libraries.  Qualcomm's -- the
# QNN libraries of its QAIRT SDK -- must be bundled by the app, and serves
# both backends, so `qualcomm` stages three things beside the shims, all of
# one QNN build:
#  - com.qualcomm.qti:qnn-runtime from Maven Central: libQnnHtp,
#    libQnnSystem, libQnnHtpPrepare and every Hexagon generation's
#    Skel/Stub, 67 MB compressed;
#  - libQnnIr.so and libQnnSaver.so from the QAIRT SDK of that build.  The
#    Maven package leaves them out, and LiteRT's compiler plugin is linked
#    against them.  It never calls either -- LiteRT opens libQnnIr only for
#    its IR backend and libQnnSaver only when saver_output_dir is set, and
#    clpeak asks for neither -- but Android's linker refuses a library one
#    of whose DT_NEEDED entries it cannot find, and without the plugin
#    LiteRT compiles nothing for the NPU.  The SDK is a 2.6 GB zip on
#    Qualcomm's software center (where LiteRT's own build fetches it); its
#    server answers byte ranges, so python3 reads the zip's directory and
#    the two entries, about 3 MB;
#  - com.qualcomm.qti:onnxruntime-android-qnn from Maven Central: one
#    libonnxruntime_providers_qnn.so, which the app registers on the
#    onnxruntime-android it packages (SettingsService.effectiveOnnxEpLibraries).
# The build refuses a Qualcomm set with any of the QNN libraries missing:
# it would install, run and list no Hexagon NPU.  A Play release could
# instead ship one feature module per Hexagon generation, delivered by
# device group, which is the layout Google's zip comes in (its
# fetch_qualcomm_library.sh fills them from the same SDK).
#
# Uses curl and unzip from the base system, and python3 for Qualcomm.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
readme=$here/third_party/litert/README.md
vendor=${1:-}
case "$vendor" in
    qualcomm)      vendors=qualcomm ;;
    google_tensor) vendors=google_tensor ;;
    all)           vendors="qualcomm google_tensor" ;;
    *)
        echo "usage: tools/fetch_android_npu.sh qualcomm|google_tensor|all [tag]" >&2
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

# Qualcomm's set, one QNN build for every library in the APK: the Maven
# runtime, the QAIRT SDK it was built from (2.50.0 carries
# v2.50.0.260828221209, and the SDK's LICENSE.pdf and NOTICE.txt), and the
# ONNX Runtime plugin Qualcomm validated against that runtime.  LiteRT
# 2.2.0's shims were built against QAIRT 2.47; its QnnManager accepts a
# runtime whose API minor version is newer, with a warning, and refuses
# only a major mismatch.  Bump the three together.
qnn=2.50.0
qairt=2.50.0.260828
ort_qnn=2.6.0
maven=https://repo1.maven.org/maven2/com/qualcomm/qti
qairt_url=https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$qairt/v$qairt.zip
qairt_libs="libQnnIr.so libQnnSaver.so"

staging=$(mktemp -d "${TMPDIR:-/tmp}/clpeak-android-npu.XXXXXX")
trap 'rm -rf "$staging"' EXIT INT TERM

# fetch_aar <artifact> <version> <dir>: the arm64 libraries of one of
# Qualcomm's AARs on Maven Central into <dir>.
fetch_aar() {
    curl -sfL --max-time 900 "$maven/$1/$2/$1-$2.aar" -o "$staging/$1.aar" &&
        unzip -q -j -o "$staging/$1.aar" 'jni/arm64-v8a/*.so' -d "$3"
}

# fetch_qairt <dir> <library>...: the named aarch64-android libraries of the
# QAIRT SDK into <dir>, read out of the zip on Qualcomm's server by HTTP
# range.
fetch_qairt() {
    python3 - "$qairt_url" "qairt/$qairt/lib/aarch64-android" "$@" <<'EOF'
import io, sys, urllib.request, zipfile

url, prefix, out, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]

class Ranged(io.RawIOBase):
    def __init__(self, url):
        req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 206:
                sys.exit('%s does not serve byte ranges' % url)
            self.url = r.geturl()   # past the redirect, once
            self.size = int(r.headers['Content-Range'].rsplit('/', 1)[1])
        self.pos = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + off
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={'Range': 'bytes=%d-%d' % (self.pos, self.pos + n - 1)})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

sdk = zipfile.ZipFile(io.BufferedReader(Ranged(url), 1 << 16))
for name in names:
    try:
        data = sdk.read(prefix + '/' + name)   # CRC-checked
    except KeyError:
        sys.exit('%s has no %s/%s' % (url, prefix, name))
    with open(out + '/' + name, 'wb') as f:
        f.write(data)
EOF
}

if ! curl -sfL --max-time 300 "$repo/releases/download/$tag/litert_npu_runtime_libraries_jit.zip" \
        -o "$staging/npu.zip"; then
    echo "fetch_android_npu.sh: cannot fetch litert_npu_runtime_libraries_jit.zip at $tag" >&2
    exit 1
fi
(cd "$staging" && unzip -q npu.zip)

# Qualcomm's runtime before anything is replaced: a staging that would
# leave the shims without it is no staging at all.
case " $vendors " in *" qualcomm "*)
    if ! command -v python3 >/dev/null 2>&1; then
        echo "fetch_android_npu.sh: python3 is needed to read $qairt_libs out of the QAIRT $qairt SDK" >&2
        exit 1
    fi
    mkdir "$staging/qnn"
    if ! fetch_aar qnn-runtime "$qnn" "$staging/qnn"; then
        echo "fetch_android_npu.sh: cannot fetch com.qualcomm.qti:qnn-runtime:$qnn" >&2
        exit 1
    fi
    if ! fetch_qairt "$staging/qnn" $qairt_libs; then
        echo "fetch_android_npu.sh: cannot fetch $qairt_libs from the QAIRT $qairt SDK ($qairt_url)" >&2
        exit 1
    fi
    if ! fetch_aar onnxruntime-android-qnn "$ort_qnn" "$staging/qnn"; then
        echo "fetch_android_npu.sh: cannot fetch com.qualcomm.qti:onnxruntime-android-qnn:$ort_qnn" >&2
        exit 1
    fi
    ;;
esac

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
        echo "staged $base (LiteRT $tag)"
    done
    if [ "$v" = qualcomm ]; then
        for so in "$staging"/qnn/*.so; do
            base=$(basename "$so")
            cp "$so" "$dest/$base"
            n=$((n + 1))
            case "$base" in
                libonnxruntime_providers_qnn.so) from="onnxruntime-android-qnn $ort_qnn" ;;
                libQnnIr.so|libQnnSaver.so)      from="QAIRT $qairt" ;;
                *)                               from="qnn-runtime $qnn" ;;
            esac
            echo "staged $base ($from)"
        done
    fi
done
echo "Staged $n libraries ($vendors) into ${dest#$here/};"
echo "the next Android build packages them."
