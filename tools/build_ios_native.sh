#!/usr/bin/env bash
# Builds the clpeak_ffi iOS framework (device + simulator) and stages the
# artifacts the Flutter Runner's "Embed clpeak native frameworks" build
# phase consumes:
#
#   app/ios/clpeak_native/clpeak_ffi.xcframework       (both platforms)
#   app/ios/clpeak_native/embed-device/*.framework     (optional Vulkan loader +
#                                                       MoltenVK, device only)
#   app/ios/clpeak_native/embed-device/vulkan/         (ICD resources)
#   app/ios/clpeak_native/embed-device/*.dylib         (LiteRT runtime + Metal
#   app/ios/clpeak_native/embed-simulator/*.dylib       accelerator, per slice)
#
# Vulkan is env-gated exactly like the retired native iOS app: it is enabled
# when the LunarG iOS SDK is discoverable via $VULKAN_SDK or
# ~/VulkanSDK/1.4.350.0/iOS; otherwise the framework ships Metal + CPU only.
#
# ONNX Runtime is fetched (the official static C pod) and linked into the
# framework; --no-onnx leaves it out, and CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK
# points the build at a local one instead.
#
# LiteRT is fetched (Google's iOS dylibs for the release the vendored headers
# came from) and embedded beside the framework, to be dlopen'd like on every
# other platform; --no-litert leaves the backend out, and CLPEAK_IOS_LITERT_DIR
# points at a local directory holding ios_arm64/ and ios_sim_arm64/ instead.
#
# Usage: tools/build_ios_native.sh [--no-vulkan] [--no-onnx] [--no-litert]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/app/ios/clpeak_native"
BUILD="$ROOT/build-ios"

# The build trees under build-ios/ keep their CMake cache between runs, so
# every switch is passed explicitly, ON or OFF: a --no-* from one run must
# not linger into the next.
EXTRA_ARGS=()
VULKAN=ON
NO_ONNX=0
NO_LITERT=0
for arg in "$@"; do
    case "$arg" in
        --no-vulkan) VULKAN=OFF ;;
        --no-onnx)   NO_ONNX=1 ;;
        --no-litert) NO_LITERT=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done
EXTRA_ARGS+=(-DCLPEAK_IOS_ENABLE_VULKAN=$VULKAN)

# ---- ONNX Runtime (static, from the official C pod archive) -----------------
# iOS is the one platform where the runtime cannot be dlopen'd: Apple's pod is
# a static framework and iOS will not load a library that was not built into
# the app.  So it is fetched here and linked in.  Cached under build-ios/, not
# checked in -- it is 61 MB.  Set CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK to point
# at your own build instead, or pass --no-onnx to leave the backend out.
ORT_POD_VERSION="${CLPEAK_IOS_ONNXRUNTIME_VERSION:-1.30.0}"
ORT_DIR="$BUILD/onnxruntime/$ORT_POD_VERSION"

if [[ $NO_ONNX -eq 0 ]]; then
    if [[ -n "${CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK:-}" ]]; then
        ORT_XCFRAMEWORK="$CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK"
    else
        ORT_XCFRAMEWORK="$ORT_DIR/onnxruntime.xcframework"
        if [[ ! -d "$ORT_XCFRAMEWORK" ]]; then
            echo "==> Fetching ONNX Runtime $ORT_POD_VERSION (iOS pod archive)"
            mkdir -p "$ORT_DIR"
            curl -fsSL -o "$ORT_DIR/pod.zip" \
                "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-$ORT_POD_VERSION.zip"
            unzip -oq "$ORT_DIR/pod.zip" -d "$ORT_DIR"
            rm -f "$ORT_DIR/pod.zip"
        fi
    fi
    if [[ ! -d "$ORT_XCFRAMEWORK" ]]; then
        echo "==> ONNX Runtime not available at $ORT_XCFRAMEWORK — skipping backend"
        ORT_XCFRAMEWORK=""
    fi
else
    ORT_XCFRAMEWORK=""
fi
EXTRA_ARGS+=(-DCLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK="$ORT_XCFRAMEWORK")

# ---- LiteRT (dynamic, Google's iOS binaries) --------------------------------
# Unlike ONNX Runtime, LiteRT ships iOS as plain dylibs built to live in an
# app's Frameworks directory (install name @rpath/libLiteRt.dylib, rpath
# @executable_path/Frameworks), and iOS will dlopen a library that is inside
# the signed bundle -- so the backend keeps the load-on-demand design it has
# everywhere else, and only the packaging is new.  The binaries are not on
# CocoaPods (the LiteRTC pod is the old Interpreter API) nor in the GitHub
# release; Google publishes them per version in its litert bucket, the
# runtime and the Metal accelerator for the device and the simulator, ~8 MB
# each.  The version is the one the vendored headers came from
# (third_party/litert/README.md): the ABI is checked at load, and a mismatch
# is a backend that reports itself absent.  Cached under build-ios/.
LITERT_VERSION="${CLPEAK_IOS_LITERT_VERSION:-$(sed -n 's/^- \*\*Tag:\*\* `v\([0-9.]*\)`.*/\1/p' \
    "$ROOT/third_party/litert/README.md")}"
LITERT_DIR="$BUILD/litert/$LITERT_VERSION"
LITERT_BUCKET="https://storage.googleapis.com/litert/binaries/$LITERT_VERSION"
LITERT_FILES=(libLiteRt.dylib libLiteRtMetalAccelerator.dylib)
LITERT_SRC=""

if [[ $NO_LITERT -eq 0 ]]; then
    if [[ -z "$LITERT_VERSION" ]]; then
        echo "error: could not read the LiteRT tag from third_party/litert/README.md" >&2
        exit 1
    fi
    if [[ -n "${CLPEAK_IOS_LITERT_DIR:-}" ]]; then
        LITERT_SRC="$CLPEAK_IOS_LITERT_DIR"
    else
        LITERT_SRC="$LITERT_DIR"
        for slice in ios_arm64 ios_sim_arm64; do
            for f in "${LITERT_FILES[@]}"; do
                if [[ ! -f "$LITERT_DIR/$slice/$f" ]]; then
                    echo "==> Fetching LiteRT $LITERT_VERSION $slice/$f"
                    mkdir -p "$LITERT_DIR/$slice"
                    curl -fsSL -o "$LITERT_DIR/$slice/$f.part" "$LITERT_BUCKET/$slice/$f"
                    mv "$LITERT_DIR/$slice/$f.part" "$LITERT_DIR/$slice/$f"
                fi
            done
        done
    fi
    for slice in ios_arm64 ios_sim_arm64; do
        for f in "${LITERT_FILES[@]}"; do
            if [[ ! -f "$LITERT_SRC/$slice/$f" ]]; then
                echo "==> LiteRT not available at $LITERT_SRC/$slice/$f — skipping backend"
                LITERT_SRC=""
                break 2
            fi
        done
    done
fi
if [[ -n "$LITERT_SRC" ]]; then
    EXTRA_ARGS+=(-DCLPEAK_IOS_ENABLE_LITERT=ON)
else
    EXTRA_ARGS+=(-DCLPEAK_IOS_ENABLE_LITERT=OFF)
fi

configure_and_build() {
    local dir="$1" sysroot="$2"
    cmake -B "$dir" -G Xcode \
        -DCMAKE_SYSTEM_NAME=iOS \
        -DCMAKE_OSX_SYSROOT="$sysroot" \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        "${EXTRA_ARGS[@]:-}" \
        "$ROOT/src/ffi/ios"
    cmake --build "$dir" --config Release --target clpeak_ffi -- \
        CODE_SIGNING_ALLOWED=NO
}

echo "==> Building clpeak_ffi for iphoneos"
configure_and_build "$BUILD/device" iphoneos

echo "==> Building clpeak_ffi for iphonesimulator (arm64)"
configure_and_build "$BUILD/simulator" iphonesimulator

echo "==> Creating xcframework"
rm -rf "$OUT/clpeak_ffi.xcframework" "$OUT/embed-device" "$OUT/embed-simulator"
mkdir -p "$OUT"
# clpeak_ffi's LIBRARY_OUTPUT_DIRECTORY places frameworks at
# <build>/<config>/ regardless of platform.
xcodebuild -create-xcframework \
    -framework "$BUILD/device/Release/clpeak_ffi.framework" \
    -framework "$BUILD/simulator/Release/clpeak_ffi.framework" \
    -output "$OUT/clpeak_ffi.xcframework"

# ---- Optional Vulkan runtime pieces (device only) ---------------------------
VULKAN_SDK_ROOT="$(sed -n 's/^CLPEAK_IOS_VULKAN_SDK:PATH=//p' \
    "$BUILD/device/CMakeCache.txt" 2>/dev/null || true)"
if [[ -n "$VULKAN_SDK_ROOT" && -d "$VULKAN_SDK_ROOT/lib/vulkan.framework" ]]; then
    echo "==> Staging Vulkan loader + MoltenVK from $VULKAN_SDK_ROOT"
    mkdir -p "$OUT/embed-device"
    cp -R "$VULKAN_SDK_ROOT/lib/vulkan.framework" "$OUT/embed-device/"
    # Pick the device slice out of MoltenVK.xcframework (dynamic framework).
    MVK_SLICE="$(find "$VULKAN_SDK_ROOT/lib/MoltenVK.xcframework" \
        -maxdepth 2 -name "MoltenVK.framework" -path "*ios-arm64*" \
        ! -path "*simulator*" | head -1 || true)"
    if [[ -n "$MVK_SLICE" ]]; then
        cp -R "$MVK_SLICE" "$OUT/embed-device/"
    fi
    if [[ -d "$VULKAN_SDK_ROOT/share/vulkan" ]]; then
        cp -R "$VULKAN_SDK_ROOT/share/vulkan" "$OUT/embed-device/vulkan"
    fi
else
    echo "==> Vulkan SDK not found — Metal + CPU only"
fi

# ---- LiteRT runtime + Metal accelerator (both slices) ----------------------
# Google's device slice is unsigned and the simulator's ad-hoc signed; the
# embed phase signs whatever it copies with the app's identity, as it does
# the frameworks.  The two must share a directory: the runtime finds the
# accelerator as <its own directory>/libLiteRtMetalAccelerator.dylib.
if [[ -n "$LITERT_SRC" ]]; then
    echo "==> Staging LiteRT $LITERT_VERSION from $LITERT_SRC"
    mkdir -p "$OUT/embed-device" "$OUT/embed-simulator"
    for f in "${LITERT_FILES[@]}"; do
        cp "$LITERT_SRC/ios_arm64/$f" "$OUT/embed-device/$f"
        cp "$LITERT_SRC/ios_sim_arm64/$f" "$OUT/embed-simulator/$f"
    done
fi

echo "==> Done. Artifacts staged under $OUT"
