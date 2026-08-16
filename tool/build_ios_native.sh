#!/usr/bin/env bash
# Builds the clpeak_ffi iOS framework (device + simulator) and stages the
# artifacts the Flutter Runner's "Embed clpeak native frameworks" build
# phase consumes:
#
#   app/ios/clpeak_native/clpeak_ffi.xcframework    (both platforms)
#   app/ios/clpeak_native/embed-device/*.framework  (optional Vulkan loader +
#                                                    MoltenVK, device only)
#   app/ios/clpeak_native/embed-device/vulkan/      (ICD resources)
#
# Vulkan is env-gated exactly like the retired native iOS app: it is enabled
# when the LunarG iOS SDK is discoverable via $VULKAN_SDK or
# ~/VulkanSDK/1.4.350.0/iOS; otherwise the framework ships Metal + CPU only.
#
# Usage: tool/build_ios_native.sh [--no-vulkan]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/app/ios/clpeak_native"
BUILD="$ROOT/build-ios"

EXTRA_ARGS=()
if [[ "${1:-}" == "--no-vulkan" ]]; then
    EXTRA_ARGS+=(-DCLPEAK_IOS_ENABLE_VULKAN=OFF)
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
rm -rf "$OUT/clpeak_ffi.xcframework" "$OUT/embed-device"
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

echo "==> Done. Artifacts staged under $OUT"
