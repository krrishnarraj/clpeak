cmake_minimum_required(VERSION 3.24)
project(xpupeak_ffi_ios LANGUAGES C CXX OBJCXX)

# Standalone superproject that builds xpupeak_ffi as an iOS dynamic framework.
# Driven by tools/build_ios_native.sh, which configures it twice (device +
# simulator) and combines the products into xpupeak_ffi.xcframework for the
# Flutter runner's local pod (app/ios/xpupeak_native/).
#
#   device:    Metal + CPU + Core ML + ONNX Runtime + LiteRT, and
#              Vulkan-over-MoltenVK when the LunarG iOS SDK is discoverable
#              (env-gated exactly like the retired ios/ app)
#   simulator: the same minus Vulkan (the loader framework is device-only)
#
# Configure with -DCMAKE_SYSTEM_NAME=iOS and either
# -DCMAKE_OSX_SYSROOT=iphoneos or =iphonesimulator.

set(CMAKE_OSX_DEPLOYMENT_TARGET 16.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(XPUPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
get_filename_component(XPUPEAK_ROOT "${XPUPEAK_ROOT}" ABSOLUTE)
set(XPUPEAK_GIT_ROOT "${XPUPEAK_ROOT}")
include(${XPUPEAK_ROOT}/src/common/cmake/version.cmake)

if(CMAKE_OSX_SYSROOT MATCHES "iphonesimulator")
    set(XPUPEAK_IOS_SIMULATOR ON)
else()
    set(XPUPEAK_IOS_SIMULATOR OFF)
endif()

set(XPUPEAK_ENABLE_OPENCL OFF CACHE BOOL "" FORCE)
set(XPUPEAK_ENABLE_CUDA   OFF CACHE BOOL "" FORCE)
set(XPUPEAK_ENABLE_METAL  ON  CACHE BOOL "" FORCE)
set(XPUPEAK_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)

# ---- Vulkan backend (device-only, optional, ON by default) ------------------
option(XPUPEAK_IOS_ENABLE_VULKAN "Enable Vulkan (MoltenVK) backend on iOS" ON)
set(XPUPEAK_IOS_VULKAN_AVAILABLE OFF)

if(XPUPEAK_IOS_ENABLE_VULKAN AND NOT XPUPEAK_IOS_SIMULATOR)
  if(NOT DEFINED XPUPEAK_IOS_VULKAN_SDK)
    if(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/lib/vulkan.framework")
      set(XPUPEAK_IOS_VULKAN_SDK "$ENV{VULKAN_SDK}" CACHE PATH "iOS Vulkan SDK root")
    elseif(EXISTS "$ENV{HOME}/VulkanSDK/1.4.350.0/iOS/lib/vulkan.framework")
      set(XPUPEAK_IOS_VULKAN_SDK "$ENV{HOME}/VulkanSDK/1.4.350.0/iOS" CACHE PATH "iOS Vulkan SDK root")
    else()
      set(XPUPEAK_IOS_VULKAN_SDK "" CACHE PATH "iOS Vulkan SDK root")
    endif()
  endif()

  if(XPUPEAK_IOS_VULKAN_SDK AND EXISTS "${XPUPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework")
    set(XPUPEAK_IOS_VULKAN_AVAILABLE ON)
    set(XPUPEAK_ENABLE_VULKAN ON CACHE BOOL "" FORCE)

    set(Vulkan_FOUND TRUE)
    set(Vulkan_VERSION "1.4.350")
    set(Vulkan_INCLUDE_DIRS "${XPUPEAK_IOS_VULKAN_SDK}/include")
    set(Vulkan_LIBRARIES "${XPUPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework")
  else()
    message(WARNING "Vulkan SDK for iOS not found — Vulkan backend disabled. "
                    "Set XPUPEAK_IOS_VULKAN_SDK to the SDK iOS root "
                    "(e.g. \$HOME/VulkanSDK/1.4.350.0/iOS) or pass "
                    "-DXPUPEAK_IOS_ENABLE_VULKAN=OFF to silence this warning.")
  endif()
endif()

add_subdirectory(${XPUPEAK_ROOT}/src/common ${CMAKE_BINARY_DIR}/common)
if(XPUPEAK_IOS_VULKAN_AVAILABLE)
  add_subdirectory(${XPUPEAK_ROOT}/src/vulkan ${CMAKE_BINARY_DIR}/vulkan)
endif()
add_subdirectory(${XPUPEAK_ROOT}/src/metal ${CMAKE_BINARY_DIR}/metal)

# Native CPU backend — plain C++ / std::thread, no external deps.
set(XPUPEAK_ENABLE_CPU ON CACHE BOOL "" FORCE)
add_subdirectory(${XPUPEAK_ROOT}/src/cpu ${CMAKE_BINARY_DIR}/cpu)

# Core ML backend — the Neural Engine, GPU and CPU through the system
# framework, no pod needed.  On the simulator Core ML enumerates the CPU (and
# GPU) only; a device gets the Neural Engine.
set(XPUPEAK_ENABLE_COREML ON CACHE BOOL "" FORCE)
add_subdirectory(${XPUPEAK_ROOT}/src/coreml ${CMAKE_BINARY_DIR}/coreml)

# ---- ONNX Runtime backend (CoreML / XNNPACK / CPU) -------------------------
# Statically linked, and it has to be: Apple's official pod ships
# onnxruntime.xcframework as a *static* framework (`static_framework = true`
# in its podspec), and iOS will not dlopen a library that was not built into
# the app to begin with.  So iOS is the one platform where the runtime is
# baked in -- which is also why --onnx-lib and the settings screen's library
# picker do nothing here: there is no second runtime to choose.
#
# tools/build_ios_native.sh fetches and unpacks the pod archive, then passes
# its path in.  Point XPUPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK at an existing
# onnxruntime.xcframework to use your own build; the backend is left out
# entirely when it is absent.
set(XPUPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK "" CACHE PATH
    "onnxruntime.xcframework to link into xpupeak_ffi (empty = no ONNX backend)")
set(XPUPEAK_IOS_ONNX_AVAILABLE OFF)

if(XPUPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK)
  if(XPUPEAK_IOS_SIMULATOR)
    set(_ort_slice "ios-arm64_x86_64-simulator")
  else()
    set(_ort_slice "ios-arm64")
  endif()
  set(_ort_fw
      "${XPUPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK}/${_ort_slice}/onnxruntime.framework")
  if(EXISTS "${_ort_fw}/onnxruntime")
    set(XPUPEAK_IOS_ONNX_AVAILABLE ON)
    set(XPUPEAK_ENABLE_ONNX ON)
    set(XPUPEAK_ONNX_STATIC ON)
    add_subdirectory(${XPUPEAK_ROOT}/src/onnx ${CMAKE_BINARY_DIR}/onnx)
  else()
    message(WARNING "No ${_ort_slice} slice under "
                    "${XPUPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK} - ONNX backend disabled.")
  endif()
endif()

# ---- LiteRT backend (CPU via XNNPACK / GPU via Metal) -----------------------
# Dynamic, as on every other platform: only the vendored C headers are needed
# to build, and libLiteRt.dylib is dlopen'd from the app's Frameworks
# directory at run time, where the Runner's embed phase puts it together with
# libLiteRtMetalAccelerator.dylib (tools/build_ios_native.sh fetches Google's
# iOS builds of both, device and simulator, and stages them under
# app/ios/xpupeak_native/embed-{device,simulator}/).  iOS loads a library from
# inside the signed bundle, which is the one place --litert-lib and the
# settings picker could never name -- so, as with ONNX Runtime, the runtime
# the app carries is the runtime it measures.  No NPU: Core ML is the route
# to the Neural Engine on Apple platforms.
option(XPUPEAK_IOS_ENABLE_LITERT "Enable the LiteRT backend on iOS" ON)
if(XPUPEAK_IOS_ENABLE_LITERT)
  set(XPUPEAK_ENABLE_LITERT ON)
  add_subdirectory(${XPUPEAK_ROOT}/src/litert ${CMAKE_BINARY_DIR}/litert)
endif()

# ---- The FFI bridge framework ------------------------------------------------
set(XPUPEAK_FFI_STANDALONE ON)
add_subdirectory(${XPUPEAK_ROOT}/src/ffi ${CMAKE_BINARY_DIR}/ffi)

set_target_properties(xpupeak_ffi PROPERTIES
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "16.0"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    "XCODE_ATTRIBUTE_EXCLUDED_ARCHS[sdk=iphonesimulator*]" "x86_64"
)

if(XPUPEAK_IOS_VULKAN_AVAILABLE)
  target_link_libraries(xpupeak_ffi PRIVATE "${XPUPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework")
endif()

if(XPUPEAK_IOS_ONNX_AVAILABLE)
  # The framework binary is a plain ar archive, so it links as a static
  # library by path.  The accompanying system frameworks are the ones the
  # official podspec declares -- CoreML weak-linked, exactly as the pod does,
  # since the EP is optional at run time.
  target_link_libraries(xpupeak_ffi PRIVATE
      "${_ort_fw}/onnxruntime"
      "-weak_framework CoreML"
      "-framework CoreFoundation"
      "-framework Foundation"
      "-framework Network"
      "-framework Security"
      "-framework SystemConfiguration"
      "-framework UIKit")
endif()

foreach(_dep peak_common peak_metal peak_cpu peak_vulkan peak_onnx peak_coreml peak_litert)
  if(TARGET ${_dep})
    set_target_properties(${_dep} PROPERTIES
      "XCODE_ATTRIBUTE_EXCLUDED_ARCHS[sdk=iphonesimulator*]" "x86_64"
    )
  endif()
endforeach()
