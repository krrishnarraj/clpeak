cmake_minimum_required(VERSION 3.24)
project(clpeak_ffi_ios LANGUAGES C CXX OBJCXX)

# Standalone superproject that builds clpeak_ffi as an iOS dynamic framework.
# Driven by tool/build_ios_native.sh, which configures it twice (device +
# simulator) and combines the products into clpeak_ffi.xcframework for the
# Flutter runner's local pod (app/ios/clpeak_native/).
#
#   device:    Metal + CPU + Vulkan-over-MoltenVK (when the LunarG iOS SDK
#              is discoverable; env-gated exactly like the retired ios/ app)
#   simulator: Metal + CPU only (the Vulkan loader framework is device-only)
#
# Configure with -DCMAKE_SYSTEM_NAME=iOS and either
# -DCMAKE_OSX_SYSROOT=iphoneos or =iphonesimulator.

set(CMAKE_OSX_DEPLOYMENT_TARGET 16.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
get_filename_component(CLPEAK_ROOT "${CLPEAK_ROOT}" ABSOLUTE)
set(CLPEAK_GIT_ROOT "${CLPEAK_ROOT}")
include(${CLPEAK_ROOT}/src/common/cmake/version.cmake)

if(CMAKE_OSX_SYSROOT MATCHES "iphonesimulator")
    set(CLPEAK_IOS_SIMULATOR ON)
else()
    set(CLPEAK_IOS_SIMULATOR OFF)
endif()

set(CLPEAK_ENABLE_OPENCL OFF CACHE BOOL "" FORCE)
set(CLPEAK_ENABLE_CUDA   OFF CACHE BOOL "" FORCE)
set(CLPEAK_ENABLE_METAL  ON  CACHE BOOL "" FORCE)
set(CLPEAK_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)

# ---- Vulkan backend (device-only, optional, ON by default) ------------------
option(CLPEAK_IOS_ENABLE_VULKAN "Enable Vulkan (MoltenVK) backend on iOS" ON)
set(CLPEAK_IOS_VULKAN_AVAILABLE OFF)

if(CLPEAK_IOS_ENABLE_VULKAN AND NOT CLPEAK_IOS_SIMULATOR)
  if(NOT DEFINED CLPEAK_IOS_VULKAN_SDK)
    if(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/lib/vulkan.framework")
      set(CLPEAK_IOS_VULKAN_SDK "$ENV{VULKAN_SDK}" CACHE PATH "iOS Vulkan SDK root")
    elseif(EXISTS "$ENV{HOME}/VulkanSDK/1.4.350.0/iOS/lib/vulkan.framework")
      set(CLPEAK_IOS_VULKAN_SDK "$ENV{HOME}/VulkanSDK/1.4.350.0/iOS" CACHE PATH "iOS Vulkan SDK root")
    else()
      set(CLPEAK_IOS_VULKAN_SDK "" CACHE PATH "iOS Vulkan SDK root")
    endif()
  endif()

  if(CLPEAK_IOS_VULKAN_SDK AND EXISTS "${CLPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework")
    set(CLPEAK_IOS_VULKAN_AVAILABLE ON)
    set(CLPEAK_ENABLE_VULKAN ON CACHE BOOL "" FORCE)

    set(Vulkan_FOUND TRUE)
    set(Vulkan_VERSION "1.4.350")
    set(Vulkan_INCLUDE_DIRS "${CLPEAK_IOS_VULKAN_SDK}/include")
    set(Vulkan_LIBRARIES "${CLPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework")
  else()
    message(WARNING "Vulkan SDK for iOS not found — Vulkan backend disabled. "
                    "Set CLPEAK_IOS_VULKAN_SDK to the SDK iOS root "
                    "(e.g. \$HOME/VulkanSDK/1.4.350.0/iOS) or pass "
                    "-DCLPEAK_IOS_ENABLE_VULKAN=OFF to silence this warning.")
  endif()
endif()

add_subdirectory(${CLPEAK_ROOT}/src/common ${CMAKE_BINARY_DIR}/common)
if(CLPEAK_IOS_VULKAN_AVAILABLE)
  add_subdirectory(${CLPEAK_ROOT}/src/vulkan ${CMAKE_BINARY_DIR}/vulkan)
endif()
add_subdirectory(${CLPEAK_ROOT}/src/metal ${CMAKE_BINARY_DIR}/metal)

# Native CPU backend — plain C++ / std::thread, no external deps.
set(CLPEAK_ENABLE_CPU ON CACHE BOOL "" FORCE)
add_subdirectory(${CLPEAK_ROOT}/src/cpu ${CMAKE_BINARY_DIR}/cpu)

# ---- ONNX Runtime backend (CoreML / XNNPACK / CPU) -------------------------
# Statically linked, and it has to be: Apple's official pod ships
# onnxruntime.xcframework as a *static* framework (`static_framework = true`
# in its podspec), and iOS will not dlopen a library that was not built into
# the app to begin with.  So iOS is the one platform where the runtime is
# baked in -- which is also why --onnx-lib and the settings screen's library
# picker do nothing here: there is no second runtime to choose.
#
# tool/build_ios_native.sh fetches and unpacks the pod archive, then passes
# its path in.  Point CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK at an existing
# onnxruntime.xcframework to use your own build; the backend is left out
# entirely when it is absent.
set(CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK "" CACHE PATH
    "onnxruntime.xcframework to link into clpeak_ffi (empty = no ONNX backend)")
set(CLPEAK_IOS_ONNX_AVAILABLE OFF)

if(CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK)
  if(CLPEAK_IOS_SIMULATOR)
    set(_ort_slice "ios-arm64_x86_64-simulator")
  else()
    set(_ort_slice "ios-arm64")
  endif()
  set(_ort_fw
      "${CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK}/${_ort_slice}/onnxruntime.framework")
  if(EXISTS "${_ort_fw}/onnxruntime")
    set(CLPEAK_IOS_ONNX_AVAILABLE ON)
    set(CLPEAK_ENABLE_ONNX ON)
    set(CLPEAK_ONNX_STATIC ON)
    add_subdirectory(${CLPEAK_ROOT}/src/onnx ${CMAKE_BINARY_DIR}/onnx)
  else()
    message(WARNING "No ${_ort_slice} slice under "
                    "${CLPEAK_IOS_ONNXRUNTIME_XCFRAMEWORK} - ONNX backend disabled.")
  endif()
endif()

# ---- The FFI bridge framework ------------------------------------------------
set(CLPEAK_FFI_STANDALONE ON)
add_subdirectory(${CLPEAK_ROOT}/src/ffi ${CMAKE_BINARY_DIR}/ffi)

set_target_properties(clpeak_ffi PROPERTIES
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "16.0"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    "XCODE_ATTRIBUTE_EXCLUDED_ARCHS[sdk=iphonesimulator*]" "x86_64"
)

if(CLPEAK_IOS_VULKAN_AVAILABLE)
  target_link_libraries(clpeak_ffi PRIVATE "${CLPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework")
endif()

if(CLPEAK_IOS_ONNX_AVAILABLE)
  # The framework binary is a plain ar archive, so it links as a static
  # library by path.  The accompanying system frameworks are the ones the
  # official podspec declares -- CoreML weak-linked, exactly as the pod does,
  # since the EP is optional at run time.
  target_link_libraries(clpeak_ffi PRIVATE
      "${_ort_fw}/onnxruntime"
      "-weak_framework CoreML"
      "-framework CoreFoundation"
      "-framework Foundation"
      "-framework Network"
      "-framework Security"
      "-framework SystemConfiguration"
      "-framework UIKit")
endif()

foreach(_dep peak_common peak_metal peak_cpu peak_vulkan peak_onnx)
  if(TARGET ${_dep})
    set_target_properties(${_dep} PROPERTIES
      "XCODE_ATTRIBUTE_EXCLUDED_ARCHS[sdk=iphonesimulator*]" "x86_64"
    )
  endif()
endforeach()
