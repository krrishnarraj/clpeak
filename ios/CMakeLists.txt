cmake_minimum_required(VERSION 3.24)
project(clpeak_ios LANGUAGES C CXX OBJCXX Swift)

set(CMAKE_OSX_DEPLOYMENT_TARGET 16.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_XCODE_GENERATE_SCHEME TRUE)
set(CMAKE_XCODE_SCHEME_ENABLE_GPU_API_VALIDATION FALSE)
set(CMAKE_XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE DISABLED)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
get_filename_component(CLPEAK_GIT_ROOT "${CLPEAK_ROOT}" ABSOLUTE)
include(${CLPEAK_ROOT}/src/common/cmake/version.cmake)

set(CLPEAK_ENABLE_OPENCL OFF CACHE BOOL "" FORCE)
set(CLPEAK_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(CLPEAK_ENABLE_METAL ON CACHE BOOL "" FORCE)

# ---- Vulkan backend (optional, ON by default) -------------------------------
option(CLPEAK_IOS_ENABLE_VULKAN "Enable Vulkan (MoltenVK) backend on iOS" ON)

set(CLPEAK_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
set(CLPEAK_IOS_VULKAN_AVAILABLE OFF)

if(CLPEAK_IOS_ENABLE_VULKAN)
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
    set(CLPEAK_MOLTENVK_XCFRAMEWORK "${CLPEAK_IOS_VULKAN_SDK}/lib/MoltenVK.xcframework")
    set(CLPEAK_VALIDATION_FRAMEWORK "${CLPEAK_IOS_VULKAN_SDK}/lib/VkLayer_khronos_validation.framework")
    set(CLPEAK_VULKAN_RESOURCE_DIR "${CLPEAK_IOS_VULKAN_SDK}/share/vulkan")
  else()
    message(WARNING "Vulkan SDK for iOS not found — Vulkan backend disabled. "
                    "Set CLPEAK_IOS_VULKAN_SDK to the SDK iOS root "
                    "(e.g. \$HOME/VulkanSDK/1.4.350.0/iOS) or pass "
                    "-DCLPEAK_IOS_ENABLE_VULKAN=OFF to silence this warning.")
  endif()
else()
  message(STATUS "Vulkan backend disabled (CLPEAK_IOS_ENABLE_VULKAN=OFF)")
endif()

add_subdirectory(${CLPEAK_ROOT}/src/common ${CMAKE_BINARY_DIR}/common)
if(CLPEAK_IOS_VULKAN_AVAILABLE)
  add_subdirectory(${CLPEAK_ROOT}/src/vulkan ${CMAKE_BINARY_DIR}/vulkan)
endif()
add_subdirectory(${CLPEAK_ROOT}/src/metal ${CMAKE_BINARY_DIR}/metal)

# Native CPU backend — plain C++ / std::thread, no external deps.
set(CLPEAK_ENABLE_CPU ON CACHE BOOL "" FORCE)
add_subdirectory(${CLPEAK_ROOT}/src/cpu ${CMAKE_BINARY_DIR}/cpu)

set(CLPEAK_IOS_SWIFT_SOURCES
  App/ClpeakApp.swift
  App/Models.swift
  App/BenchmarkViewModel.swift
  App/SetupView.swift
  App/ResultsView.swift
  App/AboutView.swift
)

set(CLPEAK_IOS_ASSET_CATALOG "${CMAKE_CURRENT_SOURCE_DIR}/Resources/Assets.xcassets")

# Native sources shared by both device and simulator targets
set(CLPEAK_IOS_NATIVE_SOURCES
  Native/entry_ios.mm
  Native/logger_ios.mm
  Native/clpeak_ios_bridge.h
  Native/logger_ios.h
)

# ---- Shared configuration for both iOS targets ------------------------------
set(CLPEAK_IOS_BASE_SOURCES
  ${CLPEAK_IOS_SWIFT_SOURCES}
  ${CLPEAK_IOS_NATIVE_SOURCES}
  ${CLPEAK_IOS_ASSET_CATALOG}
)

set(CLPEAK_IOS_COMMON_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/Native
  ${CLPEAK_ROOT}/include
  ${CLPEAK_ROOT}/include/common
)

set(CLPEAK_IOS_BASE_LINK_LIBS peak_common peak_metal peak_cpu)
set(CLPEAK_IOS_BASE_DEFINES ENABLE_METAL ENABLE_CPU)

# ---- Device target ----------------------------------------------------------
set(_clpeak_ios_device_sources ${CLPEAK_IOS_BASE_SOURCES})
if(CLPEAK_IOS_VULKAN_AVAILABLE)
  list(APPEND _clpeak_ios_device_sources ${CLPEAK_VULKAN_RESOURCE_DIR})
endif()

add_executable(clpeak_ios MACOSX_BUNDLE ${_clpeak_ios_device_sources})

set(_clpeak_ios_link_libs ${CLPEAK_IOS_BASE_LINK_LIBS})
if(CLPEAK_IOS_VULKAN_AVAILABLE)
  list(APPEND _clpeak_ios_link_libs peak_vulkan "${CLPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework")
endif()
target_link_libraries(clpeak_ios PRIVATE ${_clpeak_ios_link_libs})

target_include_directories(clpeak_ios PRIVATE ${CLPEAK_IOS_COMMON_INCLUDE_DIRS})

set(_clpeak_ios_defines ${CLPEAK_IOS_BASE_DEFINES})
if(CLPEAK_IOS_VULKAN_AVAILABLE)
  list(APPEND _clpeak_ios_defines ENABLE_VULKAN)
endif()
target_compile_definitions(clpeak_ios PRIVATE ${_clpeak_ios_defines})
target_compile_options(clpeak_ios PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
clpeak_setup_version(clpeak_ios GIT_ROOT ${CLPEAK_GIT_ROOT})

if(CLPEAK_IOS_VULKAN_AVAILABLE)
  set_source_files_properties(${CLPEAK_VULKAN_RESOURCE_DIR}
    PROPERTIES
    MACOSX_PACKAGE_LOCATION Resources
  )
endif()

set_source_files_properties(${CLPEAK_IOS_ASSET_CATALOG}
  PROPERTIES
  MACOSX_PACKAGE_LOCATION Resources
)

set(_clpeak_embed_frameworks "")
if(CLPEAK_IOS_VULKAN_AVAILABLE)
  list(APPEND _clpeak_embed_frameworks
    "${CLPEAK_MOLTENVK_XCFRAMEWORK}"
    "${CLPEAK_IOS_VULKAN_SDK}/lib/vulkan.framework"
  )
  if(EXISTS "${CLPEAK_VALIDATION_FRAMEWORK}")
    list(APPEND _clpeak_embed_frameworks "${CLPEAK_VALIDATION_FRAMEWORK}")
  endif()
endif()

# ---- Common target properties ----------------------------------------------
foreach(_ios_target clpeak_ios clpeak_ios_simulator)
  set_target_properties(${_ios_target} PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "16.0"
    XCODE_ATTRIBUTE_SWIFT_VERSION "5.0"
    XCODE_ATTRIBUTE_SWIFT_OBJC_BRIDGING_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/Native/clpeak_ios_bridge.h"
    XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES "YES"
    XCODE_ATTRIBUTE_LD_RUNPATH_SEARCH_PATHS "@executable_path/Frameworks"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
    XCODE_ATTRIBUTE_SKIP_INSTALL "NO"
    XCODE_ATTRIBUTE_INSTALL_PATH "$(LOCAL_APPS_DIR)"
    XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AppIcon"
    XCODE_GENERATE_SCHEME TRUE
  )
endforeach()

# Device-specific properties
set_target_properties(clpeak_ios PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "kr.clpeak.ios"
  XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS "iphoneos"
  XCODE_EMBED_FRAMEWORKS "${_clpeak_embed_frameworks}"
  XCODE_EMBED_FRAMEWORKS_CODE_SIGN_ON_COPY "YES"
  XCODE_EMBED_FRAMEWORKS_REMOVE_HEADERS_ON_COPY "YES"
)

# ---- Simulator target (Metal + CPU only, no Vulkan) -----------------------
# The Vulkan SDK 1.4.350 iOS loader framework is device-only
# (platform=IOS, arm64/arm64e). Provide a simulator scheme that keeps the same
# Swift UI and native bridge but runs the Metal backend only.
add_executable(clpeak_ios_simulator MACOSX_BUNDLE ${CLPEAK_IOS_BASE_SOURCES})

target_link_libraries(clpeak_ios_simulator PRIVATE ${CLPEAK_IOS_BASE_LINK_LIBS})
target_include_directories(clpeak_ios_simulator PRIVATE ${CLPEAK_IOS_COMMON_INCLUDE_DIRS})
target_compile_definitions(clpeak_ios_simulator PRIVATE ${CLPEAK_IOS_BASE_DEFINES})
target_compile_options(clpeak_ios_simulator PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")
clpeak_setup_version(clpeak_ios_simulator GIT_ROOT ${CLPEAK_GIT_ROOT})

# Simulator-specific properties
set_target_properties(clpeak_ios_simulator PROPERTIES
  XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "kr.clpeak.ios.simulator"
  XCODE_ATTRIBUTE_PRODUCT_NAME "clpeak-sim"
  XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS "iphonesimulator"
  "XCODE_ATTRIBUTE_EXCLUDED_ARCHS[sdk=iphonesimulator*]" "x86_64"
)

foreach(_sim_dep peak_common peak_metal peak_cpu)
  if(TARGET ${_sim_dep})
    set_target_properties(${_sim_dep} PROPERTIES
      "XCODE_ATTRIBUTE_EXCLUDED_ARCHS[sdk=iphonesimulator*]" "x86_64"
    )
  endif()
endforeach()
