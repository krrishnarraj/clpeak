# clpeak_ffi — the C-ABI shared library behind every clpeak GUI.
#
# Built in two contexts:
#  - Desktop: from the root CMakeLists when CLPEAK_ENABLE_GUI is ON and the
#    Flutter SDK was found (the clpeak-gui target consumes it).
#  - Mobile: the standalone superprojects in android/ and ios/ subdirs set
#    CLPEAK_FFI_STANDALONE and add this directory directly.

if(NOT CLPEAK_FFI_STANDALONE)
    if(NOT CLPEAK_ENABLE_GUI OR NOT FLUTTER_EXECUTABLE)
        return()
    endif()
endif()

add_library(clpeak_ffi SHARED
    clpeak_ffi.cpp
    logger_ffi.cpp
)

# Only the clpeak_* C symbols are exported (CLPEAK_FFI_EXPORT).
set_target_properties(clpeak_ffi PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
if(NOT ANDROID)
    # Place the library at the build root for easy discovery (clpeak-gui,
    # CLPEAK_FFI_PATH dev override).  Android keeps CMake defaults so the
    # Gradle plugin finds/strips/packs the .so itself.
    set_target_properties(clpeak_ffi PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"   # Windows .dll
    )
endif()

# Apple platforms ship clpeak_ffi as a framework: macOS embeds it directly
# into the .app (clpeak-gui target below); iOS wraps device+simulator builds
# into an xcframework consumed by a local CocoaPods pod (tool/build_ios_native.sh).
# On iOS CMake produces the shallow framework layout automatically.
if(APPLE)
    set_target_properties(clpeak_ffi PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER kr.clpeak.ffi
        MACOSX_FRAMEWORK_BUNDLE_VERSION "${CLPEAK_VERSION_STR}"
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING "${CLPEAK_VERSION_STR}"
        INSTALL_NAME_DIR "@rpath"
    )
endif()

# Link enabled backends (targets exist only if their CMakeLists was processed)
if(TARGET peak_opencl)
    target_link_libraries(clpeak_ffi PRIVATE peak_opencl)
    target_compile_definitions(clpeak_ffi PRIVATE ENABLE_OPENCL)
endif()
if(TARGET peak_vulkan)
    target_link_libraries(clpeak_ffi PRIVATE peak_vulkan)
    target_compile_definitions(clpeak_ffi PRIVATE ENABLE_VULKAN)
endif()
if(TARGET peak_cuda)
    target_link_libraries(clpeak_ffi PRIVATE peak_cuda)
    target_compile_definitions(clpeak_ffi PRIVATE ENABLE_CUDA)
endif()
if(TARGET peak_rocm)
    target_link_libraries(clpeak_ffi PRIVATE peak_rocm)
    target_compile_definitions(clpeak_ffi PRIVATE ENABLE_ROCM)
endif()
if(TARGET peak_metal)
    target_link_libraries(clpeak_ffi PRIVATE peak_metal)
    target_compile_definitions(clpeak_ffi PRIVATE ENABLE_METAL)
endif()
if(TARGET peak_cpu)
    target_link_libraries(clpeak_ffi PRIVATE peak_cpu)
    target_compile_definitions(clpeak_ffi PRIVATE ENABLE_CPU)
endif()
if(TARGET peak_oneapi)
    target_link_libraries(clpeak_ffi PRIVATE peak_oneapi)
    target_compile_definitions(clpeak_ffi PRIVATE ENABLE_ONEAPI)
    add_sycl_to_target(TARGET clpeak_ffi SOURCES clpeak_ffi.cpp)
endif()

if(ANDROID)
    target_link_libraries(clpeak_ffi PRIVATE log)
endif()

target_include_directories(clpeak_ffi PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/version.cmake)
if(DEFINED CLPEAK_GIT_ROOT)
    clpeak_setup_version(clpeak_ffi GIT_ROOT "${CLPEAK_GIT_ROOT}")
else()
    clpeak_setup_version(clpeak_ffi)
endif()

# ---- clpeak-gui: the Flutter desktop app -----------------------------------
# Builds the Flutter app in <root>/app and assembles a complete bundle at
# ${CMAKE_BINARY_DIR}/clpeak-gui/ with the clpeak_ffi library placed where
# the Dart resolver (app/lib/src/ffi/clpeak_library.dart) expects it.  The
# Flutter-generated runner projects are deliberately left untouched — this
# target owns the final assembly, so `flutter create` upgrades can never
# break the native wiring.  (Dev flow: `flutter run` with CLPEAK_FFI_PATH.)
if(NOT CLPEAK_FFI_STANDALONE)
    set(CLPEAK_APP_DIR "${CLPEAK_ROOT}/app")
    set(CLPEAK_GUI_OUT "${CMAKE_BINARY_DIR}/clpeak-gui")

    if(APPLE)
        # ditto preserves the framework's symlink structure; codesign the
        # embedded framework first, then re-seal the app bundle.
        add_custom_target(clpeak-gui ALL
            DEPENDS clpeak_ffi
            COMMAND ${FLUTTER_EXECUTABLE} build macos --release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${CLPEAK_GUI_OUT}/clpeak.app"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLPEAK_GUI_OUT}"
            COMMAND ditto "${CLPEAK_APP_DIR}/build/macos/Build/Products/Release/clpeak.app" "${CLPEAK_GUI_OUT}/clpeak.app"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLPEAK_GUI_OUT}/clpeak.app/Contents/Frameworks"
            COMMAND ditto "$<TARGET_BUNDLE_DIR:clpeak_ffi>" "${CLPEAK_GUI_OUT}/clpeak.app/Contents/Frameworks/clpeak_ffi.framework"
            COMMAND codesign --force --sign - "${CLPEAK_GUI_OUT}/clpeak.app/Contents/Frameworks/clpeak_ffi.framework"
            COMMAND codesign --force --sign - "${CLPEAK_GUI_OUT}/clpeak.app"
            WORKING_DIRECTORY "${CLPEAK_APP_DIR}"
            COMMENT "clpeak-gui: flutter build macos + bundle assembly"
            VERBATIM)
    elseif(WIN32)
        # Flutter's Windows bundle is a flat directory; the DLL sits next to
        # the runner executable.
        add_custom_target(clpeak-gui ALL
            DEPENDS clpeak_ffi
            COMMAND ${FLUTTER_EXECUTABLE} build windows --release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${CLPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CLPEAK_APP_DIR}/build/windows/x64/runner/Release" "${CLPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:clpeak_ffi>" "${CLPEAK_GUI_OUT}/"
            WORKING_DIRECTORY "${CLPEAK_APP_DIR}"
            COMMENT "clpeak-gui: flutter build windows + bundle assembly"
            VERBATIM)
    else()
        # Linux bundle layout: bundle/{clpeak, data/, lib/*.so}; the Dart
        # resolver loads <exeDir>/lib/libclpeak_ffi.so.
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(_clpeak_flutter_arch "arm64")
        else()
            set(_clpeak_flutter_arch "x64")
        endif()
        add_custom_target(clpeak-gui ALL
            DEPENDS clpeak_ffi
            COMMAND ${FLUTTER_EXECUTABLE} build linux --release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${CLPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CLPEAK_APP_DIR}/build/linux/${_clpeak_flutter_arch}/release/bundle" "${CLPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:clpeak_ffi>" "${CLPEAK_GUI_OUT}/lib/"
            WORKING_DIRECTORY "${CLPEAK_APP_DIR}"
            COMMENT "clpeak-gui: flutter build linux + bundle assembly"
            VERBATIM)
    endif()
endif()
