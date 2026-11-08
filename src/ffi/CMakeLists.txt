# xpupeak_ffi — the C-ABI shared library behind every xpupeak GUI.
#
# Built in two contexts:
#  - Desktop: from the root CMakeLists when XPUPEAK_ENABLE_GUI is ON and the
#    Flutter SDK was found (the xpupeak-gui target consumes it).
#  - Mobile: the standalone superprojects in android/ and ios/ subdirs set
#    XPUPEAK_FFI_STANDALONE and add this directory directly.

if(NOT XPUPEAK_FFI_STANDALONE)
    if(NOT XPUPEAK_ENABLE_GUI OR NOT FLUTTER_EXECUTABLE)
        return()
    endif()
endif()

add_library(xpupeak_ffi SHARED
    xpupeak_ffi.cpp
    engine.cpp
    logger_ffi.cpp
)

# Only the xpupeak_* C symbols are exported (XPUPEAK_FFI_EXPORT).
set_target_properties(xpupeak_ffi PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
# See src/cli/CMakeLists.txt: the run document records the configuration.
target_compile_definitions(xpupeak_ffi PRIVATE XPUPEAK_BUILD_CONFIG="$<CONFIG>")
if(NOT ANDROID)
    # Place the library at the build root for easy discovery (xpupeak-gui,
    # XPUPEAK_FFI_PATH dev override).  Android keeps CMake defaults so the
    # Gradle plugin finds/strips/packs the .so itself.
    set_target_properties(xpupeak_ffi PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"   # Windows .dll
    )
endif()

# Apple platforms ship xpupeak_ffi as a framework: macOS embeds it directly
# into the .app (xpupeak-gui target below); iOS wraps device+simulator builds
# into an xcframework consumed by a local CocoaPods pod (tools/build_ios_native.sh).
# On iOS CMake produces the shallow framework layout automatically.
if(APPLE)
    set_target_properties(xpupeak_ffi PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        MACOSX_FRAMEWORK_IDENTIFIER kr.XpuPeak.ffi
        MACOSX_FRAMEWORK_BUNDLE_VERSION "${XPUPEAK_VERSION_STR}"
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING "${XPUPEAK_VERSION_STR}"
        INSTALL_NAME_DIR "@rpath"
    )
endif()

# Every backend in this build, through the shared registry (also what the
# CLI links), so both binaries run the same backends in the same order.
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/backends.cmake)
xpupeak_link_backends(xpupeak_ffi)

if(ANDROID)
    target_link_libraries(xpupeak_ffi PRIVATE log)
endif()

target_include_directories(xpupeak_ffi PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/version.cmake)
xpupeak_setup_version(xpupeak_ffi)

# ---- xpupeak-gui: the Flutter desktop app -----------------------------------
# Builds the Flutter app in <root>/app and assembles a complete bundle at
# ${CMAKE_BINARY_DIR}/xpupeak-gui/ with the xpupeak_ffi library placed where
# the Dart resolver (app/lib/src/ffi/xpupeak_library.dart) expects it.  The
# Flutter-generated runner projects are deliberately left untouched — this
# target owns the final assembly, so `flutter create` upgrades can never
# break the native wiring.  (Dev flow: `flutter run` with XPUPEAK_FFI_PATH.)
if(NOT XPUPEAK_FFI_STANDALONE)
    set(XPUPEAK_APP_DIR "${XPUPEAK_ROOT}/app")
    set(XPUPEAK_GUI_OUT "${CMAKE_BINARY_DIR}/xpupeak-gui")

    # xpupeak-engine: the process the desktop app enumerates and runs in
    # (xpupeak_engine_main in xpupeak_ffi.h).  It dlopens the library the app
    # names, so it links nothing of ours; it sits next to the runner in every
    # bundle, and at the build root beside xpupeak_ffi for the dev loop.
    add_executable(xpupeak-engine engine_host.cpp)
    set_target_properties(xpupeak-engine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    if(WIN32)
        # GUI subsystem: no console window each time the app starts one.
        set_target_properties(xpupeak-engine PROPERTIES WIN32_EXECUTABLE TRUE)
        target_link_libraries(xpupeak-engine PRIVATE shell32)
        if(MINGW)
            target_link_options(xpupeak-engine PRIVATE -municode)
        endif()
    else()
        target_link_libraries(xpupeak-engine PRIVATE ${CMAKE_DL_LIBS})
    endif()

    # Flutter lays its desktop output out under an arch directory on Linux and
    # Windows (build/<os>/<arch>/...).  Derive the tag it uses from the host.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(_xpupeak_flutter_arch "arm64")
    else()
        set(_xpupeak_flutter_arch "x64")
    endif()

    if(APPLE)
        # ditto preserves the framework's symlink structure; codesign the
        # nested code -- the embedded framework and the engine host beside the
        # runner in Contents/MacOS -- first, then re-seal the app bundle.
        add_custom_target(xpupeak-gui ALL
            DEPENDS xpupeak_ffi xpupeak-engine
            COMMAND ${FLUTTER_EXECUTABLE} build macos --release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${XPUPEAK_GUI_OUT}/XpuPeak.app"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${XPUPEAK_GUI_OUT}"
            COMMAND ditto "${XPUPEAK_APP_DIR}/build/macos/Build/Products/Release/XpuPeak.app" "${XPUPEAK_GUI_OUT}/XpuPeak.app"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${XPUPEAK_GUI_OUT}/XpuPeak.app/Contents/Frameworks"
            COMMAND ditto "$<TARGET_BUNDLE_DIR:xpupeak_ffi>" "${XPUPEAK_GUI_OUT}/XpuPeak.app/Contents/Frameworks/xpupeak_ffi.framework"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xpupeak-engine>" "${XPUPEAK_GUI_OUT}/XpuPeak.app/Contents/MacOS/"
            COMMAND codesign --force --sign - "${XPUPEAK_GUI_OUT}/XpuPeak.app/Contents/Frameworks/xpupeak_ffi.framework"
            COMMAND codesign --force --sign - "${XPUPEAK_GUI_OUT}/XpuPeak.app/Contents/MacOS/xpupeak-engine"
            COMMAND codesign --force --sign - "${XPUPEAK_GUI_OUT}/XpuPeak.app"
            WORKING_DIRECTORY "${XPUPEAK_APP_DIR}"
            COMMENT "xpupeak-gui: flutter build macos + bundle assembly"
            VERBATIM)
    elseif(WIN32)
        # Flutter's Windows bundle is a flat directory; the DLL sits next to
        # the runner executable.  The arch dir it lands in is resolved at build
        # time (see stage_windows_bundle.cmake) so an arm64 host works whichever
        # tag the SDK picks.
        #
        # The flutter call gets a target to itself, and that is load-bearing:
        # FLUTTER_EXECUTABLE is flutter.BAT, the generators put every COMMAND of
        # a target into ONE batch script, and cmd.exe transfers control
        # permanently when a .bat invokes a .bat without `call` -- anything
        # sequenced after flutter is silently skipped, build still green.
        add_custom_target(xpupeak-gui-flutter
            COMMAND ${FLUTTER_EXECUTABLE} build windows --release
            WORKING_DIRECTORY "${XPUPEAK_APP_DIR}"
            COMMENT "xpupeak-gui: flutter build windows"
            VERBATIM)
        add_dependencies(xpupeak-gui-flutter xpupeak_ffi)

        add_custom_target(xpupeak-gui ALL
            COMMAND ${CMAKE_COMMAND}
                    -DAPP_DIR=${XPUPEAK_APP_DIR}
                    -DOUT_DIR=${XPUPEAK_GUI_OUT}
                    -DARCH=${_xpupeak_flutter_arch}
                    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/stage_windows_bundle.cmake"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xpupeak_ffi>" "${XPUPEAK_GUI_OUT}/"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xpupeak-engine>" "${XPUPEAK_GUI_OUT}/"
            WORKING_DIRECTORY "${XPUPEAK_APP_DIR}"
            COMMENT "xpupeak-gui: bundle assembly"
            VERBATIM)
        add_dependencies(xpupeak-gui xpupeak-gui-flutter xpupeak-engine)
    else()
        # Linux bundle layout: bundle/{xpupeak-gui, xpupeak-engine, data/,
        # lib/*.so}; the Dart resolver loads <exeDir>/lib/libxpupeak_ffi.so,
        # and the runner loads data/xpupeak_icon.png as its window icon (GTK
        # has no .rc-style resource embedding, see
        # linux/runner/my_application.cc).
        add_custom_target(xpupeak-gui ALL
            DEPENDS xpupeak_ffi xpupeak-engine
            COMMAND ${CMAKE_COMMAND} -E env DESTDIR= ${FLUTTER_EXECUTABLE} build linux --release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${XPUPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${XPUPEAK_APP_DIR}/build/linux/${_xpupeak_flutter_arch}/release/bundle" "${XPUPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xpupeak_ffi>" "${XPUPEAK_GUI_OUT}/lib/"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:xpupeak-engine>" "${XPUPEAK_GUI_OUT}/"
            WORKING_DIRECTORY "${XPUPEAK_APP_DIR}"
            COMMENT "xpupeak-gui: flutter build linux + bundle assembly"
            VERBATIM)
    endif()

    # ---- Install rules (feed the CPack zip) --------------------------------
    # A Flutter desktop app is a bundle, not a lone executable: the runner only
    # starts if its sibling data/ (icudtl + assets) and lib/ (engine, AOT app,
    # xpupeak_ffi) travel with it.  So the bundle ships whole under gui/ and
    # bin/xpupeak-gui is a wrapper that exec's it -- giving the release layout
    #
    #   bin/xpupeak, bin/xpupeak-gui  ->  gui/{xpupeak-gui, data/, lib/}
    #
    # On macOS the equivalent is a relocatable .app, which goes to the package
    # root (a .app under bin/ would be neither clickable nor tidy).
    include(GNUInstallDirs)
    if(APPLE)
        # ditto, not install(DIRECTORY): the .app holds framework version
        # symlinks and an ad-hoc code signature, and file(INSTALL) is not
        # guaranteed to keep the seal intact.
        install(CODE "
            set(_src \"${XPUPEAK_GUI_OUT}/XpuPeak.app\")
            set(_dst \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/XpuPeak.app\")
            message(STATUS \"Installing: \${_dst}\")
            file(REMOVE_RECURSE \"\${_dst}\")
            execute_process(COMMAND ditto \"\${_src}\" \"\${_dst}\"
                            RESULT_VARIABLE _xpupeak_ditto_rc)
            if(NOT _xpupeak_ditto_rc EQUAL 0)
                message(FATAL_ERROR \"ditto of XpuPeak.app failed (\${_xpupeak_ditto_rc}) -- build the xpupeak-gui target first\")
            endif()
        ")
    else()
        if(WIN32)
            # `start` so a console invocation isn't blocked by the GUI process.
            set(_xpupeak_gui_wrapper "${CMAKE_BINARY_DIR}/gui-wrapper/xpupeak-gui.cmd")
            file(WRITE "${_xpupeak_gui_wrapper}"
"@echo off\r\n\
rem xpupeak GUI launcher -- the Flutter bundle in ..\\gui needs its sibling\r\n\
rem data\\ and DLLs, so it stays there and this wrapper starts it.\r\n\
start \"\" \"%~dp0..\\gui\\xpupeak-gui.exe\" %*\r\n")
        else()
            set(_xpupeak_gui_wrapper "${CMAKE_BINARY_DIR}/gui-wrapper/xpupeak-gui")
            file(WRITE "${_xpupeak_gui_wrapper}"
"#!/bin/sh
# xpupeak GUI launcher -- the Flutter bundle in ../gui needs its sibling data/
# and lib/, so it stays there and this wrapper exec's it.
here=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)
exec \"$here/../gui/xpupeak-gui\" \"$@\"
")
        endif()
        # Say plainly when the bundle was never staged: file(INSTALL)'s own
        # message for a missing source directory is "cannot find <dir>: File
        # exists", which reads like a collision.
        install(CODE "
            if(NOT IS_DIRECTORY \"${XPUPEAK_GUI_OUT}\")
                message(FATAL_ERROR
                    \"xpupeak-gui bundle missing at ${XPUPEAK_GUI_OUT} -- \"
                    \"build the xpupeak-gui target before packaging.\")
            endif()
        ")
        install(DIRECTORY "${XPUPEAK_GUI_OUT}/" DESTINATION "gui"
                USE_SOURCE_PERMISSIONS)
        install(PROGRAMS "${_xpupeak_gui_wrapper}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
endif()
