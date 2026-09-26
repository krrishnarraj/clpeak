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
    engine.cpp
    logger_ffi.cpp
)

# Only the clpeak_* C symbols are exported (CLPEAK_FFI_EXPORT).
set_target_properties(clpeak_ffi PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
# See src/cli/CMakeLists.txt: the run document records the configuration.
target_compile_definitions(clpeak_ffi PRIVATE CLPEAK_BUILD_CONFIG="$<CONFIG>")
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
# into an xcframework consumed by a local CocoaPods pod (tools/build_ios_native.sh).
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

# Every backend in this build, through the shared registry (also what the
# CLI links), so both binaries run the same backends in the same order.
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/backends.cmake)
clpeak_link_backends(clpeak_ffi)

if(ANDROID)
    target_link_libraries(clpeak_ffi PRIVATE log)
endif()

target_include_directories(clpeak_ffi PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/version.cmake)
clpeak_setup_version(clpeak_ffi)

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

    # clpeak-engine: the process the desktop app enumerates and runs in
    # (clpeak_engine_main in clpeak_ffi.h).  It dlopens the library the app
    # names, so it links nothing of ours; it sits next to the runner in every
    # bundle, and at the build root beside clpeak_ffi for the dev loop.
    add_executable(clpeak-engine engine_host.cpp)
    set_target_properties(clpeak-engine PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    if(WIN32)
        # GUI subsystem: no console window each time the app starts one.
        set_target_properties(clpeak-engine PROPERTIES WIN32_EXECUTABLE TRUE)
        target_link_libraries(clpeak-engine PRIVATE shell32)
        if(MINGW)
            target_link_options(clpeak-engine PRIVATE -municode)
        endif()
    else()
        target_link_libraries(clpeak-engine PRIVATE ${CMAKE_DL_LIBS})
    endif()

    # Flutter lays its desktop output out under an arch directory on Linux and
    # Windows (build/<os>/<arch>/...).  Derive the tag it uses from the host.
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(_clpeak_flutter_arch "arm64")
    else()
        set(_clpeak_flutter_arch "x64")
    endif()

    if(APPLE)
        # ditto preserves the framework's symlink structure; codesign the
        # nested code -- the embedded framework and the engine host beside the
        # runner in Contents/MacOS -- first, then re-seal the app bundle.
        add_custom_target(clpeak-gui ALL
            DEPENDS clpeak_ffi clpeak-engine
            COMMAND ${FLUTTER_EXECUTABLE} build macos --release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${CLPEAK_GUI_OUT}/clpeak-gui.app"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLPEAK_GUI_OUT}"
            COMMAND ditto "${CLPEAK_APP_DIR}/build/macos/Build/Products/Release/clpeak-gui.app" "${CLPEAK_GUI_OUT}/clpeak-gui.app"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CLPEAK_GUI_OUT}/clpeak-gui.app/Contents/Frameworks"
            COMMAND ditto "$<TARGET_BUNDLE_DIR:clpeak_ffi>" "${CLPEAK_GUI_OUT}/clpeak-gui.app/Contents/Frameworks/clpeak_ffi.framework"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:clpeak-engine>" "${CLPEAK_GUI_OUT}/clpeak-gui.app/Contents/MacOS/"
            COMMAND codesign --force --sign - "${CLPEAK_GUI_OUT}/clpeak-gui.app/Contents/Frameworks/clpeak_ffi.framework"
            COMMAND codesign --force --sign - "${CLPEAK_GUI_OUT}/clpeak-gui.app/Contents/MacOS/clpeak-engine"
            COMMAND codesign --force --sign - "${CLPEAK_GUI_OUT}/clpeak-gui.app"
            WORKING_DIRECTORY "${CLPEAK_APP_DIR}"
            COMMENT "clpeak-gui: flutter build macos + bundle assembly"
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
        add_custom_target(clpeak-gui-flutter
            COMMAND ${FLUTTER_EXECUTABLE} build windows --release
            WORKING_DIRECTORY "${CLPEAK_APP_DIR}"
            COMMENT "clpeak-gui: flutter build windows"
            VERBATIM)
        add_dependencies(clpeak-gui-flutter clpeak_ffi)

        add_custom_target(clpeak-gui ALL
            COMMAND ${CMAKE_COMMAND}
                    -DAPP_DIR=${CLPEAK_APP_DIR}
                    -DOUT_DIR=${CLPEAK_GUI_OUT}
                    -DARCH=${_clpeak_flutter_arch}
                    -P "${CMAKE_CURRENT_LIST_DIR}/cmake/stage_windows_bundle.cmake"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:clpeak_ffi>" "${CLPEAK_GUI_OUT}/"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:clpeak-engine>" "${CLPEAK_GUI_OUT}/"
            WORKING_DIRECTORY "${CLPEAK_APP_DIR}"
            COMMENT "clpeak-gui: bundle assembly"
            VERBATIM)
        add_dependencies(clpeak-gui clpeak-gui-flutter clpeak-engine)
    else()
        # Linux bundle layout: bundle/{clpeak-gui, clpeak-engine, data/,
        # lib/*.so}; the Dart resolver loads <exeDir>/lib/libclpeak_ffi.so,
        # and the runner loads data/clpeak_icon.png as its window icon (GTK
        # has no .rc-style resource embedding, see
        # linux/runner/my_application.cc).
        add_custom_target(clpeak-gui ALL
            DEPENDS clpeak_ffi clpeak-engine
            COMMAND ${CMAKE_COMMAND} -E env DESTDIR= ${FLUTTER_EXECUTABLE} build linux --release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${CLPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CLPEAK_APP_DIR}/build/linux/${_clpeak_flutter_arch}/release/bundle" "${CLPEAK_GUI_OUT}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:clpeak_ffi>" "${CLPEAK_GUI_OUT}/lib/"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:clpeak-engine>" "${CLPEAK_GUI_OUT}/"
            WORKING_DIRECTORY "${CLPEAK_APP_DIR}"
            COMMENT "clpeak-gui: flutter build linux + bundle assembly"
            VERBATIM)
    endif()

    # ---- Install rules (feed the CPack zip) --------------------------------
    # A Flutter desktop app is a bundle, not a lone executable: the runner only
    # starts if its sibling data/ (icudtl + assets) and lib/ (engine, AOT app,
    # clpeak_ffi) travel with it.  So the bundle ships whole under gui/ and
    # bin/clpeak-gui is a wrapper that exec's it -- giving the release layout
    #
    #   bin/clpeak, bin/clpeak-gui  ->  gui/{clpeak-gui, data/, lib/}
    #
    # On macOS the equivalent is a relocatable .app, which goes to the package
    # root (a .app under bin/ would be neither clickable nor tidy).
    include(GNUInstallDirs)
    if(APPLE)
        # ditto, not install(DIRECTORY): the .app holds framework version
        # symlinks and an ad-hoc code signature, and file(INSTALL) is not
        # guaranteed to keep the seal intact.
        install(CODE "
            set(_src \"${CLPEAK_GUI_OUT}/clpeak-gui.app\")
            set(_dst \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/clpeak-gui.app\")
            message(STATUS \"Installing: \${_dst}\")
            file(REMOVE_RECURSE \"\${_dst}\")
            execute_process(COMMAND ditto \"\${_src}\" \"\${_dst}\"
                            RESULT_VARIABLE _clpeak_ditto_rc)
            if(NOT _clpeak_ditto_rc EQUAL 0)
                message(FATAL_ERROR \"ditto of clpeak-gui.app failed (\${_clpeak_ditto_rc}) -- build the clpeak-gui target first\")
            endif()
        ")
    else()
        if(WIN32)
            # `start` so a console invocation isn't blocked by the GUI process.
            set(_clpeak_gui_wrapper "${CMAKE_BINARY_DIR}/gui-wrapper/clpeak-gui.cmd")
            file(WRITE "${_clpeak_gui_wrapper}"
"@echo off\r\n\
rem clpeak GUI launcher -- the Flutter bundle in ..\\gui needs its sibling\r\n\
rem data\\ and DLLs, so it stays there and this wrapper starts it.\r\n\
start \"\" \"%~dp0..\\gui\\clpeak-gui.exe\" %*\r\n")
        else()
            set(_clpeak_gui_wrapper "${CMAKE_BINARY_DIR}/gui-wrapper/clpeak-gui")
            file(WRITE "${_clpeak_gui_wrapper}"
"#!/bin/sh
# clpeak GUI launcher -- the Flutter bundle in ../gui needs its sibling data/
# and lib/, so it stays there and this wrapper exec's it.
here=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)
exec \"$here/../gui/clpeak-gui\" \"$@\"
")
        endif()
        # Say plainly when the bundle was never staged: file(INSTALL)'s own
        # message for a missing source directory is "cannot find <dir>: File
        # exists", which reads like a collision.
        install(CODE "
            if(NOT IS_DIRECTORY \"${CLPEAK_GUI_OUT}\")
                message(FATAL_ERROR
                    \"clpeak-gui bundle missing at ${CLPEAK_GUI_OUT} -- \"
                    \"build the clpeak-gui target before packaging.\")
            endif()
        ")
        install(DIRECTORY "${CLPEAK_GUI_OUT}/" DESTINATION "gui"
                USE_SOURCE_PERMISSIONS)
        install(PROGRAMS "${_clpeak_gui_wrapper}"
                DESTINATION "${CMAKE_INSTALL_BINDIR}")
    endif()
endif()
