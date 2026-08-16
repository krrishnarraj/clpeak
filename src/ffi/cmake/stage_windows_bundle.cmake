# stage_windows_bundle.cmake — copy Flutter's Windows runner bundle into the
# clpeak-gui staging dir, without hard-coding the arch directory.
#
#   cmake -DAPP_DIR=<app> -DOUT_DIR=<build>/clpeak-gui -DARCH=<x64|arm64> \
#         -P stage_windows_bundle.cmake
#
# Flutter emits build/windows/<arch>/runner/Release, where <arch> follows the
# host (x64, arm64).  Which tag a given SDK picks on an arm64 host has moved
# around between releases, so prefer the expected one and fall back to whatever
# single Release dir exists rather than failing on a name mismatch.
cmake_minimum_required(VERSION 3.16)

set(_expected "${APP_DIR}/build/windows/${ARCH}/runner/Release")
if(EXISTS "${_expected}")
    set(_src "${_expected}")
else()
    file(GLOB _candidates "${APP_DIR}/build/windows/*/runner/Release")
    list(LENGTH _candidates _n)
    if(_n EQUAL 0)
        message(FATAL_ERROR
            "flutter build windows produced no runner/Release directory under "
            "${APP_DIR}/build/windows (expected ${_expected}).")
    endif()
    list(GET _candidates 0 _src)
    message(STATUS "clpeak-gui: ${_expected} absent, using ${_src}")
endif()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")
file(COPY "${_src}/" DESTINATION "${OUT_DIR}")
