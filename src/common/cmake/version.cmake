# cmake/version.cmake
# Derives CLPEAK_VERSION_STR from git-describe at build time.
# Falls back to a hardcoded version when building from a tarball (no .git).
#
# Callers may set CLPEAK_GIT_ROOT before including this file to override the
# git working directory (e.g., Android builds where CMAKE_SOURCE_DIR
# points deep into the NDK project tree).  When not set, defaults to
# CMAKE_SOURCE_DIR.

set(_CLPEAK_VERSION_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(CLPEAK_VERSION_FALLBACK "2.0.16")

# Allow callers to override the git root.
if(NOT DEFINED CLPEAK_GIT_ROOT)
    set(CLPEAK_GIT_ROOT "${CMAKE_SOURCE_DIR}")
endif()

find_package(Git QUIET)
include("${_CLPEAK_VERSION_CMAKE_DIR}/GitVersion.cmake")

# --- Configure-time: seed an initial version.h so the tree always has one ---
clpeak_git_describe(CLPEAK_VERSION_STR
    "${GIT_EXECUTABLE}" "${CLPEAK_GIT_ROOT}" "${CLPEAK_VERSION_FALLBACK}")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
configure_file(
  "${_CLPEAK_VERSION_CMAKE_DIR}/version.h.in"
  "${CMAKE_BINARY_DIR}/generated/version.h"
  @ONLY
)

# --- Build-time: regenerate version.h on every build (write-if-different) ---
# The GenVersion.cmake script only overwrites the file when the string changes,
# so only options.cpp recompiles when the version actually changes.
#
# GIT_ROOT keyword: optional override for the git root used at build time.
# Defaults to CMAKE_SOURCE_DIR.  Android builds pass the real repo root.
function(clpeak_setup_version target)
  set(_options "")
  set(_oneValueArgs GIT_ROOT)
  set(_multiValueArgs "")
  cmake_parse_arguments(_ver "" "${_oneValueArgs}" "${_multiValueArgs}" ${ARGN})

  if(NOT DEFINED _ver_GIT_ROOT)
    set(_ver_GIT_ROOT "${CMAKE_SOURCE_DIR}")
  endif()

  if(NOT TARGET clpeak_version_gen)
    add_custom_target(clpeak_version_gen ALL
      COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=${_ver_GIT_ROOT}
        -DBINARY_DIR=${CMAKE_BINARY_DIR}
        -DTEMPLATE_DIR=${_CLPEAK_VERSION_CMAKE_DIR}
        -DFALLBACK=${CLPEAK_VERSION_FALLBACK}
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -P ${_CLPEAK_VERSION_CMAKE_DIR}/GenVersion.cmake
      WORKING_DIRECTORY ${_ver_GIT_ROOT}
      COMMENT "Checking clpeak version..."
    )
  endif()

  add_dependencies(${target} clpeak_version_gen)
  target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()
