# cmake/version.cmake
# Derives CLPEAK_VERSION_STR from git-describe and generates version.h from it.
# Reports "unknown" when git-describe can't run (tarball build, no .git, no git
# binary): a hardcoded number there is worse than no number, because it silently
# stamps every such build with a version it isn't.
#
# A builder that knows the version for certain can pass it in instead, with
# -DCLPEAK_VERSION_OVERRIDE=<version>.  This is the distribution-packaging case:
# the source is the release tarball of exactly that tag, so the number is a fact
# the builder holds and git does not.  It is not the stale in-tree fallback this
# file used to carry -- nothing in the repository supplies a value, so a build
# that does not set it still reports "unknown".
#
# Derived once, at configure time, never during the build: the GUI build rewrites
# tracked files (Flutter owns pubspec.lock, analysis_options.yaml and the
# generated plugin registrants), which anything re-deriving later reads as
# "-dirty".  Re-run `cmake -B build` after committing to refresh the string (the
# configure summary prints it).
#
# Callers may set CLPEAK_GIT_ROOT before including this file to override the
# git working directory; it otherwise resolves to the repo root relative to
# this file, which is correct for the mobile superprojects too (their
# CMAKE_SOURCE_DIR points deep into the NDK/Xcode project tree).

set(_CLPEAK_VERSION_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(CLPEAK_VERSION_UNKNOWN "unknown")

if(NOT DEFINED CLPEAK_GIT_ROOT)
    get_filename_component(CLPEAK_GIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()

set(CLPEAK_VERSION_STR "${CLPEAK_VERSION_UNKNOWN}")

if(CLPEAK_VERSION_OVERRIDE)
    # Strip optional leading 'v' as the git path does, so both spellings of a
    # tag give the same string.
    string(REGEX REPLACE "^v" "" CLPEAK_VERSION_STR "${CLPEAK_VERSION_OVERRIDE}")
else()
    find_package(Git QUIET)

    if(GIT_FOUND AND EXISTS "${CLPEAK_GIT_ROOT}/.git")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
            WORKING_DIRECTORY ${CLPEAK_GIT_ROOT}
            OUTPUT_VARIABLE _git_version
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _git_result
        )
        if(_git_result EQUAL 0 AND NOT _git_version STREQUAL "")
            # Strip optional leading 'v' (tags mix v1.0 and 1.1.7)
            string(REGEX REPLACE "^v" "" _git_version "${_git_version}")
            set(CLPEAK_VERSION_STR "${_git_version}")
        endif()
    endif()
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
configure_file(
  "${_CLPEAK_VERSION_CMAKE_DIR}/version.h.in"
  "${CMAKE_BINARY_DIR}/generated/version.h"
  @ONLY
)

# Give <target> access to the generated version.h.
function(clpeak_setup_version target)
  target_include_directories(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()
